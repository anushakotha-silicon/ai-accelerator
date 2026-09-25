"""End-to-end serving scenarios: chat inference and agentic episodes."""
import math
from dataclasses import dataclass, replace

from .engine import best_decode_step, max_resident_seqs, prefill, weights_gb
from .hw import DTYPE_BYTES, Chip, Serving
from .workloads import AgentProfile, ChatProfile, Model


@dataclass
class ChatResult:
    batch: int
    feasible: bool
    ttft_ms: float
    tpot_ms: float            # time per output token seen by one user
    tokens_per_s: float       # chip-wide generated tokens / s
    j_per_token: float
    avg_power_w: float
    bound: str
    spec_k: int = 0

    @property
    def tokens_per_j(self) -> float:
        return 1.0 / self.j_per_token


def serve_chat(model: Model, chip: Chip, srv: Serving, prof: ChatProfile, batch: int) -> ChatResult:
    """Steady-state continuous batching of identical chat requests."""
    ctx_avg = prof.prompt_tokens + prof.output_tokens / 2
    feasible = batch <= max_resident_seqs(model, chip, srv, prof.prompt_tokens + prof.output_tokens)
    pf = prefill(model, chip, srv, prof.prompt_tokens)
    srv, step = best_decode_step(model, chip, replace(srv, spec_accept=prof.spec_accept), batch, ctx_avg)
    steps = math.ceil(prof.output_tokens / srv.tokens_per_step())

    # chip time and energy attributable to one request
    chip_time = pf.time_s + steps * step.time_s / batch
    energy = pf.total_j + steps * step.total_j / batch
    return ChatResult(
        batch=batch,
        feasible=feasible,
        ttft_ms=pf.time_s * 1e3,
        tpot_ms=step.time_s / srv.tokens_per_step() * 1e3,
        tokens_per_s=prof.output_tokens / chip_time,
        j_per_token=energy / prof.output_tokens,
        avg_power_w=energy / chip_time,
        bound=step.bound,
        spec_k=srv.spec_k,
    )


# --------------------------------------------------------------------------
# Agentic episodes
# --------------------------------------------------------------------------
POLICIES = {
    "recompute": "No KV reuse: every turn re-prefills the whole session (shared prefix cached)",
    "hbm-cache": "Session KV stays resident in HBM across tool calls",
    "tiered-kv": "KV parked in LPDDR during tool calls, restored by DMA before next turn",
}


@dataclass
class AgentResult:
    policy: str
    concurrent_agents: float
    decode_batch: float
    turns_per_s: float
    turn_latency_s: float
    j_per_turn: float
    gen_tokens_per_j: float
    avg_power_w: float
    utilization: float
    limiter: str
    spec_k: int = 0


def run_agents(model: Model, chip: Chip, srv: Serving, prof: AgentProfile, policy: str) -> AgentResult:
    if policy not in POLICIES:
        raise ValueError(policy)
    if policy == "tiered-kv" and chip.lpddr_gb <= 0:
        raise ValueError("tiered-kv needs an LPDDR tier")
    srv = replace(srv, spec_accept=prof.spec_accept) if srv.spec_k else srv
    kvb = model.kv_bytes_per_token(DTYPE_BYTES[srv.kv_dtype])
    e = chip.energy

    # ---- per-turn prefill (averaged over the episode) ----
    pf_time = pf_dyn = xfer_lat = xfer_j = 0.0
    for i in range(prof.turns):
        ctx = prof.ctx_at_turn(i)
        new = prof.new_tokens_at_turn(i)
        if policy == "recompute":
            c = prefill(model, chip, srv, ctx - prof.shared_prefix, prof.shared_prefix)
        else:
            c = prefill(model, chip, srv, new, ctx - new)
        pf_time += c.time_s
        pf_dyn += c.dynamic_j
        if policy == "tiered-kv" and i > 0:
            restore = (ctx - new) * kvb                      # LPDDR -> HBM
            park = (prof.gen_tokens + prof.tool_output_tokens) * kvb  # HBM -> LPDDR (incremental)
            xfer_lat += restore / chip.lpddr_bw
            moved = restore + park
            xfer_j += moved * (e.lpddr_pj_per_byte + e.hbm_pj_per_byte) * 1e-12
    n = prof.turns
    pf_time, pf_dyn, xfer_lat, xfer_j = pf_time / n, pf_dyn / n, xfer_lat / n, xfer_j / n

    # average live session context and the context decode attends over
    ctx_session = sum(prof.ctx_at_turn(i) + prof.gen_tokens / 2 for i in range(n)) / n
    per_agent_gb = (ctx_session - prof.shared_prefix) * kvb / 1e9
    shared_gb = prof.shared_prefix * kvb / 1e9
    free_hbm = chip.hbm_gb * 0.92 - weights_gb(model, srv) - shared_gb
    steps = math.ceil(prof.gen_tokens / srv.tokens_per_step())

    # ---- choose the decode batch B ----
    # When the chip is saturated, each turn costs chip_time(B) seconds of chip,
    # so turns/s = 1/chip_time(B). Decode wall time follows from Little's law:
    # B agents decode concurrently, so each waits B * chip_time(B).
    # Larger B -> more throughput, less J/turn, but more agents holding KV and
    # longer turns. Take the largest B that fits memory and the latency SLO.
    feasible, fail = [], None
    for b in range(1, 1025):
        s_used, step = best_decode_step(model, chip, srv, b, ctx_session)
        steps = math.ceil(prof.gen_tokens / s_used.tokens_per_step())
        chip_time = pf_time + steps * step.time_s / b
        tps = 1.0 / chip_time
        dec_lat = max(steps * step.time_s, b * chip_time - pf_time)
        turn_lat = pf_time + dec_lat + xfer_lat
        cycle = turn_lat + prof.tool_latency_s
        agents = tps * cycle
        duty = turn_lat / cycle
        # KV that must sit in HBM: every agent's (hbm-cache) or only running agents'
        hbm_need = agents * per_agent_gb * (1.0 if policy == "hbm-cache" else duty)
        lp_need = agents * per_agent_gb if policy == "tiered-kv" else 0.0
        if turn_lat > prof.max_turn_latency_s:
            fail = "latency SLO"
        elif hbm_need > free_hbm:
            fail = "HBM capacity"
        elif lp_need > chip.lpddr_gb:
            fail = "LPDDR capacity"
        if fail:
            break
        feasible.append((b, s_used, step, steps, chip_time, tps, turn_lat, agents))

    util = 1.0
    if feasible:
        # max throughput, then the lowest-latency batch within 1% of it
        top = max(f[5] for f in feasible)
        pick = min((f for f in feasible if f[5] >= 0.99 * top), key=lambda f: f[6])
        b, s_used, step, steps, chip_time, turns_per_s, turn_lat, agents = pick
        limiter = fail or "batch cap"
        if pick is not feasible[-1]:
            limiter = "compute (saturated)"
    else:
        b, s_used, step, steps, chip_time, turns_per_s, turn_lat, agents = (
            1, s_used, step, steps, chip_time, tps, turn_lat, agents)
        if fail == "latency SLO":
            limiter = "SLO missed even at B=1"
        else:
            # not enough memory to saturate the chip even at B=1: run below saturation
            limiter = fail
            cycle = turn_lat + prof.tool_latency_s
            cap = free_hbm / per_agent_gb
            agents = cap if policy == "hbm-cache" else cap * cycle / turn_lat
            if policy == "tiered-kv":
                agents = min(agents, chip.lpddr_gb / per_agent_gb)
            turns_per_s = agents / cycle
            util = turns_per_s * chip_time

    dyn_per_turn = pf_dyn + steps * step.dynamic_j / b + xfer_j
    j_per_turn = dyn_per_turn + chip.static_w / turns_per_s
    return AgentResult(
        policy=policy,
        concurrent_agents=agents,
        decode_batch=b,
        turns_per_s=turns_per_s,
        turn_latency_s=turn_lat,
        j_per_turn=j_per_turn,
        gen_tokens_per_j=prof.gen_tokens / j_per_turn,
        avg_power_w=j_per_turn * turns_per_s,
        utilization=min(1.0, util),
        limiter=limiter,
        spec_k=s_used.spec_k,
    )

