"""Analytical time/energy cost of transformer inference on a Chip.

Execution model
---------------
* Matmuls run on weight-stationary systolic arrays with double-buffered weight
  loads. Each array_dim^2 weight block costs max(stream cycles, load cycles),
  which captures why decode (few tokens per block) is bounded by how fast
  weights can be pushed into the arrays, not by MAC count.
* Decode attention runs on the per-tile attention/vector engine (GEMV-shaped,
  would waste a 128x128 array). Prefill attention runs on the arrays
  (flash-attention style; one K/V layer fits in the distributed scratchpad).
* HBM traffic, array time and attention time overlap; a step takes the max of
  the three plus a per-layer synchronisation cost.
"""
import math
from dataclasses import dataclass, field, replace

from .hw import DTYPE_BYTES, DTYPE_RATE, Chip, Serving
from .workloads import Model


@dataclass
class Cost:
    time_s: float = 0.0
    energy: dict = field(default_factory=dict)   # joules per component
    bound: str = ""

    @property
    def total_j(self) -> float:
        return sum(self.energy.values())

    @property
    def dynamic_j(self) -> float:
        return self.total_j - self.energy.get("static", 0.0)

    def scaled(self, k: float) -> "Cost":
        return Cost(self.time_s * k, {n: e * k for n, e in self.energy.items()}, self.bound)

    def __add__(self, other: "Cost") -> "Cost":
        e = dict(self.energy)
        for n, v in other.energy.items():
            e[n] = e.get(n, 0.0) + v
        return Cost(self.time_s + other.time_s, e, self.bound or other.bound)


def _matmul_time(chip: Chip, srv: Serving, weight_params: float, m_tokens: float) -> float:
    """Seconds to multiply `m_tokens` activations by `weight_params` weights."""
    rate = DTYPE_RATE[srv.compute_dtype]
    block_weights = chip.array_dim ** 2 * rate
    n_blocks = weight_params / block_weights
    load_cycles = block_weights * DTYPE_BYTES[srv.weight_dtype] / chip.weight_load_bytes_per_cycle
    cycles_per_block = max(m_tokens, load_cycles)
    return n_blocks * cycles_per_block / (chip.tiles * chip.clock_hz * chip.mm_eff)


def _energy(chip: Chip, srv: Serving, *, array_macs=0.0, vec_macs=0.0, vec_ops=0.0,
            hbm_bytes=0.0, act_bytes=0.0, time_s=0.0) -> dict:
    e = chip.energy
    pj = 1e-12
    # Every DRAM byte is written into and read out of scratchpad once; array
    # operands are re-read from scratchpad once per array_dim of reuse.
    sram_bytes = 2 * hbm_bytes + 3 * array_macs / chip.array_dim + vec_macs / 8
    noc_bytes = (hbm_bytes + act_bytes) * chip.noc_hops
    return {
        "mac": array_macs * e.mac[srv.compute_dtype] * pj,
        "attn": vec_macs * e.vec_mac_pj * pj,
        "vector": vec_ops * e.vec_op_pj * pj,
        "sram": sram_bytes * e.sram_pj_per_byte * pj,
        "noc": noc_bytes * e.noc_pj_per_byte_hop * pj,
        "hbm": hbm_bytes * e.hbm_pj_per_byte * pj,
        "static": chip.static_w * time_s,
    }


def _vector_ops_per_token(model: Model, ctx: float) -> float:
    # norms, residuals, rope, activation fn, softmax(exp+sum+scale), sampling
    per_layer = 12 * model.d_model + 2 * model.top_k * model.d_ff + 4 * model.n_heads * ctx
    return model.layers * per_layer + 3 * model.vocab


def decode_step(model: Model, chip: Chip, srv: Serving, batch: int, ctx: float) -> Cost:
    """One decode iteration for `batch` sequences at context length `ctx`.

    With speculative decoding each sequence verifies spec_k+1 positions per
    step; Serving.tokens_per_step() gives the expected accepted tokens.
    """
    q = 1 + srv.spec_k
    m = batch * q                                         # rows through the arrays
    touched = model.experts_touched(m)
    per_expert_m = m * model.top_k / touched

    # array time: always-active weights see all m rows; each expert sees its share
    t_mm = _matmul_time(chip, srv, model.always_active_params, m)
    t_mm += _matmul_time(chip, srv, model.layers * touched * model.expert_params_per_layer, per_expert_m)
    array_macs = m * model.active_params_per_token

    # attention on the vector engine; KV is read once per step and shared by the q positions
    vec_macs = m * model.attn_macs_per_query(ctx)
    t_attn = vec_macs / (chip.tiles * chip.vec_macs_per_cycle * chip.clock_hz * 0.8)
    vec_ops = m * _vector_ops_per_token(model, ctx)
    t_vec = vec_ops / (chip.tiles * chip.vec_ops_per_cycle * chip.clock_hz * 0.8)

    weight_bytes = model.weight_params_touched(m) * DTYPE_BYTES[srv.weight_dtype]
    kv_read = batch * ctx * model.kv_bytes_per_token(DTYPE_BYTES[srv.kv_dtype])
    kv_write = m * model.kv_bytes_per_token(DTYPE_BYTES[srv.kv_dtype])
    hbm_bytes = weight_bytes + kv_read + kv_write
    t_mem = hbm_bytes / chip.hbm_bw

    parts = {"memory": t_mem, "array": t_mm, "attention": t_attn + t_vec}
    bound = max(parts, key=parts.get)
    t = parts[bound] + model.layers * chip.layer_sync_us * 1e-6

    act_bytes = model.layers * m * model.d_model * 4
    energy = _energy(chip, srv, array_macs=array_macs, vec_macs=vec_macs, vec_ops=vec_ops,
                     hbm_bytes=hbm_bytes, act_bytes=act_bytes, time_s=t)
    return Cost(t, energy, bound)


def best_decode_step(model: Model, chip: Chip, srv: Serving, batch: int, ctx: float):
    """Pick the draft length (0..srv.spec_k) with the lowest energy per accepted token.

    Speculation is nearly free when decode is memory-bound and wasteful once
    the arrays saturate, so the on-chip scheduler adapts k to the live batch.
    The objective is total joules (static power x time included), so latency
    still counts, but rejected-draft MACs are no longer treated as free.
    """
    best = None
    for k in range(srv.spec_k + 1):
        s = replace(srv, spec_k=k)
        c = decode_step(model, chip, s, batch, ctx)
        score = c.total_j / s.tokens_per_step()
        if best is None or score < best[0] - 1e-12:
            best = (score, s, c)
    return best[1], best[2]


def prefill(model: Model, chip: Chip, srv: Serving, new_tokens: int, past_ctx: int = 0) -> Cost:
    """Process `new_tokens` prompt tokens appended to `past_ctx` cached tokens."""
    if new_tokens <= 0:
        return Cost(0.0, {}, "none")
    chunks = math.ceil(new_tokens / srv.prefill_chunk)
    chunk = new_tokens / chunks
    kvb = model.kv_bytes_per_token(DTYPE_BYTES[srv.kv_dtype])

    t_mm = chunks * _matmul_time(chip, srv, model.always_active_params, chunk)
    touched = model.experts_touched(chunk)
    t_mm += chunks * _matmul_time(chip, srv, model.layers * touched * model.expert_params_per_layer,
                                  chunk * model.top_k / touched)
    array_macs = new_tokens * model.active_params_per_token

    # causal attention over past + new tokens, on the arrays at ~70% efficiency
    key_pairs = new_tokens * past_ctx + new_tokens * (new_tokens + 1) / 2
    attn_macs = model.attn_macs_per_query(1) * key_pairs
    t_mm += attn_macs / (chip.peak_macs(srv.compute_dtype) * 0.7)
    exps = model.layers * model.n_heads * key_pairs
    vec_ops = new_tokens * _vector_ops_per_token(model, 0) + 4 * exps
    t_vec = vec_ops / (chip.tiles * chip.vec_ops_per_cycle * chip.clock_hz * 0.8)

    weight_bytes = chunks * model.weight_params_touched(chunk) * DTYPE_BYTES[srv.weight_dtype]
    # each chunk re-reads the KV that precedes it (one layer at a time fits in SRAM)
    kv_read = sum((past_ctx + i * chunk) * kvb for i in range(chunks))
    hbm_bytes = weight_bytes + kv_read + new_tokens * kvb
    t_mem = hbm_bytes / chip.hbm_bw

    parts = {"memory": t_mem, "array": t_mm, "attention": t_vec}
    bound = max(parts, key=parts.get)
    t = parts[bound] + chunks * model.layers * chip.layer_sync_us * 1e-6

    act_bytes = model.layers * new_tokens * model.d_model * 4
    energy = _energy(chip, srv, array_macs=array_macs + attn_macs, vec_ops=vec_ops,
                     hbm_bytes=hbm_bytes, act_bytes=act_bytes, time_s=t)
    return Cost(t, energy, bound)


def weights_gb(model: Model, srv: Serving) -> float:
    return model.total_params * DTYPE_BYTES[srv.weight_dtype] / 1e9


def max_resident_seqs(model: Model, chip: Chip, srv: Serving, ctx: float, reserve: float = 0.08) -> int:
    free = chip.hbm_gb * (1 - reserve) - weights_gb(model, srv)
    per_seq = ctx * model.kv_bytes_per_token(DTYPE_BYTES[srv.kv_dtype]) / 1e9
    return max(0, int(free / per_seq))
