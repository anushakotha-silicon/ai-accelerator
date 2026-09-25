"""Generate results/report.md: chip summary, chat + agentic results, ablations, sweeps.

    python3 -m model.run
"""
import os
from dataclasses import replace

from .engine import best_decode_step, decode_step, prefill, weights_gb
from .hw import Chip, Serving
from .scenarios import POLICIES, run_agents, serve_chat
from .workloads import LLAMA_8B, LLAMA_70B, MODELS, AgentProfile, ChatProfile

BASE_CHIP = Chip()
BASE_SRV = Serving(spec_k=4)
CHAT = ChatProfile()
CODING = AgentProfile()
RESEARCH = AgentProfile(name="research-agent", shared_prefix=12000, task_tokens=1000, turns=40,
                        tool_output_tokens=3000, gen_tokens=250, tool_latency_s=20.0,
                        spec_accept=0.6, max_turn_latency_s=15.0)

ABLATIONS = [
    ("baseline", BASE_CHIP, BASE_SRV),
    ("FP8 weights (no FP4)", BASE_CHIP, replace(BASE_SRV, weight_dtype="fp8")),
    ("BF16 weights+compute+KV", BASE_CHIP, replace(BASE_SRV, weight_dtype="bf16", compute_dtype="bf16", kv_dtype="bf16")),
    ("BF16 KV cache", BASE_CHIP, replace(BASE_SRV, kv_dtype="bf16")),
    ("no speculative decode", BASE_CHIP, replace(BASE_SRV, spec_k=0)),
    ("narrow weight port (128 B/cyc)", replace(BASE_CHIP, weight_load_bytes_per_cycle=128), BASE_SRV),
    ("no LPDDR tier", replace(BASE_CHIP, lpddr_gb=0), BASE_SRV),
    ("+ FP4 KV cache", BASE_CHIP, replace(BASE_SRV, kv_dtype="fp4")),
    ("+ 512 GB LPDDR tier", replace(BASE_CHIP, lpddr_gb=512), BASE_SRV),
]


def _t(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def chip_summary(c: Chip) -> str:
    rows = [
        ("Compute tiles", f"{c.tiles} x ({c.array_dim}x{c.array_dim} systolic array + attention/vector engine)"),
        ("Clock", f"{c.clock_ghz} GHz"),
        ("Peak dense FP4 / FP8 / BF16", f"{c.peak_tflops('fp4'):.0f} / {c.peak_tflops('fp8'):.0f} / {c.peak_tflops('bf16'):.0f} TFLOPS"),
        ("On-chip SRAM", f"{c.sram_mb:.0f} MB ({c.sram_mb_per_tile:.0f} MB/tile)"),
        ("HBM3E", f"{c.hbm_gb:.0f} GB @ {c.hbm_tbps} TB/s"),
        ("LPDDR5X capacity tier", f"{c.lpddr_gb:.0f} GB @ {c.lpddr_tbps} TB/s"),
        ("Array weight-feed BW", f"{c.tiles * c.weight_load_bytes_per_cycle * c.clock_hz / 1e12:.1f} TB/s (must exceed HBM BW)"),
        ("Ridge point (FP8)", f"{c.peak_macs('fp8') * 2 / (c.hbm_tbps * 1e12):.0f} FLOP/byte"),
        ("Static power / TDP", f"{c.static_w:.0f} W / {c.tdp_w:.0f} W"),
    ]
    return _t(rows, ["Parameter", "Value"])


def chat_table() -> str:
    rows = []
    for m in MODELS:
        for b in (1, 8, 32, 64, 128, 256):
            r = serve_chat(m, BASE_CHIP, BASE_SRV, CHAT, b)
            if not r.feasible:
                continue
            rows.append((m.name, b, f"{r.ttft_ms:.0f}", f"{r.tpot_ms:.2f}", f"{1000 / r.tpot_ms:.0f}",
                         f"{r.tokens_per_s:,.0f}", f"{r.tokens_per_j:.1f}", f"{r.avg_power_w:.0f}", r.spec_k, r.bound))
    return _t(rows, ["Model", "Batch", "TTFT ms", "ms/token", "tok/s/user", "tok/s chip",
                     "tok/J", "Avg W", "draft k", "Decode bound"])


def energy_breakdown() -> str:
    rows = []
    cases = [
        ("8B decode, B=1", lambda: decode_step(LLAMA_8B, BASE_CHIP, replace(BASE_SRV, spec_k=0), 1, 1280)),
        ("8B decode, B=128", lambda: decode_step(LLAMA_8B, BASE_CHIP, replace(BASE_SRV, spec_k=0), 128, 1280)),
        ("70B decode, B=64", lambda: decode_step(LLAMA_70B, BASE_CHIP, replace(BASE_SRV, spec_k=0), 64, 1280)),
        ("70B prefill 4k", lambda: prefill(LLAMA_70B, BASE_CHIP, BASE_SRV, 4096)),
        ("70B agent-turn prefill (1.2k new @ 26k ctx)", lambda: prefill(LLAMA_70B, BASE_CHIP, BASE_SRV, 1200, 26000)),
    ]
    keys = ["mac", "attn", "vector", "sram", "noc", "hbm", "static"]
    for name, fn in cases:
        c = fn()
        tot = c.total_j
        rows.append([name] + [f"{100 * c.energy.get(k, 0) / tot:.0f}%" for k in keys] + [f"{tot * 1e3:.1f} mJ", c.bound])
    return _t(rows, ["Case"] + keys + ["Total", "Bound"])


def ablation_table() -> str:
    rows = []
    base = None
    for name, chip, srv in ABLATIONS:
        c8 = serve_chat(LLAMA_8B, chip, srv, CHAT, 64)
        c70 = serve_chat(LLAMA_70B, chip, srv, CHAT, 64)
        pol = "tiered-kv" if chip.lpddr_gb > 0 else "hbm-cache"
        a70 = run_agents(LLAMA_70B, chip, srv, CODING, pol)
        ar = run_agents(LLAMA_8B, chip, srv, RESEARCH, pol)
        vals = (c8.tokens_per_j, c70.tokens_per_j, a70.gen_tokens_per_j, ar.turns_per_s)
        if base is None:
            base = vals
        cells = [f"{v:.2f} ({v / b:.2f}x)" for v, b in zip(vals, base)]
        if not c70.feasible:
            cells[1] = "doesn't fit"
        if a70.limiter == "weights don't fit":
            cells[2] = "doesn't fit"
        rows.append((name, *cells))
    return _t(rows, ["Variant", "Chat 8B B=64 tok/J", "Chat 70B B=64 tok/J",
                     "Coding agent 70B tok/J", "Research agent 8B turns/s"])


def agent_table(prof: AgentProfile) -> str:
    rows = []
    for m in MODELS:
        for p in POLICIES:
            r = run_agents(m, BASE_CHIP, BASE_SRV, prof, p)
            rows.append((m.name, p, f"{r.concurrent_agents:.0f}", r.decode_batch, f"{r.turns_per_s:.2f}",
                         f"{r.turn_latency_s:.1f}", f"{r.j_per_turn:.0f}", f"{r.gen_tokens_per_j:.2f}",
                         f"{r.avg_power_w:.0f}", f"{r.utilization:.2f}", r.limiter))
    return _t(rows, ["Model", "KV policy", "Agents", "Decode B", "Turns/s", "Turn lat s",
                     "J/turn", "gen tok/J", "Avg W", "Util", "Limited by"])


def sweep_table() -> str:
    rows = []
    for tiles in (16, 32, 64):
        for bw in (3.2, 4.8, 6.4, 8.0):
            chip = replace(BASE_CHIP, tiles=tiles, hbm_tbps=bw)  # static scales with tile count
            c = serve_chat(LLAMA_70B, chip, BASE_SRV, CHAT, 64)
            a = run_agents(LLAMA_70B, chip, BASE_SRV, CODING, "tiered-kv")
            rows.append((tiles, f"{chip.peak_tflops('fp8'):.0f}", bw,
                         f"{c.tokens_per_j:.1f}", f"{c.tpot_ms:.1f}",
                         f"{a.turns_per_s:.2f}", f"{a.gen_tokens_per_j:.2f}", f"{a.avg_power_w:.0f}"))
    return _t(rows, ["Tiles", "FP8 TFLOPS", "HBM TB/s", "Chat 70B tok/J", "Chat ms/token",
                     "Agent 70B turns/s", "Agent tok/J", "Agent avg W"])


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parts = [
        "# Model results\n",
        "_Generated by `python3 -m model.run`. Analytical model, not silicon: see docs/ARCHITECTURE.md for assumptions._\n",
        "## 1. Baseline chip\n", chip_summary(BASE_CHIP), "",
        f"Weights resident: 8B = {weights_gb(LLAMA_8B, BASE_SRV):.1f} GB, 70B = {weights_gb(LLAMA_70B, BASE_SRV):.1f} GB (MXFP4).\n",
        "## 2. Chat inference (1024 in / 512 out, continuous batching)\n", chat_table(), "",
        "## 3. Where the energy goes\n", energy_breakdown(), "",
        "## 4. Agentic: coding agent\n",
        f"Profile: {CODING.turns} turns, {CODING.tool_output_tokens} tool-output tokens and {CODING.gen_tokens} generated "
        f"tokens per turn, {CODING.shared_prefix}-token shared prefix, {CODING.tool_latency_s}s tool latency, "
        f"final context {CODING.final_ctx:,} tokens, turn SLO {CODING.max_turn_latency_s}s.\n",
        agent_table(CODING), "",
        "## 5. Agentic: research agent (long context, slow tools)\n",
        f"Profile: {RESEARCH.turns} turns, {RESEARCH.tool_output_tokens} tool-output tokens per turn, "
        f"{RESEARCH.tool_latency_s}s tool latency, final context {RESEARCH.final_ctx:,} tokens, turn SLO {RESEARCH.max_turn_latency_s}s.\n",
        agent_table(RESEARCH), "",
        "## 6. Ablations (remove one feature at a time)\n", ablation_table(), "",
        "## 7. Design sweep: compute vs bandwidth (70B)\n", sweep_table(), "",
        "Policy definitions: " + "; ".join(f"**{k}**: {v}" for k, v in POLICIES.items()) + "\n",
    ]
    path = os.path.join(here, "results", "report.md")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    print(open(path).read())


if __name__ == "__main__":
    main()
