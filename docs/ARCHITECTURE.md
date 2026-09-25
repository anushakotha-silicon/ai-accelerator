# Inference accelerator: architecture spec (v0.1)

Status: **phase 1: architecture + analytical model.** Every number below comes from
`python3 -m model.run` (full tables in [`results/report.md`](../results/report.md)),
not from silicon. Energy assumptions are in section 8; change them in `model/hw.py`.

---

## 1. Goals and non-goals

| | |
|---|---|
| **Primary metric** | Generated tokens per joule (tok/J) at a stated latency target |
| **Workload 1** | LLM inference, 8B–70B dense and MoE, single chip |
| **Workload 2** | Agentic loops: long, growing contexts, tool-call pauses, structured outputs |
| **Power envelope** | 350 W TDP, air-coolable card |
| **Non-goals (v1)** | Training, FP64/HPC, graphics, models >~200B on one chip |

"All workloads" is achieved by composition: a fixed-function-heavy datapath for
the ~95% of FLOPs and bytes that are matmul/attention, and programmable RISC-V
cores for everything irregular (sampling, grammar masks, scheduling, tool I/O).

---

## 2. Workload mental model

```
                 ┌──────────── one agent turn ────────────┐
 tool result ──▶ │ PREFILL new tokens │ DECODE gen tokens │ ──▶ tool call ──▶ (tool runs, 3–20 s)
                 │  M = 1k–4k rows    │  M = batch × (k+1) │
                 │  compute-bound     │  memory-bound      │
                 └────────────────────┴────────────────────┘
                    KV cache grows every turn; it must survive the tool pause
```

**Arithmetic intensity** decides everything. The chip's ridge point is
`peak FLOPs / HBM bytes/s = 1258 TFLOPS / 4.8 TB/s = 262 FLOP/byte`.

| Phase | FLOPs per weight byte (FP4 weights) | Regime |
|---|---|---|
| Decode, batch B | ≈ 2·B / 0.53 = 3.8·B | memory-bound until B ≈ 70 |
| Prefill, chunk 4096 | ≈ 2·4096 / 0.53 ≈ 15,400 | compute-bound |

**KV cache bytes per token** = `2 · layers · kv_heads · head_dim · bytes`:
8B → 64 KB, 70B → 160 KB (FP8). A 100k-token agent session on 70B is 16 GB.
For agentic serving, **KV bytes per token is the unit of cost**, just as weight
bytes are for chat decode.

**Agentic vs chat, in one line:** chat is decode-bound (output ≈ input), agent
turns are prefill-bound (1,200 new input tokens : 300 generated), and both share
a capacity problem in the KV cache.

---

## 3. Top-level block diagram

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                               SoC die                                  │
 │  ┌─────────┐ ┌─────────┐                         ┌─────────┐           │
 │  │ Tile 0  │ │ Tile 1  │  ...  8 x 4 mesh  ...   │ Tile 31 │           │
 │  └────┬────┘ └────┬────┘                         └────┬────┘           │
 │       └───────────┴──────── 2D mesh NoC ──────────────┘                │
 │         (multicast for weight broadcast, reduction for all-reduce)     │
 │                                                                        │
 │  ┌──────────────────┐ ┌──────────────────┐ ┌─────────────────────────┐ │
 │  │ Control cluster  │ │ KV / Prefix      │ │ DMA engines (x8)        │ │
 │  │ 8x RISC-V RVA23  │ │ Manager (HW)     │ │ HBM<->SRAM, HBM<->LPDDR │ │
 │  │ scheduler,       │ │ page tables,     │ │ descriptor rings        │ │
 │  │ sampling, grammar│ │ prefix hash CAM, │ └─────────────────────────┘ │
 │  │ masks, tool I/O  │ │ park/restore     │                             │
 │  └──────────────────┘ └──────────────────┘                             │
 │  ┌────────────────────────────┐  ┌───────────────┐  ┌───────────────┐  │
 │  │ HBM3E PHY+ctrl x4          │  │ LPDDR5X ctrl  │  │ PCIe6 / CXL   │  │
 │  │ 144 GB, 4.8 TB/s           │  │ 256 GB 0.5TB/s│  │ + 8x chip link│  │
 │  └────────────────────────────┘  └───────────────┘  └───────────────┘  │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Compute tile

```
 ┌──────────────────────────── Tile (x32) ────────────────────────────┐
 │  Scratchpad SRAM 4 MB, 16 banks, software-managed (no cache tags)  │
 │     │ weight port 256 B/cyc         │ activation port              │
 │     ▼                               ▼                              │
 │  ┌─────────────────────────┐   ┌──────────────────────────────┐   │
 │  │ Systolic array 128x128  │   │ Attention / vector engine    │   │
 │  │ weight-stationary,      │   │ 1024 FP8 MAC/cyc (QK, AV)    │   │
 │  │ double-buffered weights │   │ 512 FP32 op/cyc (exp, norm,  │   │
 │  │ FP4xFP8 / FP8 / BF16    │   │ rope, softmax, dequant)      │   │
 │  │ MX block-scale dequant  │   │ online-softmax state         │   │
 │  └─────────────────────────┘   └──────────────────────────────┘   │
 │  Tile sequencer: runs layer micro-programs; clock-gates idle units │
 └────────────────────────────────────────────────────────────────────┘
```

Aggregate: 524,288 PEs at 1.2 GHz = **1,258 TFLOPS FP8 / 2,517 FP4 / 629 BF16**.

**Why the arrays are weight-stationary with a 256 B/cycle weight port.** Each
128×128 block of weights costs `max(M rows streamed, load cycles)`. With FP4
weights, `load = 16,384 × 0.53 B / 256 B = 34 cycles`. In decode, M = batch is
often < 34, so the arrays are limited by *weight feed*, not MACs.
Sizing rule: weight-feed bandwidth ≥ HBM bandwidth. The break-even port is
`4.08 TB/s / (32 tiles × 1.2 GHz × 0.85) = 125 B/cycle`. 128 B/cycle meets
today's HBM exactly (the ablation shows 0% loss); 256 B/cycle is 2× headroom for
an HBM4-class (~8 TB/s) respin.

**Why decode attention is on a separate engine.** Decode attention is
matrix-vector (per KV head: `group × ctx` against `ctx × 128`). On a 128×128
array that would use 4/128 of the rows. A narrow dot-product engine does it at
full utilisation; its throughput only has to beat KV bytes/s from HBM.

---

## 5. Memory hierarchy

| Level | Size | BW | Energy | Holds |
|---|---|---|---|---|
| PE registers | – | – | in MAC cost | stationary weights, partial sums |
| Tile scratchpad | 4 MB × 32 = 128 MB | ~40 TB/s aggregate | 0.8 pJ/B | weight tiles in flight, one layer of K/V during prefill |
| HBM3E | 144 GB | 4.8 TB/s | 30 pJ/B | weights, **active** KV |
| LPDDR5X | 256 GB | 0.5 TB/s | 40 pJ/B | **parked** KV of agents waiting on tools, prefix cache |
| Host / CXL | TBs | ~64 GB/s | – | cold sessions |

**Why a second DRAM tier instead of more HBM.** An agent waiting 20 s on a
tool doesn't need its KV at 4.8 TB/s; it needs it *somewhere*. Restoring 11 GB
of 70B KV from LPDDR takes 11e9 / 0.425e12 ≈ **26 ms**, while recomputing it
takes seconds of prefill. LPDDR costs roughly 1/4–1/5 of HBM per GB.

---

## 6. Agentic-specific features

1. **Hardware KV manager.** Paged KV (e.g. 16-token pages), page tables walked by
   the DMA engines, so attention kernels see a virtual contiguous KV.
2. **Prefix cache.** A hash CAM over page contents lets 8k-token system prompts and
   tool schemas be shared across all agents (stored once: 0.5 GB on 8B, 1.3 GB on 70B).
3. **Park / restore.** When an agent emits a tool call, its KV pages are demoted
   to LPDDR in the background (only pages added since the last park are written).
   A restore runs as soon as the tool result arrives, overlapping the new prefill.
4. **Adaptive speculative decoding.** Drafts come from prompt lookup (n-gram match
   against the context, no draft model), because tool calls copy paths and
   identifiers from context (assumed 60–70% acceptance). The scheduler picks draft
   length k per step to minimise *joules* per accepted token: k=3–4 at small batch,
   0–1 once the arrays saturate.
5. **On-chip sampling + grammar masks.** Logits never leave the chip. RISC-V cores
   apply JSON-schema/grammar token masks and sample (top-p, temperature), which
   removes a PCIe round trip per token.

---

## 7. Numerics

| Tensor | Format | Why |
|---|---|---|
| Weights | MXFP4 (4-bit + E8 scale per 32) | halves decode bytes vs FP8 |
| Activations / compute | FP8 (E4M3) | FP4 weights dequantised in-array; FP32 accumulate |
| KV cache | FP8, FP4 optional | KV bytes = agent capacity |
| Softmax, norms | FP32 | accuracy-critical, tiny FLOP share |

Accuracy risk: MXFP4 weights need quantisation-aware methods for some models.
The chip keeps FP8 and BF16 paths, and the model quantifies the cost of using
them (section 9).

---

## 8. Model assumptions (5nm-class node)

| Quantity | Value | Basis |
|---|---|---|
| FP8 MAC (effective PE incl. regs/clock) | 0.25 pJ | bare multiplier ~0.08 pJ × ~3 for datapath overhead |
| FP4 / BF16 MAC | 0.14 / 0.60 pJ | scaled by multiplier area |
| SRAM access | 0.8 pJ/B | large-bank 5nm SRAM |
| NoC | 0.3 pJ/B/hop | ~2.8 mean hops on 8×4 mesh |
| HBM3E | 30 pJ/B (3.75 pJ/bit) | incl. PHY + controller |
| LPDDR5X | 40 pJ/B (5 pJ/bit) | |
| Static power | 60 W | leakage + clock trees + PHY idle + control cores |
| Achievable BW / array efficiency | 85% / 85% | |
| Per-layer sync | 0.5 µs | hardware sequencer barrier |

**Not modelled yet:** DVFS (decode leaves the arrays ~70% idle, so voltage and
frequency could drop to cut power), NoC contention, multi-chip, thermal
throttling, area/cost.

---

## 9. What the model says (design decisions)

**D1: numerics are the biggest hardware lever.** BF16 everywhere is **0.28–0.57×**
the baseline (and 70B no longer fits at batch 64). FP4 weights alone are worth
1.15–1.18× tok/J over FP8 in chat decode. → *An MX FP4/FP8 datapath is mandatory.*

**D2: decode energy is HBM bytes.** In 8B decode at batch 1, HBM is 63% of
energy and MACs are 1%. Chat 70B tok/J stays within 7.9–9.1 across a 2.5×
bandwidth sweep (3.2 → 8.0 TB/s): bandwidth buys *latency* (18.6 → 6.9 ms/token),
not efficiency. → *Size HBM for the latency target, not for tok/J.*

**D3: agentic throughput is compute-bound.** Coding agent 70B turns/s is
0.70 → 1.40 → 2.79 for 16 → 32 → 64 tiles and flat across HBM bandwidth. Prefill
MACs are 62% of agent-turn energy. → *An agentic chip wants roughly 2× the FLOP:byte
ratio of a chat chip.* 64 tiles at 4.8 TB/s: agent tok/J 2.12 → 2.60 (1.23×),
2× throughput, 322 W (at or over the 350 W TDP with faster HBM), about 2× array area.
**Open decision:** 32 vs 48 vs 64 tiles, pending area/cost numbers from phase 3.

**D4: KV reuse across tool calls is worth 4.7–6.0×** in J/turn (70B: 843 → 140 J).
Without it, work per episode grows with the square of the turn count. → *The KV
manager is a first-class hardware block.*

**D5: the LPDDR tier pays off only for long, slow sessions.** Research agent
(140k-token context, 20 s tools): removing LPDDR costs 0.63× throughput on 8B.
Tiered KV serves 49 vs 30 agents (8B) and 52 vs 25 (Mixtral) and reaches compute
saturation; HBM-only doesn't. Coding agent (45k tokens, 3 s tools): no difference.
512 GB adds nothing over 256 GB, because the chip is already compute-saturated.
→ *256 GB is right-sized.*

**D6: speculation must be adaptive and energy-aware.** Fixed k=4 made 8B chat at
batch 64 array-bound (4.5 ms/step), and a time-minimising scheduler wasted MACs
on rejected drafts whenever HBM got faster. With an energy-minimising scheduler,
speculation is worth 1.19× tok/J on the 70B coding agent and 1.16× on research
throughput, and ~1.0× on large-batch chat, where the scheduler turns it off.

**D7: FP4 KV is the cheapest next win for agents** (+3–12%), if accuracy holds.

**D8: latency vs energy is a runtime knob, not silicon.** At the same throughput
(within 1%), the 70B coding agent can run at B=6 with 4.3 s turns and 140 J/turn,
or B=15 with 9.8 s turns and 126 J/turn. The model reports the latency-first point.

---

## 10. Roadmap

| Phase | Deliverable | Tools |
|---|---|---|
| 1 ✅ | This spec + analytical model | Python |
| 1b | DVFS model, multi-chip (tensor-parallel 405B), resolve D3 | Python |
| 2 | RTL: 1 tile (systolic array + MX dequant + scratchpad + sequencer), testbench vs NumPy golden | SystemVerilog, Verilator, cocotb |
| 3 | Synthesis area/power of the tile on an open PDK; recalibrate section 8 | Yosys, OpenROAD, SKY130/GF180 |
| 4 | Multi-tile + NoC + RISC-V control core, run a real 1-layer transformer in simulation | Verilator, CVA6/Rocket |
| 5 | Tiny Tapeout / shuttle test chip of one scaled-down tile | OpenLane |
