# AI inference accelerator

An LLM inference accelerator designed for perf/W, built workload-first: first
chat inference, then agentic AI (long, growing contexts, tool-call pauses,
structured outputs).

**Phase 1:** an architecture spec and an analytical performance and energy model
that justifies every block before any RTL gets written.
**Phase 2 (in progress):** RTL for one compute tile, verified against a golden model.

| | |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | The chip: blocks, memory hierarchy, numerics, agentic features, design decisions D1–D8 |
| [results/report.md](results/report.md) | Generated tables: chat, agentic, energy breakdown, ablations, design sweep |
| [results/node_study.md](results/node_study.md) | 5nm vs 3nm variants, DVFS/power gating, levers, scored on a 7-workload suite |
| [results/sensitivity.md](results/sensitivity.md) | One-knob-at-a-time sweeps around v0.2, ranked by effect on tok/J |
| [results/memory_study.md](results/memory_study.md) | v0.3: HBM4, LPDDR weight sharing, half-HBM and LPDDR-only products |
| [docs/package-3d.html](docs/package-3d.html) | Interactive 3D package and floorplan, v0.1 and v0.2 (open in a browser) |
| [docs/PHASE2.md](docs/PHASE2.md) | Tile RTL plan and the systolic-array design notes |
| `rtl/`, `tb/`, `Makefile` | Phase 2 RTL: systolic array with wavefront weight loading, golden model, testbench |
| `model/hw.py` | Chip config + energy table (every assumption lives here) |
| `model/workloads.py` | Llama-3.1 8B/70B, Mixtral 8x7B, chat and agent profiles |
| `model/engine.py` | Cost of one decode step / one prefill: time, bound, energy by component |
| `model/scenarios.py` | Continuous-batching chat; agentic episodes under 3 KV policies |
| `model/study.py` | Process-node and power-management study across the workload suite |
| `model/sensitivity.py`, `model/memory_study.py` | Parameter sweeps and the v0.3 memory study |

Requires only Python 3.9+ with the standard library.

```bash
python3 -m model.run                      # regenerate results/report.md
python3 -m model.study                    # regenerate results/node_study.md
python3 -m model.sensitivity              # regenerate results/sensitivity.md
python3 -m model.memory_study             # regenerate results/memory_study.md
python3 -m unittest discover tests        # sanity checks
```

To explore a design: edit `Chip(...)` in `model/hw.py`, or add a variant to
`ABLATIONS` in `model/run.py`, then rerun.
