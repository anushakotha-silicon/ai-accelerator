# AI inference accelerator

An LLM inference accelerator designed for perf/W, built workload-first: first
chat inference, then agentic AI (long, growing contexts, tool-call pauses,
structured outputs).

**Phase 1 (this repo today):** an architecture spec and an analytical
performance and energy model that justifies every block before any RTL gets written.

| | |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | The chip: blocks, memory hierarchy, numerics, agentic features, design decisions D1–D8 |
| [results/report.md](results/report.md) | Generated tables: chat, agentic, energy breakdown, ablations, design sweep |
| `model/hw.py` | Chip config + energy table (every assumption lives here) |
| `model/workloads.py` | Llama-3.1 8B/70B, Mixtral 8x7B, chat and agent profiles |
| `model/engine.py` | Cost of one decode step / one prefill: time, bound, energy by component |
| `model/scenarios.py` | Continuous-batching chat; agentic episodes under 3 KV policies |

Requires only Python 3.9+ with the standard library.

```bash
python3 -m model.run                      # regenerate results/report.md
python3 -m unittest discover tests        # sanity checks
```

To explore a design: edit `Chip(...)` in `model/hw.py`, or add a variant to
`ABLATIONS` in `model/run.py`, then rerun.
