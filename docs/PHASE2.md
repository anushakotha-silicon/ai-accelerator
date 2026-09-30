# Phase 2: tile RTL

Goal: synthesizable SystemVerilog for one compute tile of the v0.2 design
(3nm, 48 tiles), verified cycle by cycle against a Python golden model, with
the cycle counts checked against the performance model in `model/engine.py`.

## Milestones

| | Deliverable | Checks |
|---|---|---|
| **M1** | Weight-stationary N×N systolic array, INT8×INT8→INT32, double-buffered weights with **wavefront loading** (`rtl/`) | Exact match vs golden; cycles/block = max(M, N/LANES); deliberate hazard fails |
| M2 | FP8 (E4M3) activations × MXFP4 weights (E2M1 + E8M0 scale per 32), FP32 accumulate | Bit-exact vs a Python reference of the same rounding |
| **M1-FPGA** | `rtl/tile_core.sv`: array + buffers + **hardware sequencer** + AXI-Lite; AWS F2 kit in `fpga/aws_f2/` | Bit-exact through AXI-Lite in sim; runs on F2 via `ia1_host.py` |
| **A1** | `rtl/kv_manager.sv`: paged KV, HBM/DDR tiers, park/restore, incremental parking, prefix sharing | Multi-agent test: data intact, exact tokens moved, no leaks |
| **A1-FPGA** | `rtl/ia1_top.sv`: AXI-Lite router + tile_core + `kv_axil` (KV registers, per-command cycle counter) | Matmul + agents through one port; F2 host `kv_host.py` |
| M3 | Scratchpad (16 banks × 256 KB), weight feeder at 256 B/cycle | Streams a full 70B layer's weights for one tile |
| M4 | Attention/vector engine: QKᵀ dot products, online softmax (exp unit), RoPE, norms | Matches reference attention to FP32 tolerance |
| M5 | Tile top-level + synthesis (Yosys/OpenROAD) → area and power to recalibrate `model/hw.py` | Phase 3 hand-off |

## M1 design notes

### Dataflow

```
            col 0     col 1          col N-1
 x[m][0] → [PE 0,0] → [PE 0,1] → … → [PE 0,N-1]      act moves right 1 PE/cycle
 x[m][1] → [PE 1,0] → …                               (row r delayed r cycles on entry)
    ⋮          ↓ psum     ↓                             psum moves down 1 PE/cycle
 x[m][N-1]→[PE N-1,0] → …        → [PE N-1,N-1]
               ↓                        ↓
            y[m][0] (delay N-1)  …  y[m][N-1] (delay 0)  → one deskewed row per cycle
```

Row m presented in cycle t reaches PE[r][c] at cycle **t + r + c**, and its
output row appears at **t + 2N − 1**. PE[r][c] holds W[r][c], so the bottom of
column c accumulates Σₖ x[m][k]·W[k][c].

### Double buffering without a global swap

Each PE has two weight registers. Every activation carries a 1-bit tag naming
the buffer it multiplies against, so the switch from block b to block b+1
moves diagonally through the array with the data. No PE ever needs a global
"swap now" signal, which would be wrong for all but one diagonal.

### Why weights must load as a wavefront

A buffer freed by block b−2 is still being read after that block's last row
enters: PE[r][c] reads it until t_last + r + c, up to **2N − 2 cycles** later.

**Naive loading** (write a buffer only after it fully drains, LR rows per cycle):
blocks alternate buffers, so the load for block b+1 has to fit between block b−1
draining and block b ending. The steady-state period is

  P = max(M, (M + 2N + N/LR − 1) / 2)

For N = 128, LR = 2, and decode-sized blocks (M ≈ 1–64), that's 160–192 cycles
per block, against the **64** the performance model assumes. That's 2.5–3× slower in decode.
Triple buffering only reaches ~107; hitting 64 would need 2·LR + 1 = 5 buffers.

**Wavefront loading** (what M1 builds): a load *lane* writes one weight row per
cycle, and the array delays column c of each lane by c cycles. Row r of the new
block lands in PE[r][c] at s₀ + r + c. The old block's last read of that PE
is at t_last + r + c, and the new block's first read is at t_first + r + c. Every
PE on the diagonal has the same window, so the only conditions are

  t_last ≤ s₀ < t_first

One lane is busy for N cycles, and a new block needs a lane every M cycles, so
with LANES lanes the period is **max(M, N / LANES)**. That is exactly the
`_matmul_time` formula in the model. Two lanes of 128 × 8-bit rows is the
spec's 256 B/cycle weight port.

Cost of the skew registers: N(N−1)/2 bytes per lane, so 2 × 8,128 B ≈ **16 KB of
flops per tile** at N = 128. That's 0.4% of the tile's 4 MB SRAM.

### Verification

- `tb/gen_vectors.py`: random INT8 blocks, including −128 and 127, with golden
  Y = X·W in INT32.
- `tb/tb_systolic_array.sv` acts as the sequencer. It starts a lane for block b
  once block b−2 has presented its last row, walks rows 0…N−1, and presents block
  b's rows starting one cycle after the lane starts. It checks every output row
  and prints measured cycles per block next to the model's max(M, N/LANES).
- `make hazard` starts lanes one row early (`+EARLY=1`). It **must fail**; a pass
  would mean the schedule has slack that the analysis says shouldn't exist.

```bash
make sim            # N=8, LANES=2, M=4
make sweep          # M = 1…32: measured vs model cycles/block
make hazard         # contract violation → expect MISMATCH
make lint           # Verilator lint
```

**Status: M1 verified** (Icarus Verilog 13.0, Verilator 5.052 lint, built from source
into `~/.local/eda/bin`).

| Check | Result |
|---|---|
| `make lint` (Verilator `-Wall`) | Clean; one documented waiver (activations leaving the right edge) |
| `make regress`: N ∈ {8, 16, 32}, LANES ∈ {1, 2, 4}, M from 1 to 2N | Bit-exact vs golden in every run |
| Cycles per block | = max(M, N/LANES) in steady state (e.g. 4.00, 8.00, 16.00, 32.00) |
| `make hazard` (lane starts one cycle early) | Fails on the last row of the old block, as predicted |

What the first simulations taught us:
1. **Lanes must start on different cycles.** Two lanes that start together walk
   the same rows in lockstep, so one of their writes is lost. Staggered starts
   keep lanes at least one row apart forever, because all lanes move at one row
   per cycle. The hardware sequencer enforces one lane start per cycle.
2. **The legal window is exactly s₀ ≥ L.** A lane may start on the cycle the old
   block presents its last row: each PE reads the old weight and the new weight
   is written on the same edge, and nonblocking assignment keeps the read on the
   old value. Starting one cycle earlier corrupts that last row.
3. **Short runs look faster than the model** when lanes are the limit (3.00 vs 4
   cycles/block with 8 blocks), because all lanes are free at start-up. With 40
   blocks the average converges (3.84 → 4).

## M1-FPGA: tile core for AWS F2

`rtl/tile_core.sv` turns the testbench's schedule into hardware, so a host that
can only write registers can run the array:

- **Sequencer:** each cycle, (1) present the next activation row if its block's lane
  started on an earlier cycle, (2) start at most one lane, once the block two back
  has presented its last row (that cycle counts, s₀ = L), (3) advance every lane one row.
- **Buffers:** weights (16 blocks × N rows), activations and outputs (16 × 64 rows).
  Reads are registered, so every array input comes from a flop one stage after the
  sequencer's decision. That keeps the schedule and gives the FPGA timing slack.
- **AXI4-Lite** register map (top of the file): the F2 OCL/BAR0 port. Counters
  `BLK1_START` and `LAST_START` let the host compute cycles/block exactly as the
  simulations do.

| Check | Result |
|---|---|
| `make core` (N = 8/16/32, LANES = 2/4, M = 1…40), all through AXI-Lite | Bit-exact; cycles/block = model (4, 8, 16, 40…) |
| `make lint-core` | Clean (unused high address bits waived: the BAR aliases above 16 MiB) |
| `ocl_hookup.inc` inside a stand-in CL with the shell's exact OCL port names | Lint clean at N = 32 |
| `setup_cl.py` on a mock of AWS's `CL_TEMPLATE` | Patches all 8 OCL tie-offs, includes the hookup once, idempotent |

Runbook: [`fpga/aws_f2/README.md`](../fpga/aws_f2/README.md). Needs an AWS account with F2 access.

## A1: agent KV manager

`rtl/kv_manager.sv` is the chip's KV/prefix manager and park/restore DMA in
small form. HBM and DDR are local arrays here; on F2 they become the card's HBM and DDR.

```
page table [session][logical page] → { HBM page, DDR page, in_hbm, in_ddr, dirty, shared }
free bitmaps for HBM and DDR pages · reference count per HBM page
commands: ALLOC · WRITE · READ · PARK · RESTORE · FREE · SHARE
```

| Mechanism | Rule in hardware |
|---|---|
| Paging | A token's page is `tok / PAGE_TOK`; any free physical page can back any logical page, so memory never fragments |
| Park | For each private resident page: if dirty or never copied, copy HBM→DDR; then free the HBM page. Shared pages stay pinned. |
| Incremental parking | Restore keeps the DDR copy valid and marks the page clean; only pages written after that are copied at the next park |
| Restore | Copy every non-resident page DDR→HBM into freshly allocated pages |
| Prefix sharing | `SHARE` maps a source session's first n pages into an empty session, increments each page's reference count, and makes them read-only; the last `FREE` releases them |

`make kv`: session 0 writes a 2-page system prompt, 7 agents share it, HBM holds
32 pages, DDR 128. The scoreboard predicts the exact tokens every park and restore
moves and checks every token after every round trip.

| Result | |
|---|---|
| Checks / errors | 20,276 / 0; `make lint-kv` clean |
| Prefix sharing | 7 agents use 2 HBM pages for the prompt instead of 16; writes to it are refused |
| Capacity | Without parking, HBM fills after 5 of 7 agents (6 private pages each); with parking all 7 finish |
| Incremental parking | 1,904 tokens written to DDR vs 8,400 for full-history parking: **4.41× less** in this short scenario; the gap grows with session length |
| Leaks | Every HBM and DDR page back on the free list after all sessions free |

The analytical model (`model/scenarios.py`, tiered-kv) makes the same assumptions:
parking writes only the tokens added this turn, and restore reads the whole history.

**Next agent-memory steps:**
1. **Faster restore:** restore moves the full private history every turn (7,328 tokens
   here). Restoring layer by layer, overlapped with the next prefill, hides most of it.
2. **Hardware prefix detection:** a hash CAM that finds identical prompt pages
   automatically, instead of the host calling `SHARE`.
3. **On F2:** back the two tiers with the card's real HBM and DDR (AXI4 bursts),
   then measure park/restore bandwidth on hardware.

## A1-FPGA: KV manager on the FPGA path

`rtl/ia1_top.sv` puts both blocks behind the F2 OCL port. `axil_split2` routes by
address bit 22 (tile below 0x400000, KV above), so every tile address is unchanged.
`kv_axil` turns register writes into KV commands (map at the top of the file) and
counts the cycles each command takes, so park/restore latency is measured by hardware.

| Check | Result |
|---|---|
| `make top` (routing, tile matmul, 4 agents × 3 turns through one AXI-Lite port) | 1,966 checks, 0 errors |
| Park / restore cost in hardware cycles | 1.26 / 1.31 cycles per token (1 token per cycle plus page-table walk) |
| `make lint-top` (full top at N = 32) | Clean |
| `setup_cl.py` + hookup | 7 RTL files copied; `ia1_top` lints against the shell's OCL port names |
| Host programs with `--fake` | `ia1_host.py` bit-exact; `kv_host.py` 1,877 and 5,224 checks, 0 errors; token counts match the RTL |
