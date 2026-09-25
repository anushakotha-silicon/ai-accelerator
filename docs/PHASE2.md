# Phase 2: tile RTL

Goal: synthesizable SystemVerilog for one compute tile of the v0.2 design
(3nm, 48 tiles), verified cycle by cycle against a Python golden model, with
the cycle counts checked against the performance model in `model/engine.py`.

## Milestones

| | Deliverable | Checks |
|---|---|---|
| **M1** | Weight-stationary N×N systolic array, INT8×INT8→INT32, double-buffered weights with **wavefront loading** (`rtl/`) | Exact match vs golden; cycles/block = max(M, N/LANES); deliberate hazard fails |
| M2 | FP8 (E4M3) activations × MXFP4 weights (E2M1 + E8M0 scale per 32), FP32 accumulate | Bit-exact vs a Python reference of the same rounding |
| M3 | Scratchpad (16 banks × 256 KB), weight feeder at 256 B/cycle, tile sequencer that issues the lane schedule | Streams a full 70B layer's weights for one tile |
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

**Status:** RTL, golden model and testbench are written but **not yet
simulated**. No HDL simulator is installed on this machine yet.
