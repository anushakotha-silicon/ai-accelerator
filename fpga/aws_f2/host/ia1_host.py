#!/usr/bin/env python3
"""Run the IA-1 tile core on an AWS F2 FPGA and check it against the golden model.

On an F2 instance with the cl_ia1 AFI loaded (fpga-load-local-image):

    sudo python3 fpga/aws_f2/host/ia1_host.py --slot 0 --m 4 --blocks 12

Talks to the core through PCIe BAR0 (the OCL AXI-Lite window) by mmap-ing the
device's resource0 file, so it needs only the Python standard library and root.
The register map is documented at the top of rtl/tile_core.sv.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tb"))
from gen_vectors import generate  # noqa: E402  (same golden model as simulation)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bar import Bar, find_bar0  # noqa: E402

ID, CTRL, STATUS, NUM_BLOCKS, ROWS, CYCLES, STALLS, BLK1, LAST, PARAMS = range(0, 0x28, 4)
WBUF, ABUF, OBUF = 0x100000, 0x200000, 0x300000


def pack_row(vals):
    """INT8 row -> 32-bit words, element i in byte i (matches rtl/tile_core.sv)."""
    words = []
    for i in range(0, len(vals), 4):
        b = [v & 0xFF for v in vals[i:i + 4]]
        words.append(b[0] | b[1] << 8 | b[2] << 16 | b[3] << 24)
    return words


def s32(v):
    return v - (1 << 32) if v & 0x80000000 else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--fake", action="store_true", help="run against fake_bar.py instead of an FPGA")
    ap.add_argument("--m", type=int, default=4, help="rows (tokens) per weight block")
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    if a.fake:
        from fake_bar import FakeBar
        bar = FakeBar()
    else:
        bar = Bar(find_bar0(a.slot))
    ident = bar.peek(ID)
    if ident != 0x1A1C0001:
        sys.exit(f"unexpected ID {ident:#010x}: is the cl_ia1 AFI loaded in slot {a.slot}?")
    p = bar.peek(PARAMS)
    n, lanes, max_blocks, max_m = p & 0xFF, (p >> 8) & 0xFF, (p >> 16) & 0xFF, (p >> 24) & 0xFF
    if not (3 <= a.blocks <= max_blocks and 1 <= a.m <= max_m):
        sys.exit(f"blocks must be 3..{max_blocks} and m 1..{max_m}")
    print(f"IA-1 tile: N={n} LANES={lanes}, running {a.blocks} blocks of M={a.m}")

    W, X, Y = generate(n, a.m, a.blocks, a.seed)
    bar.poke(NUM_BLOCKS, a.blocks)
    bar.poke(ROWS, a.m)
    wpr = n // 4
    for b in range(a.blocks):
        for k in range(n):
            for i, word in enumerate(pack_row(W[b][k])):
                bar.poke(WBUF + ((b * n + k) * wpr + i) * 4, word)
        for m in range(a.m):
            for i, word in enumerate(pack_row(X[b][m])):
                bar.poke(ABUF + ((b * a.m + m) * wpr + i) * 4, word)

    t0 = time.perf_counter()
    bar.poke(CTRL, 1)
    while not bar.peek(STATUS) & 2:
        if time.perf_counter() - t0 > 5:
            sys.exit("timeout waiting for done")
    errors = 0
    for b in range(a.blocks):
        for m in range(a.m):
            row = b * a.m + m
            for j in range(n):
                got = s32(bar.peek(OBUF + (row * n + j) * 4))
                if got != Y[b][m][j]:
                    errors += 1
                    if errors <= 5:
                        print(f"MISMATCH block {b} row {m} col {j}: got {got} expected {Y[b][m][j]}")
    cycles, stalls = bar.peek(CYCLES), bar.peek(STALLS)
    period = (bar.peek(LAST) - bar.peek(BLK1)) / (a.blocks - 2)
    model = max(a.m, -(-n // lanes))
    print(f"RESULT fpga N={n} LANES={lanes} M={a.m} blocks={a.blocks} | cycles={cycles} stalls={stalls} | "
          f"cycles/block measured={period:.2f} model={model} | errors={errors} | {'PASS' if errors == 0 else 'FAIL'}")
    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
