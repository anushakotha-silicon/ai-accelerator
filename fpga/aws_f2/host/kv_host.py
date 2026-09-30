#!/usr/bin/env python3
"""Run the agent KV scenario on the IA-1 FPGA (AWS F2) and check it.

    sudo python3 fpga/aws_f2/host/kv_host.py --slot 0 --agents 4 --turns 3

Same scenario and scoreboard as tb/tb_ia1_top.sv: session 0 writes a shared
system prompt, agents map it with SHARE, then each turn restores, appends
tokens, reads every token back, and parks. The scoreboard predicts exactly how
many tokens each park and restore moves (incremental parking). Reports the
hardware cycle counts (KV_CYCLES) and host-side wall time per operation.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bar import Bar, find_bar0  # noqa: E402

KV = 0x400000
ID, CMD, TOK, WLO, WHI, STATUS, RLO, RHI, PARKED, RESTORED, FREE, CYCLES, PARAMS, TIERS = (
    KV + o for o in (0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C, 0x20, 0x24, 0x28, 0x2C, 0x30, 0x34))
ALLOC, WRITE, READ, PARK, RESTORE, FREE_OP, SHARE = range(1, 8)
CODES = {0: "OK", 1: "NOSPACE", 2: "RANGE", 3: "STATE"}


def kvdata(s, t):
    return (0xA9E5 << 48) | (s << 40) | t


class KV:
    def __init__(self, bar):
        self.bar = bar
        self.last_cycles = 0

    def cmd(self, op, s=0, src=0, n=0, tok=0, wdata=0):
        b = self.bar
        b.poke(TOK, tok)
        b.poke(WLO, wdata & 0xFFFFFFFF)
        b.poke(WHI, wdata >> 32)
        b.poke(CMD, op | s << 4 | src << 8 | n << 16)
        t0 = time.perf_counter()
        while True:
            st = b.peek(STATUS)
            if st & 2:
                break
            if time.perf_counter() - t0 > 2:
                sys.exit(f"timeout on op {op}")
        self.last_cycles = b.peek(CYCLES)
        return (st >> 4) & 3, b.peek(RLO) | b.peek(RHI) << 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--fake", action="store_true", help="run against fake_bar.py instead of an FPGA")
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--prefix-pages", type=int, default=2)
    a = ap.parse_args()

    if a.fake:
        from fake_bar import FakeBar
        bar = FakeBar()
    else:
        bar = Bar(find_bar0(a.slot))
    if bar.peek(ID) != 0x1A1C4B56:
        sys.exit("KV manager not found at 0x400000: is the ia1_top AFI loaded?")
    p = bar.peek(PARAMS)
    S, P, PT = p & 0xFF, (p >> 8) & 0xFF, (p >> 16) & 0xFF
    t = bar.peek(TIERS)
    H, D = t & 0xFFFF, t >> 16
    print(f"KV manager: {S} sessions, {P} pages x {PT} tokens, HBM {H} pages, DDR {D} pages")
    if not 1 <= a.agents < S:
        sys.exit(f"--agents must be 1..{S - 1}")

    kv = KV(bar)
    errors = checks = 0

    def check(what, got, want):
        nonlocal errors, checks
        checks += 1
        if got != want:
            errors += 1
            if errors <= 8:
                print(f"ERROR {what}: got {got} expected {want}")

    ntok = [0] * S; npg = [0] * S; nsh = [0] * S
    dirty = [[False] * P for _ in range(S)]; hasd = [[False] * P for _ in range(S)]
    stats = {"park": [0, 0, 0.0], "restore": [0, 0, 0.0]}   # tokens, cycles, host seconds

    def append(s, k):
        for _ in range(k):
            tk = ntok[s]
            if tk % PT == 0:
                code, _ = kv.cmd(ALLOC, s)
                check("alloc", CODES[code], "OK")
                dirty[s][npg[s]] = True; hasd[s][npg[s]] = False; npg[s] += 1
            code, _ = kv.cmd(WRITE, s, tok=tk, wdata=kvdata(s, tk))
            check("write", CODES[code], "OK")
            dirty[s][tk // PT] = True; ntok[s] += 1

    def verify(s):
        for tk in range(ntok[s]):
            code, v = kv.cmd(READ, s, tok=tk)
            check("read", CODES[code], "OK")
            check(f"data s{s} t{tk}", v, kvdata(0 if tk // PT < nsh[s] else s, tk))

    def timed(kind, s, want):
        t0 = time.perf_counter()
        code, v = kv.cmd(PARK if kind == "park" else RESTORE, s)
        dt = time.perf_counter() - t0
        check(f"{kind} s{s}", (CODES[code], v), ("OK", want))
        st = stats[kind]; st[0] += want; st[1] += kv.last_cycles; st[2] += dt

    append(0, a.prefix_pages * PT)
    for ag in range(1, a.agents + 1):
        code, _ = kv.cmd(SHARE, ag, src=0, n=a.prefix_pages)
        check("share", CODES[code], "OK")
        npg[ag] = nsh[ag] = a.prefix_pages; ntok[ag] = a.prefix_pages * PT
    code, _ = kv.cmd(WRITE, 1, tok=0, wdata=1)
    check("shared prompt is read-only", CODES[code], "STATE")

    for turn in range(a.turns):
        for ag in range(1, a.agents + 1):
            if ntok[ag] + 12 + 7 * turn > P * PT:
                continue
            if npg[ag] > nsh[ag]:
                timed("restore", ag, (npg[ag] - nsh[ag]) * PT)
            append(ag, 12 + 7 * turn)
            verify(ag)
            want = sum(PT for pg in range(nsh[ag], npg[ag]) if dirty[ag][pg] or not hasd[ag][pg])
            timed("park", ag, want)
            for pg in range(nsh[ag], npg[ag]):
                dirty[ag][pg] = False; hasd[ag][pg] = True

    for s in range(a.agents, -1, -1):
        code, _ = kv.cmd(FREE_OP, s)
        check("free", CODES[code], "OK")
    check("no leaked pages", bar.peek(FREE), H | D << 16)

    for kind, (tok, cyc, sec) in stats.items():
        if tok:
            print(f"{kind:8s}: {tok} tokens, {cyc} cycles ({cyc / tok:.2f} cycles/token), "
                  f"host wall time {sec * 1e6 / tok:.1f} us/token incl. PCIe")
    print(f"RESULT fpga-kv agents={a.agents} turns={a.turns} | checks={checks} errors={errors} | "
          f"{'PASS' if errors == 0 else 'FAIL'}")
    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
