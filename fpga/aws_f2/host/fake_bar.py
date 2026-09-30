"""Software stand-in for the IA-1 FPGA's BAR0, for testing the host programs
without an F2 instance (`--fake`). Implements the same register maps and
semantics as rtl/tile_core.sv and rtl/kv_manager.sv / rtl/kv_axil.sv.
Cycle counters are plausible estimates, not simulation results.
"""
N, LANES, MAX_BLOCKS, MAX_M = 32, 2, 16, 64
S, P, PT, H, D = 8, 16, 16, 32, 128
KV = 0x400000


def s8(b):
    return b - 256 if b & 0x80 else b


class FakeBar:
    def __init__(self):
        self.tile = {"nb": 1, "m": 1, "done": 0, "cycles": 0, "blk1": 0, "last": 0}
        self.wbuf, self.abuf, self.obuf = {}, {}, {}
        self.kvr = {"tok": 0, "wlo": 0, "whi": 0, "code": 0, "rval": 0, "done": 0, "cycles": 0}
        self.pte = {}                         # (s, p) -> dict
        self.npages = [0] * S
        self.hbm_free, self.ddr_free = set(range(H)), set(range(D))
        self.refcnt = [0] * H
        self.hbm, self.ddr = {}, {}
        self.parked = self.restored = 0

    # ------------------------------------------------------------------ access
    def peek(self, a):
        if a >= KV:
            o, r = a - KV, self.kvr
            return {0x00: 0x1A1C4B56, 0x08: r["tok"], 0x0C: r["wlo"], 0x10: r["whi"],
                    0x14: (r["code"] << 4) | (r["done"] << 1), 0x18: r["rval"] & 0xFFFFFFFF,
                    0x1C: r["rval"] >> 32, 0x20: self.parked, 0x24: self.restored,
                    0x28: len(self.hbm_free) | len(self.ddr_free) << 16, 0x2C: r["cycles"],
                    0x30: S | P << 8 | PT << 16, 0x34: H | D << 16}.get(o, 0xDEADBEEF)
        t = self.tile
        if a < 0x100000:
            return {0x00: 0x1A1C0001, 0x08: t["done"] << 1, 0x0C: t["nb"], 0x10: t["m"],
                    0x14: t["cycles"], 0x18: 0, 0x1C: t["blk1"], 0x20: t["last"],
                    0x24: N | LANES << 8 | MAX_BLOCKS << 16 | MAX_M << 24}.get(a, 0xDEADBEEF)
        if a >= 0x300000:
            return self.obuf.get((a - 0x300000) >> 2, 0)
        return (self.wbuf if a < 0x200000 else self.abuf).get((a & 0xFFFFF) >> 2, 0)

    def poke(self, a, v):
        v &= 0xFFFFFFFF
        if a >= KV:
            o = a - KV
            if o == 0x04:
                self._kv_cmd(v)
            else:
                self.kvr[{0x08: "tok", 0x0C: "wlo", 0x10: "whi"}.get(o, "_")] = v
        elif a == 0x0C:
            self.tile["nb"] = v
        elif a == 0x10:
            self.tile["m"] = v
        elif a == 0x04 and v & 1:
            self._tile_run()
        elif 0x100000 <= a < 0x200000:
            self.wbuf[(a - 0x100000) >> 2] = v
        elif 0x200000 <= a < 0x300000:
            self.abuf[(a - 0x200000) >> 2] = v

    # -------------------------------------------------------------------- tile
    def _row(self, buf, r):
        wpr = N // 4
        out = []
        for w in range(wpr):
            word = buf.get(r * wpr + w, 0)
            out += [s8((word >> (8 * i)) & 0xFF) for i in range(4)]
        return out

    def _tile_run(self):
        nb, m = self.tile["nb"], self.tile["m"]
        for b in range(nb):
            W = [self._row(self.wbuf, b * N + k) for k in range(N)]
            for i in range(m):
                x = self._row(self.abuf, b * m + i)
                for j in range(N):
                    self.obuf[(b * m + i) * N + j] = sum(x[k] * W[k][j] for k in range(N)) & 0xFFFFFFFF
        period = max(m, -(-N // LANES))
        self.tile.update(done=1, blk1=period + 1, last=period * (nb - 1) + 1, cycles=period * nb + 2 * N + 3)

    # ---------------------------------------------------------------------- kv
    def _drop(self, pg):
        self.refcnt[pg] -= 1
        if self.refcnt[pg] == 0:
            self.hbm_free.add(pg)

    def _kv_cmd(self, v):
        op, s, src, n = v & 7, (v >> 4) & 0xF, (v >> 8) & 0xF, (v >> 16) & 0x1F
        r = self.kvr
        code, val, moved, work = 0, 0, 0, 2
        tok = r["tok"]; page, off = tok // PT, tok % PT
        e = self.pte.get((s, page))
        if op == 1:                                                    # ALLOC
            if self.npages[s] == P: code = 2
            elif not self.hbm_free: code = 1
            else:
                pg = min(self.hbm_free); self.hbm_free.remove(pg); self.refcnt[pg] = 1
                self.pte[(s, self.npages[s])] = dict(hbm=pg, ddr=None, in_hbm=True, in_ddr=False, dirty=True, shared=False)
                self.npages[s] += 1; val = pg
        elif op in (2, 3):                                             # WRITE / READ
            if page >= self.npages[s]: code = 2
            elif not e["in_hbm"] or (op == 2 and e["shared"]): code = 3
            elif op == 2: self.hbm[(e["hbm"], off)] = r["whi"] << 32 | r["wlo"]; e["dirty"] = True
            else: val = self.hbm.get((e["hbm"], off), 0)
        elif op in (4, 5):                                             # PARK / RESTORE
            for p in range(self.npages[s]):
                e = self.pte[(s, p)]; work += 1
                if op == 4:
                    if e["shared"] or not e["in_hbm"]: continue
                    if e["dirty"] or not e["in_ddr"]:
                        if not e["in_ddr"]:
                            if not self.ddr_free: code = 1; break
                            e["ddr"] = min(self.ddr_free); self.ddr_free.remove(e["ddr"]); e["in_ddr"] = True
                        for w in range(PT): self.ddr[(e["ddr"], w)] = self.hbm.get((e["hbm"], w), 0)
                        moved += PT; self.parked += PT
                    self._drop(e["hbm"]); e["in_hbm"] = False; e["dirty"] = False
                else:
                    if e["in_hbm"]: continue
                    if not e["in_ddr"]: code = 3; break
                    if not self.hbm_free: code = 1; break
                    pg = min(self.hbm_free); self.hbm_free.remove(pg); self.refcnt[pg] = 1; e["hbm"] = pg
                    for w in range(PT): self.hbm[(pg, w)] = self.ddr.get((e["ddr"], w), 0)
                    e["in_hbm"] = True; e["dirty"] = False; moved += PT; self.restored += PT
            val = moved
        elif op == 6:                                                  # FREE
            for p in range(self.npages[s]):
                e = self.pte.pop((s, p))
                if e["in_hbm"]: self._drop(e["hbm"])
                if e["in_ddr"]: self.ddr_free.add(e["ddr"])
            self.npages[s] = 0
        elif op == 7:                                                  # SHARE
            if self.npages[s] or n > self.npages[src] or s == src: code = 3
            else:
                for p in range(n):
                    se = self.pte[(src, p)]; se["shared"] = True; self.refcnt[se["hbm"]] += 1
                    self.pte[(s, p)] = dict(hbm=se["hbm"], ddr=None, in_hbm=True, in_ddr=False, dirty=False, shared=True)
                self.npages[s] = n
        else:
            code = 3
        r.update(code=code, rval=val, done=1, cycles=moved + work + 2)
