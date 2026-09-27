"""Golden model + test vectors for tb_systolic_array.sv (stdlib only).

Writes, into --out:
  weights.hex   W[b][k][n]  int8,  index (b*N + k)*N + n
  acts.hex      X[b][m][k]  int8,  index (b*M + m)*N + k
  expected.hex  Y[b][m][n]  int32, index (b*M + m)*N + n,  Y = X @ W
  params.svh    localparams N, LANES, M, NB for the testbench
"""
import argparse
import os
import random


def generate(n, m, nb, seed):
    """Random INT8 blocks W[b][k][n], X[b][m][k] and golden Y = X @ W (INT32)."""
    rng = random.Random(seed)

    # include the extremes so sign handling and accumulator width get exercised
    def i8():
        return rng.choice([-128, 127, rng.randint(-128, 127), rng.randint(-128, 127)])

    W = [[[i8() for _ in range(n)] for _ in range(n)] for _ in range(nb)]
    X = [[[i8() for _ in range(n)] for _ in range(m)] for _ in range(nb)]
    Y = [[[sum(X[b][i][k] * W[b][k][j] for k in range(n)) for j in range(n)] for i in range(m)]
         for b in range(nb)]
    return W, X, Y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--lanes", type=int, default=2)
    p.add_argument("--m", type=int, default=4, help="rows (tokens) per weight block")
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="build")
    a = p.parse_args()
    if a.blocks < 3:
        p.error("--blocks must be >= 3 to measure the steady-state period")
    n, m, nb = a.n, a.m, a.blocks
    W, X, Y = generate(n, m, nb, a.seed)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "weights.hex"), "w") as f:
        f.writelines(f"{v & 0xFF:02x}\n" for blk in W for row in blk for v in row)
    with open(os.path.join(a.out, "acts.hex"), "w") as f:
        f.writelines(f"{v & 0xFF:02x}\n" for blk in X for row in blk for v in row)
    with open(os.path.join(a.out, "expected.hex"), "w") as f:
        f.writelines(f"{v & 0xFFFFFFFF:08x}\n" for blk in Y for row in blk for v in row)
    with open(os.path.join(a.out, "params.svh"), "w") as f:
        f.write(f"localparam int N = {n};\nlocalparam int LANES = {a.lanes};\n"
                f"localparam int M = {m};\nlocalparam int NB = {nb};\n")


if __name__ == "__main__":
    main()
