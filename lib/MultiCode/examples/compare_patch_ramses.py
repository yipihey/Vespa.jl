#!/usr/bin/env python3
# Calibrate the patch-decomposition CICASS run against the RAMSES reference.
#
# Reads the cellcmp dumps (binary: int64 N, then 5 × N³ float64 fields:
# rho_b, x_HII, f_H2, f_HD, T[K]) written by patch_cicass.jl (tag "patch128")
# and cicass_ramses_pk.jl (tag "f32cuda"), and reports per-redshift the mean
# and rms of each field plus the patch/RAMSES ratio — and, when the two share
# the same IC realization, the cell-by-cell Pearson correlation of delta_b.
#
# Usage: python3 compare_patch_ramses.py [patch_tag] [ramses_tag] [z1 z2 ...]

import sys, os
import numpy as np

REPORTS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "reports", "multicode")
FIELDS = ["rho_b", "x_HII", "f_H2", "f_HD", "T_K"]

def read_cellcmp(path):
    with open(path, "rb") as f:
        N = int(np.fromfile(f, dtype=np.int64, count=1)[0])
        n3 = N**3
        d = {fld: np.fromfile(f, dtype=np.float64, count=n3) for fld in FIELDS}
    return N, d

def stats(v):
    m = np.mean(v)
    return m, (np.std(v) / m if m != 0 else np.nan)

def main():
    ptag = sys.argv[1] if len(sys.argv) > 1 else "patch128"
    rtag = sys.argv[2] if len(sys.argv) > 2 else "f32cuda"
    zs = [int(z) for z in sys.argv[3:]] if len(sys.argv) > 3 else [300, 100, 50, 20]

    print(f"# patch tag '{ptag}'  vs  RAMSES tag '{rtag}'")
    for z in zs:
        pp = os.path.join(REPORTS, f"patch_cellcmp_{ptag}_z{z}.bin")
        rp = os.path.join(REPORTS, f"ramses_cellcmp_{rtag}_z{z}.bin")
        if not (os.path.exists(pp) and os.path.exists(rp)):
            miss = [p for p in (pp, rp) if not os.path.exists(p)]
            print(f"\nz={z}: MISSING {', '.join(os.path.basename(m) for m in miss)}")
            continue
        Np, dp = read_cellcmp(pp)
        Nr, dr = read_cellcmp(rp)
        print(f"\n=== z={z}  (patch N={Np}, RAMSES N={Nr}) ===")
        print(f"  {'field':8s} {'patch_mean':>12s} {'ram_mean':>12s} {'ratio':>8s}"
              f" {'patch_rms':>10s} {'ram_rms':>10s}")
        for fld in FIELDS:
            pm, prms = stats(dp[fld]); rm, rrms = stats(dr[fld])
            ratio = pm / rm if rm != 0 else np.nan
            print(f"  {fld:8s} {pm:12.4e} {rm:12.4e} {ratio:8.4f} {prms:10.4e} {rrms:10.4e}")
        # cell-by-cell delta_b correlation (only meaningful if same IC realization + N)
        if Np == Nr:
            db_p = dp["rho_b"] / np.mean(dp["rho_b"]) - 1.0
            db_r = dr["rho_b"] / np.mean(dr["rho_b"]) - 1.0
            cc = np.corrcoef(db_p, db_r)[0, 1]
            slope = np.dot(db_p, db_r) / np.dot(db_r, db_r)
            print(f"  delta_b cell-by-cell: corr={cc:.4f}  slope(patch/ram)={slope:.4f}")

if __name__ == "__main__":
    main()
