#!/usr/bin/env python3
r"""Boost a CICASS `.cicass` realization into the BARYON-REST (CMB) frame.

The standard CICASS realization is written in the DM-rest frame: the gas carries the coherent v_bc bulk
velocity (`mean(gas_vel) = v_bc` along one axis) and the DM is at rest.  For a grid+particle code that's
the wrong Galilean frame — the Eulerian gas then advects bodily through the grid every step, and in f16
the physically-important peculiar velocity δv (~few km/s) rides as a ~10% ripple on a ~30 km/s pedestal,
so the small-scale streaming suppression under-develops (badly at z≳100, high μ, high k).

This subtracts `mean(gas_vel)` (per component) from BOTH species — a pure Galilean boost that puts the
baryons at mean-zero momentum and the DM streaming at −v_bc, PRESERVING the relative gas−DM streaming
(growth depends only on the relative velocity).  It is the same operation as `zero_baryon_bulk` in
MultiCodeCICASSExt's Enzo/RAMSES paths; here it's applied to a raw `.cicass` so the PatchGrid streaming
load (`scatter_cicass_gas_stream!`) stays untouched and simply reads a boosted realization.

Only the two velocity blocks change; positions, δb, and temperature are byte-identical to the input.

`.cicass` layout (see makeCosICs/capi_out.c): 8-byte magic + 2 int32 (N, nspecies) + 10 float64, then
N³-float32 blocks in order: DMpos x/y/z (1-3), DMvel x/y/z (4-6), δb (7), gasvel x/y/z (8-10), temp (11).

Run:  <anaconda python3> boost_cicass_baryon_rest.py in.cicass out.cicass
"""
import sys, os, shutil, struct
import numpy as np

def field_offset(idx, N3, header=96, fbytes=4):
    return header + (idx - 1) * N3 * fbytes            # 1-based field index

def main():
    if len(sys.argv) != 3:
        sys.exit("usage: boost_cicass_baryon_rest.py <in.cicass> <out.cicass>")
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, "rb") as f:
        f.seek(8)                                       # skip the 8-byte magic
        N, nspecies = struct.unpack("2i", f.read(8))    # grid size, #species
    N3 = N * N * N
    print(f"input {src}: N={N}³ nspecies={nspecies}")

    print(f"copying → {dst} ...")
    shutil.copyfile(src, dst)

    # coherent bulk velocity = mean of each gas-velocity component (fields 8,9,10)
    gbulk = [float(np.fromfile(src, dtype=np.float32, count=N3,
                               offset=field_offset(g, N3)).mean()) for g in (8, 9, 10)]
    print("gas bulk v_bc (km/s):", [round(x, 4) for x in gbulk])

    # subtract it from gas (8,9,10 → rest) and DM (4,5,6 → −v_bc), in place on the copy
    with open(dst, "r+b") as f:
        for c, gf in zip(range(3), (8, 9, 10)):
            v = np.fromfile(src, dtype=np.float32, count=N3, offset=field_offset(gf, N3))
            v -= np.float32(gbulk[c]); f.seek(field_offset(gf, N3)); v.tofile(f)
        for c, df in zip(range(3), (4, 5, 6)):
            v = np.fromfile(src, dtype=np.float32, count=N3, offset=field_offset(df, N3))
            v -= np.float32(gbulk[c]); f.seek(field_offset(df, N3)); v.tofile(f)

    gm = [float(np.fromfile(dst, dtype=np.float32, count=N3, offset=field_offset(g, N3)).mean()) for g in (8, 9, 10)]
    dm = [float(np.fromfile(dst, dtype=np.float32, count=N3, offset=field_offset(d, N3)).mean()) for d in (4, 5, 6)]
    rel = [round(gm[i] - dm[i], 4) for i in range(3)]
    print(f"boosted: mean(gas)={[round(x,4) for x in gm]}  mean(dm)={[round(x,4) for x in dm]} km/s")
    print(f"relative gas−dm streaming preserved: {rel} km/s  (positions/δb/temp unchanged)")

if __name__ == "__main__":
    main()
