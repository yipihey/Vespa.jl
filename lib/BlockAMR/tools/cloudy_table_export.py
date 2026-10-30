#!/usr/bin/env python3
"""Export a Grackle Cloudy cooling table (CloudyData_*.h5) to plain .npy files
for BlockAMR's tabulated cooling (src/cooling.jl) — no HDF5 dependency in Julia.

    python3 cloudy_table_export.py <CloudyData_UVB=FG2011.h5> <outdir> [redshift]

Writes, at the requested redshift (default 0: the table's z = 0 slice; other z
are linearly interpolated in z exactly like Grackle's interpolate_3Dz):
    lognH.npy (n), logT.npy (m),
    prim_cool.npy, prim_heat.npy, met_cool.npy, met_heat.npy, mmw.npy  (n, m)
Λ/n_H² and H/n_H² [erg cm^3 s^-1] as stored (linear; the loader takes log10 like
Grackle's initialize_cloudy_data); mmw linear.
"""
import sys, os
import numpy as np
import h5py

src, out = sys.argv[1], sys.argv[2]
z = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
os.makedirs(out, exist_ok=True)
with h5py.File(src, "r") as f:
    d = f["CoolingRates/Primordial/Cooling"]
    lognH = np.asarray(d.attrs["Parameter1"], dtype=np.float64)
    zs = np.asarray(d.attrs["Parameter2"], dtype=np.float64)
    logT = np.log10(np.asarray(d.attrs["Temperature"], dtype=np.float64))

    def slab(path):
        a = np.asarray(f[path], dtype=np.float64)          # (n, z, T)
        if z <= zs[0]:
            return a[:, 0, :]
        k = int(np.searchsorted(zs, z) - 1)
        w = (z - zs[k]) / (zs[k + 1] - zs[k])
        return (1 - w) * a[:, k, :] + w * a[:, k + 1, :]

    arrs = dict(prim_cool=slab("CoolingRates/Primordial/Cooling"),
                prim_heat=slab("CoolingRates/Primordial/Heating"),
                met_cool=slab("CoolingRates/Metals/Cooling"),
                met_heat=slab("CoolingRates/Metals/Heating"),
                mmw=slab("CoolingRates/Primordial/MMW"))
np.save(os.path.join(out, "lognH.npy"), lognH)
np.save(os.path.join(out, "logT.npy"), logT)
for k, v in arrs.items():
    np.save(os.path.join(out, k + ".npy"), np.ascontiguousarray(v))
with open(os.path.join(out, "SOURCE.txt"), "w") as fh:
    fh.write(f"{os.path.abspath(src)}\nredshift {z}\n")
print("exported", src, "z =", z, "->", out, {k: v.shape for k, v in arrs.items()})
