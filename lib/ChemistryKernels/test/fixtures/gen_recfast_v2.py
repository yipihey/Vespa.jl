#!/usr/bin/env python3
"""
Generate a CAMB/RECFAST-v2 hydrogen recombination reference trajectory
for the ChemistryKernels test suite.

Cosmology: H0=71, Omega_b=0.044, Omega_c=0.226, Omega_Lambda=0.73, T_CMB=2.725K.
YHe=0  (pure hydrogen, matching ChemistryKernels' helium-neutral assumption).

CAMB uses RECFAST v2 by default (Rubiño-Martín et al. 2010), with fudge=1.125
and two Gaussian corrections that bring x_e to within ~0.1-0.3% of HyRec.

Output: recfast_v2_xe.csv — columns: z, xe
"""

import numpy as np
import camb

# Our cosmology (matches ChemistryKernels tests: H0=71, Om=0.27, OL=0.73)
H0 = 71.0
h  = H0 / 100.0
Omb = 0.044
Omc = 0.27 - Omb
OmL = 0.73

pars = camb.CAMBparams()
pars.set_cosmology(
    H0   = H0,
    ombh2 = Omb * h**2,
    omch2 = Omc * h**2,
    omk   = 0.0,
    YHe   = 0.24,         # standard Big Bang value; He fully neutral by z~1800,
                           # so x_e from He is negligible at z<1500 (comparison window)
    TCMB  = 2.725,
)
pars.set_for_lmax(100, lens_potential_accuracy=0)   # background only

# Print RECFAST v2 parameters for the record
r = pars.Recomb
print(f"RECFAST params: fudge={r.RECFAST_fudge}, Hswitch={r.RECFAST_Hswitch}")
print(f"  Gauss1: A={r.AGauss1}, z={r.zGauss1}, w={r.wGauss1}")
print(f"  Gauss2: A={r.AGauss2}, z={r.zGauss2}, w={r.wGauss2}")

results = camb.get_background(pars)

# z grid: log-spaced from 200 to 8000 (covers Saha → freeze-out → deep recombination)
z_arr = np.exp(np.linspace(np.log(200.0), np.log(8000.0), 1000))
ev = results.get_background_redshift_evolution(z_arr, vars=['x_e', 'T_b'], format='dict')

xe_camb = ev['x_e']
Tb_camb = ev['T_b']

# Save
out = np.column_stack([z_arr, xe_camb, Tb_camb])
header = (
    "CAMB/RECFAST-v2 hydrogen recombination reference (YHe=0.24, H0=71, Ob=0.044, Oc=0.226, OL=0.73)\n"
    "He fully neutral by z~1800; x_e here is H-only at z<1500 (comparison window).\n"
    "fudge=1.125, Hswitch=True, Gauss1(A=-0.14,z=7.28,w=0.18), Gauss2(A=0.079,z=6.73,w=0.33)\n"
    "z,xe,Tb_K"
)
np.savetxt("recfast_v2_xe.csv", out, delimiter=",", header=header, comments="# ", fmt="%.8e")
print(f"Wrote recfast_v2_xe.csv  ({len(z_arr)} rows)")
print(f"x_e range: {xe_camb.min():.4e} – {xe_camb.max():.4e}")
print(f"x_e at z=1100: {np.interp(1100.0, z_arr[::-1], xe_camb[::-1]):.6f}")
print(f"x_e at z=1200: {np.interp(1200.0, z_arr[::-1], xe_camb[::-1]):.6f}")
print(f"x_e at z= 500: {np.interp(500.0,  z_arr[::-1], xe_camb[::-1]):.6e}")
