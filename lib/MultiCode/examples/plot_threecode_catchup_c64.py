#!/usr/bin/env python
# 3-code baryon catch-up vs CICASS two-fluid (with-pressure) linear theory, for the
# c64 (64^3, box 0.128 Mpc/h, z=1000->20, H+D chem + Compton drag) run set:
#   Enzo (native hydro/grav/particles), RAMSES, Arepo (cosmological TreePM),
# all on ONE CICASS realization at the shared output redshift list.
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(__file__), "..", "..", "..", "reports", "multicode")

def blocks(fn):
    d, cur = {}, None
    path = os.path.join(R, fn)
    if not os.path.exists(path):
        print("WARNING: missing", fn); return None
    for line in open(path):
        if line.startswith('@'): cur = line[1:].strip(); d[cur] = []
        elif line.startswith('#') or not line.strip(): continue
        elif cur:
            parts = line.split()
            if len(parts) >= 2: d[cur].append((float(parts[0]), float(parts[1])))
    return {k: np.array(v) for k, v in d.items() if len(v)}

def zs(d): return sorted({k.split()[0][2:] for k in d if k.endswith('baryon') or k.endswith('dm')},
                         key=float, reverse=True)
def nz(d, zt):
    a = np.array([float(z) for z in zs(d)]); return zs(d)[int(np.argmin(np.abs(a-zt)))]
def dbdc(d, z, kq):
    kb, pb = d[f'z={z} baryon'].T; kd, pd = d[f'z={z} dm'].T
    return np.sqrt(np.interp(kq, kb, pb/np.interp(kb, kd, pd)))

lin = blocks('cicass_linear_pk.dat')
nat = blocks('cicass_highz_pk_c64.dat')   # Enzo driver (cicass_highz_pk.jl) writes this name
ram = blocks('cicass_ramses_pk_c64.dat')
ar  = blocks('cicass_arepo_pk_c64.dat')
codes = [(n, d, st, c) for (n, d, st, c) in
         [('Enzo (native)', nat, 'o-', 'C2'), ('RAMSES', ram, '^-', 'C1'),
          ('Arepo (TreePM)', ar, 'D-', 'C3')] if d is not None]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
# panel 1: db/dc(k) at z~20
kq = np.logspace(np.log10(60), np.log10(700), 36)
if lin is not None:
    a1.loglog(kq, dbdc(lin, '20.0', kq), 'k--', lw=2, label='CICASS 2-fluid (pressure)')
for name, d, st, c in codes:
    a1.loglog(kq, dbdc(d, nz(d, 20), kq), st, c=c, ms=4, label=f'{name} z={nz(d,20)}')
a1.set_xlabel('k [h/Mpc]'); a1.set_ylabel(r'$\delta_b/\delta_c=\sqrt{P_b/P_{dm}}$')
a1.set_title(r'Baryon catch-up vs scale at z$\approx$20'); a1.legend(fontsize=9); a1.grid(alpha=.3)
# panel 2: low-k db/dc history
klow = np.array([70., 120.])
if lin is not None:
    a2.loglog(1+np.array([float(z) for z in zs(lin)]),
              [dbdc(lin, z, klow).mean() for z in zs(lin)], 'k--', lw=2, label='CICASS 2-fluid (pressure)')
for name, d, st, c in codes:
    zz = [float(z) for z in zs(d)]
    a2.loglog(1+np.array(zz), [dbdc(d, z, klow).mean() for z in zs(d)], st, c=c, ms=4, label=name)
a2.set_xlabel('1+z'); a2.set_ylabel(r'$\delta_b/\delta_c$ (k$\approx$70-120)')
a2.set_title('Baryon catch-up history'); a2.legend(fontsize=9); a2.grid(alpha=.3); a2.invert_xaxis()
fig.suptitle('CICASS 64$^3$ 128 kpc/h z=1000$\\rightarrow$20: Enzo + RAMSES + Arepo vs two-fluid linear theory (H+D chem, Compton drag)', fontsize=11)
fig.tight_layout()
out = os.path.join(R, 'threecode_catchup_c64.png'); fig.savefig(out, dpi=120); print('wrote', out)
