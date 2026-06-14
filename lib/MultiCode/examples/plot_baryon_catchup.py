#!/usr/bin/env python
# Baryon catch-up diagnostic: db/dc = sqrt(P_b/P_dm) vs the CICASS two-fluid
# (with-pressure) linear theory.  Compares native-Enzo (hydro=:enzo, correct),
# the broken :julia GPU-hydro Enzo, and RAMSES.
#   ~/Projects/disco-dj-fem/.venv/bin/python plot_baryon_catchup.py
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(__file__), "..", "..", "..", "reports", "multicode")

def blocks(fn):
    d, cur = {}, None
    for line in open(os.path.join(R, fn)):
        if line.startswith('@'): cur = line[1:].strip(); d[cur] = []
        elif line.startswith('#') or not line.strip(): continue
        elif cur: k, p = line.split(); d[cur].append((float(k), float(p)))
    return {k: np.array(v) for k, v in d.items()}

def zlist(d):
    return sorted({k.split()[0][2:] for k in d}, key=float, reverse=True)

def dbdc(d, z, kq):
    kb, pb = d[f'z={z} baryon'].T; kd, pd = d[f'z={z} dm'].T
    r = pb / np.interp(kb, kd, pd)
    return np.sqrt(np.interp(kq, kb, r))

def nearest_z(d, ztarget):
    zs = np.array([float(z) for z in zlist(d)])
    return zlist(d)[int(np.argmin(np.abs(zs - ztarget)))]

lin = blocks('cicass_linear_pk.dat')
ram = blocks('cicass_ramses_pk.dat')
jul = blocks('cicass_highz_pk_enzo.dat')           # broken :julia GPU hydro
nat = blocks('cicass_highz_pk_natenzo.dat')        # native Enzo hydro (correct)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))

# ── panel 1: db/dc(k) at z~20 ──
kq = np.logspace(np.log10(60), np.log10(2500), 40)
a1.loglog(kq, dbdc(lin, '20.0', kq), 'k--', lw=2, label='CICASS 2-fluid (pressure)')
a1.loglog(kq, dbdc(nat, nearest_z(nat, 20), kq), 'o-', c='C2', ms=4,
          label=f'Enzo native hydro (z={nearest_z(nat,20)})')
a1.loglog(kq, dbdc(jul, '20.0', kq), 's-', c='C0', ms=4, label='Enzo :julia GPU hydro (z=20)')
a1.loglog(kq, dbdc(ram, nearest_z(ram, 20), kq), '^-', c='C1', ms=4,
          label=f'RAMSES (z={nearest_z(ram,20)})')
a1.set_xlabel('k [h/Mpc]'); a1.set_ylabel(r'$\delta_b/\delta_c=\sqrt{P_b/P_{dm}}$')
a1.set_title('Baryon catch-up vs scale at z$\\approx$20'); a1.legend(fontsize=9); a1.grid(alpha=.3)

# ── panel 2: low-k db/dc vs z ──
klow = np.array([70., 120.])
def trace(d, kk=klow):
    zs = [float(z) for z in zlist(d)]
    val = [dbdc(d, z, kk).mean() for z in zlist(d)]
    return np.array(zs), np.array(val)
for d, lab, st in ((lin, 'CICASS 2-fluid (pressure)', 'k--'),
                   (nat, 'Enzo native hydro', 'o-'),
                   (jul, 'Enzo :julia GPU hydro', 's-'),
                   (ram, 'RAMSES', '^-')):
    z, v = trace(d)
    a2.loglog(1 + z, v, st, ms=4, lw=2 if st == 'k--' else 1.4, label=lab)
a2.set_xlabel('1+z'); a2.set_ylabel(r'$\delta_b/\delta_c$ (large scale, k$\approx$70-120)')
a2.set_title('Baryon catch-up history'); a2.legend(fontsize=9); a2.grid(alpha=.3)
a2.invert_xaxis()

fig.suptitle('CICASS 128 kpc/h z=1000→20: baryon catch-up vs two-fluid linear theory (with pressure)',
             fontsize=12)
fig.tight_layout()
out = os.path.join(R, 'baryon_catchup_diagnosis.png')
fig.savefig(out, dpi=120); print('wrote', out)
