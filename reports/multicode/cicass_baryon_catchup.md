# CICASS z=1000→20: baryon catch-up vs two-fluid linear theory

**Question.** In the CICASS 128 kpc/h, 128³ box (z=1000→20, H+D reduced Grackle
chemistry), do the codes reproduce the post-recombination baryon catch-up — gas
falling into the dark-matter potential wells until δ_b → δ_c — that linear theory
predicts?

## The baseline already exists: `cicass_linear_pk.dat`

CICASS (McQuinn & O'Leary 2012) integrates the **baryon–CDM transfer function
forward from recombination including gas pressure and the streaming velocity**, so
its per-output-redshift P_b(k), P_dm(k) **are the two-fluid, pressure-included
linear prediction**.  Its large-scale δ_b/δ_c = √(P_b/P_dm) catches up

| z | 1000 | 520 | 274 | 142 | 75 | 38 | 20 |
|---|---|---|---|---|---|---|---|
| δ_b/δ_c (k≈49) | 0.007 | 0.166 | 0.40 | 0.59 | 0.72 | 0.81 | **0.87** |

and at z=20 is strongly scale-dependent — 0.89 at k=49 (the fundamental) falling to
0.15 at k=835 (the Jeans/filtering scale).

> Do **not** use the driver's `theory_*_cic` columns in `cicass_highz_pk.dat` as the
> baseline: those just grow the z_init CICASS P(k) by D(a)² uniformly, giving a flat,
> artifactual ratio with no catch-up.

## Result (δ_b/δ_c at z≈20)

| k [h/Mpc] | CICASS 2-fluid | **Enzo native** | Enzo `:julia` GPU | RAMSES |
|---|---|---|---|---|
| 70  | 0.845 | **0.894** | 0.175 | 0.911 |
| 120 | 0.720 | **0.835** | 0.172 | 0.871 |
| 200 | 0.524 | **0.739** | 0.162 | 0.801 |
| 400 | 0.328 | **0.551** | 0.150 | 0.662 |
| 800 | 0.162 | **0.349** | 0.126 | 0.516 |

Plot: `baryon_catchup_diagnosis.png`.

- **Enzo native hydro (`hydro=:enzo, gravity=:enzo`) reproduces the linear catch-up
  on large scales** (0.894 vs theory 0.845 at k=70) and tracks the full
  z=1000→20 history (right panel). At high k it lies *above* the linear filtering
  prediction — expected, since those scales are going nonlinear by z=20 and the
  code's actual gas temperature sets a weaker filtering than the recfast linear
  c_s(a).
- **RAMSES** matches on large scales but over-clusters at high k more than Enzo
  (0.52 vs 0.16 at k=800) — too little baryon pressure support (effectively cold
  gas).
- **Enzo `:julia` GPU hydro is broken**: δ_b/δ_c is nearly k-independent at ~0.15 —
  baryons barely catch up at any scale.

## Root cause of the `:julia` defect

Isolation matrix over z=1000→500 (new `CIC_HYDRO`/`CIC_GRAV`/`CIC_TAG` knobs in
`cicass_highz_pk.jl`), metric = low-k δ_b/δ_c:

| hydro | gravity | δ_b/δ_c(z500) | δb_rms growth | verdict |
|---|---|---|---|---|
| `:enzo` | `:enzo` | 0.178 (≈ theory 0.166) | ×32 | ✅ catches up |
| `:julia` | `:enzo` | 0.0025 | ×1.2 | ❌ frozen |
| `:julia` | `:julia` | ~0.002 | — | ❌ frozen |

The defect is in the **`:julia` PPMKernels hydro slot's gravitational coupling in
the deep cosmological-linear regime** — *not* the gravity slot. The `:julia` hydro
reads the correct acceleration (max|g|≈1.7e-3, the very field native hydro uses
successfully; verified with `CIC_DEBUG=1`), routes it per-axis, and applies Enzo's
euler formula — yet δ_b stays frozen. The SB-cluster run (z=63, few cycles) works
with `:julia` hydro because there `gravity!` writes the acceleration via
`comp_accel!`; the cosmological run feeds Enzo-native `ComputeAccelerations`
output, and over thousands of steps the baryon perturbation never grows. Exact
root cause (suspected comoving-expansion operator-split interaction, or a
per-step-tiny-but-cumulative convention mismatch) is **not yet pinned — deferred**.

## Decision

The production Enzo CICASS campaign uses **`hydro=:enzo, gravity=:enzo`** (native
Enzo kernels driven by the Julia `evolve_level!` loop — still Vespa). This is
verified to match two-fluid linear theory. Data: `cicass_highz_pk_natenzo.dat`.
The `:julia` GPU-hydro path is an orthogonal optimization to repair later.
