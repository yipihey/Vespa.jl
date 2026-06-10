# EnzoNG framework — planned next steps

The recorded, already-scoped directions beyond the completed program
(ADR-0006 phases 0–7, Next-1…14, Phase C). Each item is grounded in work that
has *already been done* — the on-ramp is concrete, the certification pattern is
known, and the recorded gotchas are listed so a fresh session starts warm.

These are deliberately *not* in the ADR's numbered "Next" list: that list
tracks completed increments. This file tracks intent. When an item lands, move
it into the ADR status appendix as a `Next-N (done)` entry and delete it here.

Status of the program itself: every wrapper in the registry carries a live
cross-code gate; both legacy multigrids (RAMSES, Enzo) have machine-precision
KA replacements; the slot architecture produces bit-identical orbits on live
Enzo AMR. What follows extends reach, not correctness.

---

## Track A — solver convergence (the D5 endgame)

**A1. Athena++ per-stage solver slots on Enzo / HG grids.**
The guest-slot pattern proven for PPMKernels-in-RAMSES, applied the other
direction: run Athena++'s per-stage Riemann solve as a `:julia`/guest slot on
an Enzo or HierarchicalGrids mesh. Athena++ is already in-process and
re-entrant (AthenaLib), and `read_vtk` gives full-domain CellSet readback — so
the missing piece is exposing a *single VL2 stage* through the C-ABI rather
than a whole `run`. On-ramp: extend `athena_capi.cpp` with a
`athena_stage(meshblock, dt)` entry beside `athena_main`; raster an Enzo grid
into one MeshBlock (the Next-13 CellSet machinery, inverted); gate against the
existing `:hydro` full-run on the same IC.
Recorded gotcha: Athena++ statics size at first init — run the
highest-dimensional configuration first (the 1D→3D segfault from Next-13).

**A2. More GR spacetimes (now one flavor line away).**
The `:gr` flavor (Next-14) established the recipe: a `build_flavor` line
upstream + a `LazyLib` flavor + a stationary-solution gate. The obvious next
spacetimes, each with a known analytic or quasi-stationary oracle:
- **Kerr** (spinning Kerr-Schild) — `gr_bondi` already runs on Kerr-Schild
  coordinates; raise the spin and gate Bondi-like stationarity.
- **Fishbone–Moncrief torus** (`gr_torus` pgen) — the hydrostatic-equilibrium
  torus must hold its pressure maximum; gate on the inner-edge radius.
- **GR shock tube** (`gr_shock_tube` pgen) — a relativistic Riemann problem
  with an exact solution, the GR analogue of the Sod harness row.
On-ramp: one line in `athena/deps/build_athena_darwin.sh`, one `LazyLib`
entry, one stationarity/exact gate in the AthenaLib suite — `read_vtk` already
clamps singleton dims for the 1-D/2-D outputs these problems use.

---

## Track B — initial conditions (fixed-and-paired, cross-generator)

**B1. MUSIC noise-file writer (shared-phases tasks 19–20).**
The shared-NGenIC-phases work demonstrated fixed-and-paired (Angulo–Pontzen)
ICs via Fourier-noise *injection* in DISCO-DJ (paired corr −0.9999, residual =
the correct 2LPT even term) and confirmed GADGET-4's native
`NGENIC_FIX_MODE_AMPLITUDES`/`MIRROR_PHASES` flags. The remaining piece is the
symmetric one for MUSIC: a writer that emits a MUSIC-format noise file from a
given white-noise field, so one realization (and its phase-mirror) can be
forced through all three IC generators identically.
On-ramp: MUSIC reads a noise file per level (`random/file[N]`); the writer is
the inverse of `get_ngenic_noise` already wrapped in MusicLib. Gate: feed the
written file back and recover the same δ field; then cross-check MUSIC↔DISCO-DJ
phases at corr ≈ 1 (vs the current 0.99725 from shared-seed-only).

**B2. The `ka_poisson` plugin seam for MUSIC zoom.**
MUSIC's zoom multigrid is bitwise-identical across accuracy 1e-4→1e-8
(discretization-limited; its k-space path is unigrid-only). PoissonKernels now
has the certified Dirichlet machinery (Next-4/6, Phase C) that *is* a
discretization-exact MG. Wiring it in as a MUSIC gravity plugin would let the
zoom solve match the unigrid k-space reference. On-ramp: MUSIC exposes a
`poisson` plugin interface; the masked/Dirichlet `vcycle_solve!`/`masked_cg!`
already solve arbitrary refined-region shapes. This is the same "slot, not
port" move as the RAMSES gravity guest.

---

## Track C — science campaigns (run the certified machinery at scale)

**C1. Santa Barbara to refinement onset, then a GPU production campaign.**
Phase C wired the certified subgrid gravity hook into `sb_metal_amr` and ran 8
cycles from z=63 (single root grid, CPU-f32 ≡ Metal-f32). The next step is
*duration*: run far enough that the cluster collapses and the hierarchy
refines, exercising the armed subgrid path under live AMR — then a full GPU
production run with timing at depth (the kernel-level speedups are 31× hydro /
12× multigrid, currently masked by startup at 8 cycles).
On-ramp: increase `maxcyc`, let `regrid=true` build levels; the hook already
writes the root φ to `PotentialField` so child BCs read our solution.

**C2. Structure-formation run on the f32 reference build.**
The `f32` Enzo bridge flavor (p4_b4, `ENZOMODULES_GRID_LIB`-selected) is the
faithful-precision CPU reference for the Metal kernels (ρ parity 4e-4 on SB).
With Phase C's level>0 gravity hook now done (it was the open TODO in the f32
notes), a from-ICs structure-formation run becomes the end-to-end science gate:
EnzoNG-Metal vs enzo-f32 trajectories on a real cosmological volume.

---

## Track D — packaging polish (deferred, not blocked)

**D1. Extension-ify the legacy wrappers in MultiCode.**
EnzoLib/RamsesLib/ArepoLib are currently hard deps of MultiCode. The five
wrapper *generators* (dfmm, Athena, MUSIC, DISCO-DJ, GADGET-4) already moved to
package extensions; the three core wrappers can follow the same pattern
(`[weakdeps]` + `[extras]` + `[extensions]`, the rule the dfmm extension
documents). Deliberately deferred: these are lazy pure-Julia bindings (no
`dlopen` until first use, no load-time burden), so the conversion buys little
until a registry release forces the weak-dep discipline. Recorded so it is a
decision, not an oversight.

---

*Maintenance:* keep this list grounded — every entry must point at completed
work it extends and name its first concrete step. If an item cannot name its
on-ramp, it is research, not a planned step, and belongs in a notebook rather
than here.
