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
Implementation status: the sibling Athena source now has a `from_binary`
problem generator and a `:stage` flavor (`libathena_capi_stage`) that ingests
one raw conserved MeshBlock state, advances it through Athena's normal VL2
driver for a host-supplied `dt`, and returns primitive VTK readback through
`AthenaLib.advance_hydro_uniform`. The live A1 gate runs a uniform 8^3 conserved
state through the stage flavor in a fresh worker and verifies density,
momentum, energy, and final time. Remaining work is the final Enzo/HG adapter
that rasters hierarchy blocks into this stage surface and syncs them back.

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
Implementation status: the sibling Athena build script now produces
`:gr_kerr`, `:gr_torus`, and `:gr_shock` dylib flavors, and AthenaLib exposes
matching `LazyLib` entries. The live A2 gates pass: Kerr-Schild Bondi with
spin `a=0.5` stays within the measured 2% short-run drift gate, the
Fishbone-Moncrief torus smoke test preserves finite positive pressure/density,
and the relativistic shock-tube flavor evolves a finite, positive density
profile in a fresh worker. The tests deliberately isolate pgen-changing
flavors in fresh Julia workers because separately configured Athena++ dylibs
share C++ symbol/global names inside one process.

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
Implementation status: the sibling `Music.jl` checkout now has
`write_music_noise`, `read_music_noise`, `mirror_noise`, and
`MusicSpec.noise_files`, with pure round-trip/config coverage plus a live
MUSIC generation gate that writes a normalized 32^3 noise file, feeds it back
through `seed[5]`, and verifies an Enzo IC product. MusicLib is green at
`61/61`. MultiCode now has a combined MUSIC/DISCO-DJ phase audit extension;
the report at `reports/multicode/shared_phases_and_zoom_poisson.md` shows the
MUSIC fixed/mirror controls at +1/-1 and the current same-seed
MUSIC↔DISCO-DJ proxy correlation at `0.011234`, i.e. the two generators are
deterministic but do not yet share an explicit white-noise realization.

**B2. The `ka_poisson` plugin seam for MUSIC zoom.**
MUSIC's zoom multigrid is bitwise-identical across accuracy 1e-4→1e-8
(discretization-limited; its k-space path is unigrid-only). PoissonKernels now
has the certified Dirichlet machinery (Next-4/6, Phase C) that *is* a
discretization-exact MG. Wiring it in as a MUSIC gravity plugin would let the
zoom solve match the unigrid k-space reference. On-ramp: MUSIC exposes a
`poisson` plugin interface; the masked/Dirichlet `vcycle_solve!`/`masked_cg!`
already solve arbitrary refined-region shapes. This is the same "slot, not
port" move as the RAMSES gravity guest.
Implementation status: `MusicSpec.poisson_solver = :ka_poisson` now writes
`[poisson]/solver = ka_poisson` and `kspace = no`; native MUSIC honors the
solver override and registers a `ka_poisson` plugin stub that fails explicitly
with the KernelAbstractions/PoissonKernels bridge-not-linked diagnostic. The
C-ABI dylib was rebuilt with this plugin, and MusicLib's live suite is green at
`61/61`, including the native seam check.

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
Implementation status: the campaign harness now records per-cycle CSV and
Markdown diagnostics, detects refinement onset, and can stop on the first AMR
cycle via `SB_STOP_ON_REFINEMENT=1`; see
`reports/multicode/santa_barbara_campaign.md`. The measured CPU-f32 and
Metal-f32 refinement-onset pair now both reach the first level-1 grid at cycle
42 (`[1, 1, 0]`); the comparison report
`reports/multicode/santa_barbara_campaign_comparison.md` has 43 matching rows,
matching cycles/refinement, zero reported rho-max drift at the printed
precision, and a 3.142× Metal speedup including warmup.

**C2. Structure-formation run on the f32 reference build.**
The `f32` Enzo bridge flavor (p4_b4, `ENZOMODULES_GRID_LIB`-selected) is the
faithful-precision CPU reference for the Metal kernels (ρ parity 4e-4 on SB).
With Phase C's level>0 gravity hook now done (it was the open TODO in the f32
notes), a from-ICs structure-formation run becomes the end-to-end science gate:
EnzoNG-Metal vs enzo-f32 trajectories on a real cosmological volume.
Implementation status: the short f32 structure-formation gate has run for 6
cycles from the Santa Barbara ICs with native `enzo-f32` as the reference and
EnzoNG Metal-f32 as the candidate. See
`reports/multicode/santa_barbara_structure_f32.md`: both reach `t=0.933129`,
rho relL2 is `3.766e-04`, TE/GE relL2 are about `2.6e-04`, and EnzoNG Metal is
8.28× faster per cycle (15.18× in hydro, 4.10× in gravity).

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
Implementation status: this pass audited the split boundary rather than doing
a risky partial refactor. See
`reports/multicode/legacy_weakdeps_split.md` for the concrete migration order:
pure specs/helpers first, then Enzo/RAMSES/Arepo extension surfaces, then the
multi-wrapper MUSIC/DISCO-DJ/Moray extensions, and only finally moving the
three wrappers from `[deps]` to `[weakdeps]`.

---

## Track E — GPU performance & out-of-core scale

**Context (completed work this extends).** The CICASS z=1000→20 cosmology run
(`lib/MultiCode/examples/cicass_highz_pk.jl`) was taken from 391s → 111s on Metal:
a GPU CIC deposit that **bypasses Enzo's `PrepareDensityField`** (gravity source
rebuilt on the GPU, `corr=1.0` vs the live GMF — `lib/PoissonKernels/src/deposit.jl`),
the fast one-ghost **Local PPM** hydro slot with species-colour advection added to
`muscl_hancock_step_3d!`, and **CFL parity** with RAMSES (Courant 0.8, da/a 0.1, one
`CIC_COURANT` knob both codes share). Against RAMSES-Metal (45s) on **bit-identical**
CICASS ICs (`/tmp/verify_ics.jl`: same seed ⇒ `max|Δ|=0` on all fields), Enzo is now
**2.5×** (was 8.7×). The residual gap was profiled to be **structural**, not
incremental — measured dead ends (do not re-try): GPU φ→accel removes ≤4% of
`session_gravity_post` (the 2M-particle `InterpolateParticlePositions`/drift dominate),
a cross-slot field cache saves ~12ms/cyc of ~55ms (most copies are non-redundant), and
skipping `session_copy_baryon_to_old` is **unsafe** (changes baryon large-scale P(k)
~2.26× — `OldBaryonField` feeds the comoving/gravity time-centering).

**E0. Out-of-core capacity is the design goal — keep Enzo state HOST-RESIDENT (decision,
not an oversight).** RAMSES's speed edge comes from keeping the whole state GPU-resident,
which caps problem size at device memory (~a few ×128³ on this hardware). Enzo's
host-resident model + per-slot host↔device staging is deliberately retained so we can
run problems **far larger than device memory** later — the per-step cost (the 2.5×) is
the accepted price for that capacity. So do NOT pursue "make Enzo GPU-resident": it
would forfeit the capacity advantage that motivates using Enzo here at all.

**E1. Tiled / streaming GPU offload (the on-ramp when problems exceed device memory).**
The right way to get GPU throughput on larger-than-device grids is to process the grid
in **tiles that fit on the device, streamed host↔device with halos**, rather than a
one-shot full-field upload. First concrete step: make the LocalPPM hydro slot
(`hydro_localppm!`) tile the active grid into device-sized blocks with a 1-cell ghost
ring per block (it is already a one-ghost local stencil — minimal halo), looping
upload→sweep→download per tile; measure on a grid sized past device memory (e.g. 256³)
where the current whole-field path OOMs. The GPU CIC deposit and FFT Poisson already
have the structure to tile similarly (deposit is scatter-with-atomics per particle
batch; the root FFT is the one genuinely global step — keep it whole or use a
slab-decomposed transform). Until a problem actually exceeds device memory this is
research, not a step; the trigger is the first OOM in a science campaign (Track C).

---

*Maintenance:* keep this list grounded — every entry must point at completed
work it extends and name its first concrete step. If an item cannot name its
on-ramp, it is research, not a planned step, and belongs in a notebook rather
than here.
