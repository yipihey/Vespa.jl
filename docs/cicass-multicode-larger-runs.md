# Running the CICASS 3-code comparison at larger scale

This is the operational guide for the CICASS streaming-velocity multi-code
comparison (Enzo + RAMSES + Arepo on one CICASS realization, z=1000→20, shared
reduced H+D chemistry + Compton drag), and — importantly — **what has to change
to make the comparison scientifically meaningful at larger scale**. The `c64`
(64³, 128 kpc/h) run set that exists today is a *plumbing* gate: it proves the
three codes boot from the same ICs and run to z=20 with the same physics knobs.
It is **not** a valid science comparison (see "Why c64 is too small").

## The run set (what exists)

Each code has a self-contained launcher in the repo root that pins the shared
cosmology + output redshift list and writes a log + a `.dat` P(k) table:

| code   | launcher              | driver                                   | output `.dat`                |
|--------|-----------------------|------------------------------------------|------------------------------|
| Enzo   | `run_c64_enzo.sh`     | `lib/MultiCode/examples/cicass_highz_pk.jl` | `cicass_highz_pk_c64.dat`    |
| RAMSES | `run_c64_ramses.sh`   | `lib/MultiCode/examples/cicass_ramses_pk.jl`| `cicass_ramses_pk_c64.dat`   |
| Arepo  | `run_c64_arepo.sh`    | `lib/MultiCode/examples/cicass_arepo_pk.jl` | `cicass_arepo_pk_c64.dat`    |

CICASS analytic two-fluid linear P(k) at the output redshifts:
`lib/MultiCode/examples/cicass_linear_at_outputs.jl` → `cicass_linear_pk.dat`
(it **clobbers** that file — back it up first).

Shared knobs (identical across all three launchers — keep them in sync):
```
CIC_BOX=0.128 CIC_NGRID=64 CIC_OMEGAM=0.27 CIC_VBC=30.0 CIC_ZSTART=1000.0 CIC_ZEND=20.0
CIC_ZOUT="1000,680,460,315,215,145,100,65,45,30,20"   # consistent output redshifts
CIC_CHEM=1 CIC_CHEM_ENGINE=kernels CIC_COMPTON_DRAG=1  # shared in-repo chemistry + drag
```

Plot: `plot_threecode_catchup_c64.py` → `reports/multicode/threecode_catchup_c64.png`
(reads the three `.dat` files above + `cicass_linear_pk.dat`; robustly skips any
missing input). NB the Enzo file is `cicass_highz_pk_c64.dat`, *not*
`cicass_enzo_pk_c64.dat`.

## Why c64 is too small (read before trusting any plot)

128 kpc/h box, 64³:
- **k_fundamental = 2π/L ≈ 49 h/Mpc.** The interesting k-range (≈60–700 h/Mpc)
  is only ~1.4 to ~14 fundamental modes — almost no large-scale dynamic range,
  and the lowest bins rest on a handful of modes (large sample variance).
- **Shot-noise floor P_shot = V/N = L³/N_particles ≈ 8e-9 (Mpc/h)³.** The DM
  P(k) drops to ~1–3× that floor by k≈300–500, so the small-scale end is
  noise-dominated, and the codes subtract/deconvolve shot noise differently.
- **δb/δc is a confounded diagnostic.** δb/δc = √(P_b/P_dm) mixes the baryon
  *and* DM fields. The codes agree on DM P(k) at z=1000 to <4% (same
  realization) but diverge in DM growth by z=20 (at k=70: RAMSES ≈0.60×,
  Arepo ≈1.41× of Enzo's DM power — all three *under*-grow large-scale DM vs the
  linear D(a)², by 25–50%). That denominator divergence dominates the δb/δc
  plot, so it is **not** a clean baryon-catch-up comparison. Diagnose DM growth
  directly (`P_dm(z)/P_dm(z_init)` per code vs the linear D(a)²) before reading
  δb/δc.

The cross-code DM-growth divergence is the known **Vespa GPU DM under-growth**
issue (gravity / leapfrog-timestep / PM-resolution), now visible across all
three solvers. It must be understood independently of the baryon physics.

## How to scale up (what to change)

To make the comparison meaningful you need (a) many low-k modes and (b) the
signal well above the shot-noise floor across the compared k-range.

1. **Bigger box and/or more particles.**
   - More low-k dynamic range: raise `CIC_BOX` (e.g. 1–4 Mpc/h) so k_fundamental
     drops well below the scales of interest. Trade-off: larger box at fixed N
     coarsens the grid → raises the smallest resolved k_Nyquist = π N / L.
   - Lower shot noise: raise `CIC_NGRID` (P_shot = L³/N³ falls as N⁻³). 128³ is
     8× the particles of 64³ (P_shot ÷8); 256³ is 64×.
   - `CIC_SUB=s` subsamples a CICASS N³ realization to an (N/s)³ code load
     (block-avg gas, decimate DM ×s³) — use to keep the CICASS realization fixed
     while cheapening a code's particle load; note DM decimation *raises* its
     shot noise, so prefer a native large N for the science run.

2. **Arepo: escape the single-rank in-process limit.** The in-process
   `ArepoLib` bridge is one MPI rank = one core; a 128³ native run is hours
   single-threaded and `MaxMemSize` must be large (32000 MB booted 4.19M
   particles; 4000 OOMs the gravity tree). For real scale, drive Arepo as a
   **child-process MPI** job (mpiexec N ranks), like Gadget4Lib — the in-process
   path cannot parallelize. PMGRID (currently 256) should track the particle
   load.

3. **Rebuild `libarepo3d_cosmo.dylib` WITH the per-step-leak fix.** A regression
   in the PowerFoam pre-flux trace export (`bridge_preflux_capture` in
   `arepo/src/hydro/finite_volume_solver.c`) leaked ~900 MB/step (only freed at
   `Ti_Current==0`); the fix clears the snapshots at the start of each new step.
   On a fresh machine, rebuild and verify RSS plateaus:
   ```
   cd ~/Projects/arepo
   cp Config_cosmo.sh Config.sh
   PATH=/opt/homebrew/bin:$PATH SYSTYPE=Darwin-Homebrew make shared   # ~30s incremental
   cp libarepo.dylib libarepo3d_cosmo.dylib
   ```
   (On Linux: set the appropriate `SYSTYPE` in the Makefile; deps = mpicc, fftw3,
   gsl, hdf5, gmp. `SHARED_EXT` becomes `so`, so the product is `libarepo.so`.)

4. **Memory discipline in the Arepo driver (`cicass_arepo_pk.jl`).** Already
   wired and should stay:
   - `OutputListOn` defaults **off** — Arepo HDF5 snapshots are redundant
     (`record!` captures P(k) from the live Voronoi cells in-process). Re-enable
     with `CIC_AREPO_SNAPSHOTS=1` only if you actually want the dumps.
   - Per-step `Sys.maxrss` logging (free) so memory is always visible.
   - Hard ceiling `CIC_RSS_CEIL_MB` (default 50000) aborts the step loop cleanly
     before a leak can exhaust RAM. **Raise it on a big-RAM machine.**

5. **Chemistry / GPU.** `CIC_CHEM_ENGINE=kernels` (in-repo ChemistryKernels) runs
   in-process; `CIC_CHEM_BACKEND=metal|cpu` and `BACKEND=metal|cpu`. CPU f64 ≡
   Metal f32 to all digits and chem is ~0.5% of runtime, so on a non-Apple box
   set `BACKEND=cpu CIC_CHEM_BACKEND=cpu` (or a CUDA backend if added). The
   Metal chem path itself does **not** leak (verified by an isolated 80-iter
   probe).

6. **Keep `CIC_ZOUT` identical across the three launchers** so all codes write
   P(k) at the same redshifts. Enzo and RAMSES cap their timestep onto the exact
   output a; Arepo (`OutputListOn` off) captures at the first step with
   `a ≥ a_out`, a ≤`MaxSizeTimestep` (≤2%) overshoot — fine for P(k), and visible
   in the Arepo `.dat` z labels (e.g. 675 vs 680).

## Repository manifest (exact clone list)

The whole framework is a path-`[sources]` tree rooted at `~/Projects/` (the
parent of `enzo-dev/`). Clone **all** of these as siblings — every one is a
`[sources]` path that must exist for `lib/MultiCode/test` to instantiate, even
the codes the CICASS run doesn't exercise. All are under `github.com/yipihey`:

| local dir (`~/Projects/…`) | clone | branch | needed for |
|---|---|---|---|
| `enzo-dev` | `yipihey/enzo-dev` | `enzong-amr-subcycling-refluxing` | Enzo + Vespa.jl (this tree) |
| `Arepo.jl` | `yipihey/Arepo.jl` | `main` | ArepoLib wrapper |
| `RamsesNG.jl` | `yipihey/RamsesNG.jl` | `main` | RamsesLib wrapper |
| `CICASS.jl` | `yipihey/CICASS.jl` | `main` | CICASSLib (IC wrapper) |
| `r3djl` | `yipihey/r3djl` | `main` | R3D (transitive dep) |
| `Gadget4.jl` | `yipihey/Gadget4.jl` | `main` | instantiate (Gadget4Lib) |
| `DiscoDJ.jl` | `yipihey/DiscoDJ.jl` | `main` | instantiate (DiscoDJLib) |
| `Music.jl` | `yipihey/Music.jl` | `main` | instantiate (MusicLib) |
| `Athena.jl` | `yipihey/Athena.jl` | `main` | instantiate (AthenaLib) |
| `dfmm` | `yipihey/dfmm` | `main` | optional engine ext |
| `HierarchicalGrids.jl` | `yipihey/HierarchicalGrids.jl` | `feat-cell-average-fieldset` | HGBackend (not the CICASS path) |

Native-code source repos (the compiled libs do **not** move — rebuild on host):

| local dir | clone | branch | builds |
|---|---|---|---|
| `arepo` | `yipihey/arepo` | `arepo-jl-bridge` | `libarepo3d_cosmo.dylib` (incl. the per-step leak fix) |
| `mini-ramses` | `yipihey/mini-ramses-metal` | **`cicass-cosmology`** | RAMSES `bin64*` flavors |
| `cicass` | `yipihey/CICASS` | `main` | CICASS C IC tool (transfer.x) |

⚠️ **mini-ramses branch matters.** The validated `bin64*` libs and every CICASS
run were built from the `cicass-cosmology` branch, which is the old `develop`
base + the `ramses_boost_particles` capi commit. The remote `develop` is ~165
commits ahead (a large upstream GPU/CUB merge) and would change behavior — so
clone **`cicass-cosmology`** to reproduce, and decide separately whether/when to
rebase onto the new upstream (re-validation required).

Native libs to build on the new host (Julia binary is juliaup-managed and not on
the non-interactive PATH — use the absolute path):
- **Enzo grid dylib**: `EnzoModules/deps/build_grid_darwin.sh` (Linux: the
  equivalent; deps = mpicc, hdf5).
- **Arepo cosmo dylib**: see step 3 above (deps = mpicc, fftw3, gsl, hdf5, gmp).
- **RAMSES flavor `bin64sc_chem`** (the one `run_c64_ramses.sh` uses) from
  `mini-ramses` (gfortran). Verify it does not hard-link grackle, since the run
  uses the in-repo ChemistryKernels engine, not the C grackle.
- **grackle is NOT required** for the default CICASS run (`CIC_CHEM_ENGINE=kernels`).
  Only needed if you switch a code to `engine=:grackle` (the C reduced lib).
- Python for the plots: any env with `numpy`+`matplotlib` (the scripts hard-code
  `~/Projects/disco-dj-fem/.venv/bin/python` — repoint it on the new host).

## Moving to a bigger machine — checklist

- Clone every repo in the manifest above as a sibling of `enzo-dev/`.
- Build the native libs on the new host: Enzo grid dylib
  (`EnzoModules/deps/build_grid_darwin.sh`, or the Linux equivalent), RAMSES
  flavors (`bin64h`/`bin64hrt`/`bin64sc`), and the patched Arepo cosmo dylib
  (step 3 above). The Julia binary is juliaup-managed and not on the
  non-interactive PATH — use the absolute path.
- For real science volumes: prefer a multi-core/MPI host (Arepo + RAMSES scale
  with ranks), large RAM (N³ particles + Voronoi mesh), and a capable GPU if
  using the Metal/CUDA gravity+chem paths. Pick (box, N) so the compared
  k-range has P ≫ P_shot = L³/N³ and ≳ tens of modes per bin at the largest
  scale of interest.
- Re-run the launchers (bump `CIC_NGRID`/`CIC_BOX`, raise `CIC_RSS_CEIL_MB`),
  regenerate `cicass_linear_pk.dat` at the chosen cosmology, and plot.
