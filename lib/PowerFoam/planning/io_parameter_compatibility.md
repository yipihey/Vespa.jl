# AREPO IO And Parameter Compatibility Audit

Date: 2026-06-15

## Scope

This note audits the current PowerFoam-side evidence for AREPO-style runtime
compatibility.  It only covers file/parameter surfaces already visible in this
repo: example case writers, snapshot analyzers, planning docs, and the current
runtime scaffold.  It does not claim that package-source parity exists today.

## What Exists Today

The repo already has two example families that speak in AREPO's file language
rather than only through the live bridge:

| Surface | Evidence in repo | What it proves |
| --- | --- | --- |
| Sedov proxy case generation | `lib/PowerFoam/examples/arepo_sedov_proxy/` | PowerFoam can emit matched AREPO case directories with `IC.hdf5`, `param.txt`, `Config.sh`, and optional `output_list.txt` |
| Noh proxy case generation | `lib/PowerFoam/examples/arepo_noh_proxy/` | Same case-directory pattern, plus a generated `build_and_run.sh` helper |
| Snapshot analysis | `profile_snapshots.py` + `profile_*snapshot.c` in both proxy dirs | The repo already knows the minimal snapshot layouts and HDF5 fields needed for post-run parity diagnostics |
| Runtime planning | `lib/PowerFoam/src/arepo_runtime_scaffold.jl` and `lib/PowerFoam/planning/arepo_jl_full_rewrite_master_plan.md` | The rewrite plan already names parameter parsing, snapshot IO, and output policy as first-class runtime workstreams, but not as implemented runtime modules yet |

Most other `lib/PowerFoam/examples/arepo_*.jl` files are bridge or parity gates
for geometry, gradients, traces, hierarchy, and standard-problem comparisons.
They are important physics gates, but they do not add general parameter-file or
snapshot compatibility on their own.

## Current Compatibility Surfaces

### 1. Parameter-file surface (`param.txt`)

The two `write_arepo_cases.py` helpers define the only explicit AREPO parameter
templates in this repo today.  The minimum shared runtime surface visible there
is:

- input/output routing:
  - `InitCondFile`
  - `ICFormat`
  - `OutputDir`
  - `SnapshotFileBase`
  - `SnapFormat`
  - `NumFilesPerSnapshot`
  - `NumFilesWrittenInParallel`
  - `OutputListOn`
  - `OutputListFilename`
- time/output control:
  - `TimeBegin`
  - `TimeMax`
  - `TimeBetSnapshot`
  - `TimeOfFirstSnapshot`
  - `TimeBetStatistics`
  - `CpuTimeBetRestartFile`
  - `TimeLimitCPU`
- domain/cosmology toggles:
  - `BoxSize`
  - `PeriodicBoundariesOn`
  - `ComovingIntegrationOn`
  - `Omega0`
  - `OmegaBaryon`
  - `OmegaLambda`
  - `HubbleParam`
- hydro timestep and stability knobs:
  - `CourantFac`
  - `MaxSizeTimestep`
  - `MinSizeTimestep`
  - `TypeOfTimestepCriterion`
  - `LimitUBelowThisDensity`
  - `LimitUBelowCertainDensityToThisValue`
  - `InitGasTemp`
  - `MinGasTemp`
  - `MinEgySpec`
  - `MinimumDensityOnStartUp`
- mesh/domain-decomposition knobs:
  - `DesNumNgb`
  - `MaxNumNgbDeviation`
  - `MultipleDomains`
  - `TopNodeFactor`
  - `ActivePartFracForNewDomainDecomp`
  - `CellShapingSpeed`
  - `CellMaxAngleFactor`
- gravity/softening placeholders needed by these AREPO runs:
  - `ErrTolIntAccuracy`
  - `ErrTolTheta`
  - `ErrTolForceAcc`
  - `GasSoftFactor`
  - `SofteningComovingType0..5`
  - `SofteningMaxPhysType0..5`
  - `SofteningTypeOfPartType0..5`
  - `GravityConstantInternal`

For Arepo.jl parity, a Julia runtime needs to accept at least this subset of
AREPO-style parameters and map them onto a structured runtime state.  Right now
that mapping exists only inside example-local Python writers, not in a reusable
Julia runtime module.

### 2. Compile/runtime option surface (`Config.sh` plus solver staging)

The examples also encode a second compatibility layer in `Config.sh`.  The
currently exercised knobs are:

- shared hydro/build toggles:
  - `TWODIMS`
  - `DOUBLEPRECISION=1`
  - `INPUT_IN_DOUBLEPRECISION`
  - `OUTPUT_IN_DOUBLEPRECISION`
  - `HAVE_HDF5`
  - `OUTPUT_CENTER_OF_MASS`
  - `OUTPUT_VOLUME`
  - `OUTPUT_PRESSURE`
  - `OUTPUT_VERTEX_VELOCITY` (Sedov proxy)
- mesh-motion/regularization toggles:
  - `REGULARIZE_MESH_CM_DRIFT`
  - `REGULARIZE_MESH_CM_DRIFT_USE_SOUNDSPEED`
  - `REGULARIZE_MESH_FACE_ANGLE`
  - `FORCE_EQUAL_TIMESTEPS` (Sedov proxy)
  - `TREE_BASED_TIMESTEPS` (Noh proxy)
- hydro algorithm toggles:
  - `LOCAL_PPM`
  - `RIEMANN_HLL`
  - `ARTIFICIAL_BULK_VISCOSITY`
  - `ARTIFICIAL_BULK_VISCOSITY_QUAD`
  - `ARTIFICIAL_BULK_VISCOSITY_LINEAR`
  - `ARTIFICIAL_BULK_VISCOSITY_PRESSURE_JUMP`
  - `ARTIFICIAL_BULK_VISCOSITY_PRESSURE_CAP`
- proxy-specific experimental hook:
  - `SHOCK_FOLLOWING_MESH`
  - `SHOCK_FOLLOWING_MESH_GAIN`
  - `SHOCK_FOLLOWING_MESH_WIDTH`
  - `SHOCK_FOLLOWING_MESH_RMIN`

For rewrite parity, Arepo.jl needs a compatibility story for both:

- compile-time AREPO flags that become Julia runtime features or backend
  selectors, and
- runtime solver choices that should no longer be hidden behind ad hoc example
  scripts.

### 3. Initial-condition compatibility (`IC.hdf5`)

`lib/PowerFoam/examples/arepo_sedov_proxy/csv_to_arepo_ic.c` defines the only
explicit AREPO IC writer contract in this repo.  The current generated HDF5 IC
surface is:

- group `Header` with attributes:
  - `NumPart_ThisFile`
  - `NumPart_Total`
  - `NumPart_Total_HighWord`
  - `MassTable`
  - `Time`
  - `Redshift`
  - `BoxSize`
  - `NumFilesPerSnapshot`
  - `Omega0`
  - `OmegaB`
  - `OmegaLambda`
  - `HubbleParam`
  - `Flag_Sfr`
  - `Flag_Cooling`
  - `Flag_StellarAge`
  - `Flag_Metals`
  - `Flag_Feedback`
  - `Flag_DoublePrecision`
  - custom marker `PowerFoamProxy`
- group `PartType0` with datasets:
  - `ParticleIDs`
  - `Coordinates`
  - `Masses`
  - `Velocities`
  - `InternalEnergy`

That is enough for the current proxy cases, but it is still a narrow,
example-local writer.  Arepo.jl parity needs a general Julia-side IC reader and
writer for this schema rather than a C helper compiled from the example
directory.

### 4. Snapshot compatibility

The two `profile_snapshots.py` analyzers show the snapshot layouts and HDF5
fields that current parity diagnostics already depend on.

Supported snapshot path layouts:

- direct single-file snapshots:
  - `output/snap_###.hdf5`
- split-directory snapshots:
  - `output/snapdir_###/snap_###.0.hdf5`

Required or currently consumed snapshot fields:

- `Header/Time`
- `Header/BoxSize` (Noh analyzer)
- `PartType0/Density`
- `PartType0/Masses`
- `PartType0/InternalEnergy`
- `PartType0/Velocities`
- `PartType0/Volume` or a fallback derived from `mass / density` (Noh only)
- `PartType0/Pressure` or a fallback derived from EOS (Sedov only)
- `PartType0/CenterOfMass` if present, otherwise `PartType0/Coordinates`

This is already enough to define a minimum snapshot-compatibility target for
Arepo.jl.  The missing part is packaging it as reusable runtime IO instead of
example-specific analysis code.

### 5. Diagnostic/output compatibility

Current diagnostics are split between:

- bridge-facing parity artifacts under `lib/PowerFoam/examples/out/`
- proxy analysis outputs such as:
  - `radial_profiles.csv`
  - `metrics.csv`
  - `evolution_metrics.csv`
  - `noh_cell_values.csv`
  - `noh_radial_bins.csv`
  - `noh_metrics.csv`
  - `noh_evolution_metrics.csv`
  - SVG summary plots

For Arepo.jl parity, the runtime needs a stable output policy that can emit:

- AREPO-compatible snapshots for cross-code comparison,
- lightweight comparison-friendly CSV/JSON summaries, and
- restart/output cadence driven by the same parameter layer as the run itself.

## Required Missing Surfaces

The file inventory and scaffold docs point to a consistent gap list:

1. No general Julia parameter/config parser
   - The current parameter logic lives in example-local Python templates.
   - The runtime scaffold still treats parameter parsing as an unsupported
     surface.

2. No general Julia snapshot/IC IO module
   - IC writing is currently delegated to `csv_to_arepo_ic.c`.
   - Snapshot reading is currently delegated to example-local C analyzers.
   - There is no `src` module dedicated to AREPO HDF5 schema compatibility.

3. No restart-file compatibility layer
   - The examples mention restart cadence (`CpuTimeBetRestartFile`) but there is
     no visible runtime surface for reading or writing restart files.

4. No reusable output/diagnostic policy layer
   - CSV/SVG analysis exists, but only as example tooling.
   - The runtime scaffold explicitly still lists snapshot IO and output policy
     as unsupported.

5. No unified mapping from `Config.sh` feature flags to Julia runtime features
   - Solver choice, `LOCAL_PPM`, mesh-regularization switches, equal-timestep
     vs tree-based timestep modes, and proxy-only hooks are still staged by
     hand in example scripts.

6. No single entrypoint that binds parameter parsing, IC loading, run policy,
   and snapshot writing together
   - The rewrite master plan says `Arepo.jl` should eventually provide this
     through a real runtime/API layer.
   - The current `arepo_run_scaffold` exists only as a planning stub.

## What Must Exist For Arepo.jl Parity

Before the rewrite can claim AREPO-style IO/parameter compatibility, the repo
needs at least these durable Julia-side surfaces:

1. `param.txt` reader and normalizer for the currently exercised proxy keys.
2. `Config.sh` compatibility mapper for the already used hydro/mesh flags.
3. HDF5 IC reader/writer for the current `Header` + `PartType0` schema.
4. Snapshot reader that supports both `snap_###.hdf5` and
   `snapdir_###/snap_###.0.hdf5`.
5. Snapshot writer with the fields already consumed by the proxy analyzers.
6. Output-list and snapshot-cadence policy equivalent to the current
   `OutputListOn` / `OutputListFilename` / `TimeBetSnapshot` flow.
7. Lightweight diagnostic writers so bridge parity and pure-Julia parity can
   share the same post-processing contract.
8. A runtime entrypoint that accepts either a problem spec or an AREPO-style
   parameter file and drives the same output surfaces.

## Practical Conclusion

PowerFoam already has strong bridge and physics-gate coverage, and it already
has two narrow AREPO-compatible file workflows for Sedov and Noh proxy cases.
What it does not yet have is a reusable Julia runtime layer that owns those
same parameter, IC, snapshot, restart, and diagnostic surfaces centrally.

That is the compatibility gap the main Arepo.jl rewrite needs to close: move
from example-local writers/analyzers plus bridge gates to a package-level
runtime that can ingest AREPO-style inputs and emit AREPO-style comparison
artifacts on demand.
