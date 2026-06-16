# PowerFoam IO Runtime Module Design

Date: 2026-06-15

## Current Status

As of 2026-06-15, `lib/PowerFoam/src/arepo_io_snapshots.jl` contains a real
minimal HDF5 snapshot backend, and `lib/PowerFoam/Project.toml` now declares
`HDF5`. The smoke gate should now preflight dependency visibility explicitly so
that a missing `HDF5` declaration, a stale manifest, or an unresolved active
project each point to a concrete `Pkg` action rather than collapsing into a
generic "backend unavailable" message.

## Purpose

This note proposes a package-level IO module skeleton for the future Julia
runtime surface described in `runtime_api_scaffold.md` and audited in
`io_parameter_compatibility.md`. It names concrete types, functions, and future
source files, but does not implement HDF5 dependencies, exports, or package
wiring yet.

The scope here is deliberately narrow:

- plan new IO/runtime modules only
- keep AREPO-facing naming explicit
- preserve a dependency-free staging path until HDF5 integration is approved
- avoid edits to `src/`, tests, or existing examples for now

## Audit Constraints That Drive The Design

The current audit establishes five facts that the runtime surface must respect:

1. `param.txt` compatibility already exists only in example-local writers, so
   the package needs a reusable parameter parser and normalized runtime
   parameter set.
2. `Config.sh` carries compile-time features that should become explicit Julia
   runtime capability flags rather than hidden example-local text templates.
3. `IC.hdf5` support currently covers a narrow `Header` + `PartType0` schema,
   so the first package-level IC API should target that exact minimum shape.
4. Snapshot compatibility already has a minimum read surface from the proxy
   analyzers: direct or split snapshot layouts, `Header/Time`, gas fields, and
   a few derived-field fallbacks.
5. Diagnostic outputs are split across parity artifacts and proxy CSV/SVG
   outputs, so the runtime needs one stable policy for where summaries,
   profiles, and reports land.

These constraints imply that the first IO package layer should normalize file
semantics and policy, not immediately implement every format detail.

## Proposed Future Source Skeleton

The future package-level IO layer should be split into five focused source
files plus one umbrella include:

```text
lib/PowerFoam/src/arepo_io_runtime.jl
lib/PowerFoam/src/arepo_io_parameters.jl
lib/PowerFoam/src/arepo_io_ic.jl
lib/PowerFoam/src/arepo_io_snapshots.jl
lib/PowerFoam/src/arepo_io_restart.jl
lib/PowerFoam/src/arepo_io_diagnostics.jl
```

Recommended ownership:

- `arepo_io_runtime.jl`
  - top-level include-only seam for the runtime IO surface
  - shared enums/symbol policies
  - lightweight feature/capability checks
- `arepo_io_parameters.jl`
  - `param.txt` parsing and normalization
  - `Config.sh` feature decoding
- `arepo_io_ic.jl`
  - initial-condition schema types
  - read/write contracts for the current `Header` + `PartType0` minimum
- `arepo_io_snapshots.jl`
  - snapshot path resolution
  - header/field read contracts
  - derived fallback rules for `Volume`, `Pressure`, and center positions
- `arepo_io_restart.jl`
  - restart compatibility checks
  - version stamps and state equivalence metadata
- `arepo_io_diagnostics.jl`
  - output directory tags
  - CSV/JSON report policy
  - artifact naming rules

## Proposed Top-Level Module Surface

The future umbrella file should provide one explicit namespace:

```julia
module ArepoIORuntime
```

That namespace can stay internal to `PowerFoam` until package exports are ready.
The key point is to avoid scattering IO/runtime naming across hydro or bridge
files.

## Parameter Parsing And Runtime Feature Names

### Core types

```julia
struct ArepoConfigFlags
    enabled::Set{Symbol}
    values::Dict{Symbol,String}
end

struct ArepoParameterSet
    raw::Dict{String,String}
    normalized::NamedTuple
    config_flags::ArepoConfigFlags
end

struct ArepoRuntimeFeatureSet
    hydro_solver::Symbol
    reconstruction::Symbol
    mesh_regularization::Symbol
    precision::Symbol
    io_format::Symbol
    restart_mode::Symbol
end
```

### Parsing functions

Recommended entrypoints:

- `read_arepo_param_file(path::AbstractString)`
- `parse_arepo_param_text(text::AbstractString)`
- `read_arepo_config_flags(path::AbstractString)`
- `parse_arepo_config_text(text::AbstractString)`
- `normalize_arepo_parameters(raw::AbstractDict, flags::ArepoConfigFlags)`
- `arepo_runtime_features(params::ArepoParameterSet)`
- `validate_arepo_parameters(params::ArepoParameterSet)`

### First implemented slice

The first package-level implementation slice should stay narrower than the full
future skeleton above:

- one include-only source file: `src/arepo_io_parameters.jl`
- no `PowerFoam.jl` wiring yet
- no HDF5 dependency or backend hooks yet
- no test-tree edits yet

This slice should implement only the dependency-free text surface:

- `parse_arepo_param_text`
- `parse_arepo_config_text`
- `normalize_arepo_parameters`
- `validate_arepo_parameters`

Recommended include-time records for this slice:

```julia
struct ArepoConfigFlags
    enabled::Set{Symbol}
    values::Dict{Symbol,String}
end

struct ArepoParameterSet
    raw::NamedTuple
    normalized::NamedTuple
    config_flags::ArepoConfigFlags
end
```

### First-slice examples

Parameter example:

```text
InitCondFile ics.hdf5
ICFormat 3
OutputDir output
SnapshotFileBase snap
SnapFormat 3
NumFilesPerSnapshot 1
TimeBegin 0.0
TimeMax 0.2
CourantFac 0.4
BoxSize 1.0
PeriodicBoundariesOn 1
ComovingIntegrationOn 0
OutputListOn 1
OutputListFilename output_list.txt
```

Config example:

```text
TWODIMS
DOUBLEPRECISION=1
OUTPUT_PRESSURE
RIEMANN_HLL
```

Expected direct-include usage:

```julia
include("lib/PowerFoam/src/arepo_io_parameters.jl")

raw = parse_arepo_param_text(param_text)
flags = parse_arepo_config_text(config_text)
params = normalize_arepo_parameters(raw, flags)
validation = validate_arepo_parameters(params)
```

### Direct include smoke command

The first smoke gate for this slice should remain package-independent:

```bash
julia -e 'include("lib/PowerFoam/src/arepo_io_parameters.jl"); param_text = "InitCondFile ics.hdf5\nICFormat 3\nOutputDir output\nSnapshotFileBase snap\nSnapFormat 3\nNumFilesPerSnapshot 1\nTimeBegin 0.0\nTimeMax 0.2\nCourantFac 0.4\nBoxSize 1.0\nPeriodicBoundariesOn 1\nComovingIntegrationOn 0\nOutputListOn 1\nOutputListFilename output_list.txt\n"; config_text = "TWODIMS\nDOUBLEPRECISION=1\nOUTPUT_PRESSURE\nRIEMANN_HLL\n"; raw = parse_arepo_param_text(param_text); flags = parse_arepo_config_text(config_text); params = normalize_arepo_parameters(raw, flags); validation = validate_arepo_parameters(params); @assert validation.valid; @assert params.normalized.io.output_list_on; @assert :TWODIMS in params.config_flags.enabled; println("arepo_io_parameters smoke: ok")'
```

### Package-exported smoke example

Now that the parser surface is package-exported, the lightweight usage path
should also exist as a package-level example:

```bash
julia --project=lib/PowerFoam lib/PowerFoam/examples/arepo_parameter_parser_smoke.jl
```

That example should keep the runtime contract narrow:

- use `using PowerFoam`
- parse representative `param.txt` and `Config.sh` text
- normalize and validate the resulting `ArepoParameterSet`
- write a tiny `README.md` plus CSV summary under
  `examples/out/arepo_parameter_parser_smoke/<timestamp>/`
- print a short key-field summary for human inspection

### Parameter access helpers

- `arepo_output_schedule(params::ArepoParameterSet)`
- `arepo_domain_spec(params::ArepoParameterSet)`
- `arepo_timestep_policy(params::ArepoParameterSet)`
- `arepo_mesh_policy(params::ArepoParameterSet)`
- `arepo_cosmology_policy(params::ArepoParameterSet)`

### Notes on naming

- Use `ArepoParameterSet` rather than a generic `RuntimeConfig` because the
  audit is specifically about compatibility with AREPO-style text surfaces.
- Keep `Config.sh` decoding separate from `param.txt` decoding because the audit
  shows they encode different concepts: compile/runtime capability flags versus
  run-instance settings.
- `normalize_arepo_parameters` should be the only place that maps strings like
  `SnapFormat`, `OutputListOn`, or `ComovingIntegrationOn` into typed Julia
  values.

## Initial-Condition Read/Write Names

### Core types

```julia
struct ArepoICHeader
    time::Float64
    redshift::Float64
    box_size::Float64
    num_files_per_snapshot::Int
    omega0::Float64
    omega_baryon::Float64
    omega_lambda::Float64
    hubble_param::Float64
    double_precision::Bool
    markers::NamedTuple
end

struct ArepoGasICBlock
    particle_ids
    coordinates
    masses
    velocities
    internal_energy
end

struct ArepoICData
    header::ArepoICHeader
    gas::ArepoGasICBlock
    extras::NamedTuple
end
```

### Read/write entrypoints

- `read_arepo_ic(path::AbstractString; part_type::Symbol = :gas)`
- `write_arepo_ic(path::AbstractString, ic::ArepoICData)`
- `validate_arepo_ic(ic::ArepoICData)`
- `arepo_ic_schema_version(ic::ArepoICData)`

### Low-level contracts

- `read_arepo_ic_header(reader, source)`
- `read_arepo_gas_ic_block(reader, source)`
- `write_arepo_ic_header(writer, header::ArepoICHeader)`
- `write_arepo_gas_ic_block(writer, gas::ArepoGasICBlock)`

### Why this shape

The audit shows the current package evidence only needs `Header` plus
`PartType0` fields for proxy cases. `ArepoICData` should therefore stay narrow
at first and accept future extensions through `extras` rather than forcing a
premature particle-family abstraction.

## Snapshot Read/Write Names

### Core types

```julia
struct ArepoSnapshotLocator
    root::String
    snapshot_index::Int
    layout::Symbol
    resolved_paths::Vector{String}
end

struct ArepoSnapshotHeader
    time::Float64
    box_size::Union{Nothing,Float64}
    num_files::Int
    fields_present::Set{Symbol}
end

struct ArepoGasSnapshotBlock
    density
    masses
    internal_energy
    velocities
    volume
    pressure
    center
    particle_ids
end

struct ArepoSnapshotData
    locator::ArepoSnapshotLocator
    header::ArepoSnapshotHeader
    gas::ArepoGasSnapshotBlock
    derived::NamedTuple
end
```

### Path resolution and read/write entrypoints

- `locate_arepo_snapshot(root::AbstractString, snapshot_index::Integer)`
- `read_arepo_snapshot(root::AbstractString, snapshot_index::Integer)`
- `write_arepo_snapshot(path::AbstractString, snapshot::ArepoSnapshotData)`
- `validate_arepo_snapshot(snapshot::ArepoSnapshotData)`

### Field-level helpers

- `read_arepo_snapshot_header(reader, locator::ArepoSnapshotLocator)`
- `read_arepo_gas_snapshot_block(reader, locator::ArepoSnapshotLocator)`
- `snapshot_available_fields(header::ArepoSnapshotHeader)`
- `derive_arepo_snapshot_volume!(gas::ArepoGasSnapshotBlock)`
- `derive_arepo_snapshot_pressure!(gas::ArepoGasSnapshotBlock; gamma::Real)`
- `resolve_arepo_snapshot_centers!(gas::ArepoGasSnapshotBlock)`

### Layout policy

`locate_arepo_snapshot` should be the only function that knows about:

- `output/snap_###.hdf5`
- `output/snapdir_###/snap_###.0.hdf5`

That keeps direct/split snapshot layout logic out of diagnostics and runtime
drivers.

## Restart Compatibility Names

### Core types

```julia
struct ArepoRestartStamp
    format::Symbol
    codegen_tag::String
    parameter_digest::String
    feature_digest::String
    schema_version::VersionNumber
end

struct ArepoRestartAssessment
    compatible::Bool
    status::Symbol
    reasons::Vector{String}
    warnings::Vector{String}
end
```

### Compatibility entrypoints

- `build_arepo_restart_stamp(params::ArepoParameterSet, features::ArepoRuntimeFeatureSet)`
- `read_arepo_restart_stamp(path::AbstractString)`
- `write_arepo_restart_stamp(path::AbstractString, stamp::ArepoRestartStamp)`
- `assess_arepo_restart_compatibility(current::ArepoRestartStamp, saved::ArepoRestartStamp)`
- `require_arepo_restart_compatibility(current::ArepoRestartStamp, saved::ArepoRestartStamp)`

### Runtime-facing helpers

- `arepo_restart_mode(params::ArepoParameterSet)`
- `restart_requires_fresh_mesh(params::ArepoParameterSet)`
- `restart_requires_same_output_layout(params::ArepoParameterSet)`

### Why this belongs in its own file

The runtime scaffold already uses metadata such as `requires_restart`, but the
audit shows restart compatibility is still a missing package-level surface.
Keeping restart checks separate from generic snapshot reads avoids mixing:

- "can I decode this file?" with
- "is it valid to continue this run with the current runtime choices?"

## Diagnostic Output Policy Names

### Core types

```julia
struct ArepoDiagnosticPolicy
    root::String
    run_tag::String
    write_csv::Bool
    write_json::Bool
    write_svg::Bool
    write_snapshots::Bool
    verbosity::Symbol
end

struct ArepoRunArtifactLayout
    run_dir::String
    diagnostics_dir::String
    figures_dir::String
    snapshots_dir::String
    logs_dir::String
end
```

### Policy entrypoints

- `default_arepo_diagnostic_policy(; root = "lib/PowerFoam/examples/out", kwargs...)`
- `build_arepo_run_tag(spec, params::ArepoParameterSet, features::ArepoRuntimeFeatureSet)`
- `materialize_arepo_artifact_layout(policy::ArepoDiagnosticPolicy)`
- `write_arepo_run_summary(path::AbstractString, summary)`
- `write_arepo_metrics_csv(path::AbstractString, table)`
- `write_arepo_profile_csv(path::AbstractString, table)`
- `write_arepo_diagnostic_json(path::AbstractString, payload)`

### Policy helpers

- `arepo_should_write_snapshots(policy::ArepoDiagnosticPolicy)`
- `arepo_should_write_figures(policy::ArepoDiagnosticPolicy)`
- `arepo_log_event!(buffer, level::Symbol, message::AbstractString)`

### Output policy recommendation

The first runtime policy should standardize around one artifact root with
stable subdirectories and run tags encoding the knobs the audit already treats
as meaningful:

- backend
- solver
- reconstruction
- mesh algorithm
- precision
- restart/fresh-start mode

That gives the bridge, proxy, and pure-Julia paths one common output contract
without requiring them to share implementation details.

## Dependency Staging Plan

Because HDF5 integration is explicitly out of scope for this task, the future
implementation should stage dependencies in two layers:

1. dependency-free API and validation layer
   - all type definitions above
   - path resolution
   - normalization and compatibility checks
   - output policy
2. backend readers/writers added later
   - `HDF5.jl` adapters for IC/snapshot/restart read/write
   - optional CSV/JSON helpers if Base is insufficient

Recommended stub hook names:

- `arepo_hdf5_reader()`
- `arepo_hdf5_writer()`
- `has_arepo_hdf5_support()`

These should fail clearly until a real backend is wired in.

## Immediate Next Steps

1. Land the first include-only `src/arepo_io_parameters.jl` slice and smoke it
   through direct `include(...)`.
2. Export nothing at first; keep the API internal until the orchestrator owns
   the package boundary.
3. Implement parameter/config parsing before any HDF5 work, since that path is
   dependency-free and already grounded in repo-local evidence.
4. Implement snapshot path resolution and restart-stamp compatibility before
   binary IO so smoke gates can verify policy without reading HDF5 payloads.
5. Add IC/snapshot read/write adapters only after the HDF5 dependency decision
   is explicit.

## Minimal Planning Checklist

- [x] concrete future source-file names
- [x] concrete Julia type names
- [x] concrete Julia function names
- [x] parameter parsing surface
- [x] IC read/write surface
- [x] snapshot read/write surface
- [x] restart compatibility surface
- [x] diagnostic output policy surface
- [x] explicit no-HDF5-yet staging

## 2026-06-15 First Snapshot Runtime Slice

The first implementation slice for snapshot/runtime IO now exists as a
standalone source file and smoke gate:

- `src/arepo_io_snapshots.jl`
  - typed in-memory `ArepoSnapshotData` / `ArepoGasSnapshotBlock` schema
  - direct vs split snapshot path preflight via `locate_arepo_snapshot`
  - required-field validation plus derived fallbacks for `Volume`,
    `Pressure`, and center positions from `Coordinates`
  - `write_arepo_snapshot` preflight that validates target paths and fields even
    when `HDF5.jl` is not part of the active project
  - optional minimal HDF5 read/write adapter only when `HDF5.jl` is already
    loadable in the active project
- `examples/arepo_snapshot_io_smoke.jl`
  - constructs a small gas snapshot payload
  - emits machine-readable `PASS` / `BLOCKER` rows
  - writes `examples/out/arepo_snapshot_io_smoke/<timestamp>/rows.csv`

This keeps runtime feature-completeness moving without forcing package wiring
or a new hard HDF5 dependency into `PowerFoam.jl`.
