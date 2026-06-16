# Noh2D Executable Gate Conversion Plan

Date: 2026-06-15

Purpose: convert the existing `lib/PowerFoam/examples/arepo_noh_proxy/`
material from a proxy/readiness surface into one canonical executable
final-field parity gate against original AREPO for the 2-D Noh problem.

## Current Implementation Snapshot

As of 2026-06-15, the first step of this promotion is now in place:

- `examples/arepo_noh2d_gate.jl` is no longer inspect-only.
- It generates a small bounded `Noh2D` initial condition directly in Julia.
- It advances a short PowerFoam 2-D hydro run and writes:
  - `metrics_powerfoam.csv`
  - `radial_bins_powerfoam.csv`
  - `powerfoam.log`
  - `README.md`
- The executable rung is intentionally labeled `calibration-PENDING`, not
  `physics-PASS`, because it still lacks the planned original-AREPO staging and
  analytic threshold enforcement described below.

This note leaves the rest of the document as the next-step promotion plan from
the new executable PowerFoam-only rung to a full AREPO-backed physics gate.

## Scope

- Planning only. Do not edit `src/`, `test/`, or example drivers in this step.
- Reuse the existing proxy assets where they already encode the right problem
  definition, IC writer path, and radial analyzer.
- Keep the first promoted gate narrow: one mesh, one solver choice, one final
  time, one artifact contract.
- Use original AREPO's `examples/noh_2d/check.py` as the reference definition
  of success.

## What Already Exists

Proxy/readiness material already available in-repo:

- `examples/arepo_noh_proxy/generate_tables.jl`
  - writes `standard_ic.csv`, `powerfoam_ic.csv`, and `metadata.txt`
  - freezes the proxy problem at `BoxSize=6`, `gamma=5/3`, `rho0=1`,
    `p0=1e-4`, `v_r=-1`, `t_final=2`
- `examples/arepo_noh_proxy/write_arepo_cases.py`
  - already knows how to turn the CSV tables into AREPO-readable `IC.hdf5`
  - already knows the `Config.sh`/`param.txt` shape for the proxy runs
- `examples/arepo_noh_proxy/profile_noh_snapshot.c`
  - already defines the radial diagnostics used in the stored proxy results
- `examples/arepo_noh2d_proxy_gate.jl`
  - proves the local source/results surface exists, but does not run a real
    parity comparison
- `examples/arepo_noh_proxy/results/`
  - records the proxy metric scale at `64 x 64`, especially the small deltas
    between the best `standard-*-hll` and `powerfoam-*-hll` branches

Upstream AREPO reference behavior:

- `examples/noh_2d/check.py`
  - computes volume-weighted `L1_dens` over cells with `r < 0.8`
  - compares against the analytic 2-D Noh density
  - enforces `L1_dens <= 0.05 * time`
  - at `t = 2`, the final analytic pass threshold is therefore `0.10`

## Promotion Target

Promote `Noh2D` by replacing the readiness shell with one executable gate that:

1. stages one original-AREPO reference run,
2. stages one PowerFoam run from the same initial state,
3. exports final-field tables for both,
4. evaluates analytic Noh metrics and direct AREPO-vs-PowerFoam parity metrics,
5. writes a stable artifact bundle under
   `lib/PowerFoam/examples/out/arepo_noh2d_gate/<run-tag>/`,
6. exits nonzero when the documented pass criteria fail.

## Freeze The First Gate Surface

Do not promote the whole proxy matrix. Freeze the first executable rung to:

- mesh source: proxy `standard_ic.csv` only
- resolution: `N = 64`
- final time: `t_final = 2.0`
- outputs: final snapshot only
- solver: `HLL`
- reconstruction: MUSCL / repo-default
- artificial viscosity: off
- comparison mode: CPU `Float64`

Why this exact slice:

- it keeps the smoke-scale `64 x 64` size already present in the proxy archive
- it uses the same HLL branch that was the cleanest proxy stabilizer
- it avoids turning `powerfoam_ic.csv`, Local PPM, or artificial-viscosity
  variants into blocking requirements before the stock parity rung exists

`powerfoam_ic.csv` should stay as a non-blocking follow-on diagnostic after the
standard-mesh gate is working.

## Required Inputs

The real gate should consume these frozen inputs:

- `lib/PowerFoam/examples/arepo_noh_proxy/out/metadata.txt`
- `lib/PowerFoam/examples/arepo_noh_proxy/out/standard_ic.csv`
- original AREPO source tree at `${AREPO_DIR:-/Users/tabel/Projects/arepo}`
- Python with `h5py` and `numpy`
- `h5cc` for the existing profiler / IC writer helper path

The first executable version should not depend on `powerfoam_ic.csv`,
multi-case proxy fanout, or extra snapshots at intermediate times.

## Canonical Command Shape

The planned gate command should be a single-entry Julia driver:

```bash
AREPO_DIR=/Users/tabel/Projects/arepo \
AREPO_PYTHON=/Users/tabel/Projects/arepo/.venv/bin/python \
julia --project=lib/PowerFoam \
  lib/PowerFoam/examples/arepo_noh2d_gate.jl 64 2.0 48 hll
```

Planned arguments:

1. `N = 64`
2. `t_final = 2.0`
3. `nbins = 48`
4. `riemann = hll`

The driver should internally call the existing table generator only when the
expected `standard_ic.csv`/`metadata.txt` inputs are absent or stale.

## First Skeleton Before The Real Gate

Before any staged AREPO/PowerFoam execution, land one non-heavy skeleton
driver at `examples/arepo_noh2d_gate.jl` that:

- accepts the planned `N t_final nbins solver` arguments,
- exits `0` only for the frozen default rung `64 2.0 48 hll`,
- inspects the required proxy/reference files without running either code,
- writes `examples/out/arepo_noh2d_gate/N64_t2p0_hll/preflight.csv`,
  with one row set for inspected inputs and one row set for the still-planned
  executable phases,
- writes `examples/out/arepo_noh2d_gate/N64_t2p0_hll/README.md`,
- records the exact future command lines and artifact filenames expected once
  the real AREPO/PowerFoam phases are wired in,
- states clearly that the executable run phases below are still missing.

This gives the registry a stable entry point and artifact location before the
real parity phases are wired in.

### Optional Preflight Rung

The skeleton may add one strictly optional `--run-proxy` mode, as long as the
default no-flag path stays inspect-only and fast.

Allowed behavior for `--run-proxy`:

- rerun `examples/arepo_noh_proxy/generate_tables.jl` for the frozen rung
- refresh `arepo_noh_proxy/out/metadata.txt` and the local IC CSVs
- copy the parsed key/value metadata into a machine-readable gate artifact
  such as `proxy_metadata.csv`
- capture the helper stdout/stderr in `proxy_generate.log`

Not allowed for this rung:

- staging or running original AREPO
- staging or running a PowerFoam evolution solve
- compiling or invoking the HDF5 snapshot profiler
- claiming executable parity

This optional preflight is worthwhile because it records one real PowerFoam-side
artifact surface from the existing Noh proxy without making the default gate
heavy.

## Output Contract

The gate should always emit:

- `README.md`
- `arepo_check.log`
- `powerfoam.log`
- `metrics_arepo.csv`
- `metrics_powerfoam.csv`
- `radial_bins_arepo.csv`
- `radial_bins_powerfoam.csv`
- `final_fields_arepo.csv`
- `final_fields_powerfoam.csv`
- `field_compare.csv`

When `--run-proxy` is used, the gate should additionally emit:

- `proxy_generate.log`
- `proxy_metadata.csv`

Artifact root:

```text
lib/PowerFoam/examples/out/arepo_noh2d_gate/N64_t2p0_hll/
```

`field_compare.csv` should follow the existing local convention:

- columns:
  `reference,candidate,field,cells,l1,linf,rel_l1,rel_linf,atol,rtol,status`
- row set:
  `rho`, `pressure`, `vx`, `vy`, `vrad`, `energy_density`
- optional diagnostic-only rows:
  `x`, `y`, `volume`, `mass`, `mx`, `my`, `energy`, `r`

## What Must Be Compared To Original AREPO

The blocking parity rung should compare PowerFoam to original AREPO in three
separate ways.

### 1. AREPO's Own Analytic Success Check

The staged original AREPO reference run must pass its own success surface:

- run the equivalent of `examples/noh_2d/check.py <run-dir> False`
- record the final analytic metrics in `metrics_arepo.csv`
- require final `L1_dens <= 0.10` at `t = 2.0`

If the staged AREPO reference does not pass this self-check, the gate fails
before comparing PowerFoam.

### 2. Analytic Noh Metrics For Both Codes

Reuse `profile_noh_snapshot.c` semantics for both final snapshots and store:

- `time`
- `shock_radius = time / 3`
- `l1_density`
- `l2_density`
- `postshock_mean`
- `postshock_std`
- `mass`
- `energy`
- `volume_sum`

PowerFoam should pass the first executable rung only if all of these hold:

- `l1_density_powerfoam <= 0.10`
- `l1_density_powerfoam <= l1_density_arepo + 0.01`
- `l2_density_powerfoam <= l2_density_arepo + 0.03`
- `abs(postshock_mean_powerfoam - postshock_mean_arepo) <= 0.10`
- `postshock_std_powerfoam <= postshock_std_arepo + 0.10`
- `abs(volume_sum_powerfoam - volume_sum_arepo) <= 1e-10 * volume_sum_arepo`

These tolerances are intentionally tied to the current proxy evidence:

- the upstream analytic threshold at `t=2` is `0.10`
- the stored `64 x 64` HLL proxy deltas are only a few `1e-3` in `l1_density`
  and a few `1e-2` to `1e-1` in postshock scatter

### 3. Final-Field Parity Against AREPO

Both runs should export one final-field CSV sorted by persistent cell id.
Required columns:

- `label`
- `id`
- `t`
- `x`
- `y`
- `volume`
- `rho`
- `vx`
- `vy`
- `pressure`
- `mass`
- `mx`
- `my`
- `energy_density`
- `energy`
- `r`
- `vrad`

Comparison rules:

- use exact `id` matching as the primary join
- fail closed if the PowerFoam side cannot emit the same id set as the AREPO
  reference
- do not use nearest-neighbor remapping for the blocking parity metric
- nearest-neighbor or radial-bin remaps are acceptable only for extra plots

For the first executable rung, the gate should write `field_compare.csv` for
inspection but keep the blocking thresholds on the analytic/radial metrics
above. After 3 repeated green runs at the frozen `N64_t2p0_hll` setting,
promote these field thresholds into blocking `Q1` criteria:

- `rho`: `rel_l1 <= 0.02`, `rel_linf <= 0.10`
- `pressure`: `rel_l1 <= 0.03`, `rel_linf <= 0.15`
- `vx`, `vy`, `vrad`: `rel_l1 <= 0.03`, `rel_linf <= 0.15`
- `energy_density`: `rel_l1 <= 0.03`, `rel_linf <= 0.15`

## Exact Conversion Checklist

1. Split the proxy helpers into reusable pieces instead of calling the current
   multi-case proxy fanout directly.
   - keep `generate_tables.jl`
   - keep the existing CSV-to-HDF5 writer path
   - keep `profile_noh_snapshot.c`
   - do not make the real gate build all proxy branches
2. Stage one original AREPO reference case from `standard_ic.csv`.
   - use HLL
   - write only the final snapshot
   - keep `N=64`, `t_final=2.0`
3. Run original AREPO and immediately verify its own `noh_2d/check.py`
   semantics.
4. Export the AREPO final snapshot to `final_fields_arepo.csv`.
5. Run PowerFoam from the same initial state and same final time.
6. Export the PowerFoam final state to `final_fields_powerfoam.csv`.
7. Run the existing radial profiler on both final snapshots.
8. Write `metrics_arepo.csv`, `metrics_powerfoam.csv`,
   `radial_bins_arepo.csv`, and `radial_bins_powerfoam.csv`.
9. Join the final fields by `id` and write `field_compare.csv`.
10. Make the driver fail on any of the documented `Q0` conditions:
    - AREPO self-check failure
    - PowerFoam positivity/nonfinite failure
    - PowerFoam analytic/radial tolerance failure
    - missing artifact or mismatched id coverage
11. Write one summary `README.md` with:
    - command
    - input tables used
    - AREPO self-check result
    - final metric table for both codes
    - pass/fail table for every threshold above
12. Only after repeated green reruns, promote the per-field thresholds to
    blocking `Q1` status.

## Explicit Non-Goals For The First Rung

Do not bundle these into the first executable gate:

- `powerfoam_ic.csv`
- Local PPM
- artificial viscosity
- LLF or exact-solver sweeps
- intermediate snapshot evolution plots
- GPU / Metal parity
- a full proxy matrix over mesh objectives

Those belong in separate diagnostics once the stock `standard_ic.csv` parity
gate is real and repeatable.
