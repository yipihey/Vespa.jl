# ADR-0004: Optional MPI for the Enzo substrate (MPItrampoline standard)

- **Status:** DONE (substrate + toolchain), but the **in-process** MPI delivery this
  ADR assumes (MPI.jl owns `MPI_Init`, the bridge dlopen'd into the Julia process) was
  **superseded by ADR-0005** for the multi-rank case: loading the gcc/libstdc++ MPI
  `libenzo` into the libc++ Julia process aborts in C++ static init (a runtime
  collision). The MPI substrate now runs in a **subprocess worker** (ADR-0005,
  COMPLETE) — read that ADR for how to build/run multi-rank. The source changes here
  (the `#ifdef USE_MPI` bridge/substrate work, the dual-artifact build, the
  MPItrampoline toolchain) all stand and are what the subprocess worker hosts; only
  the *in-process* hosting was abandoned. Serial path: core 203/203, EnzoLib green,
  zero impact.
- **Date:** 2026-06-04
- **Builds on:** ADR-0002 (method-slot registry), ADR-0003 (conservative `:julia` AMR),
  the EnzoNG↔Enzo C-ABI bridge (`EnzoModules/src/enzomodules_problem_bridge.C` ↔
  `EnzoNG.jl/lib/EnzoLib/src/session.jl`).

---

## Context

EnzoNG drives a live Enzo hierarchy through a C-ABI bridge. That bridge was
**deliberately serial-only**: it hardcoded `NumberOfProcessors=1` / `MyProcessorNumber=0`,
forced `UnigridTranspose=0`, and every accessor assumed all grids were local and resident.
That blocked (1) running original Enzo under MPI, and (2) multi-host capability in general —
even though EnzoNG-native solvers target shared-memory / GPU / Rust per node.

This ADR makes the **Enzo substrate** MPI-capable, **optionally** and with the serial path
unchanged. It does NOT distribute the EnzoNG-native seam (the `RemoteNeighbor` case that
ADR-0001 deferred remains future work — see "Amends ADR-0001").

## Key enabling fact

Enzo's source is already fully `#ifdef USE_MPI`-gated, and Enzo's own AMR routines
(`SetBoundaryConditions`, `UpdateFromFinerGrids`, `CommunicationTransferSubgridFluxes`, …)
already do all cross-rank communication when built with `-DUSE_MPI`. Crucially,
**distribution is already inside Enzo**: `InitializeNew` calls `CommunicationPartitionGrid`
(a no-op when `NumberOfProcessors==1`) to partition the initial grids, and `RebuildHierarchy`
calls `CommunicationLoadBalanceGrids` on regrid. So the bridge does not implement
distribution; it only had to **stop suppressing it** and let the `:julia` hooks touch only
local grids.

## Decisions

1. **Scope:** Enzo substrate + bridge become MPI-capable; under MPI each rank runs its own
   EnzoNG on its **local** grids (mirroring Enzo's `SolveHydroEquations`, which skips grids
   whose `ProcessorNumber != MyProcessorNumber`), and Enzo's existing machinery moves
   grid/flux data between ranks. EnzoNG-native solvers stay shared-memory.

2. **Build model — dual artifacts, serial default.** Both a serial and an MPI `libenzo` +
   matching bridge dylib coexist; serial stays the default so the current dev/test loop is
   untouched. Serial libenzo in `src/enzo/`, MPI libenzo in `src/enzo/mpi/`; bridges are
   `libenzomodules_grid{,_mpi}.dylib`. Selected at session init via `ENV["ENZONG_ENZO_MPI"]`.

3. **MPI ownership — Julia/MPI.jl owns the lifecycle.** Launch via `mpiexec -n N julia`;
   `CommunicationInitialize` is patched to `MPI_Initialized`-guard `MPI_Init` (and a new
   `CommunicationOwnsMPI` global gates `MPI_Finalize`), so MPI.jl owns init/finalize and Enzo
   reuses the same `MPI_COMM_WORLD`. Standalone Enzo is unaffected (it still inits).

4. **MPI provider — MPItrampoline is the standard, everywhere.** The `mpi` flavor compiles
   Enzo + the bridge against the MPItrampoline ABI; MPI.jl uses `MPItrampoline_jll`; a
   locally-built **MPIwrapper** (wrapping the host's real MPI, e.g. Homebrew open-mpi 5.0.9)
   is selected at runtime via `MPITRAMPOLINE_LIB`. This guarantees the Enzo↔MPI.jl ABI match
   by construction (the plan's biggest risk) and makes the binary cluster-portable: build
   once, run on any host's MPI. There is no open-mpi-direct path. Note: `MPItrampoline_jll`
   reports `is_available()==false` on arm64-darwin in this depot, so MPItrampoline + MPIwrapper
   are **built from source** here and MPI.jl is pointed at them via `MPIPreferences`.

## What landed (serial-verified)

- **`src/enzo/CommunicationInitialize.C`** (+`communication.h`): `MPI_Initialized`/`Finalized`
  guards + `CommunicationOwnsMPI` global — all inside `#ifdef USE_MPI`, so the serial dylib is
  behaviorally unchanged.
- **`EnzoModules/src/enzomodules_problem_bridge.C`**: `UnigridTranspose=0` gated to
  `#ifndef USE_MPI`; `compute_dt` skips non-local grids and reduces globally via
  `CommunicationMinValue` (`#ifdef USE_MPI`); new `session_my_rank` / `session_num_ranks` /
  `problem_grid_processor` accessors (return 0/1/0 in serial). No init-partition or
  load-balance code needed (Enzo already does both).
- **`EnzoNG.jl/lib/EnzoLib/src/session.jl`**: `enzo_mpi_enabled()` + `_mpi`-suffixed dylib
  selection (serial default); `session_my_rank`/`session_num_ranks`/`problem_grid_processor`
  bindings; `local_grids_on_level` (== `grids_on_level` in serial).
- **`EnzoNG.jl/lib/EnzoLib/test/test_julia_reflux.jl`**: the `:julia` hydro hook skips grids
  not resident on this rank (no-op in serial).
- **`EnzoModules/deps/build_grid_darwin.sh`**: `serial`|`mpi` flavor parameter.

## Remaining work

- Build MPItrampoline + MPIwrapper from source (arm64-darwin); point MPI.jl at them via
  `MPIPreferences` (this mutates the depot's global MPI.jl config).
- Build the `mpi` libenzo + bridge (slow full Enzo build under `-DUSE_MPI`).
- Multi-rank smoke gate `lib/EnzoLib/test/test_mpi_session.jl` (guarded on `session_num_ranks>1`):
  `:enzo` Sod under `mpiexec -n 2` matches serial; the `:julia` hook conserves globally
  (`MPI_Allreduce` over per-rank composite totals — note `read_root_totals` reads the root
  grid, which is partitioned under MPI, so the test must reduce across ranks). Verify grids
  actually land on >1 rank.

## Amends ADR-0001

ADR-0001 lists "Non-goals. MPI / distributed memory." That non-goal is **narrowed** to the
EnzoNG-**native** seam (RefMesh/HGBackend and the `RemoteNeighbor` distributed backend remain
future work). MPI now lives in the **Enzo substrate path** only: the native solvers stay
shared-memory (chunk-based → Rust/Rayon/GPU), while distribution across hosts is provided by
Enzo's existing communication under the optional MPI build flavor.
