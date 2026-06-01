# ADR-0001: Architecture for the Next-Generation Enzo

- **Status:** Proposed
- **Date:** 2026-05-31 (rev. 2)
- **Deciders:** T. Abel
- **Supersedes:** The monolithic C++/Fortran Enzo (`enzo-dev`)
- **Change in rev. 2:** Build strategy pivots to *HierarchicalGrids.jl-first*.
  The AMR substrate is no longer a from-scratch Rust core but a swappable
  backend behind an interface, with HG.jl as the first implementation. A Rust
  backend becomes a later, measured, drop-in alternative rather than a
  prerequisite.

---

## Context

Enzo is a ~200,000-line C++/Fortran AMR cosmology and astrophysics code with ~30
years of accumulated design debt. Its core (`Grid.h`) is a god-class fusing data
storage, physics, parallelism, AMR topology, and I/O. Build configuration,
parameter parsing, and problem setup are three separate bespoke systems
(`Make.*`, `ReadParameterFile.C`, `ProblemType` integer dispatch) that drift from
each other and from documentation. Coordinates are stored as absolute floating
point, forcing `__float128` and a compile-time precision matrix to support deep
refinement.

The codebase was shaped by constraints that no longer hold: broken template
support, slow virtual dispatch, expensive memory, and MPI as the only path to
performance. We are rebuilding to target a specific, modern hardware envelope
and a specific user experience.

A working, tested substrate already exists. **HierarchicalGrids.jl** (HG.jl) is a
dimension-generic hierarchical-mesh library that independently arrived at the
design principles below: mesh purity (structure only — no fields, physics, or
I/O), integer-exact geometry with relative-to-LCA coordinates, Taichi-inspired
layout-flexible field storage (`SoA`/`AoS`/`Blocked{B}`), conservative
restrict/prolong, and chunk-based parallelism. It supports cell-, block-, and
patch-based refinement, and ships with a finite-volume **cell-average** field
model (added as a first-class sibling to its polynomial and point-sample
models). It deliberately contains no solvers — exactly the layer this project
supplies.

This changes the build calculus: the hard, bug-prone AMR engine is done and
tested. What remains is the physics we know, written on top.

## Goals

1. **Three hardware targets, one codebase:** Apple Silicon (CPU + GPU), large
   shared-memory AMD nodes, and unified-memory accelerators (MI300A, Grace
   Hopper). **No MPI.** Single-host, shared/unified memory only.
2. **First development milestone:** classic-Enzo equivalence on Apple Silicon
   CPU — passing all existing Enzo hydro, MHD, and cosmology test problems —
   built on HG.jl.
3. **A joy to use.** All complex control flow, setup, and orchestration is
   compact, readable Julia. The performance-critical surface is small and stable.
4. **Interactive visualization and analysis are first-class**, built into the
   runtime, not bolted on in post-processing.
5. **Correctness decoupled from performance, in time.** Establish a correct,
   complete reference first; defer every performance decision to a point where
   it can be measured against that reference rather than guessed.

## Decision

A layered architecture with a hard separation between *orchestration* (large,
fluid, scientific, in Julia) and *kernels* (small, stable, performance-critical),
mediated by a swappable **AMR substrate** behind a single interface.

```
┌──────────────────────────────────────────────────────────┐
│  Julia driver + science layer                             │
│  Problem specs, time integration, AMR policy, timestep     │
│  control, I/O scheduling, in-situ analysis & visualization │
│  Solvers / kernels written ONLY against the substrate API  │
└───────────────────────────┬──────────────────────────────┘
                            │  AbstractMeshBackend  (the seam)
        ┌───────────────────┴───────────────────┐
        ▼                                        ▼
┌──────────────────────┐              ┌──────────────────────────┐
│  HGBackend (FIRST)    │              │  RustBackend (LATER)      │
│  HierarchicalGrids.jl │              │  Rust + Rayon core,       │
│  pure Julia substrate │              │  AdaptiveCpp/Metal kernels│
│  reference / oracle    │              │  measured drop-in        │
└──────────────────────┘              └──────────────────────────┘
```

The substrate interface is the project's load-bearing boundary. Solvers,
problem specs, the driver loop, and analysis are written against it and never
against a concrete backend. HG.jl is the first backend and the correctness
oracle; alternative backends (a Rust+Rayon core, a Metal/AdaptiveCpp GPU path,
future block- or patch-optimized variants) are isolated, measurable, swap-in
projects validated against the working HG.jl version test-for-test.

> **Milestone-1 note (this repository).** Per the project owner's direction, the
> interface ("the API we like") was designed first with an in-repo **`RefMesh`**
> reference backend, then **`HGBackend`** — a thin adapter over HierarchicalGrids.jl
> — was added behind the same seam and validated test-for-test against `RefMesh`,
> including a cross-backend agreement test on the Sod tube. Both backends pass.
>
> Building the adapter surfaced (early and cheaply, as the vertical slice is meant
> to) a real design fact: HierarchicalGrids.jl is **ghost-free** — cells are
> opaque ids with BC-aware face-neighbor queries, not a ghosted Cartesian array.
> The seam was therefore made **handle-based and neighbor-driven** (no ghost cells
> in the interface; boundaries resolved per-face by `neighbor`), and the hydro
> driver rewritten as an unsplit **conservative flux-divergence** scheme (PLM +
> SSP-RK2). That single solver now runs unchanged on the uniform `RefMesh` and on
> HG's tree — which is the whole point of the seam, proven rather than asserted.

---

## Principles

### P1 — Orchestration in Julia; kernels are a small, stable surface

The performance-critical surface is small (~15–20 kernels: hydro/MHD sweeps,
Riemann solver, Poisson, cooling, particle deposit, RT absorption). These are
stateless functions over typed memory. Everything else — refinement criteria,
subcycling structure, operator-split ordering, timestep limits, output
scheduling, star formation and feedback policy — is *science*, changes between
papers, and lives in Julia where it is short and inspectable.

The litmus test: if it runs once per coarse step or less, it belongs in Julia.
If it runs in the innermost cell loop, it is a kernel. In the first milestone
the kernels are Julia too (running on HG.jl); a later backend may reimplement
that same small kernel set in Rust/AdaptiveCpp without changing the call sites.

### P2 — The AMR substrate is swappable behind `AbstractMeshBackend`

A single interface defines everything the driver and kernels require of an AMR
layer: topology (`n_cells`, `leaf_cells`, `level_of`, `refine!`, `coarsen!`),
integer-exact geometry (`cell_extent`, `neighbors` with BC resolution),
layout-parametric field storage (`allocate_fields`, zero-copy `field_view`), and
parallel iteration (`for_each_cell`, `for_each_face`). HG.jl already implements
all of these. A backend is selected by a single type parameter at the
`Simulation` constructor; nothing above the seam knows which backend is beneath.

The interface is fixed at the **cell-average (finite-volume)** field level for
the first milestone — the field model HG.jl now provides and the one the classic
Enzo test problems require. HG.jl's higher-order polynomial representation
remains available as a superset for later research paths but is not part of the
milestone interface.

**The interface is shaped by operations, not by the profiler or by any one
backend.** Do not granularize it to expose internals; sub-operation detail comes
from the measurement tier (P10), not from interface splits.

### P3 — Layout is a per-field choice via the substrate's layout machinery (Taichi-inspired)

Logical indexing is decoupled from physical memory layout; switching layout is a
constructor change, not a kernel change. HG.jl supplies this directly with
`SoA` / `AoS` / `Blocked{B}` (its `Blocked{B}` is the Taichi
`ti.root.dense(M).dense(B)` analog). Layout is assigned by *physics*: chemistry
favors AoS (all species per cell, one cache line); hydro/MHD favors SoA or
blocked (directional sweeps); coarse regions favor Morton/blocked ordering. A
later backend may add `BlockedAoSoA{W}` for SIMD/GPU widths as an additive layout
without touching kernels. Empirical layout selection is done via the benchmark
harness, comparing layouts under a fixed backend and kernels under a fixed
layout.

### P4 — Integer-exact geometry; no stored edges, no float128

AMR geometry is integer-exact. HG.jl stores cells in **relative-to-parent /
relative-to-LCA** coordinates, so arithmetic bit-width tracks the local
refinement gradient, not absolute depth — deep zoom does not widen integer
types. Consequences, all inherited from the substrate:

- Cell edges/widths are **derived**, never stored as absolute floats.
- Sibling/parent/neighbor/overlap queries are exact integer/bitwise operations:
  no epsilon tolerances, no `fabs(a-b) < tiny`.
- Particle positions are fixed-point relative to the lattice; sub-cell precision
  is constant regardless of refinement depth.
- `__float128`, the `FLOAT/PSYM/FSYM` precision macros, and the compile-time
  precision matrix are **eliminated**. Physical coordinates exist only as a thin
  conversion at the I/O boundary.

Deep hierarchies (40+ levels) are supported without precision configuration.

### P5 — Conservative refinement is provided by the substrate

Prolongation and restriction are the substrate's responsibility and are
conservative by construction. For the cell-average field model:
prolongation is injection (each child inherits the parent average); restriction
is the volume-weighted mean of children (the arithmetic mean for equal-volume
splits), preserving `Σ value×volume` to round-off, including under anisotropic
splits. This is exactly the inter-level transfer that flux-corrected AMR needs;
the solver layer builds refluxing on top of it rather than reimplementing it.

### P6 — Shared-memory parallelism; future-parallel-ready, not parallel-coupled

Parallelism is chunk-based and shared-memory, exposed through the substrate's
`for_each_cell` / `for_each_face`. HG.jl's chunked partitioning mirrors how
domains would later split, so distribution is an additive change, not a rewrite.
**No MPI.** A later Rust backend replaces chunk-threading with Rayon
work-stealing and compile-time data-race freedom; on large AMD nodes it adds
explicit NUMA-local allocation and thread pinning. The substrate interface is
unchanged by either.

### P7 — Unified memory is the common substrate; the GPU split is hidden (later backends)

On every hardware target there is a coherent memory view shared by CPU and
accelerator (physically unified on Apple Silicon and MI300A; coherence-fabric
unified on Grace Hopper). When a GPU-capable backend is built, field arrays are
allocated once and the same pointer is valid in a kernel and in Julia — no
explicit host/device copy, collapsing the `#ifdef ECUDA` duplicate-API pattern
to one kernel selected at build time. The first (HG.jl, CPU) milestone does not
exercise this, but the substrate seam is where it plugs in.

### P8 — Work scales with content, never with capacity

No pipeline stage iterates the full grid to find sparse work. Radiation is
packet-driven (O(N_packets)); rates deposit into a sparse store
(O(cells_crossed)); refinement transitions are O(depth) lookups that stay
cache-warm along a ray. An empty photon list, an unrefined region, or a
quiescent cell costs (near) nothing. This is a requirement on every subsystem,
above and below the seam.

### P9 — Problem specification is source code, not a parameter file

A problem is a Julia file: domain, solvers, refinement, physics, and initial
conditions expressed as typed values and plain functions (units via `Unitful`,
cosmology via `Cosmology.jl`). Initial conditions are functions `(x,y,z) -> state`,
JIT-compiled to native code. Cross-parameter validation is constructor-time type
checking with actionable errors, not a runtime failure deep in a 2000-line
parser. `ProblemType` integer dispatch and the separately-drifting documentation
are eliminated: the spec *is* the document. Specs are written against the
substrate interface and run unchanged on any backend.

### P10 — The substrate seam is the measurement boundary

`AbstractMeshBackend` is also where performance is instrumented, because its
methods are the meaningful, coarse units of work (`for_each_cell`, `for_each_face`,
`neighbors`/halo, `refine!`/`coarsen!`, restrict/prolong) and timing them does
not perturb the kernels. Instrumentation is an `Instrumented{B}` wrapper that
satisfies the same interface and **compiles away when unused** (it is a distinct
type specialization, not a runtime flag). Because the same wrapper wraps every
backend, measurements are directly comparable: the same span names and units
quantify exactly what a Rust or GPU backend buys, per operation, against the
HG.jl reference. Fine-grained hardware-counter attribution (cache misses, etc.)
is a second tier, exposed *through* the interface but read at the kernel level,
platform-specific and opt-in. Measurement never reshapes the interface.

### P11 — Visualization and analysis are first-class, in-situ, zero-copy

Because field memory is held by Julia (directly, on the HG.jl backend; via
zero-copy array views over coherent memory on later backends), analysis and
visualization run *inside* the live simulation with no dump/reload cycle. Power
spectra, halo finding, slices, projections, and interactive Makie plots are
ordinary Julia over the running state, scheduled from the same loop as the
physics. Designed in from the start, not added later.

---

## Build sequencing

1. **Vertical slice first.** Implement the Sod shock tube and one MHD CT test as
   the *first* solver code, written against `AbstractMeshBackend`, green on
   `HGBackend`. This exercises the whole stack (spec → mesh → fields → kernel →
   BC → output → conservation check) and surfaces the degree-0 finite-volume and
   constrained-transport questions while they are cheap.
2. **Feature-complete on HG.jl (CPU).** Build out hydro, MHD, self-gravity,
   cooling, and the cosmology path until the classic Enzo hydro/MHD/cosmology
   test suite passes to established tolerances. This is the first milestone and
   the correctness oracle.
3. **Instrument.** Wrap the backend with `Instrumented{B}` and capture the
   per-operation cost breakdown on the real test problems.
4. **Measure, then optimize.** Stand up alternative backends (Rust+Rayon core;
   Metal/AdaptiveCpp GPU path) as drop-in replacements, validated test-for-test
   against the HG.jl oracle and compared operation-by-operation via the
   instrumented wrapper. Port to Rust only what the measurements justify, in the
   order the measurements dictate.

## First Milestone (definition of done)

- **Backend:** `HGBackend` (HierarchicalGrids.jl), Apple Silicon, **CPU**.
- **Correctness:** passes the existing Enzo hydro, MHD, and cosmology
  test-problem suite (Sod and variants, MHD CT tests, Zel'dovich, cosmological
  collapse, etc.) to established tolerances. MHD additionally meets a divergence
  bound (|∇·B| at machine-precision floor) as an explicit acceptance criterion.
- **Architecture:** all solver/spec/driver/analysis code is written against
  `AbstractMeshBackend`, with `HGBackend` as the only dependency that names
  HierarchicalGrids.jl. The seam is enforced by package boundaries (a direct
  HG.jl call from solver code is a missing-import compile error).
- **Experience:** at least one in-situ analysis and one interactive Makie
  visualization demonstrated live during a run; at least one problem spec
  running unchanged with a layout swap (SoA ↔ Blocked) to prove layout
  independence.
- **Measurement:** `Instrumented{HGBackend}` produces a per-operation cost
  report on the test suite.

> Note: Apple-GPU (Metal) execution moves to a **second** milestone behind the
> same seam. The first milestone is CPU-only, prioritizing a correct, complete
> reference over the second kernel dialect. (Earlier rev. 1 folded GPU into
> milestone one; rev. 2 separates them to de-risk.)

## Consequences

**Positive.** Fastest path to a correct, complete, classic-Enzo-equivalent code,
because the tested AMR engine (HG.jl) is reused rather than rebuilt. The working
version is simultaneously the correctness oracle for all later backends. Every
performance decision is made against measured data, not guesses. Scientific
logic is short and legible; geometry is exact with no precision configuration;
analysis is first-class. Backends, layouts, and GPU paths are isolated,
measurable, swap-in efforts.

**Negative / risks.**
- *Field-model corners.* HG.jl's degree-0 finite-volume path must behave
  correctly for flux-conservative hydro with refluxing, an application it was
  not originally exercised on. The Sod tube is the early smoke test (build step 1).
- *MHD constrained transport on a hierarchical mesh.* Face-centered B /
  edge-centered E staggering with ∇·B preservation across hanging-node
  refinement boundaries is the most genuinely new design work and the highest
  technical risk. Spike it early; confirm the substrate's face/edge adjacency
  exposes what CT needs before committing a timeline.
- *Degree-0 performance through general machinery.* HG.jl's storage and
  orchestrators were built with the polynomial/overlap case in mind; the
  cell-average path may carry abstraction overhead. The instrumented wrapper
  attributes this directly, so "HG.jl is slow here" is never mistaken for
  "Julia is slow."
- *Later FFI surface.* A Rust/AdaptiveCpp backend reintroduces a thin FFI
  boundary and (on Apple) a Metal kernel dialect distinct from SYCL — contained
  to that backend, not the milestone.
- *Migration tail.* Porting 30 years of problem initializers, the HDF5 schema,
  and yt interoperability is real, ongoing work; a compatibility shim eases
  transition but is not free.

**Non-goals.** MPI / distributed memory. Backward source compatibility with the
existing `Grid` API. A restricted single-language kernel DSL (kernels stay
ordinary Julia now, and full C++/SYCL later, for expressiveness; Julia stays the
orchestration language). Putting any Enzo-specific concept — backend traits,
solvers, application goals — *into* HierarchicalGrids.jl: it remains a
domain-agnostic substrate, consumed as a dependency, ignorant of this project.
