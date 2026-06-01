# EnzoNG.jl

Next-generation Enzo: a shared-memory AMR astrophysics code with **orchestration
in Julia** and a small, stable kernel surface, built on a **swappable AMR
substrate behind one interface**. See [`docs/adr/0001-architecture.md`](docs/adr/0001-architecture.md)
for the full architecture decision record.

This repository runs an end-to-end finite-volume hydro stack (spec → mesh →
fields → kernel → BC → flux-divergence update → conservation check) on **two
interchangeable backends** behind one interface: a pure-Julia reference
(`RefMesh`) and an adapter over [HierarchicalGrids.jl](https://github.com/)
(`HGBackend`). The same solver, problem spec, and driver run unchanged on both.
Current capability:

* **Sod shock tube** on both backends, matched to the exact Riemann solution,
  with a cross-backend agreement test proving identical physics (P2 oracle).
* **Hierarchical AMR** on `HGBackend`: a Julia refinement policy drives dynamic
  regridding; the substrate conservatively remaps fields on refine and enumerates
  coarse↔fine **hanging-node** sub-faces, so the flux-divergence scheme stays
  conservative across level jumps. Validated by conservation to round-off and by
  convergence against a uniform-fine `RefMesh` oracle (refined Sod: 130 cells
  reach L1 0.014, between the 64-cell uniform 0.024 and the 256-cell 0.006).
* **2D Sedov–Taylor blast**: dynamic AMR tracks the expanding circular shock
  (base 48² → ~4.8k leaves), preserving mass/energy to round-off and four-fold
  symmetry, with shock growth ≈ the self-similar R ∝ t^{1/2} law.

## The seam is ghost-free and neighbor-driven

The substrate interface (`AbstractMeshBackend`) is shaped by the operations a
finite-volume solver performs on an AMR mesh, not by any one backend's storage. A
cell is an **opaque handle**; there are **no ghost cells** in the interface.
Boundaries are resolved per face by `neighbor(backend, cell, axis, side; bcs)`,
which returns either an interior neighbor (including a periodic wrap) or a
`DomainBoundary` carrying the BC the solver applies. This is exactly the model a
hierarchical, hanging-node mesh exposes, and it degenerates correctly to a
uniform grid — which is why one solver runs on both `RefMesh` and the
HierarchicalGrids.jl tree.

## Layout

```
lib/MeshInterface   The seam. AbstractMeshBackend: handle-based topology,
                    integer-exact derived geometry, neighbor-with-BC resolution,
                    layout-parametric cell-average fields, conservative
                    restrict/prolong, and the Instrumented{B} measurement
                    wrapper (P10). No implementations.

lib/RefMesh         Small, correct, pure-Julia reference backend + oracle.
                    Uniform mesh, SoA/AoS/Blocked field storage, conservative
                    2:1 restrict/prolong.

lib/HGBackend       Thin adapter implementing the seam on HierarchicalGrids.jl
                    (the architecture's target substrate). Uniform for this
                    milestone; validated test-for-test against RefMesh.

src/                The science layer (depends on MeshInterface ONLY — never on a
                    concrete backend; this is what enforces the seam). HLLC
                    Riemann solver, PLM/minmod reconstruction, ideal-gas EOS, a
                    ghost-free conservative flux-divergence driver with SSP-RK2,
                    the Problem type (a problem is source code, not a parameter
                    file), diagnostics, and an exact Riemann solver (the oracle).

problems/           Problem specs as source code (ADR P9).
test/               Interface conformance (run on both backends), Sod-vs-exact,
                    layout swap, instrumentation, and cross-backend agreement.
```

The seam is enforced by package boundaries: solver/spec/driver code cannot name a
concrete backend (a direct call would be a missing-import compile error). The
backend is injected at the `Simulation` constructor.

## Run it

Julia 1.10+ (developed on 1.12 via `juliaup`). From this directory:

```julia
julia --project=. -e 'using Pkg; Pkg.test()'
```

Interactively, swapping the backend is a one-line change:

```julia
using EnzoNG
prob = sod_problem_defaults(n = 256)

using RefMesh                       # reference backend …
sim = Simulation(UniformMesh(prob.dims, prob.domain), prob)

using HGBackend                     # … or HierarchicalGrids.jl, same solver
sim = Simulation(HGMesh(prob.dims, prob.domain), prob)

evolve!(sim; verbose = true)
d = dump_fields(sim)                # primitive fields + coordinates (1D), sorted

using MeshInterface                 # measure at the seam (P10), zero overhead off
sim = Simulation(Instrumented(HGMesh(prob.dims, prob.domain)), prob)
evolve!(sim); span_report(sim.backend)
```

Adaptive run — a Julia refinement policy drives dynamic regridding on the tree:

```julia
using EnzoNG, HGBackend
include("problems/sedov_blast.jl")
prob = sedov_problem(n = 48, tfinal = 0.04)
sim  = Simulation(HGMesh(prob.dims, prob.domain), prob)
policy = RefinementPolicy(refine_above = 0.2, max_level = 2, every = 4)
evolve!(sim; policy = policy, verbose = true)   # mesh follows the expanding shock
```

## Status & next steps

Hydro is a correct reference on two backends, and **hierarchical AMR works on the
HierarchicalGrids.jl substrate** (dynamic regridding, conservative remap, hanging
nodes), validated by conservation and by convergence against the uniform `RefMesh`
oracle, with the 2D Sedov blast as the dynamic-AMR demonstration. Next, per the
ADR: an MHD constrained-transport spike (highest risk; HG already exposes
face/edge adjacency), then self-gravity, cooling, and cosmology until the classic
Enzo suite passes; then Rust/GPU backends added only where the instrumented
measurements justify. Visualization/analysis are first-class and in-situ (P11).
```
