# ADR-0006: The Unified Multi-Code Framework (Enzo · Ramses · Arepo · …)

- **Status:** Accepted — Phases 0–3 implemented (see *Implementation status*)
- **Date:** 2026-06-09
- **Deciders:** T. Abel
- **Builds on:** ADR-0001 (layered architecture, the `MeshInterface` seam),
  ADR-0002 (method-slot registry), ADR-0003 (conservative `:julia` AMR),
  ADR-0004/0005 (MPI substrate, subprocess worker boundary)
- **Incorporates:** RamsesNG.jl (`RamsesLib`), Arepo.jl (`ArepoLib`),
  mini-ramses Metal solvers, dfmm, HierarchicalGrids.jl, Veusz.jl,
  PPMKernels / PoissonKernels

---

## Context

Seven threads now exist, developed separately but converging on the same
idioms:

1. **EnzoNG.jl** — orchestration-in-Julia over the `AbstractMeshBackend` seam;
   three backends (`RefMesh` oracle, `HGBackend` on HierarchicalGrids.jl,
   `EnzoBackend` on the *live* legacy hierarchy); the `EquationSet` variable
   model; the `EngineConfig` method-slot registry (`:enzo` | `:julia` per
   physics step, certified per-slot); the `@xcall` manifest-driven bridge with
   two transports (in-process `ccall`, subprocess worker over control channel +
   shm) that solved the C++/MPI runtime-collision problem once and for all.
2. **PPMKernels / PoissonKernels** — the first entries in a *portable solver
   library*: KernelAbstractions.jl kernels (one source, CPU f64/f32 + Metal
   f32), certified bit-tight against the live Fortran/C they replace, holding
   31× (hydro) and 11.9× (256³ multigrid/FFT) speedups on Apple Silicon.
3. **RamsesNG.jl** (`RamsesLib`) — the same wrapper pattern independently
   re-derived for RAMSES: `dlopen`/`dlsym` of a C-ABI library
   (`ramses_capi.f90`, ISO_C_BINDING), opaque integer handles (registry of up
   to 8 live states), a precision/ndim contract checked at load, per-level
   field get/set, wrapped gravity (full DMO stack, validated on a 2.1M-particle
   cosmological run) and hydro (godunov_fine et al.). Crucially it already does
   **dual-library loading** — a CPU and a Metal build of the same code, same
   handles, for bit-level CPU-vs-GPU diffing.
4. **mini-ramses Metal** — hand-written MSL kernels (gravity complete, hydro in
   progress) behind an Objective-C++ bridge; a second, independent GPU port of
   the same physics PPMKernels covers, in a different kernel dialect.
5. **Arepo.jl** (`ArepoLib`) — the pattern a third time, for a *moving-mesh*
   code: dlopen/ccall, lifecycle (`init`/`run_step!`/`run!`), particle and
   gas-cell field accessors, Voronoi mesh introspection, a bridge
   (`arepo_bridge.c`) that tames Arepo's process-hostile habits (guards
   `MPI_Init`, `setjmp`s around `endrun`). One live instance per process —
   Arepo keeps state in C globals.
6. **dfmm** — a genuinely new solver family (variational moment methods,
   symplectic by construction, cold-limit collisionless unification), already
   built *on HierarchicalGrids.jl*, 1D production-ready, 2D under way. The
   first solver born inside the new ecosystem rather than ported into it.
7. **Veusz.jl + EnzoViz** — interactive, daemonized plotting (JSON-RPC to
   `veuszd`) and in-situ visualization with self-contained interactive HTML
   reports.

The convergence is not an accident — it is the same architecture discovered
three times: *a legacy code is a backend behind a thin C-ABI seam; Julia owns
orchestration; kernels are small, portable, and certified against the legacy
reference.* What is missing is the layer that lets these wrappers see each
other: today EnzoLib, RamsesLib, and ArepoLib are three private dialects of one
unwritten contract, and there is no shared notion of "the state of a
simulation" that two codes could exchange.

The scientific goal that forces the unification:

- **Compare:** start from one set of ICs (e.g. RAMSES `grafic` ICs) and run
  Enzo, RAMSES, and Arepo on the same problem, with common diagnostics and one
  report.
- **Mix and match:** run RAMSES-RT inside an Enzo simulation; use Enzo's Moray
  ray tracing inside an Arepo run; drive RAMSES's hydro with EnzoNG's `:julia`
  gravity; replace any code's hydro with the PPMKernels Metal sweep.
- **Converge:** use the experience to rewrite solvers one at a time into a
  portable, code-agnostic library (the PPMKernels pattern), until the legacy
  codes are optional backends rather than hosts — with new methods (dfmm,
  moving meshes, dual-frame adaptivity) entering as first-class citizens.

## Decision

EnzoNG.jl becomes the umbrella of a **federated framework**. Its code-neutral
seams are promoted to shared substrates; the per-code wrappers become
interchangeable implementations of two small contracts; and a canonical-state
layer with conservative remap operators makes cross-code exchange a defined
operation rather than an ad-hoc script.

```
┌────────────────────────────────────────────────────────────────────────┐
│ Science layer (Julia): Problem specs · driver · EngineConfig slots ·    │
│ diagnostics · comparison harness · in-situ viz (EnzoViz/Veusz.jl)       │
├────────────────────────────────────────────────────────────────────────┤
│ Canonical state & exchange: EquationSet roles · unit contracts ·        │
│ conservative remap (AMR↔AMR, AMR↔Voronoi via R3D) · particle sets       │
├──────────────────────────────┬─────────────────────────────────────────┤
│ MeshInterface                │ CodeBridge                              │
│ (AbstractMeshBackend seam)   │ (AbstractCodeSession contract)          │
│  RefMesh · HGBackend ·       │  EnzoLib · RamsesLib · ArepoLib · …     │
│  EnzoGridMesh · (Voronoi)    │  local ccall │ subprocess worker(s)     │
├──────────────────────────────┴─────────────────────────────────────────┤
│ Portable kernel library (KernelAbstractions): PPMKernels ·              │
│ PoissonKernels · (RTKernels, dfmm, …)  — CPU/Metal/CUDA from one source │
├────────────────────────────────────────────────────────────────────────┤
│ Substrates: HierarchicalGrids.jl (+R3D exact overlap) · legacy codes    │
│ (enzo-dev, mini-ramses, arepo) each behind its own C-ABI bridge         │
└────────────────────────────────────────────────────────────────────────┘
```

Five concrete moves, in order of leverage:

### D1 — Extract `CodeBridge`: one contract for all legacy wrappers

EnzoLib, RamsesLib, and ArepoLib each hand-roll the same five mechanisms.
These move into a small shared package, `lib/CodeBridge`, and the three
wrappers become its clients:

- **Library loading** — `dlopen`/`dlsym` with env-var override, lazy symbol
  cache, and *multi-flavor* support (RamsesLib's CPU/Metal dual-library and
  EnzoLib's serial/MPI dual-dylib are the same feature).
- **Handle registry** — opaque `Handle` over per-code state. Codes that
  support multiple live states (RAMSES, Enzo sessions) and codes that are
  global-state singletons (Arepo) both fit: a singleton is a registry of
  capacity 1 — or capacity N once instances live in worker processes (D2).
- **Contract handshake** — generalize RamsesLib's precision/ndim probe and
  ArepoLib's `DOUBLEPRECISION`/IDType check into one declared
  `ContractSpec` (precision, ndim, nvar, index width, build-flag hash)
  verified at load/connect. Mismatch is a refused connection, never silent
  corruption (the ADR-0005 `contract_hash` policy, applied to all codes).
- **`@xcall` + manifest + worker transport** — the single most valuable asset
  of ADR-0005, today Enzo-only, becomes code-agnostic: any wrapper written in
  `@xcall` style gets, for free, the in-process `ccall` fast path, the
  generated RPC stubs, the C-side typed dispatch, and the subprocess worker
  with shm bulk transport. The differential parity oracle (`:local` ≡
  `:remote` bit-identical) is part of the contract.
- **Field descriptors** — the (dtype, rank, shape, layout, units) descriptor
  used by the RPC layer becomes the universal currency for field get/set
  across all bridges.

`AbstractCodeSession` is deliberately tiny — lifecycle (`init`, `step!`,
`finalize!`), introspection (`info`, `levels`, `fields`), field access
(`get_field`, `set_field!` via descriptors), and a per-code capability list
(`provides(::EnzoSession) ⊇ (:hydro, :gravity, :mhd_ct, :moray_rt, …)`).
Everything else stays code-specific. The wrappers keep their identity and
their repos; they shed only the duplicated plumbing.

### D2 — Multi-worker sessions: N legacy codes alive in one Julia session

The subprocess worker boundary (ADR-0005) was built to dodge one C++/MPI
runtime collision. Its real payoff is bigger: **each legacy code runs in its
own worker process**, so any number of codes — including N instances of
single-instance Arepo, including codes with mutually incompatible MPI builds,
Fortran runtimes, or signal handlers — coexist under one Julia orchestrator.

```julia
enzo   = connect(EnzoWorker(); params="sedov.enzo")
ramses = connect(RamsesWorker(); namelist="sedov.nml")
arepo  = connect(ArepoWorker(); params="param.txt")   # its globals isolated per process
```

The in-process `ccall` path remains for single-code work (it is the fast path
and the parity oracle); multi-code is always multi-worker. RamsesLib and
ArepoLib gain worker binaries the way EnzoModules did — the manifest generator
emits their dispatch tables from their `@xcall` sites.

### D3 — Canonical state + conservative exchange operators

Cross-code coupling and comparison need a representation no code owns:

- **Canonical fields** = `EquationSet` role indices (density, momentum,
  energy, species, …) as cell-averaged conserved quantities in **declared
  physical units**. Every bridge ships a `to_canonical`/`from_canonical` pair
  that converts at the boundary (Enzo comoving, RAMSES supercomoving, Arepo
  code units — conversion lives in the bridge, nothing downstream ever sees a
  code's internal units; the same convert-at-the-boundary rule that governs
  precision today).
- **Mesh adapters**: `EnzoGridMesh` already presents the live Enzo hierarchy
  as an `AbstractMeshBackend`; `RamsesMesh` does the same for the RAMSES oct
  hierarchy (per-level `ckey`+cell-value layout maps cleanly onto the
  handle-based, ghost-free seam). Arepo's Voronoi mesh is *not* forced through
  `MeshInterface`; it is an unstructured `CellSet` (positions, volumes,
  conserved state) — the seam stays honest.
- **Exchange operators**: conservative remap between any two representations.
  AMR↔AMR is restriction/prolongation plus tree intersection (HG.jl
  machinery). AMR↔Voronoi is *exact geometric clipping* — precisely what
  HierarchicalGrids' Layer-4 overlap module with R3D was built for: clip
  Voronoi cells (or a Delaunay/simplicial proxy) against tree cells, deposit
  moment-preserving. Particles exchange as `ParticleSet` (id, pos, vel, mass)
  with deterministic-injection entry points (RamsesLib's `init_particles`
  pattern, ArepoLib's setters).
- Every exchange is **measured**: `Σ value×volume` before/after to round-off
  is an assertion, not a hope.

### D4 — Generalize the slot registry across codes

ADR-0002's `EngineConfig` slot values widen from `{OFF, ENZO, JULIA}` to *any
provider*: a slot implementation is `(provider, method)` where provider ∈
{`:host`, `:julia`, `:enzo`, `:ramses`, `:arepo`, …}. One code is the **host**
— it owns the mesh, the timestep ladder, and the AMR/conservation machinery
(the ADR-0002 rule that structural steps are never swapped, now per-host). A
**guest slot** runs sync-out → canonicalize → remap (if meshes differ) → guest
step → remap back → sync-in, through D3's operators.

```julia
cfg = EngineConfig(host = :enzo,
                   hydro = :host,                  # Enzo PPM
                   gravity = (:julia, :poissonkernels),
                   radiation = (:ramses, :rt),     # RAMSES-RT as a guest
                   model = IdealHydroPlusRT())
```

The certification discipline is unchanged and non-negotiable: every guest
slot is gated per-step and per-run against the host's native implementation
(where one exists) or against an analytic/oracle problem (where it doesn't).
Slot boundaries remain data contracts; slots whose contracts interlock swap
as a unit.

### D5 — The portable solver library is the convergence point

PPMKernels is the template: port a solver into KernelAbstractions, certify
bit-tight against the live legacy kernel via the fixture harness, then serve
it back to *all* codes as a `:julia` slot. The library grows along three
fronts:

- **From Enzo:** done — PPM, MUSCL/Hancock, PPML, multigrid+FFT gravity.
- **From mini-ramses:** the hand-written MSL kernels (gravity validated, hydro
  in progress) are re-expressed in KA — the physics and the parity
  methodology (PORT_MAP discipline) carry over; the dialect does not. RAMSES's
  unsplit MUSCL on octs becomes the second hydro family; its multigrid the
  second gravity family. mini-ramses remains the *reference*, not the kernel
  source.
- **New methods:** dfmm enters as an `EquationSet` (Cholesky-sector moment
  variables) plus solver against `MeshInterface` — it already lives on
  HierarchicalGrids, so it is closest to native. Moving-mesh/dual-frame
  adaptivity enters through the `CellSet`/remap layer rather than by forcing
  Voronoi through the AMR seam.

End state: a request like "Sedov, RAMSES host, PPMKernels Metal hydro,
PoissonKernels gravity, Veusz live report" is an `EngineConfig`, not a
project.

## Package topology

Federation, not a monorepo merge. Repos keep their owners, tests, and release
cadence; EnzoNG.jl composes them via Julia package extensions (weak deps), so
`using EnzoNG, RamsesLib` activates the Ramses provider with zero cost when
absent.

```
EnzoNG.jl                      # umbrella: driver, slots, exchange, problems, harness
├── lib/MeshInterface          # the mesh seam (unchanged)
├── lib/CodeBridge             # NEW (D1): loading, handles, contracts, @xcall+worker
├── lib/EnzoLib                # Enzo bridge       → CodeBridge client
├── lib/RefMesh, lib/HGBackend, lib/EnzoBackend
├── lib/PPMKernels, lib/PoissonKernels   # portable KA solver library
├── lib/EnzoViz                # in-situ viz
├── ext/EnzoNGRamsesExt        # activated by RamsesLib  (RamsesNG.jl repo)
├── ext/EnzoNGArepoExt         # activated by ArepoLib   (Arepo.jl repo)
├── ext/EnzoNGVeuszExt         # activated by Veusz.jl
└── ext/EnzoNGDfmmExt          # activated by dfmm
External, unchanged owners:
RamsesNG.jl/lib/RamsesLib · Arepo.jl/lib/ArepoLib · HierarchicalGrids.jl ·
Veusz.jl · dfmm · legacy checkouts (enzo-dev, mini-ramses, arepo)
```

Future codes (Athena++, GIZMO, Gadget-4) are one `CodeBridge` client each:
write the C-ABI shim in the legacy tree (the `arepo_bridge.c` /
`ramses_capi.f90` recipe is now documented practice), declare the contract,
write the wrapper in `@xcall` style, get the worker and the parity oracle for
free, then implement `to_canonical`/`from_canonical`.

## Flagship use cases (the acceptance tests of the architecture)

1. **One IC, three codes.** Ingest RAMSES `grafic` ICs once into canonical
   state; inject into Enzo (fixture path), RAMSES (native), Arepo
   (`set_particle_field!`/cell setters). Evolve all three as workers; common
   diagnostics (profiles, spectra, conservation ledgers) on canonical state;
   one Veusz/EnzoViz report. *Exercises: D1, D2, D3-ingest, harness.*
2. **RAMSES-RT inside Enzo.** Enzo host; per coarse step, canonicalize the
   Enzo level state, remap AMR↔AMR onto a RAMSES hierarchy held by a Ramses
   worker, advance RT, remap ionization/heating back as source terms.
   *Blocked on: wrapping the RT entry points in `ramses_capi` (currently
   stubs). Exercises: D2, D3, D4.*
3. **Moray inside Arepo.** Arepo host; deposit Voronoi gas onto a scratch HG
   hierarchy via R3D exact clipping; run Enzo's Moray through EnzoLib on that
   hierarchy; remap photo-rates back to cells. *Exercises the AMR↔Voronoi
   operator — the hardest exchange, hence scheduled after 1–2.*
4. **PPMKernels hydro inside RAMSES.** Replace `godunov_fine!` per level with
   the KA Metal sweep through `RamsesMesh` (RamsesLib already exposes
   uold/unew get/set per level — this is the cheapest cross-code slot and the
   first to certify). *Exercises: D4, D5 in the guest direction.*

## Roadmap (each phase gated, in the ADR-0002 certification style)

- **Phase 0 — CodeBridge extraction.** Mechanical: factor the five shared
  mechanisms out of EnzoLib; port RamsesLib and ArepoLib onto them. Gate:
  every existing test in all three wrappers stays green; EnzoLib's
  `:local`≡`:remote` parity is bit-identical before/after.
- **Phase 1 — Multi-worker.** Worker binaries for RamsesLib/ArepoLib via the
  manifest generator; two codes stepping concurrently in one session. Gate:
  Enzo+RAMSES Sedov both advance 100 steps under one driver, each matching its
  single-code run bit-for-bit.
- **Phase 2 — Canonical state + comparison harness** (flagship 1). Gate:
  conservation ledger to round-off through every `to_canonical` round-trip;
  three-code Sod/Sedov report generated by one Problem spec.
- **Phase 3 — First cross-code slot** (flagship 4). Gate: per-step
  certification of PPMKernels-in-RAMSES vs native `godunov_fine!` to scheme
  tolerance; per-run Sedov agreement.
- **Phase 4 — Exchange physics** (flagships 2, 3). Includes wrapping
  RAMSES-RT entry points and standing up the R3D AMR↔Voronoi operator with
  conservation assertions. Gate: Iliev-style RT comparison tests
  cross-checked between Moray and RAMSES-RT *on the same density field*.
- **Phase 5 — Library convergence.** KA re-expression of mini-ramses kernels;
  dfmm `EquationSet` integration; per-solver retirement of duplicates as
  certification permits. Ongoing; gated per solver, never big-bang.

## Consequences

**Positive.** Each thread keeps its momentum and its tests; the framework is
assembled from certified parts rather than rewritten. The wrapper recipe
becomes a documented, repeatable pattern with most of the cost paid once (in
CodeBridge). Cross-code comparison — the scientifically novel capability —
arrives in Phase 2, early. The portable library inherits a *three-code*
certification surface, which is a stronger correctness statement than any
single code can make. New methods (dfmm) and new codes (Athena, GIZMO) have a
defined, bounded on-ramp.

**Negative / risks.**
- *Units and conventions are the silent killer.* Comoving vs supercomoving vs
  code units, dual-energy formalisms, γ conventions, sign conventions on
  potentials. Mitigation: units are part of the field descriptor and the
  contract handshake; `to_canonical` round-trip tests are mandatory per
  bridge; nothing downstream of a bridge ever sees code units.
- *Remap accuracy vs conservation.* AMR↔Voronoi exchange is conservative by
  construction (R3D) but diffusive; running a guest slot through a remap every
  step has a cost in effective resolution. Mitigation: the comparison harness
  *measures* it (guest-slot runs vs native runs); some couplings will be
  source-term-only (RT rates) precisely to avoid remapping the hydro state.
- *Capability gaps in the legacy bridges.* RAMSES RT and particles are
  unwrapped; Arepo exposes no step-internal hooks (only whole sync-steps), so
  fine-grained slot swapping inside Arepo needs bridge work in the C tree.
  Scheduled, not assumed.
- *Worker-process MPI lifecycles.* N workers each owning an MPI world is fine
  (separate `mpiexec`s); a future *distributed* multi-code run (two codes
  sharing one allocation) is out of scope here and would need its own ADR.
- *Drift between federated repos.* The contract hash refuses skew at connect
  time, but coordinated changes (descriptor schema, EquationSet roles) now
  span repos. Mitigation: CodeBridge owns the schema and versions it; the
  umbrella CI runs the cross-repo gates.

**Non-goals.** Merging the repos. Forcing Voronoi/moving-mesh through
`MeshInterface`. A universal parameter file (Problems are source code,
P9 of ADR-0001). Distributed multi-code runs. Replacing the legacy codes'
own developer communities — the bridges are additive shims in their trees,
never forks of their cores.

## Implementation status (2026-06-09)

- **Phase 0 — done.** `lib/CodeBridge` extracted (LazyLib loading, Bridge,
  `@xcall`, manifest/contract, worker RPC; own harness 27/27 incl. a compiled-
  fixture local≡remote parity oracle). EnzoLib ported with its 55+ `@xcall`
  sites untouched: contract hash **bit-identical** (`0x1d40ca524b8336a4`, 63
  symbols — the prebuilt C++ workers handshake unchanged), full suite 229/229
  incl. parity vs both the Julia AND the C++ worker. RamsesLib ported +
  converted to `@xcall` (36-symbol contract; dual CPU/Metal flavors via
  `flavor!`; 10/10). ArepoLib ported + converted (12-symbol contract; suite
  green after the separately-landed Config/param-error fix).
- **Phase 1 — done.** Multi-worker sessions (`test/multicode/`): Enzo (C++
  worker, Sod AMR) + RAMSES (Julia worker, 2M-oct Sedov) stepped INTERLEAVED
  from one driver, each **bit-identical** to its single-code local run; Arepo
  joined as a third live worker (14/14). Lesson encoded in CodeBridge: the
  worker now carves the control channel out of fd 1 and repoints fd 1 at
  stderr, so legacy banners (RAMSES's Fortran unit-6 prints) can never corrupt
  the wire protocol.
- **Phase 2 — done.** `lib/MultiCode`: `CellSet` canonical state + ledger,
  per-code `extract`/`inject!` adapters, exact Riemann oracle, and the
  three-code Sod harness (29/29). One spec, each code's NATIVE setup path:
  mass/energy conserved through every adapter to round-off (1.6e-15 Enzo,
  3.2e-12 RAMSES@2M cells, 4e-16 Arepo); extract→inject!→extract bit-identical
  on all three; L1(ρ) vs exact 0.0029 (Enzo PPM) / 0.0094 (RAMSES HLLC 128³) /
  0.0063 (Arepo moving mesh); report at `reports/multicode/sod_comparison.md`.
  Design note: a step IC in a periodic box carries a mirror Riemann problem at
  the wrap seam — t̂=0.1 + a double-length RAMSES domain + windowed profiles
  keep the comparison seam-clean (Enzo's outflow BCs need none of it).
- **Phase 3 — done.** The first cross-code slot (`MultiCode.ramses_ppmk_hydro_step!`):
  PPMKernels' Enzo-certified MUSCL-Hancock (PLM+HLLC) replaces `godunov_fine!`
  per level inside a live RAMSES run — raster (exact, the oct ckey IS the cell
  address) → KA step with the HOST's CFL dt → deraster (15/15). Per-step vs
  native from the same developed state: both conserve to 1e-12, inter-scheme
  difference 9.6% of the step's own update (≤15% gate), max|Δρ|=0.0037 at the
  shock. Per-run guest Sod: conservation to round-off and **L1=0.0130 vs the
  host's native 0.0161** — the guest slot beats the host on its own problem.
- **Phase 4 — done** (exchange physics; `lib/MultiCode` moray/exchange/ramsesrt):
  - *4.1 Moray as a service:* `run_moray_stromgren` drives Enzo's PhotonTest
    (= Iliev Test 1) through the certified radiation+cooling slots, with an
    optional injected density field. I-front tracks the analytic Strömgren
    front to ~1–5% from t = 3 Myr (lesson: the outer dt must stay small — the
    photons subcycle but the chemistry advances once per outer cycle).
  - *4.2 Exchange + flagship 3 (Moray inside Arepo):* conservative
    `deposit_to_grid` (NGP/CIC, conservation asserted not hoped) +
    `sample_at_points`; exact R3D Voronoi clipping slots in when the Arepo
    bridge exports 3-D cell geometry. Live demo: the Arepo Sod tube's density
    broadcast onto the RT grid (light gas toward the source), Moray's front
    runs 1.63× farther than in uniform gas, kphHI/PhotoGamma sampled back onto
    the Voronoi cells and injected as utherm heating (exact write-back), and
    Arepo keeps stepping.
  - *4.3 RAMSES-RT wrapped (flagship 2 core):* `ramses_rt_step/rt_setup/
    rt_neq_updates/get_rt/set_rt/nrtvar` added to `ramses_capi.f90` (#ifdef-RT
    guarded), `bin64hrt` build flavor (`HYDRO=1 GRAV=1 RT=1 NRTGRP=1 NION=1`),
    RamsesLib `:rt` LazyLib flavor + `rt.jl` bindings. Iliev namelist
    generated in kpc–mp–Myr units. Debug findings now encoded: (a)
    `read_rt_params` re-read argv → capi override mirrored in; (b) the σ·c
    chemistry tables are only filled by `r_rt_neq_updates` (update_time) —
    must be called once after init; (c) ion fractions are stored
    density-weighted (`uold₆ = xHII·ρ`); (d) upstream mini-ramses bugs: point-
    source CIC y/z weights use the x-center variable, and a corner source's
    CIC cloud clips to 1/8 — worked around with cell-centered placement,
    flagged upstream.
  - *4.4 The cross-check gate:* the SAME Iliev field through Moray (ray
    tracing, 32³) and RAMSES-RT (M1, reduced c, 64³): both within 12% of the
    analytic front at t = 3, 5 Myr, code-vs-code agreement within the Iliev
    inter-code band; joint report at `reports/multicode/rt_crosscheck.md`.
- **Phase 5 — done** (flagship 2 + GPU guest + hardening; full suite 101/101):
  - *Flagship 2 — RAMSES-RT inside an Enzo simulation*
    (`MultiCode.run_enzo_host_ramsesrt`, `test_rt_guest.jl` 13/13): Enzo hosts
    PhotonTest (grid, fields, clock); a PERSISTENT RAMSES-RT guest provides
    the radiation+chemistry slot, advanced once per Enzo cycle with the host's
    dt, its ionization state written into Enzo's live HII/HI/e⁻ fields every
    step (the slot data contract). Uniform field: the front measured FROM THE
    ENZO-HELD FIELDS tracks the analytic Strömgren solution (≤12%); the host
    density is bit-untouched; Enzo's own boundary + native cooling machinery
    run on the guest-written state. Structured field (light gas at the
    source): Moray and the guest agree the front sweeps the light half and
    stalls at the dense interface at exactly x̂ = 0.500 (ratio 1.000).
    Observable lesson: M1 wrap-around deposits an ionized skin at far
    periodic corners — fronts must be measured as the contiguous ionized run
    from the source.
  - *Metal guest slot* (`ramses_ppmk_hydro_step!(device = :metal)`): the
    f32 GPU sweep inside f64 RAMSES — raster → device → certified Metal
    kernels → host — with per-run Sod L1 agreeing with the CPU guest to NINE
    significant figures (0.012996783 vs 0.012996773) and conservation at the
    f32 floor. RAMSES's mesh, Enzo's kernels, Apple's GPU, one keyword.
  - *Protocol hardening:* zero-argument `@xcall`s over RPC (Arepo's
    `run!`/`run_step`) sent a trailing space that corrupted worker parsing —
    fixed both sides, covered in the CodeBridge fixture harness; a second
    Arepo `init` in one process crashes its C-global allocator (the D2
    singleton problem, observed live) — `run_arepo_sod(worker = true)` runs
    it in its own worker process through the same `@xcall` surface.
  - *Regression sweep* after all protocol changes: EnzoLib 229/229, Phase-1
    multi-worker 14/14, RamsesLib 10/10, ArepoLib 21/21, CodeBridge 29/29.
  - *Docs:* CLAUDE.md now carries the CodeBridge/MultiCode section (packages,
    run commands, the cross-repo `[sources]` coupling, the hard-won gotchas).
- **Phase 6 (partial) — the EXACT exchange + upstream fixes; all committed:**
  - *3-D Voronoi export:* `arepo_get_voronoi_3d` in the Arepo bridge (the
    voronoi_3d.c edge-ring traversal: every face as its ordered circumcenter
    ring + the outward generator direction), a 3-D library flavor
    (`Config_3d.sh`, `make shared CONFIG=Config_3d.sh BUILD_DIR=build3d
    LIBRARY=arepo3d`) selected per-WORKER via the `AREPO_LIB` env — the
    LazyLib override makes the flavor switch zero-code. The mesh must be
    live (after init / between steps; a completed `run!` frees it).
  - *`MultiCode.deposit_exact`:* signed fan tetrahedra over outward-oriented
    rings (divergence theorem — exact even for the wandering rings of a
    degenerate lattice Delaunay, where unsigned fans overlap by 100×), each
    tet clipped by the grid's axis-aligned planes via R3D. Numerical lesson
    encoded: clip the TET by the BOX — a sliver tet's face planes (crosses of
    nearly parallel edges) are pure noise, and the lattice Delaunay produces
    swarms of slivers. Gate: on the live 27000-cell noh_3d mesh (199423
    faces, 1.02M ring vertices), every cell's clipped volume reproduces
    Arepo's own `SphP.Volume`; the box tiles exactly; ledgers conserved.
    Suite: 108/108 across eight gates.
  - *Upstream mini-ramses RT injection fixed* (was a chip): point-source CIC
    y/z weights used the x-center variable, and boundary sources clipped
    their CIC cloud (a corner source emitted 1/8 of rt_n_source) — now
    minimum-periodic-image weights; a corner source reproduces the analytic
    Strömgren front (0.88 at t=3 Myr level 5, the documented M1 lag) where
    it gave 0.50 before.
- **Phase 7 — the science-grade run + the guest under AMR (done):**
  - *One Sedov IC, four engines* (`sedov_compare.jl`, report at
    `reports/multicode/sedov_comparison.md`): the SAME discrete thermal-bomb
    IC injected through each code's live-field bridge (apples-to-apples at
    the cell level, no per-code initializer quirks) into Enzo PPM, RAMSES
    unsplit MUSCL+HLLC, and the PPMKernels guest on RAMSES's mesh (CPU f64 +
    Metal f32), 64³, with the Sedov–Taylor R(t) computed from each run's
    MEASURED injected energy. Result: Enzo and RAMSES land on the same
    R/Rₐ = 0.886 to 4 digits; the two guest runs land on 0.866 to 4 digits
    (f32 ≡ f64); the common deficit from 1 is the 3-cell bomb at 64³, shared
    by all engines — the cross-code agreement is the measurement. Wall-clock
    on identical meshes: RAMSES 7.8 s, guest-Metal 16.5 s, guest-CPU 20.8 s,
    Enzo-through-the-bridge 67 s. Conservation 1e-12…1e-8(f32).
  - *The guest slot under AMR* (`ramses_composite_raster/_deraster!`,
    `ramses_ppmk_hydro_step_amr!`, `test_amr_guest.jl`): correctness-first
    COMPOSITE coupling — the live multi-level hierarchy rasters onto the
    uniform finest grid (leaf injection, coverage asserted), the guest
    advances it, and the result restricts back to every level. The host
    keeps owning refinement (its flag/refine runs between steps; the refined
    region follows the blast). Gate: a 2-level Sedov with live regridding is
    **bit-identical** to the guest-on-uniform-fine run (R to 17 digits) with
    coarse ≡ restricted-fine exactly — conservation by construction. Two
    lessons encoded: RAMSES's `newdt_fine` returns 0 on a level whose time
    state the guest manages, so the guest owns its CFL
    (`courant·dx/max_wavespeed` on the composite); and ghosts must be filled
    before the wavespeed scan (ρ = 0 ⇒ NaN). The per-level fast path (raster
    each level with coarse-interpolated ghosts + flux registers) remains the
    optimization track.
- **Next-1 — persistent Metal device residency (done):**
  `run_ramses_sedov(resident = true)` rasters ONCE, advances the whole run
  on-device inside `PPMKernels.with_pool()` (the guest owning its CFL via
  `max_wavespeed` on the composite), and derasters ONCE at the end.
  Six-engine Sedov: guest-metal-resident 1.68 s vs RAMSES native 7.2 s
  (4.3× faster than the host's own solver on the identical mesh) vs
  guest-metal round-tripping 15.0 s — the raster round-trip WAS the cost.
  CPU-resident 8.4 s vs 20.3 s round-tripping. Same R/Rₐ to f32 round-off.
- **Next-2 — the cosmology gate: one particle set, Enzo + RAMSES vs the
  exact trajectory (done):** a Zel'dovich plane wave with ZERO initial
  velocities (no velocity-unit convention can enter), the SAME
  Julia-generated 32³ lattice + sinusoidal displacement injected through
  both codes' particle bridges, evolved a_i → 4a_i in EdS, measured against
  the closed-form mixed-mode growth b(x) = (3/5)x + (2/5)x^{-3/2}
  (`zeldovich.jl`, `test_zeldovich.jl`, report
  `reports/multicode/zeldovich_comparison.md`). Enzo runs its real
  CosmologySimulation machinery (new bridge particle setters
  `enzomodules_problem_set_particle_pos/vel` — a contract-hash change, all
  workers rebuilt; EdS-patched `dm_only` parameter file; certified
  EvolveLevel slots with gravity+cosmology): growth ratio 0.9893, shape
  residual 0.024·A, 71 steps, 0.8 s. RAMSES runs its production `amr_step`
  on a new `UNITS=COSMO` build (`bin64sc`, the `:cosmo` LazyLib flavor)
  with grafic headers written from Julia (zero-content velocity planes —
  particles injected directly): ratio 0.9942, residual 0.030·A, 14 steps,
  0.3 s. Cross-code growth-normalized displacement amplitudes agree to
  0.5%·A. Two engines, zero shared code, one analytic answer.
- **Next-3 — the per-level AMR fast path (done):**
  `ramses_ppmk_hydro_step_amr_fast!` advances each level on its own
  bounding-box raster — coarse at coarse cost, fine only over the refined
  region — with coarse-injected ghosts (every non-level cell re-injected
  from the frozen time-t parent before each directional sweep) and FLUX
  REGISTERS: the guest records per-axis face fluxes (`fluxrec`), and every
  leaf cell facing a refined cell replaces its face flux with the area mean
  of the 4 child-face fluxes (ΔU = ±dt/dx·(F − F̄)), after which the
  composite flux telescopes — every physical face crossed by exactly one
  flux.  Refined cells restrict bottom-up (coarse ≡ average of children).
  Gate (`test_amr_fastpath.jl`): one isolated step conserves the composite
  mass BIT-EXACTLY (Δm = 0.0, ΔE = 4.4e-16); the full 77-step Sedov run
  with live host regridding conserves to 1e-10 with upload error 0.0 and
  lands the shock radius in the identical bin as the composite path —
  at **2.34×** the composite's speed (2.6 s vs 6.2 s, two-level 32³/64³).
  2:1 grading (RAMSES enforces it) keeps the child side of every
  coarse-fine face a leaf, so one register level per level pair suffices.
  Per-level subcycling and a Metal variant are the remaining optimization
  track (the registers and masks currently live host-side).
- **Next-4 (first slice) — the gravity guest slot: KA Poisson inside RAMSES
  (done):** the start of the mini-ramses-kernel KA re-expression, as a SLOT
  rather than a line-by-line port: RAMSES keeps its deposit (`rho_fine`) and
  force differencing (`force_fine`, the 4th-order 5-point gradient); the
  guest (`ramses_ka_poisson!`, `gravity_slot.jl`) replaces
  `multigrid`/`phi_fine_cg` with PoissonKernels' FFT solve using a NEW
  `greens=:discrete7` kernel — the discrete 7-point Laplacian's
  eigenvalues, i.e. the EXACT solution of the very linear system RAMSES's
  MG iterates on (no O(h²) spectral-vs-discrete gap).  Certification
  (`test_gravity_slot.jl`): one injected density mode, host-deposited; at
  RAMSES ε=1e-12 (namelist `epsilon_base` — levelmin uses it, NOT
  `epsilon`) the two potentials agree to **6.7e-13** and the `force_fine`
  accelerations likewise — and at a loose ε=1e-3 the diff is ~1e-3, so the
  gate tracks the oracle's convergence.  An analytic anchor closes the
  loop: the injected single mode is a discrete eigenfunction, and φ(KA)
  matches the closed-form discrete solution to **9.4e-15** — independent
  of both solvers.- **Next-4 (slice 2) — the refined-level Dirichlet solve (done):**
  RAMSES's fine-level system (`cmp_residual_cg`) is, for a CUBOID refined
  region, exactly a Dirichlet box problem: the 7-point Laplacian over the
  level's cells with ghosts fixed at `interpol_phi` of the 27 coarse
  parents around each missing oct (tfrac=0 outside subcycling).  The
  guest assembles that system on the bbox raster — the ghost ring
  reconstructed through the SAME `interpol_phi` kernel the host runs
  (RamsesLib wraps it pure) — and solves with `vcycle_solve!(dirichlet =
  true)`.  Two independent gates: the ORACLE's converged `phi_fine_cg`
  solution satisfies OUR assembled system at **1.3e-16** (replication
  certified without the KA solver), and the KA Dirichlet W-cycle from
  scratch lands on the oracle at **2.4e-12**.  The cuboid hierarchy is
  built by writing `flag1` through the bridge (new capi setter case) and
  calling the host's own `refine_fine` — `m_flag_fine` RESETS flag1, so
  drive refine directly.  Remaining in this track: the oct-native KA
  smoother for tree-irregular (non-cuboid) levels.
- **Next-5 — dfmm as a harness engine, via MultiCode's first package
  extension (done):** dfmm (the dual-frame moment method — variational,
  symplectic, Lagrangian segments on HierarchicalGrids; the first solver
  born inside the ecosystem) runs the harness's Sod spec (γ = 5/3, its
  closure's index; the [0,1] problem mirrored onto a [0,2] periodic
  domain so neither seam carries a jump) and is gated against the SAME
  exact-Riemann oracle as the legacy engines: L1(ρ) = 0.042 (inside its
  documented Tier-A.1 band), mass conserved BIT-exactly, total momentum
  at 1e-16 — the variational exactness claims reproduced in the
  harness's own terms (`reports/multicode/dfmm_sod.md`).  The seam is
  `MultiCodeDfmmExt` (weakdep + `[extras]` for the `[sources]` path
  rule): `using dfmm` activates `run_dfmm_sod`; dfmm's heavy dependency
  tree (Makie, Enzyme) never burdens a legacy-only user — this IS the
  extension-ifying pattern for MultiCode.  The full EquationSet
  embedding in EnzoNG's FV driver is deliberately NOT claimed: dfmm is
  not a flux-form method (its state is per-segment moments and its
  integrator is variational), so it enters as an ENGINE; an EquationSet
  facade would misrepresent the seam.  Revisit when dfmm-2d's Eulerian
  remap lands.
- **Next-6 — the IRREGULAR refined region (done):** the fine-level solve
  generalized beyond cuboids (`ramses_ka_poisson_fine!`): masked bbox
  raster of an arbitrary blob region, `interpol_phi` ghosts on every
  region-adjacent missing-oct cell (mask-driven, same kernel), and
  matrix-free conjugate gradients on the masked SPD system
  `6φ − Σφ_nbr = −4π·dx²·(ρ−ρ̄) + Σ ghosts`.  Certification on a
  SPHERICAL blob (840 octs in a 24³ bbox — genuinely non-cuboid,
  asserted): the oracle's converged `phi_fine_cg` solution satisfies OUR
  masked system at **4.3e-15**, and the masked CG lands on the oracle at
  **7.3e-13** in 62 iterations.  The gravity guest now solves ANY
  refined-region shape RAMSES hands it; a KA-kernelized masked smoother
  (GPU execution of the same system) is the performance follow-up.
- **Next-7 — the KA-kernelized masked CG (done):**
  `PoissonKernels.masked_cg!` — the masked 7-point apply as a single
  branch-free `@kernel` (the covered mask travels as a FIELD, 1/0, so
  the stencil multiplies instead of branching) + a device CG driver
  whose vectors stay zero outside the mask by construction (plain
  `dot`/broadcast reductions, no masked-reduction kernel needed), with
  the `vcycle_solve!`-style stagnation guard catching the f32 floor.
  `ramses_ka_poisson_fine!` now runs on it with a `device` kwarg: CPU
  f64 reproduces the host CG's blob certification at **7.3e-13**;
  Metal f32 solves the SAME blob system on the GPU to **4.5e-7** (the
  f32 residual floor) in 39 iterations — one source, two devices, the
  D5 contract applied to the irregular-domain solver.  Gravity-slot
  gate now 14/14 (root, cuboid Dirichlet, blob CPU, blob Metal).
- **Wrapper-registry audit (2026-06-10):** the parallel-session wrappers
  are integrated and verified consistent with the contract: **Music.jl
  (MusicLib)**, **Athena.jl (AthenaLib)**, **Gadget4.jl (Gadget4Lib)** —
  all CodeBridge clients with `[sources]` pointing at this repo's
  `lib/CodeBridge`, each clean on `main` with its own green suite — plus
  **DiscoDJ.jl (DiscoDJLib)** (PythonCall, deliberately not CodeBridge:
  JAX must stay in-process and traced for differentiability).  Six
  CodeBridge clients now ride one substrate; the full registry and
  capability table is `docs/framework-surface.md` (linked from the
  README), and CLAUDE.md carries the ripple warning (a CodeBridge change
  touches six sibling repos).  Their MultiCode cross-code gates
  (injectors, comparison rows) are the recorded on-ramp work, following
  the Phase-2 pattern.
- **Next-8 — Athena++ in the Sod harness (the first registry on-ramp,
  done):** `run_athena_sod` via `MultiCodeAthenaExt` (the second package
  extension — `using AthenaLib` activates it): the stock `athinput.sod`
  (γ = 1.4, interface recentred to the harness frame) run IN-PROCESS,
  profile from the final `.tab`, conservation from the `.hst`.  Gates:
  **L1(ρ) = 0.0019** / L1(u) = 0.0034 vs the shared exact-Riemann oracle
  — the sharpest engine in the harness — with **exact** mass
  conservation, in 0.02 s wall-clock (in-process, 256 cells).  Report:
  `reports/multicode/athena_sod.md`.  Remaining on-ramps: MUSIC injector
  validation (one MusicSpec → Enzo+RAMSES live comparison), Gadget4
  ICs/halo gates, DiscoDJ LPT cross-check vs the Zel'dovich machinery.
- **Next-9 — the MUSIC injector validation (done):** ONE `MusicSpec`
  realization (deterministic seeds), generated in-process in two formats;
  Enzo booted on MUSIC's own `parameter_file.txt` + particle ICs, RAMSES
  (UNITS=COSMO) on the grafic2 level directory (symlinked short — Fortran
  filename buffers truncate at 80 chars); the INITIAL CIC density fields
  compared with no evolution (`run_music_crosscheck`, `MultiCodeMusicExt`,
  the third package extension).  Result: correlation **1 − 4e-15**, rms
  residual **1.0e-7 of sigma** — exactly the float32 precision of the
  grafic planes, the only difference between the two injection chains.
  The IC chain MUSIC → format writers → each code's cosmological init is
  certified end-to-end.  TRAP: MUSIC's in-process white-noise generation
  segfaults in a process already hosting a dozen live codes (competing
  OpenMP/FFTW runtimes) — the gate therefore runs FIRST in the suite;
  the durable fix (recorded polish) is the D2 one: MUSIC in its own
  CodeBridge worker.  Also: Fortran filename buffers truncate at 80
  chars — symlink long grafic paths short.  Remaining on-ramps: Gadget4 IC/halo gates,
  DiscoDJ LPT cross-check (NGenIC-phase work already landed in the
  shared-phases session).
- **Next-10 — DISCO-DJ ICs through both codes vs the exact growth
  (done):** the differentiable-IC flagship (`run_discodj_growth`,
  `MultiCodeDiscoDJExt`, the fourth package extension): DISCO-DJ's JAX
  1LPT (≡ Zel'dovich) displacement field becomes ZERO-velocity particles
  (the Next-2 trick — no velocity-unit convention can enter) injected
  into Enzo AND RAMSES and evolved aᵢ → 4aᵢ in EdS; the WHOLE linear
  field follows b(x) = ⅗x + ⅖x^{−3/2} mode-independently.  Gates: the
  large-scale (res/4-deposit) growth tracks b at each engine's measured
  epoch to 0.963 (Enzo) / 0.977 (RAMSES) — the few-% deficit is the
  shared 32³ PM resolution, while the full-field rms is
  Nyquist-suppressed as expected; the field keeps its shape (corr vs ICs
  ≈ 0.97); and the two codes land on the SAME field, Enzo↔RAMSES
  correlation **0.998**.  Report `reports/multicode/discodj_growth.md`.
  `cic_density` promoted to MultiCode core (the shared measurement
  operator of the injection gates).  Remaining on-ramp: Gadget4 IC/halo
  gates; then MUSIC-in-a-worker (the durable D2 fix).
- **Next-11 — the GADGET-4 halo service in the harness (done):** the
  LAST wrapper on-ramp (`run_gadget4_halos`, `MultiCodeGadget4Ext`, the
  fifth package extension): FOF+SUBFIND as a SERVICE on particles in
  MultiCode's conventions (N×3 rows, [0,1)³; the particle mass set
  cosmologically consistently so the linking length is 0.2× the mean
  spacing — the recorded G4 trap).  Gates: three planted clumps →
  EXACTLY three groups of ~500; a LIVE RAMSES Zel'dovich dump at
  a/aᵢ = 4 → zero groups (pre-caustic) — a real foreign-code particle
  dump through the service.  TWO process-boundary traps closed: an
  extension sees only its parent's deps + triggers (HDF5 reaches the ext
  THROUGH Gadget4Lib); and HDF5.jl + Enzo's grid dylib each carry a
  libhdf5 whose interposed symbols ABORT whichever loads second — the
  gate therefore runs in its OWN process from runtests (the D2
  philosophy applied to tests; HDF5_DISABLE_VERSION_CHECK does NOT
  help), and the live-code half uses RAMSES (plain Fortran I/O).
  EVERY wrapper in the registry now has a live cross-code gate.
- **Next-12 — MUSIC in a CodeBridge worker (done):** the durable D2 fix
  replacing the run-first ordering workaround: `run_music_crosscheck(
  worker = true)` spawns MusicLib's Julia reference worker
  (`CodeBridge.connect_worker!` around the two `generate` calls) so
  MUSIC's white-noise generation runs in its OWN process, immune to the
  OpenMP/FFTW runtime pollution of a many-code host.  The worker path
  reproduces the in-process result BIT-identically (corr and rms equal
  to the last digit) — the @xcall transport seam doing exactly what D1/D2
  promised: one call site, two transports, same answer.  The suite gate
  now runs in worker mode.
- **Next-13 — Athena++ 3-D → the canonical state (done):** `read_vtk`
  in AthenaLib (legacy BINARY RECTILINEAR_GRID, big-endian float32, one
  MeshBlock = whole domain in one file — no HDF5 anywhere, dodging the
  libhdf5∥libenzo conflict by construction) + `run_athena_sod3d`: the
  stock Sod extruded to 32³ (a derived athinput appends the `<output3>`
  vtk block — command-line overrides cannot ADD blocks), read back into
  a `CellSet`.  Gates: ledger mass 8.3e-10 / energy 1.3e-9 vs the
  analytic reference (the f32 VTK floor), transverse scatter EXACTLY 0.0
  (bit-identical columns — 1-D physics in a 3-D box), L1(ρ) = 0.0129,
  0.04 s wall-clock.  TRAP: Athena++ re-entrancy survives same-or-lower
  dimensionality (3D→1D, re-runs) but 1D→3D in one process SEGFAULTS (a
  static sized at first init) — run higher-dimensional cases first.
  Athena++ now has the full Phase-2 treatment: engine row, canonical
  state, ledger, profile.
- **Phase C (slice 1) — the SUBGRID gravity solve certified on live Enzo
  (done):** the recorded "wire the level>0 hook" step.  Enzo's own
  subgrid chain (PrepareDensityField interpolates the parent potential
  into the subgrid PotentialField = the Dirichlet BC + initial guess;
  SolveForPotential iterates MG) is replicated through the bridge on the
  GravityTest fixture (32³ root + nested subgrid + 5000 particles,
  PotentialIterations raised to 30 so the oracle converges deeply).
  Gates (`test_gravity_subgrid_slot.jl` in the EnzoLib suite): the
  scalar α in (Σφ−6φ) = α·GMF fitted from Enzo's OWN solution leaves a
  4.6e-8 residual (system replication certified solver-free), and
  `vcycle_solve!(dirichlet = true)` from the same BC + rhs lands on
  Enzo's φ at **1.6e-15** — machine precision: the BC and rhs determine
  the solution and both solvers reach the same fixed point.  Remaining
  Phase C: the production gravity=:julia hook for all levels (root FFT
  + this subgrid solve) in a full AMR evolve vs gravity=:enzo, then the
  Santa Barbara CPU-vs-GPU run.
- **Phase C (slice 2) — the production gravity=:julia hook (done):**
  `EnzoLib.poisson_gravity_hook()` — the slot swaps ONLY the Poisson
  solve, the RAMSES-slot design replicated on Enzo: level 0 runs Enzo's
  own chain (PrepareDensityField carries the root FFT internally);
  level > 0 deposits + interpolates the BC through Enzo, solves every
  subgrid with the certified Dirichlet W-cycle, writes the potential
  back through the NEW `set_potential` bridge entry, and Enzo's own
  differencing (the NEW `session_gravity_post` = ComputeAccelerations +
  CopyPotentialToBaryonField + external) turns OUR φ into baryon AND
  particle forces.  Contract hash → 67 symbols, workers rebuilt.
  Full-evolve gate on GravityTest (5 root steps with subcycling):
  particle velocities — the integrated force field; ProblemType 23
  deliberately FREEZES positions ("don't move the particles",
  UpdateParticlePositions.C) — are **BIT-IDENTICAL** (max Δv = 0.0 at
  vmax = 0.041) between gravity=:julia and gravity=:enzo: the two
  potentials agree at the f64 ulp, so the differenced forces coincide
  exactly.  Phase C closed with the SB run: the example's
  subgrid TODO replaced by this hook (root φ additionally written to
  PotentialField so child BCs read OUR solution); 8 cycles from z=63
  with hydro(PPM)+gravity(FFT) as :julia slots — CPU-f32 and Metal-f32
  trajectories IDENTICAL to the printed digits, mass drift 2e-10,
  Metal 21.7 s vs CPU 27.4 s (startup-dominated; kernel speedups are
  the recorded 31×/12×).
- **Next-14 — the Athena++ :gr flavor: the first general-relativistic
  engine (done):** one `build_flavor` table line upstream (gr_bondi +
  hlle + kerr-schild, `configure -g`), a `:gr` LazyLib flavor in
  AthenaLib (env ATHENA_LIB_GR), and a `read_vtk` fix (singleton
  dimensions write one node plane — clamp).  Gate: Schwarzschild Bondi
  accretion STATIONARITY — the problem generator initializes the exact
  stationary flow; evolving to t = 100 preserves it to 0.107%
  (vl2+PLM at 100 radial cells), gated at 2e-3.  AthenaLib 20/20.
  GR spacetimes are now one flavor line away for the whole harness.
- **Planned next steps (`docs/roadmap.md`):** the already-scoped
  directions beyond this program, each grounded in completed work with a
  concrete on-ramp — Athena++ per-stage solver slots on Enzo/HG grids;
  more GR spacetimes (Kerr, Fishbone–Moncrief torus, GR shock tube — one
  flavor line each now); the MUSIC noise-file writer + the `ka_poisson`
  plugin seam for fixed-and-paired cross-generator ICs; the Santa
  Barbara run to refinement onset + a GPU production campaign + the
  f32-reference structure-formation gate; and (deferred polish)
  extension-ifying the LEGACY wrappers (EnzoLib/RamsesLib/ArepoLib) in
  MultiCode — lazy pure-Julia bindings, so the weak-dep conversion buys
  little until a registry release forces it; the dfmm extension
  documents the pattern.
