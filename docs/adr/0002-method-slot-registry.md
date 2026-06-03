# ADR-0002: Method-Slot Registry (mix-and-match legacy/Julia physics)

- **Status:** Proposed (scope)
- **Date:** 2026-06-03
- **Deciders:** T. Abel
- **Builds on:** ADR-0001 (architecture), the EnzoLib integration layer
  (`lib/EnzoLib`, the Julia-driven `EvolveLevel` over the legacy C-ABI bridge),
  the `EnzoBackend` seam adapter (`lib/EnzoBackend`), and the `EquationSet`
  variable-set abstraction.

---

## Context

EnzoLib already lets a **Julia-reimplemented `EvolveLevel`** drive the *live*
Enzo hierarchy, calling Enzo's own certified steps for each physics operation
(`session_solve_hydro`, `session_gravity`, `session_comoving_expansion`, the CT
EMF refluxing, …). Across the quicksuite this reproduces `EvolveHierarchy`
bit-for-bit on hydro tubes, ~1e-5 on AMR, and now bit-for-bit/near on
constrained-transport MHD and AMR+CT+cosmology.

One slot is **already swappable**: `evolve_level!(…; hydro!)` takes the hydro
step as a function — the default calls the legacy bridge, and a `:julia` closure
runs EnzoNG's HLLC/PLM/SSP-RK2 on the live grid (proved in E3/E5 via
`EnzoBackend`). Every *other* step is hard-wired to the legacy bridge and gated
by a boolean flag (`gravity`, `cooling`, `radiation`, `cosmology`, `mhdct`, …).

The goal of ADR-0001's "careful, certified incremental rewrite" needs this
generalized: **each physics step becomes a slot that resolves to `:enzo`
(legacy bridge) or `:julia` (EnzoNG kernel) on the shared live hierarchy**, with
per-slot certification against the legacy reference. That is the method-slot
registry.

## Decision (shape)

Replace the ad-hoc `hydro!` argument + boolean flags with a single
**`EngineConfig`** — a table mapping each physics slot to an implementation
symbol — threaded through `evolve_level!`/`run_amr`.

```julia
@enum SlotImpl OFF ENZO JULIA

struct EngineConfig
    hydro::SlotImpl              # ENZO (session_solve_hydro) | JULIA (EnzoBackend kernel)
    gravity::SlotImpl
    cooling::SlotImpl
    comoving_expansion::SlotImpl
    mhd_ct::SlotImpl             # OFF unless UseMHDCT; ENZO = the EMF refluxing
    radiation::SlotImpl
    star_formation::SlotImpl
    model::EquationSet           # the variable set the JULIA slots operate on
end
all_enzo(prob) = EngineConfig(ENZO, ...)        # full replication (today's default)
```

Each slot dispatches through one function that knows how to run a step *either*
way on the live grid:

```julia
run_slot(::Val{:gravity}, impl, h, level, cfg) =
    impl === ENZO  ? session_gravity(h, level) :
    impl === JULIA ? julia_gravity!(h, level, cfg.model) :   # EnzoNG CG Poisson on Enzo memory
                     nothing                                  # OFF
```

`evolve_level!` becomes a fixed orchestration skeleton (set_boundary, create
fluxes, advance_time, the AMR conservation machinery, regrid) that calls
`run_slot(...)` at each physics point. The skeleton/conservation plumbing stays
`:enzo` — only **physics** slots are swappable.

### Slots (initial set)

| Slot | `:enzo` (today) | `:julia` source | Notes |
|---|---|---|---|
| `hydro` | `session_solve_hydro` | EnzoNG HLLC+PLM+SSP-RK2 (E3/E5, done) | the proof of concept |
| `gravity` | `session_gravity` | EnzoNG matrix-free CG Poisson | EnzoNG already has the solver |
| `comoving_expansion` | `session_comoving_expansion` | EnzoNG `apply_expansion_terms!` | role-driven via EquationSet |
| `cooling` | `session_solve_cooling` | (future) | needs chemistry port |
| `mhd_ct` | EMF refluxing (done) | (far future) | face-B CT in Julia is a large port |
| `radiation`/`star_*` | bridge | (out of scope) | |

### What a `:julia` slot requires

A `:julia` slot must run an EnzoNG kernel **on Enzo-owned memory** and leave the
state in exactly the layout the *next* (possibly `:enzo`) step expects:

1. **Read** the live grid fields (`problem_get_field`) and assemble EnzoNG
   conserved state via the `EquationSet` role indices (the `EnzoBackend` sync).
2. **Run** the kernel through the `MeshInterface` seam (`EnzoBackend` aliases the
   grid) — the same unchanged `Simulation`/`step!` path RefMesh/HGBackend use.
3. **Write back** (`problem_set_field`) in Enzo's field layout, including any
   side arrays the legacy pipeline reads (e.g. a `:julia` gravity slot must fill
   `AccelerationField` the way `ComputeAccelerations` does, or gravity+hydro must
   be swapped together so the contract stays internal).

This is the central design constraint: **slot boundaries are data contracts.**
A slot is independently swappable only if its inputs/outputs match the legacy
format exactly; otherwise adjacent slots must be swapped as a unit.

### Certification

Each `:julia` slot is certified against its `:enzo` twin using the existing
harness: run the same problem with the slot `=ENZO` (reference) and `=JULIA`,
compare all fields with `_max_field_error`. Two granularities:

- **Per-step** (tight): one step, only that slot swapped → should match to the
  scheme's tolerance (bit-for-bit where the kernels are equivalent).
- **Per-run** (loose): full evolution with the slot swapped → physics-level
  agreement (the hydro `:julia` slot already passes the quicksuite this way).

The fixtures/diff policy from `lib/EnzoFixtures` is the per-kernel gate; the
quicksuite harness is the per-run gate.

## Scope

**In scope (this milestone):**

- **Phase A — formalize the registry (refactor, no behavior change).** Introduce
  `EngineConfig`, convert the `hydro!` arg + boolean flags into slots, route
  `evolve_level!`/`run_amr` through `run_slot`. Default = all-`:enzo`; the whole
  quicksuite must stay green bit-for-bit (pure refactor).
- **Phase B — hydro slot via the registry.** Wire the existing E5 `EnzoBackend`
  hydro path in as `hydro=JULIA`; certify Sod/Toro through the registry.
- **Phase C — first *new* port: `gravity=JULIA`.** Run EnzoNG's CG Poisson on the
  live grid, fill `AccelerationField`, certify vs `gravity=ENZO` on a
  self-gravity problem (per-step first, then per-run). This is the template every
  later port reuses.

**Out of scope (later ADRs):**

- ND/AMR `EnzoBackend` — `:julia` slots on **subgrids** and in 2D/3D need the
  seam over multi-grid Enzo memory (today: 1D single-grid). Until then `:julia`
  slots are certified on single-grid problems and fall back to `:enzo` under AMR.
- Swapping the **structural/AMR** steps (boundary set, flux registers,
  `update_from_finer`, regrid) — these are the conservation machinery, not
  physics; they stay `:enzo`.
- `cooling`/chemistry and `mhd_ct` Julia ports (large, separate efforts).
- A user-facing config surface (parameter-file → `EngineConfig`); start with a
  Julia-constructed config.

## Consequences / risks

- **Data-contract coupling** is the main risk: a slot that doesn't reproduce the
  exact side-effects (auxiliary arrays, energy toggles like
  `MHDCT_ConvertEnergyToConservedS`, ghost-zone state) breaks the next step
  silently. Mitigation: per-step certification catches it immediately, and slots
  with shared contracts are swapped as a unit.
- **AMR gating:** `:julia` physics under AMR is blocked on the ND/AMR backend, so
  the registry must cleanly fall back to `:enzo` when `level > 0` (or refuse the
  config) until that lands.
- **Refactor risk in Phase A** is low but real — it touches the hot orchestration
  path; the all-`:enzo` default + the bit-for-bit quicksuite gate de-risk it.

## Status of prerequisites

Done: Julia `EvolveLevel`, the bridge step surface, `EnzoBackend` (1D
single-grid), `EquationSet`, the certification harness, hydro `:julia` slot.
Needed for the registry itself: only Phase A's refactor. Needed to make it
*valuable*: at least one new `:julia` port (Phase C, gravity).
