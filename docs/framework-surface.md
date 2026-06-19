# The Vespa multi-code framework — full code surface

The one-page registry of everything the federated framework (ADR-0006) now
wraps, certifies, and runs. Updated 2026-06-10 after the Next-list close-out
and the Music/Athena/Gadget4/DiscoDJ wrapper integration audit.

## The substrate (in this repository)

| Package | What it is |
|---|---|
| `lib/CodeBridge` | The shared legacy-wrapper substrate: `LazyLib` multi-flavor loading, `Bridge`, the `@xcall` macro (in-process ccall OR subprocess worker from ONE call site), source-parsed manifest + FNV-1a contract hash, worker RPC. Every wrapper below is a client. |
| `lib/MultiCode` | The cross-code layer: `CellSet` canonical state + per-code adapters, conservation ledgers, the comparison harness (one spec → N engines → one report), the guest slots, and the conservative exchange operators (incl. exact R3D Voronoi↔AMR). Package extensions: `MultiCodeDfmmExt`, `MultiCodeAthenaExt`, `MultiCodeMusicExt`, `MultiCodeDiscoDJExt`, `MultiCodeGadget4Ext`, `MultiCodeCICASSExt` — every registry wrapper has a live cross-code gate. |
| `lib/PPMKernels` | KernelAbstractions hydro: Enzo PPM (certified bit-tight vs live Fortran), MUSCL-Hancock (PLM/PPM × HLL/HLLC/two-shock, `fluxrec` flux recording for AMR registers), HD_RK MUSCL, PPML. CPU f64/f32 + Metal f32. |
| `lib/PoissonKernels` | KernelAbstractions gravity: Enzo multigrid (bit-tight), periodic FFT root solve (`greens = :spectral` Enzo / `:discrete7` — the exact solution of the 7-point system RAMSES MG iterates on), Dirichlet V/W-cycle, and `masked_cg!` (irregular-domain Dirichlet solve, CPU f64 + Metal f32). |
| `lib/EnzoLib` | Live Enzo through the EnzoModules C-ABI bridge: certified EvolveLevel slots, `:julia` slot swaps, particle injection, flux registers, MPI worker. |
| `Vespa.jl` core | The native ghost-free FV driver (RefMesh + HGBackend), `EquationSet` model seam, cosmology units, reflux. |

## The wrapper registry (sibling repositories, all `[sources]` → this CodeBridge)

| Repo / package | Code wrapped | Transport | Capabilities certified |
|---|---|---|---|
| `RamsesNG.jl` / RamsesLib | mini-ramses | in-process, flavors `:cpu` `:metal` `:rt` `:cosmo` | hydro/gravity/RT step drivers, field+particle access, `interpol_phi` pure kernel; host for the hydro+gravity guest slots |
| `Arepo.jl` / ArepoLib | Arepo (moving mesh) | in-process (worker for re-init; per-process singleton) | Sod, Voronoi 3-D export (`libarepo3d`), Moray-inside-Arepo |
| `Music.jl` / MusicLib | MUSIC (music20) | in-process (`libmusic_capi`) | one `MusicSpec` → Enzo/RAMSES/Arepo zoom ICs; injector validated live via `MultiCodeMusicExt` (Enzo↔RAMSES initial fields corr 1−4e-15); MUSIC-format white-noise writer/reader + phase mirror + `MusicSpec.noise_files`, live-gated by feeding a generated 32^3 noise file through MUSIC's Enzo output |
| `Athena.jl` / AthenaLib | Athena++ | in-process, re-entrant for one configured flavor; flavor-per-dylib (pgen/coord/flux/eos, GR spacetimes), pgen-changing tests isolated in fresh workers | Sod-harness engine via `MultiCodeAthenaExt` (L1(ρ)=0.0019, exact conservation, 0.02 s); 3-D → `CellSet` via legacy-VTK readback (ledger at the f32 floor, scatter ≡ 0); `:stage` uniform conserved-state bridge; GR gates for Schwarzschild/Kerr Bondi, Fishbone-Moncrief torus, and relativistic shock tube |
| `Gadget4.jl` / Gadget4Lib | GADGET-4 | child process (G4 owns exit()/MPI) | NGenIC 2LPT ICs, TreePM runs, FOF/SUBFIND halo-finder-as-a-service — live in the harness via `MultiCodeGadget4Ext` (planted clumps → exactly 3 groups; RAMSES dump → 0) |
| `DiscoDJ.jl` / DiscoDJLib | DISCO-DJ (JAX) | in-process PythonCall (NOT CodeBridge) | differentiable LPT ICs + lightcones, gradients preserved; growth-gated live via `MultiCodeDiscoDJExt` (1LPT → Enzo+RAMSES, cross-corr 0.998, large scales ≥0.96·b(a)) |
| `CICASS.jl` / CICASSLib | CICASS (McQuinn & O'Leary) | in-process (`libcicass_capi`), HDF5-free `.cicass` raw dump | one `CICASSSpec` → baryon–DM **streaming-velocity** ICs (2D (k⊥,k∥) TF; the offset MUSIC/DISCO-DJ can't express); cross-code gate via `MultiCodeCICASSExt` — live Enzo (BaryonField inject) ≡ live RAMSES (grafic + coherent `set_hydro!`), gas–DM offset 3.027 km/s both, v_bc=0→0. Valid z≲1000 (post-recombination) |
| `dfmm` (sibling) | dual-frame moment method | native Julia, `MultiCodeDfmmExt` | the Sod harness engine: L1(ρ)=0.042, mass bit-exact, momentum 1e-16 |

Build hints live in each wrapper's `LazyLib` declaration (the error message IS
the build command). The Julia-side integration contract is identical across
CodeBridge clients: declare a `Bridge`, write `@xcall` sites, get the manifest,
contract hash, worker, and parity oracle for free.

## Certified capabilities (the gates, all green)

- **Cross-code comparison**: one Sod spec through Enzo/RAMSES/Arepo (+ dfmm and Athena++ via extensions);
  one Sedov IC through six engines; one Zel'dovich particle set through Enzo +
  RAMSES vs the exact mixed-mode growth (0.989/0.994).
- **Guest slots (mix-and-match)**: PPMKernels hydro inside RAMSES — uniform,
  composite-AMR (bit-identical to uniform-fine), per-level fast path with flux
  registers (Δm = 0.0 bit-exact, 2.34×), device-resident Metal (1.68 s vs
  native 7.2 s); RAMSES-RT inside Enzo; Moray inside Arepo; KA Poisson inside
  RAMSES (root 9.4e-15 anchor, cuboid Dirichlet 2.4e-12, irregular blob
  7.3e-13 CPU / f32-floor Metal).
- **Exchange**: conservative deposit/sample, exact R3D Voronoi↔AMR remap
  (clipped signed-fan tets ≡ `SphP.Volume`), Moray vs RAMSES-RT cross-check on
  one density field.
- **Multi-worker sessions**: N legacy codes alive at once, each in its own
  process, one Julia driver.

Reports land in `reports/multicode/`. Run the cross-code suite with
`<julia> --project=lib/MultiCode/test lib/MultiCode/test/runtests.jl`
(needs the Enzo grid dylib, mini-ramses `bin64h`+`bin64hrt`+`bin64sc`, the
sibling arepo + dfmm checkouts; gates skip cleanly where a library is absent).

## Status ledger

ADR-0006 phases 0–7, Next-1…14, and Phase C complete; the per-phase implementation record
(numbers, traps, commit references) is the status appendix of
[`docs/adr/0006-unified-multicode-framework.md`](adr/0006-unified-multicode-framework.md).
The wrapper-registry on-ramp is COMPLETE: all eight wrappers carry live
cross-code gates, and MUSIC generates through its own CodeBridge worker
(bit-identical to in-process — the D1/D2 transport seam certified once more).
The eighth, CICASS, adds the baryon–dark-matter streaming velocity (the offset
the other IC generators structurally cannot express), validated live into both
Enzo and RAMSES (see `reports/multicode/cicass_streaming.md`).

The already-scoped directions beyond this — the final Enzo/HG adapter for the
Athena stage bridge, the cross-generator fixed-and-paired IC report, the
deeper Santa Barbara campaign, and the deferred wrapper extension-ification — are the
**[planned next steps](roadmap.md)**, each grounded in completed work with a
concrete on-ramp.
