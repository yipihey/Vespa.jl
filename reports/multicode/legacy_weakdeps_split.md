# MultiCode legacy wrapper weakdep split

D1 is an extension split, not a single import deletion.  `MultiCode` currently
loads EnzoLib, RamsesLib, and ArepoLib at core module load time because their
handle types are used in method signatures across the main source tree.

## Core That Can Stay Hard-Free

- `canonical.jl`: `CellSet`, `ledger`, `ledger_drift`
- `exact_sod.jl`: exact Riemann oracle
- `exchange.jl`: geometry-agnostic deposition/sampling helpers, except comments
  and callers that pass Arepo Voronoi geometry
- `report.jl`: report rendering
- Problem specs and pure IC/profile helpers from `sod.jl`, `sedov_compare.jl`,
  and `zeldovich.jl`

## Enzo Extension Surface

Move to `MultiCodeEnzoExt`:

- `enzo_extract`, `enzo_inject!`
- `run_enzo_sod`
- `run_enzo_sedov`
- `run_enzo_zeldovich`
- Moray/Stromgren host routines in `moray.jl`

These functions depend on `EnzoLib.Handle`, field indexing, session lifecycle,
and grid bridge availability checks.

## RAMSES Extension Surface

Move to `MultiCodeRamsesExt`:

- `ramses_extract`, `ramses_inject!`
- `run_ramses_sod`, `run_ramses_sedov`, `run_ramses_zeldovich`
- `ramses_ppmk_hydro_step!`, AMR raster/reflux helpers, and fast path
- `ramses_ka_poisson!`, `ramses_ka_poisson_fine!`, gravity comparison runners
- RAMSES-RT helpers in `ramsesrt.jl`

This is the largest split because RAMSES owns both hydro guest slots and the
gravity guest-slot validation.

## Arepo Extension Surface

Move to `MultiCodeArepoExt`:

- `arepo_extract`
- `arepo_roundtrip_conserved`
- `run_arepo_sod`
- Arepo-side Moray exchange tests/helpers that call `ArepoLib.get_voronoi_3d`

## Multi-Wrapper Extensions

The current MUSIC and DISCO-DJ extensions import Enzo/RAMSES through
`MultiCode` because they boot generated ICs into live solvers.  After the split,
they should become multi-dependency extensions:

- `MultiCodeMusicEnzoRamsesExt = ["MusicLib", "EnzoLib", "RamsesLib"]`
- `MultiCodeDiscoDJEnzoRamsesExt = ["DiscoDJLib", "EnzoLib", "RamsesLib"]`
- `MultiCodeMorayArepoExt = ["EnzoLib", "ArepoLib"]`

The extension bodies should import `EnzoLib`, `RamsesLib`, and `ArepoLib`
directly rather than through `MultiCode`.

## Safe Migration Order

1. Split pure specs/helpers out of `sod.jl`, `sedov_compare.jl`, and
   `zeldovich.jl` into core files.
2. Add empty stubs in `MultiCode.jl` for all exported legacy functions.
3. Move Enzo-only methods into `MultiCodeEnzoExt` and run Enzo-focused tests.
4. Move RAMSES-only methods into `MultiCodeRamsesExt` and run RAMSES-focused
   hydro/gravity/RT tests.
5. Move Arepo-only methods into `MultiCodeArepoExt`.
6. Convert MUSIC/DISCO-DJ/Moray cross-code extensions to explicit multi-dep
   extensions.
7. Only then move `EnzoLib`, `RamsesLib`, and `ArepoLib` from `[deps]` to
   `[weakdeps]`/`[extras]`.

The key risk is method-signature invalidation at load time: any remaining
`EnzoLib.Handle`, `RamsesLib.Handle`, or `ArepoLib.Handle` annotation in core
will keep the package from loading without that wrapper.  The split should be
verified by creating a temporary environment that depends on `MultiCode` alone
and imports it without the three wrapper projects available on the load path.
