# ChemistryKernels — session handoff & plan

Last updated: 2026-06-16 (session 2). Branch `enzong-amr-subcycling-refluxing`.
Commits `ffda32b6`/`55abef3a` are **pushed**; **Task A (UVB-into-solver) work below is
committed locally? NO — it is UNCOMMITTED in the working tree** (staged only mentally:
all changes are confined to `lib/ChemistryKernels/`). Run the suite, then commit
`lib/ChemistryKernels` only (don't sweep the parallel WIP — see Notes).

## What this module is

A pure-Julia, table-free, KernelAbstractions implementation of the **Abel, Anninos,
Zhang & Norman (1997, NewA 2, 181)** + **Anninos, Zhang, Abel & Norman (1997, NewA
2, 209)** primordial+deuterium chemistry/cooling network — the original 1990s Enzo
chemistry (the same physics later libraried as *grackle*; we are re-grounding the
code on those primary papers, not grackle). Reduced model: advects HII, H2I, HDI;
H⁻/H₂⁺/D⁺ algebraic equilibrium; **helium in collisional-radiative ionisation
equilibrium** (or optionally advected He⁺); nₑ from charge conservation.

## Conventions you must know

- **Mass-equivalent ×N species convention**: `yHI=n_HI`, `yHII=n_HII`, `yde=n_e`,
  but `yH2I=2·n(H₂)`, `yH2II=2·n(H₂⁺)`, `yHDI=3·n(HD)`, and **`yHeX=4·n(HeX)`** (He
  mass = 4 m_H). All the literal `/2`, `/3`, `/4` factors follow from this.
- **Recombination physics is RECFAST-v2 / HyRec-validated** (<0.1% vs HyRec across
  z=700–1100). Key facts a new session must not "fix":
  - RECFAST fudge multiplies α_B (it enters the Peebles C-factor as
    `C = fu·(1+KL)/(1+KL+fu·KB)`), NOT the Λ₂γ term. v2: `fu=1.125` +
    Gaussian-on-K (`recfast_gauss_factor`).
  - `network_step` **deliberately extends** the original network: it adds the k28
    H₂⁺→H+H⁺ photodissociation return to the HI/HII equations (the original drops
    it; the k9 radiative-association leaks ~1.5% of recombination at z~1100). Do
    not "restore grackle parity" here.
  - T_CMB = **2.725** K (Fixsen 2009), in `comp2_cmb` (was 2.73).
- **Helium ionisation** — `helium_equilibrium(she1,she2,k3,k4,k5,k6,ne,nHe;
  GamHeI,GamHeII)` in `equilibrium.jl`:
  ```
  n_HeII /n_HeI  = she1/ne + k3/k4 + Γ_HeI /(k4·ne)
  n_HeIII/n_HeII = she2/ne + k5/k6 + Γ_HeII/(k6·ne)
  ```
  Saha/CMB (`she1,she2 = helium_saha_pair(T_rad)`, detailed balance) + collisional
  (k3/k5 ion, k4/k6 recomb, T_matter) + optional external photoionisation Γ.
  `network_step(...; GamHeI, GamHeII)` consumes it (default Γ=0). He⁺⁺ is always
  Saha-fast; only He⁺ ever needs a rate equation (the z≈2000 freeze-out).
- **UVB rate convention** (`uv_background.jl`): `k24=Γ_HI`, `k25=Γ_HeII`,
  `k26=Γ_HeI` [s⁻¹]; `piHI/piHeI/piHeII` photoheating [erg/s]. TREECOOL column
  order is (log10(1+z), ΓHI, ΓHeI, ΓHeII, q̇HI, q̇HeI, q̇HeII) — note HeI before
  HeII; `read_treecool` does the mapping. `fg20_uvb()` loads the shipped FG20 data.
- **Two unit gotchas already fixed** (don't reintroduce): Verner-Ferland α_He is in
  m³/s → ×1e6 for cm³/s; the He Saha quantum concentration is `NQ·T^1.5` (NQ=
  2.415e15), NOT `(NQ·T)^1.5`.

## What is DONE (committed, validated)

1. **Recombination accuracy** (earlier commits): H <0.1% vs HyRec/CAMB-RECFAST-v2
   across z=700–1100; Saha He; opt-in advected He⁺ freeze-out (`helium_HeI_rate_AB`,
   a transcription of HyRec `helium.c`); `total_electron_fraction` output helpers.
2. **De-grackle** (`ffda32b6`): all 19 src files re-attributed to the 1997 papers
   (comments only). One lineage mention left in `ChemistryKernels.jl`.
3. **He collisional-radiative equilibrium** (`ffda32b6`): `helium_equilibrium`,
   wired into `network_step` as the default He path. Tested (Saha/CIE/photo limits).
4. **FG20 UVB machinery** (`ffda32b6`): `src/uv_background.jl` + shipped FG20
   TREECOOL data under `data/`. `fg20_uvb()`; validated against published FG20
   z=0 rates; feeds the He Γ hooks → He photoionisation equilibrium at low z.

5. **UVB wired into the solver (Task A, DONE this session, UNCOMMITTED)**: a UV
   background now drives H photoionisation, He photoionisation equilibrium, and
   photoheating end-to-end. Validated: a one-zone cell under `fg20_uvb()` at z=3 mean
   IGM density relaxes to **T≈1.5×10⁴ K and x_HI≈3×10⁻⁶** (textbook IGM), with the
   photoionisation balance Γ_HI·n_HI ≈ k2·n_HII·n_e holding to ~3% and no x_HII>1
   overshoot. Changes (all in `lib/ChemistryKernels`):
   - `solve_chem_mixing!(...; uvb::Union{Nothing,UVBackground}=nothing)`. When given,
     `(k24,k25,k26,piHI,piHeI,piHeII)=uvb_rates(uvb,z)` once per step → threaded through
     `_evolve_mixing_k!` → `evolve_cell_mixing` (new kwargs `uvb,GamHI,GamHeI,GamHeII,
     piHI,piHeI,piHeII`) → `network_step` (new `GamHI` kwarg: Γ_HI in the HI `ac` /
     HII `sc`).
   - `evolve_cell_mixing`: when `uvb`, solves the He photoionisation equilibrium ONCE up
     front (so cooling, photoheating and electrons share one He state) and hands it to
     `network_step`; adds photoheating `piHI·nHI+piHeI·nHeI+piHeII·nHeII` to `edot`;
     enforces H-nuclei conservation (`make_consistent`-style renormalisation) — all gated
     on `uvb`, so the no-UVB path is **bit-identical** (the recombination suite is
     unchanged).
   - `_de_hi_dot` gained a `GamHI` kwarg (the sub-step limiter sees photoionisation).
   - **β₁s bug fix (latent, important):** `k_beta1s` (CMB photoionisation of H 1s) was
     evaluated at the MATTER temperature in `build_rates`/`build_rates_mixing`; it is a
     *radiative* rate and must use **Trad**. At recombination T≈Trad so nothing changes
     (CAMB-RECFAST-v2 accuracy preserved, still <0.1% at z=700–1100), but under a low-z
     UVB the gas heats to T≫Trad and `beta1s_freq(T)` spuriously drove H to Saha
     equilibrium at the hot matter T (x_HI→10⁻¹⁶, x_HII→thousands). Now `beta1s_freq(Trad)`.
     (Only side effect: the benign z≈1665 Saha-tracking bump in the monotonicity test
     grew 2e-4→2.7e-4; tolerance loosened to 3e-4.)

**Test status: 103/103** in `test/test_recombination_mixing.jl` (the standalone,
grackle-free suite; was 96, +7 in the new `uvb_solver_equilibrium` testset). Run it with:
```
julia --project=lib/ChemistryKernels/test lib/ChemistryKernels/test/test_recombination_mixing.jl
```
(The full `runtests.jl` still depends on a macOS-only C-grackle oracle `.dylib`
and cannot run on Linux — see task C below.)

## PLAN — remaining work, in priority order

### A. Wire the UVB into the solver — ✅ DONE (this session, uncommitted). See item 5
above. Residual follow-ups (optional, lower priority):
- The non-mixing `solve_chem!`/`subcycle.jl::evolve_cell` path was left UVB-free
  (production UVB goes through `solve_chem_mixing!`). `network_step` already accepts
  `GamHI`, so wiring `evolve_cell` is a small mirror if a UVB is ever needed there.
- Performance: with a UVB the sub-cycle still uses the net-rate 10% limiter; very
  optically-thin cells reach photoionisation equilibrium in a few sub-steps, but a
  coupled implicit HI⇌HII solve (vs the current operator-split + conservation
  renormalisation) would be cheaper and exactly conservative if profiling flags it.
- Validate He photoheating in the z≲3 HeII-reionisation regime (piHeII·n_HeII), and
  consider self-shielding for dense gas (currently optically thin only).

### B. Generalise the advected He path / non-equilibrium detection
Currently `helium_HeI_rate_AB` (advected He⁺) is the radiative-transfer He I
recombination only. To match the equilibrium generality:
1. Add collisional ionisation (k3·ne) and external photoionisation (Γ_HeI) to the
   advected He⁺ up-rate; add He⁺⁺ as an (optionally advected) species with its own
   rate (k5 ion, k6 recomb, Γ_HeII), else Saha.
2. **Non-equilibrium detector**: compare the equilibration time `t_eq ≈ 1/(loss
   freq)` to the step / Hubble time; where `t_eq ≪ dt` pin to `helium_equilibrium`
   (cheap, exact), else integrate. This is the "checking for non-equilibrium
   effects" the user asked for; it gates when the ODE is actually needed.

### C. Replace the C-grackle test oracle (finishes the de-grackle)
The test suite (`test/oracle.jl`, `harness.jl`, and the `test_rates_*`,
`test_cooling_*`, `test_temperature`, `test_edot`, `test_network_step`,
`test_smoke` files) parity-checks every rate/cooling value against a macOS-only
C-grackle `.dylib` (`oracle/libchem_oracle.dylib`) — ~100 grackle/oracle refs, and
it cannot run on Linux. Replace with **literature-pinned reference values** (the
published Abel/Anninos 1997, Hui-Gnedin, Galli-Palla fits evaluated at fixed T),
analytic/asymptotic checks, and the existing HyRec validation. This removes the
last grackle dependency and makes the whole suite run anywhere. Parallelises well
across the per-test files (subagents) once a reference-value scheme is chosen.

## Notes for the new session
- **Parallel subagents** work well for the mechanical file sweeps (they did the
  de-grackle), but the API was intermittently throwing `529 Overloaded` — retry.
- Don't commit the unrelated dirty working-tree files (EnzoLib, MultiCode, PowerFoam
  Project.tomls, `run_*.sh`) — they are parallel WIP. Stage only `lib/ChemistryKernels`.
- HyRec-2 is cloned/built at `~/Projects/HYREC-2` (`./hyrec_built < input_match.dat`,
  matched cosmology) for recombination cross-checks.
- The running cross-session state is also in the memory file
  `project_enzong_chemkernels.md`.

## Key references
- Abel, Anninos, Zhang & Norman 1997, NewA 2, 181 (the network).
- Anninos, Zhang, Abel & Norman 1997, NewA 2, 209 (the solver/cooling).
- Hui & Gnedin 1997 (recombination); Galli & Palla 1998 (H₂ cooling).
- HyRec-2: Lee & Ali-Haïmoud 2020; Ali-Haïmoud & Hirata 2011 (recombination ref).
- RECFAST: Seager et al. 1999/2000; Wong et al. 2008 (v2 Gaussians).
- Faucher-Giguère 2020, MNRAS 493, 1614 (arXiv:1903.08657) — the FG20 UVB;
  rescaled heating per Gaikwad et al. 2020 (arXiv:2009.00016).
