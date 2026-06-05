# PPMKernels MUSCL solver as the :julia HYDRO SLOT under Enzo's AMR hierarchy.
#
# Enzo owns the AMR machinery (flag → refine → evolve recursion, ghost interpolation,
# projection); the per-grid HYDRO is our PPMKernels MUSCL solver instead of Enzo's
# HD_RK. The :julia slot of evolve_level! calls our hook (h, level, dt) once per
# timestep; the hook loops the grids on that level, reads Enzo's BaryonField, runs one
# unsplit-RK2 MUSCL step (dual-energy), and writes the result back into the live grid.
#
# v1 is the NON-conservative slot: each grid steps independently on Enzo-provided
# ghosts (Enzo fills them from the parent / domain BC before the hook). Coarse↔fine
# flux correction (ADR-0003 part B: record PPMKernels boundary fluxes into Enzo's
# registers) is the conservative follow-on.
#
# Run (from lib/PPMKernels, which has BOTH PPMKernels + EnzoLib):
#   <juliaup-julia> --project=test ../EnzoLib/examples/turbulence_amr/run_ppm_under_amr.jl [mach] [overd] [cycles]

using EnzoLib, PPMKernels, Random, LinearAlgebra, Printf
const NG = 3; const GAMMA = 1.4
const PF = joinpath(@__DIR__, "decaying_turbulence_amr.enzo")
# Enzo BaryonField positional indices (field order [Density,Vel1,Vel2,Vel3,TotalE,GasE])
const iD, iV1, iV2, iV3, iTE, iGE = 0, 1, 2, 3, 4, 5

include(joinpath(@__DIR__, "ic_inject.jl"))            # inject_turbulence!  (shared)

# ── the :julia hydro hook: one PPMKernels MUSCL step per grid on `level` ──────
function ppm_hydro!(h, level, dt)
    ng = EnzoLib.session_num_grids_on_level(h, level)
    for gi in 0:ng-1
        g = EnzoLib.problem_grid_index_on_level(h, level, gi)
        dims = Tuple(Int.(EnzoLib.problem_grid_dims(h, g)))          # incl ghosts
        le, re = EnzoLib.problem_grid_edge(h, g)
        dx = (re[1] - le[1]) / (dims[1] - 2NG)                       # cell width on this grid
        # Enzo (primitive-ish) → PPMKernels conserved (CPU f64; Enzo layout = PPM layout)
        D  = EnzoLib.problem_get_field(h, iD,  g)
        v1 = EnzoLib.problem_get_field(h, iV1, g); v2 = EnzoLib.problem_get_field(h, iV2, g)
        v3 = EnzoLib.problem_get_field(h, iV3, g)
        TE = EnzoLib.problem_get_field(h, iTE, g); GE = EnzoLib.problem_get_field(h, iGE, g)
        S1 = D .* v1; S2 = D .* v2; S3 = D .* v3; Tau = D .* TE; Ge = D .* GE
        # one unsplit-RK2 MUSCL step with the dual-energy formalism (matches HD_RK class)
        PPMKernels.muscl_step_3d!(D, S1, S2, S3, Tau, dims, NG;
                                  dt = dt, gamma = GAMMA, dx = dx, ge = Ge)
        # conserved → primitive, write back into the live Enzo grid (grid is a KEYWORD!)
        EnzoLib.problem_set_field(h, iD,  D;        grid = g)
        EnzoLib.problem_set_field(h, iV1, S1 ./ D;  grid = g); EnzoLib.problem_set_field(h, iV2, S2 ./ D; grid = g)
        EnzoLib.problem_set_field(h, iV3, S3 ./ D;  grid = g)
        EnzoLib.problem_set_field(h, iTE, Tau ./ D; grid = g); EnzoLib.problem_set_field(h, iGE, Ge ./ D; grid = g)

        # reflux registers: Enzo's FinalizeFluxes / UpdateFromFinerGrids dereference the
        # SubgridFluxesEstimate planes whenever fine grids exist. SolveHydroEquations
        # would fill them; our :julia slot must too. v1 writes ZEROS (planes allocated,
        # zero flux correction ⇒ projection-only). [Conservative follow-on: write PPM's
        # actual coarse-face + outer-boundary fluxes here.]  `gi` is the on-level index.
        nf = EnzoLib.problem_num_fields(h, g)
        nsub = EnzoLib.problem_num_subgrids(h, level, gi)
        for sub in 0:nsub-1, dim in 0:2
            sz = EnzoLib.problem_subgrid_flux_size(h, level, gi, sub, dim)
            sz <= 0 && continue
            z = zeros(Float64, sz)
            for fld in 0:nf-1, side in 0:1
                EnzoLib.problem_set_subgrid_flux(h, level, gi, sub, fld, dim, side, z)
            end
        end
    end
    return nothing
end

function main()
    mach   = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 5.0
    overd  = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 2.0
    maxcyc = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 60
    EnzoLib.grid_available() || error("grid dylib not built")
    work = mktempdir(); pf = joinpath(work, "p.enzo")
    write(pf, replace(read(PF, String), r"MinimumOverDensityForRefinement = .*" =>
                                        "MinimumOverDensityForRefinement = $overd"))
    cd(work) do
        h = EnzoLib.session_init(pf); h == C_NULL && error("session_init failed")
        try
            inject_turbulence!(h; mach = mach, gamma = GAMMA, ng = NG)
            @printf("\nPPMKernels MUSCL as :julia hydro UNDER Enzo AMR — Mach0=%.1f  ρ>%.1f flags\n", mach, overd)
            # :julia hydro = PPMKernels, with reflux=true so Enzo CREATES the flux
            # registers (update_from_finer needs them); we leave them zero for now ⇒
            # projection-only AMR (runs under the hierarchy, coarse↔fine NOT yet
            # flux-corrected — the conservative follow-on fills these from PPM's fluxes).
            eng = EnzoLib.EngineConfig(; hydro = :julia, reflux = true,
                                       hooks = Dict{Symbol,Function}(:hydro => ppm_hydro!))
            EnzoLib.session_rebuild(h, 0)
            ngl() = Int[EnzoLib.session_num_grids_on_level(h, l) for l in 0:2]
            m0 = EnzoLib.session_global_field_integral(h, 0)
            @printf("%-5s %-9s %-16s %-7s %-10s\n", "cyc", "t", "grids/level", "ρmax", "Δmass/M")
            cyc = 0
            while EnzoLib.session_time(h) < EnzoLib.session_stop_time(h) && cyc < maxcyc
                EnzoLib.evolve_level!(h, 0, 0.0; engine = eng, regrid = true)
                EnzoLib.session_rebuild(h, 0)
                ρ = EnzoLib.problem_get_field(h, iD, 0)
                m = EnzoLib.session_global_field_integral(h, 0)
                @printf("%-5d %-9.4f %-16s %-7.2f %-10.1e\n",
                        cyc, EnzoLib.session_time(h), ngl(), maximum(ρ), abs(m - m0) / m0)
                (any(isnan, ρ)) && (println("  NaN — abort"); break)
                cyc += 1
            end
        finally
            EnzoLib.free_problem(h)
        end
    end
end

main()
