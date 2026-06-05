# PPMKernels MUSCL solver as the :julia HYDRO SLOT under Enzo's AMR hierarchy.
#
# Enzo owns the AMR machinery (flag → refine → evolve recursion, ghost interpolation,
# projection); the per-grid HYDRO is our PPMKernels MUSCL solver instead of Enzo's
# HD_RK. The :julia slot of evolve_level! calls our hook (h, level, dt) once per
# timestep; the hook loops the grids on that level, reads Enzo's BaryonField, runs one
# unsplit-RK2 MUSCL step (dual-energy), and writes the result back into the live grid.
#
# v2 is the CONSERVATIVE slot (ADR-0003 part B for PPMKernels): each grid steps on
# Enzo-provided ghosts (Enzo fills them from the parent / domain BC before the hook),
# AND the MUSCL-Hancock solver records its per-grid interface fluxes (`fluxrec=`),
# which we write into Enzo's flux registers so UpdateFromFinerGrids /
# CorrectForRefinedFluxes restore coarse↔fine conservation — Enzo's own machinery,
# PPMKernels' numbers. Set CONSERVATIVE=false to recover the zero-fill baseline.
#
# Run (from lib/PPMKernels, which has BOTH PPMKernels + EnzoLib):
#   <juliaup-julia> --project=test ../EnzoLib/examples/turbulence_amr/run_ppm_under_amr.jl [mach] [overd] [cycles]

using EnzoLib, PPMKernels, Random, LinearAlgebra, Printf
const NG = 3; const GAMMA = 1.4
const CONSERVATIVE = get(ENV, "PPM_AMR_CONS", "1") != "0"   # write real fluxes (vs zeros)
const PF = joinpath(@__DIR__, "decaying_turbulence_amr.enzo")
# Enzo BaryonField positional indices (field order [Density,Vel1,Vel2,Vel3,TotalE,GasE])
const iD, iV1, iV2, iV3, iTE, iGE = 0, 1, 2, 3, 4, 5

include(joinpath(@__DIR__, "ic_inject.jl"))            # inject_turbulence!  (shared)

# Enzo BaryonField `fld` → the conserved-flux slot in frec[axis] (1=D,2=S1,3=S2,4=S3,
# 5=τ,6=Ge in GRID order); 0 ⇒ this field carries no hydro flux (write zeros).
@inline flux_slot(fld) = fld == iD ? 1 : fld == iV1 ? 2 : fld == iV2 ? 3 :
                         fld == iV3 ? 4 : fld == iTE ? 5 : fld == iGE ? 6 : 0

# Rasterize ONE (dim, side) flux-register plane from the recorded grid-frame fluxes.
# `frec[axis][slot][c]` = flux through the −axis face of (column-major, with-ghost)
# grid cell c; the plane fixes the flux-dim grid coordinate at `ng + m` and sweeps the
# two orthogonal dims over the global extent (st,en). The value is scaled to Enzo's
# register units (enzo_value = F·dt/dx, matching the certified `bflux/Vcell` path).
# Mirrors EnzoNG.bflux_plane's column-major (dim-0 fastest) linearization.
function frec_plane(frec, slot, dims, ng, dim, m, st, en, g0, dtdx)
    nx, ny, _ = dims
    Dim = ntuple(d -> en[d] - st[d] + 1, 3)
    out = Vector{Float64}(undef, prod(Dim))
    f = slot == 0 ? nothing : frec[dim + 1][slot]
    @inbounds for lin in 0:length(out)-1
        rem = lin
        gidx = ntuple(3) do d
            od = rem % Dim[d]; rem ÷= Dim[d]
            g = st[d] + od
            d == dim + 1 ? ng + m : ng + (g - g0[d] + 1)     # flux dim fixed; orthogonal mapped
        end
        out[lin+1] = f === nothing ? 0.0 :
            f[gidx[1] + nx * (gidx[2] - 1) + nx * ny * (gidx[3] - 1)] * dtdx
    end
    return out
end

# Write grid `g`'s recorded fluxes into its Enzo flux registers (the SubgridFluxes-
# Estimate the AMR machinery consumes): the coarse InitialFluxes at each subgrid's
# coarse–fine faces (sub 0..nsub-2) + the grid's own outer-boundary flux (sub nsub-1).
# `gi` is the on-level index; `dxd` is the per-dim cell width (cubic here, kept general).
function write_ppm_fluxes!(h, level, gi, g, dims, frec, dt, dxd)
    g0 = EnzoLib.problem_grid_global_start(h, g)
    nf = EnzoLib.problem_num_fields(h, g)
    nsub = EnzoLib.problem_num_subgrids(h, level, gi)
    active = ntuple(d -> dims[d] - 2NG, 3)
    setp(sub, dim, side, m, st, en) = for fld in 0:nf-1
        pl = frec_plane(frec, flux_slot(fld), dims, NG, dim, m, st, en, g0, dt / dxd[dim+1])
        EnzoLib.problem_set_subgrid_flux(h, level, gi, sub, fld, dim, side, pl)
    end
    # proper subgrids: the coarse interior flux at each coarse–fine interface face. The
    # flux-dim cell `m` is the −axis face of the boundary coarse cell (Left) or of the
    # cell just past it (Right) — m = (st−g0) + side + 1 (the verified EnzoNG mapping).
    for sub in 0:nsub-2, dim in 0:2, side in 0:1
        st, en = EnzoLib.problem_subgrid_flux_extent(h, level, gi, sub, dim, side)
        setp(sub, dim, side, (st[dim+1] - g0[dim+1]) + side + 1, st, en)
    end
    # own-boundary (last): the grid's outer-face flux. lo ⇒ the −face of active cell 1;
    # hi ⇒ the +face of the last active cell (= −face of cell active+1).
    own = nsub - 1
    for dim in 0:2, side in 0:1
        st, en = EnzoLib.problem_subgrid_flux_extent(h, level, gi, own, dim, side)
        setp(own, dim, side, side == 0 ? 1 : active[dim+1] + 1, st, en)
    end
    return nothing
end

# ── the :julia hydro hook: one PPMKernels MUSCL-Hancock step per grid on `level` ──
function ppm_hydro!(h, level, dt)
    ng = EnzoLib.session_num_grids_on_level(h, level)
    for gi in 0:ng-1
        g = EnzoLib.problem_grid_index_on_level(h, level, gi)
        dims = Tuple(Int.(EnzoLib.problem_grid_dims(h, g)))          # incl ghosts
        le, re = EnzoLib.problem_grid_edge(h, g)
        dxd = ntuple(d -> (re[d] - le[d]) / (dims[d] - 2NG), 3)      # per-dim cell width
        # Enzo (primitive-ish) → PPMKernels conserved (CPU f64; Enzo layout = PPM layout)
        D  = EnzoLib.problem_get_field(h, iD,  g)
        v1 = EnzoLib.problem_get_field(h, iV1, g); v2 = EnzoLib.problem_get_field(h, iV2, g)
        v3 = EnzoLib.problem_get_field(h, iV3, g)
        TE = EnzoLib.problem_get_field(h, iTE, g); GE = EnzoLib.problem_get_field(h, iGE, g)
        S1 = D .* v1; S2 = D .* v2; S3 = D .* v3; Tau = D .* TE; Ge = D .* GE
        # one dim-split MUSCL-Hancock step (dual energy), recording interface fluxes so
        # the coarse↔fine boundaries can be flux-corrected to conservation below.
        frec = CONSERVATIVE ? ntuple(_ -> ntuple(_ -> zeros(Float64, length(D)), 6), 3) : nothing
        PPMKernels.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, dims, NG;
                                          dt = dt, gamma = GAMMA, dx = dxd[1], ge = Ge, fluxrec = frec)
        # conserved → primitive, write back into the live Enzo grid (grid is a KEYWORD!)
        EnzoLib.problem_set_field(h, iD,  D;        grid = g)
        EnzoLib.problem_set_field(h, iV1, S1 ./ D;  grid = g); EnzoLib.problem_set_field(h, iV2, S2 ./ D; grid = g)
        EnzoLib.problem_set_field(h, iV3, S3 ./ D;  grid = g)
        EnzoLib.problem_set_field(h, iTE, Tau ./ D; grid = g); EnzoLib.problem_set_field(h, iGE, Ge ./ D; grid = g)

        # reflux registers: Enzo's FinalizeFluxes / UpdateFromFinerGrids dereference the
        # SubgridFluxesEstimate planes whenever fine grids exist — they MUST be filled
        # (SolveHydroEquations normally does). CONSERVATIVE ⇒ write PPMKernels' actual
        # coarse-face + outer-boundary fluxes (round-off coarse↔fine conservation);
        # else write ZEROS (allocated, no NULL deref, projection-only baseline).
        if CONSERVATIVE
            write_ppm_fluxes!(h, level, gi, g, dims, frec, dt, dxd)
        else
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
