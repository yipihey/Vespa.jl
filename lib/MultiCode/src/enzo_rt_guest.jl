# ── flagship 2: RAMSES-RT inside an Enzo simulation (ADR-0006 Phase 5) ────────
#
# Enzo is the HOST: it owns the problem (PhotonTest's grid, fields, clock) and
# its EvolveLevel-style cycle loop runs as always — except the radiation +
# photo-chemistry slot is a PERSISTENT RAMSES-RT guest.  Once per Enzo cycle:
#
#   Enzo dt ──▶ RAMSES-RT (set_dt! → rt_setup! → rt_step!: M1 transport +
#               non-eq chemistry, subcycled at the reduced-c CFL)
#   xHII    ◀── rastered back and written into Enzo's LIVE species fields
#               (HII = x·ρ, HI = (1−x)·ρ, e⁻ = HII), the slot data contract
#
# The density field flows host → guest ONCE (PhotonTest runs HydroMethod=-1,
# so ρ is static — for a hydro host this sync moves into the loop).  Both
# codes run in Myr time units, so the host dt passes through unscaled.

"""
    run_enzo_host_ramsesrt(; density=nothing, t_end_myr=5.0, snapshots=[3.0,5.0],
                           dt_max_myr=0.25, c_fraction=0.005)
        -> (; history, xHII, fields, t, diag)

Run the PhotonTest problem with RAMSES-RT as Enzo's radiation slot.  `density`
optionally injects a structured field (n³, Enzo code units) into the HOST
first; the guest is initialized from the host's live density either way.
Returns the I-front history measured FROM THE ENZO-HELD FIELDS (the proof the
write-back contract works), the final Enzo-side xHII grid, and diagnostics.
The guest level matches Enzo's 32³ grid (level 5), so the field exchange is a
1:1 raster.
"""
function run_enzo_host_ramsesrt(; density = nothing,
                                t_end_myr::Real = 5.0, snapshots = [3.0, 5.0],
                                dt_max_myr::Real = 0.25, c_fraction::Real = 0.005,
                                paramfile::AbstractString = ENZO_PHOTONTEST_PF,
                                lib::Symbol = :rt)
    EnzoLib.grid_available() || error("Enzo grid bridge not built")
    CodeBridge.available(RamsesLib.BRIDGE, lib) ||
        error("RAMSES RT library not found (build bin64hrt or set RAMSES_LIB_RT)")

    # ── the HOST: PhotonTest staged to t_end, optional structured density ────
    dir = mktempdir()
    par = read(paramfile, String)
    par = replace(par, r"StopTime\s*=\s*\S+" => "StopTime                = $(float(t_end_myr))")
    pf = joinpath(dir, "PhotonTest.enzo")
    write(pf, par)
    # the GUEST's namelist (level 5 = 32³, matching the host grid)
    level = 5
    write(joinpath(dir, "iliev1.nml"),
          ramsesrt_iliev_namelist(; level = level, c_fraction = c_fraction))
    snaps = sort(unique(vcat(Float64.(snapshots), Float64(t_end_myr))))

    return cd(dir) do
        hE = EnzoLib.session_init(pf)
        hE == C_NULL && error("session_init failed for PhotonTest")
        try
            g = _enzo_active(hE)
            n = length(g.sl[1])
            n == 2^level || error("host grid $(n)³ ≠ guest level $level")
            if density !== nothing
                size(density) == (n, n, n) ||
                    error("density must be $(n)³, got $(size(density))")
                ratio = density ./ _enzo_field_active(hE, FT_DENSITY)
                for ft in (FT_DENSITY, FT_HI, FT_HII, 7)
                    fi = EnzoLib.field_index(hE, ft)
                    full = reshape(EnzoLib.problem_get_field(hE, fi, 0), g.dims...)
                    full[g.sl...] .*= ratio
                    EnzoLib.problem_set_field(hE, fi, vec(full))
                end
            end
            ρE = _enzo_field_active(hE, FT_DENSITY)        # the host's density, code units

            # ── the GUEST: initialized FROM the host's live field ────────────
            hR = RamsesLib.init("iliev1.nml"; lib = lib)
            RamsesLib.nrtvar(; lib = lib) >= 4 || error("guest library has no RT state")
            lev = RamsesLib.info(hR; lib = lib).levelmin
            ramsesrt_set_density!(hR, lev, ρE; nH = 1e-3, lib = lib)   # 1.0 code ≡ 1e-3 H/cc
            RamsesLib.rt_neq_updates!(hR, 0; lib = lib)

            # write the guest's chemical state into the HOST's live fields
            fiHI = EnzoLib.field_index(hE, FT_HI)
            fiHII = EnzoLib.field_index(hE, FT_HII)
            fiDe = EnzoLib.field_index(hE, 7)              # ElectronDensity
            fiD = EnzoLib.field_index(hE, FT_DENSITY)
            function sync_back!()
                x = ramsesrt_xhii_grid(hR, lev; lib = lib)
                ρfull = reshape(EnzoLib.problem_get_field(hE, fiD, 0), g.dims...)
                for (fi, val) in ((fiHII, x), (fiHI, 1.0 .- x), (fiDe, x))
                    full = reshape(EnzoLib.problem_get_field(hE, fi, 0), g.dims...)
                    full[g.sl...] .= val .* ρfull[g.sl...]
                    EnzoLib.problem_set_field(hE, fi, vec(full))
                end
                return x
            end

            history = Tuple{Float64,Float64}[]
            x = nothing
            ncyc = 0
            for target in snaps
                while EnzoLib.session_time(hE) < target * (1 - 1e-12) && ncyc < 100_000
                    EnzoLib.session_set_boundary(hE, 0)
                    dt = min(EnzoLib.session_compute_dt(hE, 0), Float64(dt_max_myr),
                             target - EnzoLib.session_time(hE))
                    EnzoLib.session_set_dt(hE, dt, 0)
                    # the radiation+chemistry SLOT, provided by the guest:
                    RamsesLib.set_dt!(hR, lev, dt; lib = lib)
                    RamsesLib.rt_setup!(hR, lev; lib = lib)
                    RamsesLib.rt_step!(hR, lev; lib = lib)
                    x = sync_back!()                       # the slot data contract
                    EnzoLib.session_advance_time(hE, 0)
                    ncyc += 1
                end
                # measure the front FROM THE HOST's fields (Moray's own observable)
                push!(history, (EnzoLib.session_time(hE), moray_ifront_radius(hE).r_I))
            end

            fields = (xHII = _enzo_field_active(hE, FT_HII) ./
                             (_enzo_field_active(hE, FT_HI) .+ _enzo_field_active(hE, FT_HII)),
                      density = _enzo_field_active(hE, FT_DENSITY))
            return (history = history, xHII = x, fields = fields,
                    t = EnzoLib.session_time(hE),
                    diag = (cycles = ncyc, level = lev, c_fraction = c_fraction),
                    enzo = hE, ramses = hR,
                    free = () -> (RamsesLib.finalize(hR; lib = lib); EnzoLib.free_problem(hE)))
        catch
            EnzoLib.free_problem(hE)
            rethrow()
        end
    end
end
