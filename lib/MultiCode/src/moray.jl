# ── Moray (Enzo's adaptive ray tracing) as a service (ADR-0006 Phase 4) ───────
#
# Enzo's PhotonTest (ProblemType 50) IS the Iliev Test-1 Strömgren setup:
# hydrogen-only, isothermal (γ=1.0001), n_H = 1e-3 cm⁻³, T = 1e4 K, a 5e48
# photons/s source in the box corner, 6.6 kpc box, no hydro (HydroMethod=-1).
# The driver below runs it through the certified EvolveLevel machinery
# (radiation + cooling slots), optionally with the DENSITY FIELD INJECTED from
# another code's canonical state — which is exactly the "Moray inside Arepo"
# coupling (ADR-0006 flagship 3): the host supplies the gas, Moray supplies
# the radiation, the rates flow back.

const ENZO_PHOTONTEST_PF = normpath(joinpath(@__DIR__, "..", "..", "..", "..",
    "run", "RadiationTransport", "PhotonTest", "PhotonTest.enzo"))

# Enzo FieldType codes (src/enzo/typedefs.h)
const FT_DENSITY = 0
const FT_HI = 8
const FT_HII = 9
const FT_KPHHI = 23
const FT_PHOTOGAMMA = 24

# Iliev Test-1 physical constants (the PhotonTest parameter values)
const STROMGREN = (nH = 1e-3,                     # cm⁻³
                   Ndot = 5e48,                   # photons/s
                   alphaB = 2.59e-13,             # case-B at 1e4 K, cm³/s
                   LengthUnits = 2.03676e22,      # cm (6.6 kpc box)
                   TimeUnits = 3.1557e13)         # s (Myr)

"Strömgren radius [box units] and recombination time [Myr] of the PhotonTest setup."
function stromgren_scales(p = STROMGREN)
    Rs = (3 * p.Ndot / (4π * p.alphaB * p.nH^2))^(1 / 3)    # cm
    trec = 1 / (p.alphaB * p.nH)                            # s
    return (Rs_box = Rs / p.LengthUnits, trec_myr = trec / p.TimeUnits)
end

"Analytic I-front radius [box units] at time t [Myr]: r = R_s·(1−e^{−t/t_rec})^{1/3}."
stromgren_radius(t_myr; p = STROMGREN) =
    (s = stromgren_scales(p); s.Rs_box * (1 - exp(-t_myr / s.trec_myr))^(1 / 3))

# ── field geometry helpers (unigrid, ghost-aware — the enzo_extract logic) ────
function _enzo_active(h; grid = 0)
    dims = EnzoLib.problem_grid_dims(h, grid)
    l, r = EnzoLib.problem_grid_edge(h, grid)
    act(d) = dims[d] > 1 ? ((_ENZO_GHOST + 1):(dims[d] - _ENZO_GHOST)) : (1:1)
    return (dims = dims, l = l, r = r, sl = (act(1), act(2), act(3)))
end

function _enzo_field_active(h, ftype; grid = 0)
    g = _enzo_active(h; grid = grid)
    fi = EnzoLib.field_index(h, ftype; grid = grid)
    A = reshape(EnzoLib.problem_get_field(h, fi, grid), g.dims...)
    return Array(A[g.sl...])
end

"""
    moray_ifront_radius(h; grid=0) -> (; r_I, profile)

The I-front radius [box units] from the live hierarchy: spherically bin
x_HII = HII/(HI+HII) about the source corner and interpolate the x_HII = 0.5
crossing.
"""
function moray_ifront_radius(h; grid = 0)
    g = _enzo_active(h; grid = grid)
    hi = _enzo_field_active(h, FT_HI; grid = grid)
    hii = _enzo_field_active(h, FT_HII; grid = grid)
    x = hii ./ (hi .+ hii)
    n = size(x, 1)
    dx = (g.r[1] - g.l[1]) / n
    # radial bins about the corner source (PhotonTestSourcePosition ≈ origin)
    nb = 2n
    sums = zeros(nb); counts = zeros(Int, nb)
    for c in CartesianIndices(x)
        r = sqrt(sum(abs2, (Tuple(c) .- 0.5) .* dx))
        b = min(nb, 1 + floor(Int, r / (dx / 2)))
        sums[b] += x[c]; counts[b] += 1
    end
    rb = [(b - 0.5) * dx / 2 for b in 1:nb]
    xb = [counts[b] > 0 ? sums[b] / counts[b] : NaN for b in 1:nb]
    # first downward crossing of 0.5 (profile decreases outward)
    r_I = NaN
    for b in 2:nb
        (isnan(xb[b - 1]) || isnan(xb[b])) && continue
        if xb[b - 1] >= 0.5 > xb[b]
            f = (xb[b - 1] - 0.5) / (xb[b - 1] - xb[b])
            r_I = rb[b - 1] + f * (rb[b] - rb[b - 1])
            break
        end
    end
    return (r_I = r_I, r = rb, xHII = xb)
end

"""
    run_moray_stromgren(; t_end_myr=30.0, snapshots=[…], density=nothing,
                        paramfile=ENZO_PHOTONTEST_PF)
        -> (; history, fields, t)

Run Moray on the PhotonTest setup to `t_end_myr` through the certified
EvolveLevel (radiation + cooling slots; HydroMethod=-1 ⇒ the hydro slot
no-ops), recording the I-front radius at each `snapshots` epoch [Myr].

`density`: an optional n³ array (box units, n = TopGridDimensions) injected
into the live grid BEFORE evolution — the cross-code coupling point.  The
hydrogen species (HI, HII, e⁻) are scaled per cell by the density ratio, so
the injected field carries a consistent chemical state.

Returns the I-front `history` [(t_myr, r_I)], the final rate fields
(`kphHI`, `photogamma`, `xHII` on the active grid) and the final time.
"""
function run_moray_stromgren(; t_end_myr::Real = 30.0,
                             snapshots = [5.0, 10.0, 20.0, 30.0],
                             density = nothing,
                             dt_max_myr::Real = 0.25,
                             paramfile::AbstractString = ENZO_PHOTONTEST_PF,
                             maxcycle::Integer = 10_000)
    EnzoLib.grid_available() || error("Enzo grid bridge not built")
    isfile(paramfile) || error("PhotonTest parameter file not found at $paramfile")
    # stage with the requested stop time (code units = Myr for this setup)
    dir = mktempdir()
    par = read(paramfile, String)
    par = replace(par, r"StopTime\s*=\s*\S+" => "StopTime                = $(float(t_end_myr))")
    pf = joinpath(dir, "PhotonTest.enzo")
    write(pf, par)
    eng = EnzoLib.engine_from_flags(; hydro = :enzo, radiation = true, cooling = true)
    snaps = sort(unique(vcat(Float64.(snapshots), Float64(t_end_myr))))

    return cd(dir) do
        h = EnzoLib.session_init(pf)
        h == C_NULL && error("session_init failed for PhotonTest")
        try
            if density !== nothing
                g = _enzo_active(h)
                n = length(g.sl[1])
                size(density) == (n, n, n) ||
                    error("density must be $(n)³ (active grid), got $(size(density))")
                ratio = zeros(size(density))
                rho0 = _enzo_field_active(h, FT_DENSITY)
                ratio .= density ./ rho0
                for ft in (FT_DENSITY, FT_HI, FT_HII, 7)        # 7 = ElectronDensity
                    fi = EnzoLib.field_index(h, ft)
                    full = reshape(EnzoLib.problem_get_field(h, fi, 0), g.dims...)
                    full[g.sl...] .*= ratio
                    EnzoLib.problem_set_field(h, fi, vec(full))
                end
            end
            history = Tuple{Float64,Float64}[]
            n = 0
            for target in snaps
                while EnzoLib.session_time(h) < target * (1 - 1e-12) && n < maxcycle
                    EnzoLib.session_set_boundary(h, 0)
                    # the CHEMISTRY advances once per outer cycle (the photons
                    # subcycle internally), so the outer dt must stay small or
                    # the ionization lags the radiation
                    dt = min(EnzoLib.session_compute_dt(h, 0), Float64(dt_max_myr),
                             target - EnzoLib.session_time(h))
                    EnzoLib.session_set_dt(h, dt, 0)
                    EnzoLib.run_slot(:radiation, eng, h, 0, dt)
                    EnzoLib.run_slot(:hydro, eng, h, 0, dt)      # no-op (HydroMethod=-1)
                    EnzoLib.run_slot(:cooling, eng, h, 0, dt)    # rate/ionization solve
                    EnzoLib.session_advance_time(h, 0)
                    n += 1
                end
                push!(history, (EnzoLib.session_time(h), moray_ifront_radius(h).r_I))
            end
            fields = (kphHI = _enzo_field_active(h, FT_KPHHI),
                      photogamma = _enzo_field_active(h, FT_PHOTOGAMMA),
                      xHII = _enzo_field_active(h, FT_HII) ./
                             (_enzo_field_active(h, FT_HI) .+ _enzo_field_active(h, FT_HII)),
                      density = _enzo_field_active(h, FT_DENSITY))
            return (history = history, fields = fields, t = EnzoLib.session_time(h),
                    cycles = n, handle = h, free = () -> EnzoLib.free_problem(h))
        catch
            EnzoLib.free_problem(h)
            rethrow()
        end
    end
end
