# ── the science-grade comparison: one Sedov blast, every engine (ADR-0006) ────
#
# The SAME discrete initial condition — uniform ρ = 1 gas with a thermal bomb
# of measured energy E₀ in a small central sphere — injected through each
# code's live-field bridge into: Enzo's PPM, RAMSES's unsplit MUSCL, and the
# PPMKernels guest slot on RAMSES's mesh (CPU and Metal).  Injecting the
# identical cell-level IC (rather than each code's own initializer) makes
# this a comparison of the SCHEMES, with no per-code injection quirks; the
# oracle is the Sedov–Taylor similarity solution R(t) = ξ₀ (E t²/ρ)^{1/5}
# evaluated with each run's MEASURED injected energy.

"ξ₀ for γ = 1.4 (Sedov 1959; the standard tabulated value)."
const SEDOV_XI0_G14 = 1.1517

Base.@kwdef struct SedovCompareSpec
    E0::Float64    = 1.0
    rho0::Float64  = 1.0
    p0::Float64    = 1e-5      # cold background
    gamma::Float64 = 1.4
    r0::Float64    = 0.05      # bomb radius (box units; spread over a few cells)
    t::Float64     = 0.05      # compare epoch: R ≈ 0.35 (waves well inside the box)
end

"Analytic shock radius at time t [box units], given the measured injected energy."
sedov_radius(spec::SedovCompareSpec, t, E) =
    SEDOV_XI0_G14 * (E * t^2 / spec.rho0)^(1 / 5)

"""
    sedov_bomb(spec, n) -> (; te, E_in)

The discrete IC on an n³ grid: SPECIFIC total energy per cell (background +
the bomb spread uniformly over the cells whose centers lie within `r0` of the
box center), and the exactly-injected energy `E_in = Σ ρ·(te−te_bg)·dV` the
analytic gate uses.
"""
function sedov_bomb(spec::SedovCompareSpec, n::Integer)
    te_bg = spec.p0 / ((spec.gamma - 1) * spec.rho0)
    inside = falses(n, n, n)
    cnt = 0
    for c in CartesianIndices(inside)
        r2 = sum(d -> ((c[d] - 0.5) / n - 0.5)^2, 1:3)
        (inside[c] = r2 < spec.r0^2) && (cnt += 1)
    end
    cnt > 0 || error("sedov_bomb: r0 smaller than a cell at n=$n")
    dV = 1.0 / n^3
    te_bomb = spec.E0 / (spec.rho0 * cnt * dV)          # specific energy added per bomb cell
    te = fill(te_bg, n, n, n)
    te[inside] .+= te_bomb
    return (te = te, E_in = spec.rho0 * te_bomb * cnt * dV)
end

"Radial density profile about the box center + the shock radius (peak location)."
function sedov_profile(cs::CellSet; nbins::Integer = 128)
    sums = zeros(nbins); counts = zeros(Int, nbins)
    for i in 1:ncells(cs)
        r = sqrt(sum(d -> (cs.pos[i, d] - 0.5)^2, 1:3))
        b = min(nbins, 1 + floor(Int, r / (0.5 * sqrt(3)) * nbins))
        sums[b] += cs.rho[i]; counts[b] += 1
    end
    rb = [(b - 0.5) * 0.5 * sqrt(3) / nbins for b in 1:nbins]
    ρb = [counts[b] > 0 ? sums[b] / counts[b] : NaN for b in 1:nbins]
    good = findall(b -> counts[b] > 0 && rb[b] < 0.5, 1:nbins)
    R = rb[good[argmax(ρb[good])]]
    return (r = rb, rho = ρb, R_shock = R)
end

# ── Enzo: a uniform 3-D box (ProblemType 1, identical states) + injection ────
function _enzo_uniform3d_param(spec::SedovCompareSpec, n::Integer)
    return """
    ProblemType            = 1
    TopGridRank            = 3
    TopGridDimensions      = $n $n $n
    HydroMethod            = 0
    StopTime               = $(spec.t)
    dtDataDump             = 10.0
    LeftFaceBoundaryCondition  = 3 3 3
    RightFaceBoundaryCondition = 3 3 3
    Gamma                  = $(spec.gamma)
    CourantSafetyNumber    = 0.8
    StaticHierarchy        = 1
    HydroShockTubesInitialDiscontinuity  = 0.5
    HydroShockTubesLeftDensity           = $(spec.rho0)
    HydroShockTubesLeftPressure          = $(spec.p0)
    HydroShockTubesRightDensity          = $(spec.rho0)
    HydroShockTubesRightPressure         = $(spec.p0)
    """
end

"""
    run_enzo_sedov(spec=SedovCompareSpec(); n=64) -> (; cs, t, E_in, seconds, steps)

Enzo PPM on the injected Sedov IC: a uniform periodic 3-D box (ProblemType 1
with identical left/right states), the bomb written into the live TotalEnergy
field, the Julia-driven cycle loop to `spec.t`.
"""
function run_enzo_sedov(spec::SedovCompareSpec = SedovCompareSpec(); n::Integer = 64)
    EnzoLib.grid_available() || error("Enzo grid bridge not built")
    dir = mktempdir()
    pf = joinpath(dir, "uniform3d.enzo")
    write(pf, _enzo_uniform3d_param(spec, n))
    bomb = sedov_bomb(spec, n)
    return cd(dir) do
        h = EnzoLib.session_init(pf)
        h == C_NULL && error("session_init failed for the uniform 3-D box")
        try
            g = _enzo_active(h)
            length(g.sl[1]) == n || error("active grid $(length(g.sl[1])) ≠ $n")
            fi = EnzoLib.field_index(h, 1)               # TotalEnergy (specific)
            full = reshape(EnzoLib.problem_get_field(h, fi, 0), g.dims...)
            full[g.sl...] .= bomb.te
            EnzoLib.problem_set_field(h, fi, vec(full))
            steps = 0
            seconds = @elapsed while EnzoLib.session_time(h) < spec.t * (1 - 1e-12) && steps < 100_000
                EnzoLib.session_set_boundary(h, 0)
                dt = min(EnzoLib.session_compute_dt(h, 0), spec.t - EnzoLib.session_time(h))
                EnzoLib.session_set_dt(h, dt, 0)
                EnzoLib.session_solve_hydro(h, 0)
                EnzoLib.session_advance_time(h, 0)
                steps += 1
            end
            cs = enzo_extract(h)
            return (cs = cs, t = EnzoLib.session_time(h), E_in = bomb.E_in,
                    seconds = seconds, steps = steps, free = () -> EnzoLib.free_problem(h))
        catch
            EnzoLib.free_problem(h)
            rethrow()
        end
    end
end

# ── RAMSES: a uniform box + injection; native or guest (CPU/Metal) hydro ─────
function _ramses_uniform_namelist(spec::SedovCompareSpec; level::Integer)
    return """
    Uniform box for the injected Sedov comparison (MultiCode.jl)

    &RUN_PARAMS
    hydro=.true.
    ncontrol=1
    nrestart=0
    nremap=0
    nsubcycle=10*1
    nstepmax=100000
    nsuperoct=2
    verbose=.false.
    /

    &AMR_PARAMS
    levelmin=$(level)
    levelmax=$(level)
    ngridtot=3000000
    ncachemax=30000
    nexpand=1
    boxlen=1.0
    /

    &INIT_PARAMS
    nregion=1
    region_type(1)='square'
    x_center=0.5
    y_center=0.5
    z_center=0.5
    length_x=10.0
    length_y=10.0
    length_z=10.0
    exp_region=10.0
    d_region=$(spec.rho0)
    u_region=0.0
    v_region=0.0
    w_region=0.0
    p_region=$(spec.p0)
    /

    &OUTPUT_PARAMS
    foutput=0
    tout=100.0
    /

    &HYDRO_PARAMS
    gamma=$(spec.gamma)
    courant_factor=0.8
    slope_type=1
    riemann='hllc'
    /

    &REFINE_PARAMS
    interpol_var=0
    interpol_type=0
    /
    """
end

"""
    run_ramses_sedov(spec=SedovCompareSpec(); level=6, engine=:native, device=:cpu)
        -> (; cs, t, E_in, seconds, steps)

RAMSES on the injected Sedov IC.  `engine = :native` runs `godunov_fine!`
(unsplit MUSCL + HLLC); `engine = :guest` runs the PPMKernels slot
(`device = :cpu` or `:metal`).  Identical IC, identical host CFL clock —
the timing column is scheme-vs-scheme on the same mesh.
"""
function run_ramses_sedov(spec::SedovCompareSpec = SedovCompareSpec(); level::Integer = 6,
                          engine::Symbol = :native, device::Symbol = :cpu,
                          lib::Symbol = :cpu)
    RamsesLib.available() || error("RAMSES library not found (set RAMSES_LIB to the bin64h hydro build)")
    n = 2^level
    bomb = sedov_bomb(spec, n)
    dir = mktempdir()
    write(joinpath(dir, "sedov_uniform.nml"), _ramses_uniform_namelist(spec; level = level))
    return cd(dir) do
        h = RamsesLib.init("sedov_uniform.nml"; lib = lib)
        lev = RamsesLib.info(h; lib = lib).levelmin
        # inject: E (var 5) = ρ·te on the bomb cells, via the ckey-mapped setter
        ck, U = RamsesLib.get_hydro_all(h, :uold, lev; lib = lib)
        noct = size(ck, 1)
        Enew = Matrix{Float64}(undef, noct, 8)
        @inbounds for o in 1:noct, c in 1:8
            i = 2 * ck[o, 1] + ((c - 1) & 1) + 1
            j = 2 * ck[o, 2] + ((c - 1) >> 1 & 1) + 1
            k = 2 * ck[o, 3] + ((c - 1) >> 2 & 1) + 1
            Enew[o, c] = spec.rho0 * bomb.te[i, j, k]
        end
        RamsesLib.set_hydro!(h, :uold, 5, lev, ck, Enew; lib = lib)
        t = 0.0; steps = 0
        seconds = @elapsed while t < spec.t * (1 - 1e-12) && steps < 100_000
            RamsesLib.newdt_fine!(h, lev; lib = lib)
            dt = min(RamsesLib.get_dt(h, lev; lib = lib).dtnew, spec.t - t)
            RamsesLib.set_dt!(h, lev, dt; lib = lib)
            if engine === :native
                RamsesLib.hydro_step!(h, lev; dt = dt, lib = lib)
            else
                ramses_ppmk_hydro_step!(h; lev = lev, dt = dt, gamma = spec.gamma,
                                        boxlen = 1.0, lib = lib, device = device)
            end
            t += dt; steps += 1
        end
        cs = ramses_extract(h; lev = lev, boxlen = 1.0, lib = lib)
        return (cs = cs, t = t, E_in = bomb.E_in, seconds = seconds, steps = steps,
                free = () -> RamsesLib.finalize(h; lib = lib))
    end
end

# ── the report ────────────────────────────────────────────────────────────────
"""
    sedov_report(rows, spec; dir) -> path

The cross-engine Sedov page: per engine the measured shock radius vs the
analytic R(t) (with that run's measured E₀), conservation, wall-clock; plus
the radial-profile overlay SVG.
"""
function sedov_report(rows, spec::SedovCompareSpec; dir::AbstractString)
    mkpath(dir)
    md = joinpath(dir, "sedov_comparison.md")
    # profile overlay SVG
    svg = joinpath(dir, "sedov_profiles.svg")
    w, hgt, pad = 760, 460, 55
    ρmax = maximum(maximum(filter(!isnan, r.profile.rho)) for r in rows)
    sx(x) = pad + x / 0.55 * (w - 2pad)
    sy(y) = hgt - pad - y / (1.1 * ρmax) * (hgt - 2pad)
    open(svg, "w") do io
        print(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$w" height="$hgt" viewBox="0 0 $w $hgt">
        <rect width="$w" height="$hgt" fill="white"/>
        <text x="$(w ÷ 2)" y="22" text-anchor="middle" font-family="sans-serif" font-size="15">Sedov blast at t = $(spec.t) — radial density (one IC, every engine)</text>
        <line x1="$pad" y1="$(hgt - pad)" x2="$(w - pad)" y2="$(hgt - pad)" stroke="black"/>
        <line x1="$pad" y1="$pad" x2="$pad" y2="$(hgt - pad)" stroke="black"/>
        <text x="$(w ÷ 2)" y="$(hgt - 14)" text-anchor="middle" font-family="sans-serif" font-size="12">r</text>
        """)
        colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
        ly = 50
        for (ri, r) in enumerate(rows)
            good = findall(!isnan, r.profile.rho)
            pts = join(("$(round(sx(r.profile.r[b]); digits=2)),$(round(sy(r.profile.rho[b]); digits=2))"
                        for b in good if r.profile.r[b] < 0.55), " ")
            c = colors[mod1(ri, length(colors))]
            print(io, """<polyline points="$pts" fill="none" stroke="$c" stroke-width="1.6"/>\n""")
            ly += 18
            print(io, """<rect x="$(w - pad - 230)" y="$(ly - 10)" width="12" height="3" fill="$c"/>
            <text x="$(w - pad - 210)" y="$ly" font-family="sans-serif" font-size="12">$(r.label) (R=$(round(r.profile.R_shock; digits=3)))</text>\n""")
        end
        # analytic shock radius of the first row's energy
        Ra = sedov_radius(spec, spec.t, rows[1].E_in)
        print(io, """<line x1="$(sx(Ra))" y1="$pad" x2="$(sx(Ra))" y2="$(hgt - pad)" stroke="black" stroke-dasharray="6,3"/>
        <text x="$(sx(Ra) + 4)" y="$(pad + 14)" font-family="sans-serif" font-size="11">analytic R(t)</text>
        </svg>\n""")
    end
    open(md, "w") do io
        println(io, "# One Sedov blast, every engine (ADR-0006)\n")
        println(io, "The SAME discrete IC — uniform ρ = $(spec.rho0), a thermal bomb of measured ",
                "energy in a sphere of radius $(spec.r0) at the box center — injected through ",
                "each code's live-field bridge and evolved to t = $(spec.t).  Oracle: the ",
                "Sedov–Taylor radius R(t) = ξ₀(E t²/ρ)^{1/5}, ξ₀ = $(SEDOV_XI0_G14) (γ = $(spec.gamma)), ",
                "with each run's measured injected E₀.\n")
        println(io, "![profiles](sedov_profiles.svg)\n")
        println(io, "| engine | cells | steps | wall-clock [s] | E₀ (measured) | R_shock | R analytic | R/Rₐ | Δmass/mass | Δenergy/E |")
        println(io, "|--------|-------|-------|----------------|----------------|---------|------------|------|-----------|-----------|")
        for r in rows
            lg = ledger(r.cs)
            Ra = sedov_radius(spec, r.t, r.E_in)
            E_tot0 = spec.p0 / (spec.gamma - 1) + r.E_in
            @printf(io, "| %s | %d | %d | %.2f | %.6g | %.4f | %.4f | %.3f | %.2e | %.2e |\n",
                    r.label, ncells(r.cs), r.steps, r.seconds, r.E_in,
                    r.profile.R_shock, Ra, r.profile.R_shock / Ra,
                    abs(lg.mass - spec.rho0) / spec.rho0,
                    abs(lg.energy - E_tot0) / E_tot0)
        end
        println(io, "\nAll engines run the identical injected IC on their own mesh/scheme: Enzo ",
                "PPM (DirectEuler), RAMSES unsplit MUSCL+HLLC, and the PPMKernels guest slot ",
                "(PLM+HLLC, Hancock) on RAMSES's mesh — on the CPU in f64 and on the Metal GPU in f32.")
    end
    return md
end
