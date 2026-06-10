# ── the cosmology comparison: one particle set, Enzo and RAMSES (ADR-0006) ────
#
# A Zel'dovich plane wave with ZERO initial velocities: x(q, a_i) = q + ψ(q),
# v = 0, ψ(q) = A sin(2π q_x) x̂, in an EdS (Ω_m = 1) universe.  In 1-D plane
# symmetry the Zel'dovich form is EXACT before shell crossing, and the
# zero-velocity start excites the mixed growing+decaying mode with the fully
# analytic amplitude
#
#     b(a/a_i) = 3/5·(a/a_i) + 2/5·(a/a_i)^{-3/2},   b(1) = 1,  b'(a_i) = 0,
#
# so every particle's trajectory is known in closed form — and because the
# initial velocities are ZERO, no per-code velocity-unit convention enters the
# comparison at all.  The SAME lattice + displacement goes into both codes
# through their particle-injection bridges; the readback gates each code
# against the analytic trajectory and against the other code.

Base.@kwdef struct ZeldovichSpec
    n::Int          = 32       # particles (and force cells) per dimension
    A::Float64      = 0.0325   # displacement amplitude [box units]; caustic-safe
    a_ratio::Float64 = 4.0     # evolve a_i → a_ratio·a_i
    z_init::Float64 = 49.0     # both codes start here (a_i = 0.02)
    box_mpch::Float64 = 32.0   # comoving box (matched; the gates are box-unit)
end

"The zero-velocity mixed-mode growth factor, b(1) = 1 (EdS)."
zeldovich_growth(x) = 0.6 * x + 0.4 * x^(-1.5)

"Caustic forms when b·A·2π = 1; the spec must stay below it."
zeldovich_caustic_ok(spec::ZeldovichSpec) =
    zeldovich_growth(spec.a_ratio) * spec.A * 2π < 0.8

"""
    zeldovich_particles(spec) -> (; q, x)

The shared IC: an n³ lattice `q` (cell-centered) and the displaced positions
`x = q + A·sin(2π q_x)·x̂`, both `N×3` in box units.  Velocities are zero by
construction.
"""
function zeldovich_particles(spec::ZeldovichSpec)
    n = spec.n
    N = n^3
    q = Matrix{Float64}(undef, N, 3)
    x = Matrix{Float64}(undef, N, 3)
    p = 0
    for k in 1:n, j in 1:n, i in 1:n
        p += 1
        q[p, 1] = (i - 0.5) / n; q[p, 2] = (j - 0.5) / n; q[p, 3] = (k - 0.5) / n
        x[p, 1] = mod(q[p, 1] + spec.A * sin(2π * q[p, 1]), 1.0)
        x[p, 2] = q[p, 2]; x[p, 3] = q[p, 3]
    end
    return (q = q, x = x)
end

"""
    zeldovich_measure(xp, spec) -> (; bA, rms_resid, lines)

Identity-free per-particle analysis: y and z never change (the displacement
and the forces are x-only), so each particle's lattice line is exact; within a
line the x→q map is monotonic before the caustic, so SORTING the x positions
pairs them with the sorted lattice q.  Returns the least-squares amplitude
`bA` of the measured displacement field against sin(2π q) and the rms residual
from that pure mode (both in box units).
"""
function zeldovich_measure(xp::AbstractMatrix, spec::ZeldovichSpec)
    n = spec.n
    qlat = [(i - 0.5) / n for i in 1:n]
    slat = sin.(2π .* qlat)
    ss = sum(abs2, slat)
    # bucket particles by their (y, z) lattice line
    lines = Dict{Tuple{Int,Int},Vector{Float64}}()
    for p in 1:size(xp, 1)
        j = round(Int, xp[p, 2] * n + 0.5); k = round(Int, xp[p, 3] * n + 0.5)
        push!(get!(() -> Float64[], lines, (j, k)), xp[p, 1])
    end
    length(lines) == n^2 || error("zeldovich_measure: $(length(lines)) lines ≠ $(n^2) " *
                                  "(y/z moved — not a pure plane wave?)")
    num = 0.0; den = 0.0
    disp = Vector{Float64}(undef, n)
    resid2 = 0.0; cnt = 0
    for xs in values(lines)
        length(xs) == n || error("zeldovich_measure: line with $(length(xs)) particles")
        sort!(xs)
        for i in 1:n
            d = xs[i] - qlat[i]
            d -= round(d)                      # periodic wrap to [-0.5, 0.5)
            disp[i] = d
        end
        a = sum(disp .* slat) / ss             # per-line LSQ amplitude
        num += a; den += 1
        resid2 += sum(abs2, disp .- a .* slat); cnt += n
    end
    bA = num / den
    return (bA = bA, rms_resid = sqrt(resid2 / cnt), lines = length(lines))
end

# ── Enzo: the dm_only CosmologySimulation, EdS-patched, particles injected ───
const ENZO_DMONLY_DIR = normpath(joinpath(@__DIR__, "..", "..", "..", "..",
                                          "run", "CosmologySimulation", "dm_only"))

"""
    run_enzo_zeldovich(spec=ZeldovichSpec()) -> (; xp, a, steps, seconds)

Enzo's CosmologySimulation (the dm_only setup, cosmology patched to EdS and
the hierarchy frozen), with the shared particle set injected through the new
bridge setters right after init, evolved by the certified EvolveLevel
machinery (gravity + comoving expansion) until a/a_i ≥ `spec.a_ratio`.
Enzo's `session_cosmology` reports a normalized to the initial redshift, so
the stop condition is exactly `a ≥ a_ratio`.
"""
function run_enzo_zeldovich(spec::ZeldovichSpec = ZeldovichSpec())
    EnzoLib.grid_available() || error("Enzo grid bridge not built")
    isdir(ENZO_DMONLY_DIR) || error("dm_only ICs not found at $ENZO_DMONLY_DIR")
    zeldovich_caustic_ok(spec) || error("spec crosses the caustic")
    ics = zeldovich_particles(spec)
    dir = mktempdir()
    for f in readdir(ENZO_DMONLY_DIR)
        p = joinpath(ENZO_DMONLY_DIR, f)
        isfile(p) && filesize(p) < 10^7 && !endswith(f, ".enzo") && cp(p, joinpath(dir, f))
    end
    par = read(joinpath(ENZO_DMONLY_DIR, "dm_only.enzo"), String)
    zf = 1 / (spec.a_ratio / (1 + spec.z_init)) - 1
    for (pat, rep) in (r"CosmologyOmegaMatterNow\s*=\s*\S+" => "CosmologyOmegaMatterNow    = 1.0",
                       r"CosmologyOmegaLambdaNow\s*=\s*\S+" => "CosmologyOmegaLambdaNow    = 0.0",
                       r"CosmologySimulationOmegaCDMNow\s*=\s*\S+" => "CosmologySimulationOmegaCDMNow = 1.0",
                       r"CosmologyInitialRedshift\s*=\s*\S+" => "CosmologyInitialRedshift   = $(spec.z_init)",
                       r"CosmologyFinalRedshift\s*=\s*\S+" => "CosmologyFinalRedshift     = $(max(zf - 1, 0.0))",
                       r"CosmologyComovingBoxSize\s*=\s*\S+" => "CosmologyComovingBoxSize   = $(spec.box_mpch)")
        par = replace(par, pat => rep)
    end
    par *= "\nStaticHierarchy = 1\nMaximumRefinementLevel = 0\n"
    pf = joinpath(dir, "zeldovich.enzo")
    write(pf, par)
    eng = EnzoLib.engine_from_flags(; hydro = :enzo, gravity = true, cosmology = true)
    return cd(dir) do
        h = EnzoLib.session_init(pf)
        h == C_NULL && error("session_init failed (EdS-patched dm_only)")
        try
            np = EnzoLib.problem_num_particles(h, 0)
            np == spec.n^3 || error("dm_only has $np particles; spec wants $(spec.n^3)")
            for d in 0:2
                EnzoLib.problem_set_particle_pos(h, d, ics.x[:, d + 1])
                EnzoLib.problem_set_particle_vel(h, d, zeros(np))
            end
            steps = 0
            seconds = @elapsed while EnzoLib.session_cosmology(h)[1] < spec.a_ratio && steps < 100_000
                steps < 5000 || error("Enzo Zel'dovich did not reach a_ratio in 5000 steps")
                EnzoLib.evolve_level!(h, 0, 0.0; engine = eng, regrid = false)
                steps += 1
            end
            xp = EnzoLib.read_particles(h)
            return (xp = xp, a = EnzoLib.session_cosmology(h)[1], steps = steps,
                    seconds = seconds, free = () -> EnzoLib.free_problem(h))
        catch
            EnzoLib.free_problem(h)
            rethrow()
        end
    end
end

# ── RAMSES: the UNITS=COSMO flavor, the same particles injected at init ──────
#
# RAMSES's cosmological init reads aexp/Ω/h/dx FROM THE GRAFIC FILE HEADERS
# (init_time.f90 init_cosmo) — namelist values do not feed that path.  So we
# write a minimal, valid grafic set from Julia: correct headers, ZERO velocity
# planes (the lattice particles they would generate are immediately replaced
# by the injected set; the grafic-derived particle MASS — 1/N for Ω_m = 1 —
# is exactly what the injection inherits).  This is also the general
# infrastructure for the shared-grafic-ICs flagship.

"Write a grafic velocity file: the 44-byte header record + n³ zero planes."
function _write_grafic_zero(path::AbstractString; n::Integer, dx_mpc::Real, astart::Real,
                            omega_m::Real, omega_l::Real, h0::Real)
    open(path, "w") do io
        # Fortran sequential unformatted: [len][payload][len]
        write(io, Int32(44))
        write(io, Int32(n), Int32(n), Int32(n))
        write(io, Float32(dx_mpc), Float32(0), Float32(0), Float32(0))
        write(io, Float32(astart), Float32(omega_m), Float32(omega_l), Float32(h0))
        write(io, Int32(44))
        plane = zeros(Float32, n * n)
        for _ in 1:n
            write(io, Int32(4 * n * n))
            write(io, plane)
            write(io, Int32(4 * n * n))
        end
    end
    return path
end

"Write the grafic IC directory (ic_velcx/y/z) for the spec; returns the dir."
function write_grafic_ics(dir::AbstractString, spec::ZeldovichSpec)
    mkpath(dir)
    a_i = 1 / (1 + spec.z_init)
    h0 = 70.4
    dx_mpc = spec.box_mpch / (h0 / 100) / spec.n        # grafic dx in Mpc
    for f in ("ic_velcx", "ic_velcy", "ic_velcz")
        _write_grafic_zero(joinpath(dir, f); n = spec.n, dx_mpc = dx_mpc,
                           astart = a_i, omega_m = 1.0, omega_l = 0.0, h0 = h0)
    end
    return dir
end

function _ramses_zeldovich_namelist(spec::ZeldovichSpec; level::Integer)
    return """
    Zel'dovich plane wave, EdS (generated by MultiCode.jl — ADR-0006)

    &RUN_PARAMS
    cosmo=.true.
    pic=.true.
    poisson=.true.
    hydro=.false.
    nrestart=0
    nremap=0
    nsubcycle=10*1
    nstepmax=100000
    ncontrol=1
    verbose=.false.
    /

    &AMR_PARAMS
    levelmin=$(level)
    levelmax=$(level)
    ngridtot=3000000
    ncachemax=30000
    npartmax=$(2 * spec.n^3)
    nexpand=1
    boxlen=1.0
    /

    &INIT_PARAMS
    filetype='grafic'
    initfile(1)='ics'
    /

    &OUTPUT_PARAMS
    foutput=0
    aend=1.0
    /

    &POISSON_PARAMS
    epsilon=1.d-5
    /

    &REFINE_PARAMS
    m_refine=10*1.d30
    /
    """
end

"""
    run_ramses_zeldovich(spec=ZeldovichSpec(); level=round(Int, log2(spec.n)))
        -> (; xp, a, steps, seconds)

RAMSES (the `UNITS=COSMO` supercomoving flavor) with the SAME particle set
injected via `init_particles` (mass defaults to 1/N — exactly Ω_m = 1), the
production `amr_step` loop driving Friedmann + PM gravity until
aexp ≥ a_ratio·aexp_ini.
"""
function run_ramses_zeldovich(spec::ZeldovichSpec = ZeldovichSpec();
                              level::Integer = round(Int, log2(spec.n)),
                              lib::Symbol = :cosmo)
    CodeBridge.available(RamsesLib.BRIDGE, lib) ||
        error("RAMSES cosmo library not found (build bin64sc or set RAMSES_LIB_COSMO)")
    2^level == spec.n || error("level $level grid ≠ $(spec.n) particles per dim")
    zeldovich_caustic_ok(spec) || error("spec crosses the caustic")
    ics = zeldovich_particles(spec)
    a_i = 1 / (1 + spec.z_init)
    a_f = spec.a_ratio * a_i
    dir = mktempdir()
    write(joinpath(dir, "zeldovich.nml"), _ramses_zeldovich_namelist(spec; level = level))
    write_grafic_ics(joinpath(dir, "ics"), spec)
    return cd(dir) do
        h = RamsesLib.init_particles("zeldovich.nml", collect(1:spec.n^3), ics.x,
                                     zeros(spec.n^3, 3); lib = lib)
        lev = RamsesLib.info(h; lib = lib).levelmin
        steps = 0
        seconds = @elapsed while RamsesLib.get_dt(h, lev; lib = lib).aexp < a_f * (1 - 1e-12)
            steps < 5000 || error("RAMSES Zel'dovich did not reach a_f in 5000 steps " *
                                  "(aexp=$(RamsesLib.get_dt(h, lev; lib = lib).aexp))")
            RamsesLib.amr_step!(h, lev, 1; lib = lib)
            steps += 1
        end
        p = RamsesLib.get_particles(h, spec.n^3; lib = lib)
        return (xp = p.xp, a = RamsesLib.get_dt(h, lev; lib = lib).aexp / a_i,
                steps = steps, seconds = seconds,
                free = () -> RamsesLib.finalize(h; lib = lib))
    end
end
