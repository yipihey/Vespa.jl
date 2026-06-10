# ── one problem spec, three codes (ADR-0006 flagship 1, Sod variant) ──────────
#
# The SAME normalized Sod shock tube — (ρ,u,p) = (1,0,1) | (0.125,0,0.1),
# γ=1.4, discontinuity at x̂=0.5, compared at t̂=0.25 — run through each code's
# NATIVE setup path: Enzo's SodShockTube parameter file, a generated RAMSES
# namelist (two square regions), and Arepo's shocktube_1d example (box 20,
# TimeMax 5 ⇒ the identical normalized problem).  Each runner returns the
# canonical CellSet plus diagnostics; nothing downstream knows which code ran.

"""
    SodSpec(; rhoL=1.0, uL=0.0, pL=1.0, rhoR=0.125, uR=0.0, pR=0.1,
            gamma=1.4, x0=0.5, t=0.1)

The normalized Sod problem all three codes run.  The defaults are the standard
Sod values shared by Enzo's `SodShockTube.enzo` and Arepo's `shocktube_1d`.

The comparison time defaults to t̂ = 0.1: a step IC in a PERIODIC box (RAMSES,
Arepo) carries a second, mirror Riemann problem at the wrap seam, and t̂ = 0.1
keeps the seam waves (fastest ≈ 1.75) away from the Sod structure.  The RAMSES
runner additionally doubles its domain so the seam never enters the comparison
window; the Arepo runner windows its profile.  Enzo uses outflow BCs and is
clean over the whole box.
"""
Base.@kwdef struct SodSpec
    rhoL::Float64 = 1.0
    uL::Float64   = 0.0
    pL::Float64   = 1.0
    rhoR::Float64 = 0.125
    uR::Float64   = 0.0
    pR::Float64   = 0.1
    gamma::Float64 = 1.4
    x0::Float64   = 0.5
    t::Float64    = 0.1
end

exact_sod(spec::SodSpec, ξ) = exact_sod(ξ; rhoL = spec.rhoL, uL = spec.uL, pL = spec.pL,
                                        rhoR = spec.rhoR, uR = spec.uR, pR = spec.pR,
                                        gamma = spec.gamma)

"Analytic conserved totals of the spec on the unit box (the ledger reference)."
function sod_reference_ledger(spec::SodSpec)
    eL = spec.pL / (spec.gamma - 1) + 0.5 * spec.rhoL * spec.uL^2
    eR = spec.pR / (spec.gamma - 1) + 0.5 * spec.rhoR * spec.uR^2
    return (mass = spec.x0 * spec.rhoL + (1 - spec.x0) * spec.rhoR,
            energy = spec.x0 * eL + (1 - spec.x0) * eR,
            momentum_x = spec.x0 * spec.rhoL * spec.uL + (1 - spec.x0) * spec.rhoR * spec.uR)
end

# ── Enzo runner ───────────────────────────────────────────────────────────────

const ENZO_SOD_PF = normpath(joinpath(@__DIR__, "..", "..", "..", "..",
                                      "run", "Hydro", "Hydro-1D", "SodShockTube", "SodShockTube.enzo"))

"""
    run_enzo_sod(spec=SodSpec(); paramfile=ENZO_SOD_PF) -> (; cs, t, profile, diag)

Enzo's native Sod: `session_init` + the Julia-driven cycle loop (set_boundary →
compute_dt → set_dt → solve_hydro → advance_time), with dt capped to land
exactly on `spec.t`, then canonicalize the root grid.  `diag.mass_bridge` is
Enzo's OWN volume-weighted density integral — an independent check on the
adapter's geometry.  Outflow BCs ⇒ the whole box is the comparison window.
"""
function run_enzo_sod(spec::SodSpec = SodSpec(); paramfile::AbstractString = ENZO_SOD_PF)
    EnzoLib.grid_available() || error("Enzo grid bridge not built")
    isfile(paramfile) || error("Enzo Sod parameter file not found at $paramfile")
    pf = abspath(paramfile)
    return cd(EnzoLib._workdir(pf)) do
        h = EnzoLib.session_init(pf)
        h == C_NULL && error("session_init failed for $pf")
        try
            n = 0
            while EnzoLib.session_time(h) < spec.t * (1 - 1e-12) && n < 100_000
                EnzoLib.session_set_boundary(h, 0)
                dt = min(EnzoLib.session_compute_dt(h, 0), spec.t - EnzoLib.session_time(h))
                EnzoLib.session_set_dt(h, dt, 0)
                EnzoLib.session_solve_hydro(h, 0)
                EnzoLib.session_advance_time(h, 0)
                n += 1
            end
            cs = enzo_extract(h)
            di = EnzoLib.field_index(h, 0; grid = 0)
            mass_bridge = EnzoLib.session_global_field_integral(h, di)
            t = EnzoLib.session_time(h)
            return (cs = cs, t = t, profile = profile_x(cs),
                    diag = (cycles = n, mass_bridge = mass_bridge), handle = h,
                    free = () -> EnzoLib.free_problem(h))
        catch
            EnzoLib.free_problem(h)
            rethrow()
        end
    end
end

# ── RAMSES runner ─────────────────────────────────────────────────────────────

# RAMSES is periodic, so the Sod step carries a SECOND (mirror) Riemann problem
# at the wrap seam.  The runner therefore uses a DOUBLE domain (boxlen 2, code
# units): left state on x ∈ [0,1], right state on x ∈ [1,2], discontinuity at
# x = 1.  The seam waves (from x = 0 ≡ 2) travel ≈ 1.75·t and never reach the
# comparison window x ∈ [0.5, 1.5] for t ≤ 0.28 — which maps to the spec's
# normalized window via x̂_w = x_code − 0.5.
#
# Two square regions: region 1 covers the box with the LEFT state; region 2
# overwrites x ∈ [1, 2] with the RIGHT state (squares overwrite in order).
function ramses_sod_namelist(spec::SodSpec; level::Integer)
    nx2 = 1.5                          # center of the right region (code units)
    lx2 = 1.0                          # its x extent
    return """
    Sod shock tube (generated by MultiCode.jl — ADR-0006 Phase 2)

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
    boxlen=2.0
    /

    &INIT_PARAMS
    nregion=2
    region_type(1)='square'
    region_type(2)='square'
    x_center=1.0,$(nx2)
    y_center=1.0,1.0
    z_center=1.0,1.0
    length_x=10.0,$(lx2)
    length_y=10.0,10.0
    length_z=10.0,10.0
    exp_region=10.0,10.0
    d_region=$(spec.rhoL),$(spec.rhoR)
    u_region=$(spec.uL),$(spec.uR)
    v_region=0.0,0.0
    w_region=0.0,0.0
    p_region=$(spec.pL),$(spec.pR)
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
    run_ramses_sod(spec=SodSpec(); level=7) -> (; cs, t, profile, diag)

RAMSES's native Sod on a uniform `level` grid (2^level cells per dim, 3-D,
double-length domain — see `ramses_sod_namelist`), generated namelist, HLLC.
The Julia driver owns the clock: each step takes `min(CFL dt, t_end − t)` via
`set_dt!`, so the run lands exactly on `spec.t` (code time = normalized time;
the double box does not rescale the states).  The returned `profile` is the
x-profile mapped to the spec's window (x̂_w = x_code − 0.5, discontinuity at
0.5) and restricted to [0, 1]; `cs` is the full normalized box (the ledger).
"""
function run_ramses_sod(spec::SodSpec = SodSpec(); level::Integer = 7, lib::Symbol = :cpu)
    RamsesLib.available() || error("RAMSES library not found (set RAMSES_LIB to the bin64h hydro build)")
    spec.t <= 0.28 || error("run_ramses_sod: t̂ > 0.28 lets the periodic seam waves reach the window")
    dir = mktempdir()
    write(joinpath(dir, "sod3d.nml"), ramses_sod_namelist(spec; level = level))
    return cd(dir) do
        h = RamsesLib.init("sod3d.nml"; lib = lib)
        lev = RamsesLib.info(h; lib = lib).levelmin
        t = 0.0; n = 0
        while t < spec.t * (1 - 1e-12) && n < 100_000
            RamsesLib.newdt_fine!(h, lev; lib = lib)
            dt = min(RamsesLib.get_dt(h, lev; lib = lib).dtnew, spec.t - t)
            RamsesLib.set_dt!(h, lev, dt; lib = lib)
            RamsesLib.hydro_step!(h, lev; dt = dt, lib = lib)
            t += dt; n += 1
        end
        cs = ramses_extract(h; lev = lev, boxlen = 2.0, lib = lib)
        # window: code x ∈ [0.5, 1.5] → x̂_w ∈ [0, 1] (cs positions are x_code/2)
        full = profile_x(cs)
        xw = 2.0 .* full.x .- 0.5
        keep = findall(x -> 0.0 <= x <= 1.0, xw)
        profile = (x = xw[keep], rho = full.rho[keep], u = full.u[keep],
                   scatter = full.scatter)
        return (cs = cs, t = t, profile = profile, diag = (steps = n, level = lev), handle = h,
                free = () -> RamsesLib.finalize(h; lib = lib))
    end
end

# ── Arepo runner ──────────────────────────────────────────────────────────────

"Find a python with numpy+h5py (AREPO_PYTHON, the arepo/.venv, else PATH)."
function _arepo_python(arepo_dir)
    ok(exe) = try
        run(pipeline(`$exe -c "import numpy, h5py"`; stdout = devnull, stderr = devnull))
        true
    catch; false end
    env = get(ENV, "AREPO_PYTHON", "")
    !isempty(env) && isfile(env) && ok(env) && return env
    venv = joinpath(arepo_dir, ".venv", "bin", "python")
    isfile(venv) && ok(venv) && return venv
    for exe in ("python3", "python")
        p = Sys.which(exe)
        p !== nothing && ok(p) && return p
    end
    return nothing
end

"""
    run_arepo_sod(spec=SodSpec(); worker=false) -> (; cs, t, profile, diag)

Arepo's native Sod: the `examples/shocktube_1d` setup (box 20, discontinuity at
10 — the identical problem in normalized units), ICs from the example's own
`create.py`, `TimeMax` patched to `20·spec.t` (the test_shocktube pattern),
evolved with `run!`, canonicalized with the box length.  Arepo's box is
periodic, so the returned `profile` is windowed to the seam-clean region
(the full-box `cs` still carries the exact conservation ledger).

`worker = true` runs Arepo in its OWN worker process (CodeBridge `:remote`,
ADR-0006 D2) — Arepo keeps its state in C globals and cannot re-`init` cleanly
in one process, so every run after the first in a session must be a worker.
The `r.free()` closure disconnects the worker.
"""
function run_arepo_sod(spec::SodSpec = SodSpec(); worker::Bool = false)
    ArepoLib.available() || error("Arepo library not built")
    (spec.rhoL, spec.uL, spec.pL, spec.rhoR, spec.uR, spec.pR, spec.gamma, spec.x0) ==
        (1.0, 0.0, 1.0, 0.125, 0.0, 0.1, 1.4, 0.5) ||
        error("run_arepo_sod uses the stock shocktube_1d example — standard Sod states only")
    arepo_dir = normpath(dirname(ArepoLib.libpath()))
    example = joinpath(arepo_dir, "examples", "shocktube_1d")
    isdir(example) || error("Arepo example not found at $example")
    py = _arepo_python(arepo_dir)
    py === nothing && error("no python with numpy+h5py for Arepo IC generation (set AREPO_PYTHON)")

    dir = mktempdir()
    param = read(joinpath(example, "param.txt"), String)
    param = replace(param, r"TimeMax\s+\S+" => "TimeMax                                   $(20 * spec.t)")
    write(joinpath(dir, "param.txt"), param)
    mkpath(joinpath(dir, "output"))
    run(pipeline(`$py $(joinpath(example, "create.py")) $dir`; stdout = devnull))
    isfile(joinpath(dir, "IC.hdf5")) || error("Arepo create.py produced no IC.hdf5")

    if worker
        # Arepo in its own process: param/IC paths resolve against the WORKER's
        # cwd (spawned in the staged run dir); every ArepoLib call routes over
        # the bridge RPC transparently (the whole surface is @xcall).
        shm = tempname()
        wfile = joinpath(dir, "arepo_worker.jl")
        write(wfile, "using ArepoLib; ArepoLib.serve(; shm = ARGS[1])\n")
        jl = Base.julia_cmd()
        CodeBridge.connect_worker!(ArepoLib.BRIDGE,
            setenv(`$jl --startup-file=no --project=$(pkgdir(ArepoLib)) $wfile $shm`; dir = dir);
            shm = shm)
    end
    release = () -> (worker && CodeBridge.disconnect_worker!(ArepoLib.BRIDGE); nothing)

    return cd(dir) do
        h = ArepoLib.init("param.txt")
        status = ArepoLib.run!(h)
        status == :done || error("Arepo run ended with $status")
        L = ArepoLib.box_size(h)
        cs = arepo_extract(h; boxlen = L)
        t = ArepoLib.sim_time(h) / L                 # normalized
        # seam-clean window: mirror-Sod waves from the wrap travel ≤ 1.76·t from
        # x̂ ∈ {0, 1} (rarefaction-into-dense from 0, shock-into-light from 1)
        full = profile_x(cs)
        lo = 1.19 * t + 0.02; hi = 1.0 - (1.76 * t + 0.02)
        keep = findall(x -> lo <= x <= hi, full.x)
        profile = (x = full.x[keep], rho = full.rho[keep], u = full.u[keep],
                   scatter = full.scatter)
        return (cs = cs, t = t, profile = profile,
                diag = (status = status, boxsize = L, window = (lo, hi), worker = worker),
                handle = h,
                free = () -> (ArepoLib.finalize(h); release()))
    end
end

# ── profiles + error norms ────────────────────────────────────────────────────

"""
    profile_x(cs::CellSet; digits=9) -> (; x, rho, u, scatter)

Volume-weighted x-profile: cells sharing an x (to `digits`) are averaged —
the y/z collapse for the 3-D RAMSES tube (and a no-op for the 1-D sets).
`scatter` is the max in-bin density spread, a transverse-symmetry check.
"""
function profile_x(cs::CellSet; digits::Integer = 9)
    bins = Dict{Float64,NTuple{4,Float64}}()    # x => (Σv, Σρv, Σρuv, ρspread placeholder)
    lo = Dict{Float64,Float64}(); hi = Dict{Float64,Float64}()
    for i in 1:ncells(cs)
        x = round(cs.pos[i, 1]; digits = digits)
        v = cs.vol[i]; ρ = cs.rho[i]; ρu = cs.mom[i, 1]
        s = get(bins, x, (0.0, 0.0, 0.0, 0.0))
        bins[x] = (s[1] + v, s[2] + ρ * v, s[3] + ρu * v, 0.0)
        lo[x] = min(get(lo, x, Inf), ρ); hi[x] = max(get(hi, x, -Inf), ρ)
    end
    xs = sort!(collect(keys(bins)))
    rho = [bins[x][2] / bins[x][1] for x in xs]
    u = [bins[x][3] / bins[x][2] for x in xs]
    scatter = maximum(hi[x] - lo[x] for x in xs)
    return (x = xs, rho = rho, u = u, scatter = scatter)
end

"""
    sod_l1(profile, spec, t) -> (; rho, u)

Cell-averaged L1 errors of the profile against the exact solution at time `t`.
"""
function sod_l1(profile, spec::SodSpec, t::Real)
    n = length(profile.x)
    e_rho = 0.0; e_u = 0.0
    for i in 1:n
        ex = exact_sod(spec, (profile.x[i] - spec.x0) / t)
        e_rho += abs(profile.rho[i] - ex.rho)
        e_u += abs(profile.u[i] - ex.u)
    end
    return (rho = e_rho / n, u = e_u / n)
end
