# ── the DISCO-DJ cross-check (the wrapper-registry on-ramp) ──────────────────
#
# DISCO-DJ's differentiable JAX LPT generates the displacement field; the two
# legacy codes evolve it.  The velocity-unit trap is dodged the Next-2 way:
# take ψ(aᵢ) at first order (1LPT ≡ Zel'dovich), build q + ψ with ZERO
# velocities, and the WHOLE linear field then follows the closed-form
# mixed-mode growth b(a/aᵢ) = ⅗x + ⅖x^{−3/2} — mode-independent, so it gates
# the full random field, not just a plane wave.  Gates per code: the final
# density field's shape stays that of the ICs (correlation) and its amplitude
# follows b; cross-code: Enzo and RAMSES land on the SAME field.
#
# A package extension: `using DiscoDJLib` activates it (PythonCall — set
# JULIA_PYTHONCALL_EXE before loading DiscoDJLib; see its module docs).

module MultiCodeDiscoDJExt

using MultiCode
using MultiCode: EnzoLib, RamsesLib, CodeBridge, ZeldovichSpec, zeldovich_growth,
                 cic_density, ENZO_DMONLY_DIR
using DiscoDJLib

# DISCO-DJ 1LPT particles, normalized to [0,1)³ rows
function _discodj_particles(; res::Integer, boxsize::Real, a_i::Real, seed::Integer)
    spec = DiscoSpec(res = res, boxsize = Float64(boxsize), n_order = 1, seed = seed)
    b = build(spec)
    ic = lpt_ics(b, a_i; n_order = 1)
    n3 = res^3
    xp = Matrix{Float64}(undef, n3, 3)
    pos = Float64.(ic.pos)
    @inbounds for k in 1:res, j in 1:res, i in 1:res
        p = i + res * (j - 1) + res^2 * (k - 1)
        for d in 1:3
            xp[p, d] = mod(pos[i, j, k, d] / boxsize, 1.0)
        end
    end
    return xp
end

# Enzo: the run_enzo_zeldovich body with an ARBITRARY zero-velocity particle set
function _enzo_growth(xp::Matrix{Float64}; n::Integer, z_init::Real, a_ratio::Real,
                      box_mpch::Real)
    EnzoLib.grid_available() || error("Enzo grid bridge not built")
    isdir(ENZO_DMONLY_DIR) || error("dm_only ICs not found")
    dir = mktempdir()
    for f in readdir(ENZO_DMONLY_DIR)
        p = joinpath(ENZO_DMONLY_DIR, f)
        isfile(p) && filesize(p) < 10^7 && !endswith(f, ".enzo") && cp(p, joinpath(dir, f))
    end
    par = read(joinpath(ENZO_DMONLY_DIR, "dm_only.enzo"), String)
    zf = (1 + z_init) / a_ratio - 1
    for (pat, rep) in (r"CosmologyOmegaMatterNow\s*=\s*\S+" => "CosmologyOmegaMatterNow    = 1.0",
                       r"CosmologyOmegaLambdaNow\s*=\s*\S+" => "CosmologyOmegaLambdaNow    = 0.0",
                       r"CosmologySimulationOmegaCDMNow\s*=\s*\S+" => "CosmologySimulationOmegaCDMNow = 1.0",
                       r"CosmologyInitialRedshift\s*=\s*\S+" => "CosmologyInitialRedshift   = $(z_init)",
                       r"CosmologyFinalRedshift\s*=\s*\S+" => "CosmologyFinalRedshift     = $(max(zf - 1, 0.0))",
                       r"CosmologyComovingBoxSize\s*=\s*\S+" => "CosmologyComovingBoxSize   = $(box_mpch)")
        par = replace(par, pat => rep)
    end
    par *= "\nStaticHierarchy = 1\nMaximumRefinementLevel = 0\n"
    pf = joinpath(dir, "discodj.enzo")
    write(pf, par)
    eng = EnzoLib.engine_from_flags(; hydro = :enzo, gravity = true, cosmology = true)
    return cd(dir) do
        h = EnzoLib.session_init(pf)
        h == C_NULL && error("session_init failed")
        try
            np = EnzoLib.problem_num_particles(h, 0)
            np == n^3 || error("expected $(n^3) particles, session has $np")
            for d in 0:2
                EnzoLib.problem_set_particle_pos(h, d, xp[:, d + 1])
                EnzoLib.problem_set_particle_vel(h, d, zeros(np))
            end
            steps = 0
            while EnzoLib.session_cosmology(h)[1] < a_ratio && steps < 100_000
                steps += 1
                EnzoLib.evolve_level!(h, 0, 0.0; engine = eng, regrid = false)
            end
            return (xp = EnzoLib.read_particles(h), a = EnzoLib.session_cosmology(h)[1],
                    steps = steps)
        finally
            EnzoLib.free_problem(h)
        end
    end
end

# RAMSES: the run_ramses_zeldovich body with an ARBITRARY zero-velocity set
function _ramses_growth(xp::Matrix{Float64}; n::Integer, z_init::Real, a_ratio::Real,
                        box_mpch::Real)
    CodeBridge.available(RamsesLib.BRIDGE, :cosmo) ||
        error("RAMSES cosmo library not found (bin64sc)")
    zspec = ZeldovichSpec(n = n, z_init = z_init, a_ratio = a_ratio, box_mpch = box_mpch)
    level = round(Int, log2(n))
    a_i = 1 / (1 + z_init)
    a_f = a_ratio * a_i
    dir = mktempdir()
    write(joinpath(dir, "discodj.nml"), MultiCode._ramses_zeldovich_namelist(zspec; level = level))
    MultiCode.write_grafic_ics(joinpath(dir, "ics"), zspec)
    return cd(dir) do
        h = RamsesLib.init_particles("discodj.nml", collect(1:n^3), xp, zeros(n^3, 3);
                                     lib = :cosmo)
        lev = RamsesLib.info(h; lib = :cosmo).levelmin
        steps = 0
        try
            while RamsesLib.get_dt(h, lev; lib = :cosmo).aexp < a_f * (1 - 1e-12)
                steps < 5000 || error("RAMSES did not reach a_f in 5000 steps")
                RamsesLib.amr_step!(h, lev, 1; lib = :cosmo)
                steps += 1
            end
            p = RamsesLib.get_particles(h, n^3; lib = :cosmo)
            return (xp = p.xp, a = RamsesLib.get_dt(h, lev; lib = :cosmo).aexp / a_i,
                    steps = steps)
        finally
            RamsesLib.finalize(h; lib = :cosmo)
        end
    end
end

function MultiCode.run_discodj_growth(; res::Integer = 32, z_init::Real = 49.0,
                                      a_ratio::Real = 4.0, box_mpch::Real = 32.0,
                                      seed::Integer = 42)
    a_i = 1 / (1 + z_init)
    xp0 = _discodj_particles(res = res, boxsize = box_mpch, a_i = a_i, seed = seed)
    d0 = cic_density(xp0, res) .- 1.0
    s0 = sqrt(sum(abs2, d0) / res^3)
    nc = max(res ÷ 4, 4)                            # well-resolved large scales
    dc0 = cic_density(xp0, nc) .- 1.0
    sc0 = sqrt(sum(abs2, dc0) / nc^3)
    out = NamedTuple[]
    for (label, runner) in (("enzo", _enzo_growth), ("ramses", _ramses_growth))
        r = runner(xp0; n = res, z_init = z_init, a_ratio = a_ratio, box_mpch = box_mpch)
        d = cic_density(r.xp, res) .- 1.0
        s = sqrt(sum(abs2, d) / res^3)
        corr0 = sum(d .* d0) / (res^3 * s * s0)     # shape vs the ICs
        dc = cic_density(r.xp, nc) .- 1.0
        sc = sqrt(sum(abs2, dc) / nc^3)
        push!(out, (label = label, a = r.a, steps = r.steps, sigma = s,
                    growth = s / s0,                 # full field: PM-resolution-limited
                    growth_coarse = sc / sc0,        # large scales: tracks b(a)
                    growth_exact = zeldovich_growth(r.a),   # b at the MEASURED epoch
                    corr_ic = corr0, delta = d))
    end
    cx = sum(out[1].delta .* out[2].delta) /
         (res^3 * out[1].sigma * out[2].sigma)      # Enzo ↔ RAMSES
    return (engines = [(; r..., delta = nothing) for r in out], cross_corr = cx,
            sigma_ic = s0, res = res)
end

end # module
