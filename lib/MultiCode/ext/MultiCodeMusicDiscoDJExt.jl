module MultiCodeMusicDiscoDJExt

using MultiCode
using Random
using Printf
using MusicLib
using DiscoDJLib

function _normed_noise(res::Integer, seed::Integer)
    rng = MersenneTwister(seed)
    f = randn(rng, res, res, res)
    f .-= sum(f) / length(f)
    f ./= sqrt(sum(abs2, f) / length(f))
    return f
end

function _corr(a, b)
    aa = a .- sum(a) / length(a)
    bb = b .- sum(b) / length(b)
    sa = sqrt(sum(abs2, aa) / length(aa))
    sb = sqrt(sum(abs2, bb) / length(bb))
    return sum(aa .* bb) / (length(aa) * sa * sb)
end

function _discodj_density_proxy(; res::Integer, boxlength::Real, seed::Integer, a::Real)
    spec = DiscoSpec(res = Int(res), boxsize = Float64(boxlength), n_order = 1, seed = Int(seed))
    ic = lpt_ics(build(spec), a; n_order = 1)
    ψ = ic.psi
    dx = Float64(boxlength) / res
    δ = Array{Float64}(undef, res, res, res)
    @inbounds for k in 1:res, j in 1:res, i in 1:res
        ip = i == res ? 1 : i + 1
        im = i == 1 ? res : i - 1
        jp = j == res ? 1 : j + 1
        jm = j == 1 ? res : j - 1
        kp = k == res ? 1 : k + 1
        km = k == 1 ? res : k - 1
        divψ = (ψ[ip, j, k, 1] - ψ[im, j, k, 1] +
                ψ[i, jp, k, 2] - ψ[i, jm, k, 2] +
                ψ[i, j, kp, 3] - ψ[i, j, km, 3]) / (2dx)
        δ[i, j, k] = -divψ
    end
    δ .-= sum(δ) / length(δ)
    return δ
end

function _write_phase_report(path::AbstractString, r)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# MUSIC ↔ DISCO-DJ fixed-phase audit\n")
        println(io, "MUSIC is exercised through its direct white-noise file inlet; the ",
                "mirrored file is the Angulo-Pontzen control. DISCO-DJ is currently ",
                "seed-driven through its NGenIC-compatible generator, so this report ",
                "compares MUSIC's explicit white-noise field with DISCO-DJ's 1LPT ",
                "finite-difference density proxy at the same integer seed.\n")
        println(io, "| res | seed | corr(noise, readback) | corr(noise, mirror) | corr(MUSIC noise, DISCO-DJ proxy) | corr(proxy seed, seed+1) |")
        println(io, "|-----|------|-----------------------|---------------------|----------------------------------|--------------------------|")
        @printf(io, "| %d³ | %d | %.15f | %.15f | %.6f | %.6f |\n",
                r.res, r.seed, r.music_readback_corr, r.music_mirror_corr,
                r.music_discodj_corr, r.discodj_seed_cross_corr)
        println(io, "\nInterpretation: the MUSIC fixed/mirror path should be +1/-1 to round-off. ",
                "A small same-seed MUSIC↔DISCO-DJ proxy correlation means the two wrappers ",
                "do not yet share an explicit white-noise realization, even if their seed ",
                "interfaces are both deterministic.")
    end
    return path
end

function MultiCode.run_music_discodj_phase_report(; res::Integer = 32, seed::Integer = 42,
                                                  boxlength::Real = 20.0,
                                                  zstart::Real = 50.0,
                                                  report_path::Union{Nothing,AbstractString} = nothing,
                                                  workdir::AbstractString = mktempdir())
    DiscoDJLib.available() || error("DISCO-DJ Python environment is not available")
    mkpath(workdir)
    noise = _normed_noise(res, seed)
    mirror = MusicLib.mirror_noise(noise)
    p = joinpath(workdir, "music_noise_$(lpad(string(seed), 4, '0')).bin")
    pm = joinpath(workdir, "music_noise_$(lpad(string(seed), 4, '0'))_mirror.bin")
    MusicLib.write_music_noise(p, noise; seed = seed)
    MusicLib.write_music_noise(pm, mirror; seed = seed)
    rb = MusicLib.read_music_noise(p).field
    rmb = MusicLib.read_music_noise(pm).field
    a = 1 / (1 + zstart)
    ddj = _discodj_density_proxy(res = res, boxlength = boxlength, seed = seed, a = a)
    ddj2 = _discodj_density_proxy(res = res, boxlength = boxlength, seed = seed + 1, a = a)
    r = (res = Int(res), seed = Int(seed), boxlength = Float64(boxlength),
         zstart = Float64(zstart), music_noise_file = p, music_mirror_file = pm,
         music_readback_corr = _corr(noise, rb),
         music_mirror_corr = _corr(noise, rmb),
         music_discodj_corr = _corr(noise, ddj),
         discodj_seed_cross_corr = _corr(ddj, ddj2))
    if report_path !== nothing
        _write_phase_report(report_path, r)
    end
    return r
end

end # module
