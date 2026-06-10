# ── the MUSIC injector validation (the wrapper-registry on-ramp) ──────────────
#
# ONE MusicSpec realization (identical deterministic seeds), TWO output
# formats, TWO live codes booted on MUSIC's own files — Enzo on the generated
# `parameter_file.txt` + particle ICs, RAMSES (UNITS=COSMO) on the grafic2
# level directory — and the two codes' INITIAL particle density fields
# CIC-deposited and correlated.  No evolution: this gates the IC injection
# chain itself (MUSIC → format writers → each code's cosmological init).
#
# A package extension: `using MusicLib` activates it.

module MultiCodeMusicExt

using MultiCode
using MultiCode: EnzoLib, RamsesLib, CodeBridge
using MusicLib

function MultiCode.run_music_crosscheck(; boxlength::Real = 20.0, zstart::Real = 50.0,
                                        level::Integer = 5, worker::Bool = false)
    MusicLib.available() || error("libmusic_capi not found")
    # worker=true: MUSIC generates in its OWN process (CodeBridge's Julia
    # reference worker) — immune to the OpenMP/FFTW runtime pollution that
    # makes in-process generation segfault once many live codes share the
    # host (the D2 fix; the gate then needs no suite-ordering care).
    if worker
        wcmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e "using MusicLib; MusicLib.serve()" $(tempname())`
        CodeBridge.connect_worker!(MusicLib.BRIDGE, wcmd)
    end
    EnzoLib.grid_available() || error("Enzo grid bridge not built")
    CodeBridge.available(RamsesLib.BRIDGE, :cosmo) ||
        error("RAMSES cosmo library not found (bin64sc)")
    n = 2^level
    mk(format, fname) = MusicSpec(boxlength = boxlength, zstart = zstart,
                                  levelmin = level, levelmax = level,
                                  format = format, filename = fname)
    re, rr = try
        a = MusicLib.generate(mk(:enzo, "ic_enzo"); workdir = mktempdir())
        b = MusicLib.generate(mk(:ramses, "ics_ramses"); workdir = mktempdir())
        (a, b)
    finally
        worker && CodeBridge.disconnect_worker!(MusicLib.BRIDGE)
    end
    re.rc == 0 && rr.rc == 0 || error("MUSIC generation failed")
    # ── Enzo boots on MUSIC's own parameter file + particle ICs ───────────────
    par = read(joinpath(re.output, "parameter_file.txt"), String)
    par *= "\nStaticHierarchy = 1\nMaximumRefinementLevel = 0\n"
    pf = joinpath(re.output, "music.enzo")
    write(pf, par)
    xp_e = cd(re.output) do
        h = EnzoLib.session_init(pf)
        h == C_NULL && error("session_init failed on the MUSIC parameter file")
        try
            EnzoLib.read_particles(h)
        finally
            EnzoLib.free_problem(h)
        end
    end
    # ── RAMSES boots on MUSIC's grafic2 level directory ───────────────────────
    leveldir = joinpath(rr.output, "level_" * lpad(level, 3, '0'))
    isdir(leveldir) || error("MUSIC grafic level dir not found at $leveldir")
    dir = mktempdir()
    # Fortran's filename buffers truncate at 80 chars — keep the namelist path
    # SHORT and relative (initfile(1)='ics'), symlinked to the MUSIC level dir
    symlink(leveldir, joinpath(dir, "ics"))
    nml = MultiCode._ramses_zeldovich_namelist(ZeldovichSpec(n = n); level = level)
    write(joinpath(dir, "music.nml"), nml)
    xp_r = cd(dir) do
        h = RamsesLib.init("music.nml"; lib = :cosmo)
        try
            RamsesLib.get_particles(h, n^3; lib = :cosmo).xp
        finally
            RamsesLib.finalize(h; lib = :cosmo)
        end
    end
    size(xp_e, 1) == n^3 || error("Enzo read $(size(xp_e,1)) particles, expected $(n^3)")
    size(xp_r, 1) == n^3 || error("RAMSES read $(size(xp_r,1)) particles, expected $(n^3)")
    # ── the gate: CIC density fields of the SAME realization ─────────────────
    de = MultiCode.cic_density(xp_e, n) .- 1.0
    dr = MultiCode.cic_density(xp_r, n) .- 1.0
    se = sqrt(sum(abs2, de) / n^3); sr = sqrt(sum(abs2, dr) / n^3)
    corr = sum(de .* dr) / (n^3 * se * sr)
    rms = sqrt(sum(abs2, de .- dr) / n^3) / se
    return (corr = corr, rms = rms, sigma_enzo = se, sigma_ramses = sr,
            n = n, free = () -> nothing)
end

end # module
