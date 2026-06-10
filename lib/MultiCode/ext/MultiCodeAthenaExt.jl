# ── the Athena++ engine (the wrapper-registry on-ramp, Phase-2 pattern) ───────
#
# Athena++ enters the Sod harness as the FOURTH legacy engine: the stock
# `athinput.sod` (γ = 1.4, domain [−0.5, 0.5], interface at 0 — the harness
# spec in a shifted frame) run IN-PROCESS through AthenaLib, profiles read
# from the final .tab, conservation from the .hst history, gated against the
# same exact-Riemann oracle as Enzo/RAMSES/Arepo/dfmm.
#
# A package extension like MultiCodeDfmmExt: `using AthenaLib` activates it.

module MultiCodeAthenaExt

using MultiCode
using MultiCode: SodSpec
using AthenaLib

function MultiCode.run_athena_sod(spec::SodSpec = SodSpec();
                                  athinput::AbstractString = normpath(joinpath(
                                      dirname(AthenaLib.libpath()), "..",
                                      "inputs", "hydro", "athinput.sod")),
                                  nx1::Integer = 256)
    AthenaLib.available() || error("libathena_capi not found (build the :hydro flavor)")
    spec.gamma == 1.4 && spec.x0 == 0.5 ||
        error("run_athena_sod: the stock athinput.sod is γ=1.4 with a centered interface")
    isfile(athinput) || error("run_athena_sod: athinput.sod not found at $athinput")
    t0 = time()
    rd = AthenaLib.run(athinput;
                       overrides = "time/tlim=$(spec.t) mesh/nx1=$(nx1) output2/dt=$(spec.t)")
    seconds = time() - t0
    h = AthenaLib.read_hst(joinpath(rd, "Sod.hst"))
    tabs = sort(filter(f -> endswith(f, ".tab"), readdir(rd)))
    t = AthenaLib.read_tab(joinpath(rd, last(tabs)))
    profile = (x = t.x1v .+ 0.5,                     # the harness frame: interface at 0.5
               rho = t.rho, u = t.vel1, scatter = 0.0)
    return (profile = profile, t = spec.t, seconds = seconds,
            mass_drift = abs(h.mass[end] - h.mass[1]) / abs(h.mass[1]),
            t_end = h.time[end], diag = (nx1 = Int(nx1), outdir = rd),
            free = () -> nothing)
end

end # module
