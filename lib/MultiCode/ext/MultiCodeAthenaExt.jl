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
using MultiCode: SodSpec, CellSet
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

"""
3-D Sod through Athena++ → the CANONICAL state: the stock athinput extruded to
n³ (a derived athinput carries an appended `<output3>` vtk block — command-line
overrides cannot ADD blocks), one MeshBlock so the whole domain lands in one
legacy-VTK file, read back into a `CellSet` (positions recentred to [0,1]³).
VTK stores float32, so ledgers gate at the f32 floor, not f64 round-off.
"""
function MultiCode.run_athena_sod3d(spec::SodSpec = SodSpec(); n::Integer = 32,
                                    athinput::AbstractString = normpath(joinpath(
                                        dirname(AthenaLib.libpath()), "..",
                                        "inputs", "hydro", "athinput.sod")))
    AthenaLib.available() || error("libathena_capi not found (build the :hydro flavor)")
    spec.gamma == 1.4 && spec.x0 == 0.5 ||
        error("run_athena_sod3d: the stock athinput.sod is γ=1.4, centered interface")
    d = mktempdir()
    pf = joinpath(d, "athinput.sod3d")
    write(pf, read(athinput, String) *
              "\n<output3>\nfile_type = vtk\nvariable  = prim\ndt        = $(spec.t)\n")
    t0 = time()
    rd = AthenaLib.run(pf; overrides = "time/tlim=$(spec.t) mesh/nx1=$(n) " *
        "mesh/nx2=$(n) mesh/nx3=$(n) mesh/x2min=-0.5 mesh/x2max=0.5 " *
        "mesh/x3min=-0.5 mesh/x3max=0.5")
    seconds = time() - t0
    v = AthenaLib.read_vtk(joinpath(rd, last(sort(filter(f -> endswith(f, ".vtk"),
                                                          readdir(rd))))))
    rho3 = v.scalars["rho"]; press = v.scalars["press"]; vel = v.vectors["vel"]
    n3 = n^3
    pos = Matrix{Float64}(undef, n3, 3)
    mom = Matrix{Float64}(undef, n3, 3)
    rho = Vector{Float64}(undef, n3); etot = Vector{Float64}(undef, n3)
    gm1 = spec.gamma - 1
    q = 0
    for k in 1:n, j in 1:n, i in 1:n
        q += 1
        pos[q, 1] = (i - 0.5) / n; pos[q, 2] = (j - 0.5) / n; pos[q, 3] = (k - 0.5) / n
        r = rho3[i, j, k]
        rho[q] = r
        v2 = 0.0
        for dd in 1:3
            mom[q, dd] = r * vel[i, j, k, dd]
            v2 += vel[i, j, k, dd]^2
        end
        etot[q] = press[i, j, k] / gm1 + 0.5 * r * v2
    end
    cs = CellSet(:athena, pos, fill(1.0 / n3, n3), rho, mom, etot,
                 (length = 1.0, time = 1.0, density = 1.0), (;))
    prof = MultiCode.profile_x(cs)
    return (cs = cs, t = v.time, seconds = seconds,
            profile = (x = prof.x, rho = prof.rho, u = prof.u, scatter = prof.scatter),
            diag = (n = Int(n), outdir = rd), free = () -> nothing)
end

end # module
