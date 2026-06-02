# Confirm that the NEW Julia driver, evolving a Sod shock tube with the OLD Enzo
# PPM kernel (`ppm_sweep_1d`, verbatim legacy Fortran via ccall), gives results
# identical — to machine precision — to the reference driver
# (`EnzoModules/enzomodules/examples/ppm_sod.py`) running the same legacy kernel.
#
# Both drivers are algorithmically identical (NGHOST=3, outflow ghosts, the same
# pressure / CFL-limited dt) and call the IDENTICAL compiled Fortran kernel with
# the same deterministic timestep sequence, so the only difference is last-bit
# reduction round-off accumulated over the run.
#
# Run:
#   ENZOMODULES_LIB=<repo>/EnzoModules/deps/libenzomodules_pilot.so \
#   julia --project=lib/EnzoLib/test lib/EnzoLib/examples/sod_replication_check.jl
#
# (Build the .so first: julia --project=lib/EnzoLib/test lib/EnzoLib/deps/build.jl)

using EnzoLib

const NG = 3
const REPO = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))

# The new Julia driver (mirrors examples/ppm_sod.py exactly).
function julia_ppm_sod(; n = 200, tfinal = 0.2, cfl = 0.4, γ = 1.4)
    idim = n + 2NG; i1, i2 = NG + 1, NG + n; dx = 1.0 / n
    xc(k) = (k - 0.5) * dx
    d = zeros(idim); e = zeros(idim); u = zeros(idim); v = zeros(idim); w = zeros(idim); p = zeros(idim)
    for k in 1:n
        ρ0, p0 = xc(k) < 0.5 ? (1.0, 1.0) : (0.125, 0.1)
        d[NG + k] = ρ0; p[NG + k] = p0; e[NG + k] = p0 / ((γ - 1) * ρ0)
    end
    refresh!(a) = (for g in 1:NG; a[g] = a[i1]; a[i2 + g] = a[i2]; end)
    t = 0.0
    while t < tfinal
        for a in (d, e, u, v, w); refresh!(a); end
        @. p = (γ - 1) * d * (e - 0.5 * (u^2 + v^2 + w^2)); refresh!(p)
        amax = maximum(abs(u[k]) + sqrt(γ * max(p[k], 1e-20) / d[k]) for k in i1:i2)
        dt = cfl * dx / amax; t + dt > tfinal && (dt = tfinal - t)
        EnzoLib.ppm_sweep_1d!(d, e, u, v, w, p; i1 = i1, i2 = i2, dx = dx, dt = dt, gamma = γ); t += dt
    end
    return d[i1:i2]
end

# Reference: run the Python driver over the same legacy kernel, capture density.
function reference_ppm_sod()
    py = something(filter(isfile, ["/Users/tabel/Projects/veusz/.venv/bin/python",
                                   Sys.which("python3")])..., nothing)
    py === nothing && error("no python found to run the reference driver")
    out = tempname()
    code = """
import sys; sys.path.insert(0, r'$(joinpath(REPO, "EnzoModules"))')
from enzomodules.examples import ppm_sod
g, n = ppm_sod.run(0.2, nx=200, cfl=0.4)
open(r'$out','w').write(' '.join(repr(x) for x in g.active(g.d)))
"""
    withenv("ENZOMODULES_LIB" => EnzoLib.libpath()) do
        run(`$py -c $code`)
    end
    return parse.(Float64, split(read(out, String)))
end

dj = julia_ppm_sod()
dref = reference_ppm_sod()
linf = maximum(abs.(dj .- dref))
nexact = count(dj .== dref)
println("Sod shock tube, 200 cells, t=0.2, legacy Enzo PPM kernel")
println("  Julia driver vs reference driver:")
println("    Linf            = ", linf)
println("    bit-identical   = ", nexact, "/", length(dj), " cells")
println("    ⇒ ", linf < 1e-12 ? "IDENTICAL to machine precision ✓" : "DIVERGENT ✗")
