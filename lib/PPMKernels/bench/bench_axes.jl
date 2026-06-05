# Per-axis micro-benchmark: is the y/z transpose a real cost?
# Times sweep_axis! for axis 1 (contiguous, NO transpose) vs axis 2/3 (transpose
# the swept axis to the lead). If 2/3 ≫ 1, transposes dominate and are worth
# attacking; if similar, the sweep compute dominates and elimination won't help.

using PPMKernels
using Printf
try; @eval using Metal; catch; end

const NG = 4
const GAMMA = 1.4
const FLAGS = (idual = 1, iflatten = 3, isteep = 0, idiff = 0, gravity = 0, eta2 = 0.1)

function ic(n)
    N = n + 2NG; tot = N^3
    d = ones(tot); p = fill(1.0 / GAMMA, tot)
    v() = 0.1 .* (rand(tot) .- 0.5)
    vx, vy, vz = v(), v(), v()
    ge = p ./ ((GAMMA - 1) .* d)
    e = ge .+ 0.5 .* (vx .^ 2 .+ vy .^ 2 .+ vz .^ 2)
    (; d, e, ge, vx, vy, vz, p, gr = zeros(tot), N, dims = (N, N, N))
end

function time_axis(be_name, n, axis, reps)
    be = PPMKernels.backend(be_name)
    s = ic(n); dev(a) = PPMKernels.to_device(be, a, Float32)
    d, e, ge = dev(s.d), dev(s.e), dev(s.ge)
    vx, vy, vz = dev(s.vx), dev(s.vy), dev(s.vz)
    p, gr = dev(s.p), dev(s.gr)
    go() = sweep_axis!(d, e, ge, vx, vy, vz, p, gr, s.dims, NG, axis;
                       dt = 0.2 / n, gamma = GAMMA, dx = 1.0 / n, FLAGS...)
    PPMKernels.with_pool() do
        go()                                                # warm
        PPMKernels.to_host(d)                               # sync
        t = @elapsed (for _ in 1:reps; go(); end; PPMKernels.to_host(d))
        t / reps
    end
end

n = isempty(ARGS) ? 128 : parse(Int, ARGS[1])
reps = 20
@printf("\nper-axis sweep_axis! @ %d^3 (Metal f32), %d reps\n", n, reps)
@printf("%-8s %-12s %-10s\n", "axis", "sec/sweep", "vs x")
println("-"^32)
PPMKernels.has_backend(:metal) || (println("no GPU"); exit())
tx = time_axis(:metal, n, 1, reps)
for (ax, name) in ((1, "x (none)"), (2, "y (transp)"), (3, "z (transp)"))
    t = time_axis(:metal, n, ax, reps)
    @printf("%-8s %-12.5g %-10s\n", name, t, @sprintf("%.2f×", t / tx))
end
println()
