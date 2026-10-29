# BlockAMR GLM-MHD vs NestedMHD (the scheme it was ported from): single-level 2-D
# Orszag–Tang in Float64, the same step schedule (NestedMHD's evened-out CFL steps;
# c_h from the CFL step; sigdims = 3 because NestedMHD's unsplit CFL counts z even
# in 2-D).  First produce the reference with bench/xcheck_nestedmhd_ref.jl, then:
#   julia --project=lib/BlockAMR/test lib/BlockAMR/bench/xcheck_nestedmhd.jl [refdir] [hlld|hll]
# Expected (Oct 2026, both solvers): identical step counts, max |Δ| ≲ 3e-8 in every
# field — the Float32 rounding of BlockAMR's level-uniform λ = dt/dx.
using BlockAMR, Printf, Serialization
const BA = BlockAMR
refdir = get(ARGS, 1, tempdir()); riemann = Symbol(get(ARGS, 2, "hlld"))
N = 64; tend = 0.1; b0 = 1 / sqrt(4π)
ot(x) = (25/(36π), -sin(2π*x[2]), sin(2π*x[1]), 0.0, 5/(12π), -b0*sin(2π*x[2]), b0*sin(4π*x[1]), 0.0)
ref = deserialize(joinpath(refdir, "nested_ot_$(riemann).jls"))
h = AMRHierarchy(; nbase = (N, N), B = 16, physics = :mhd, riemann, T = Float64, sigdims = 3)
lev = init_base_level!(h); build_level_tables!(h, 0)
n = lev.cap * lev.stride; hh = [zeros(n) for _ in 1:10]; γ = h.gamma; dx = BA.level_dx(h, 0)
for s in lev.live, j in 0:15, i in 0:15
    m = lev.meta[s]; idx = (Int(s) - 1) * lev.stride + (j + 2) * lev.nd + (i + 2) + 1
    ρ, u, v, w, P, bx, by, bz = ot(((Int(m.origin[1]) + i + 0.5) * dx, (Int(m.origin[2]) + j + 0.5) * dx))
    ge = P / (γ - 1)
    for (f, x) in enumerate((ρ, ρ*u, ρ*v, ρ*w, ge + 0.5ρ*(u^2+v^2+w^2) + 0.5(bx^2+by^2+bz^2), ge, bx, by, bz, 0.0))
        hh[f][idx] = x
    end
end
encode_mhd_from_host!(lev, hh...)
t = 0.0; nst = 0
while t < tend - 1e-12
    λc = compute_lambda!(h)                     # cfl / s_max (c_h comes from this)
    rem = tend - t; dt = rem / max(1, ceil(Int, rem / (Float64(λc) * dx) - 1e-9))
    BA.advance_level_w!(h, 0, Float32(dt / dx))
    global t += dt; global nst += 1
end
f = [Array(getfield(lev, s)) for s in (:D, :S1, :S2, :S3, :Tau, :Bx, :By, :Bz, :Psi)]
worst = zeros(9)
for s in lev.live, j in 0:15, i in 0:15
    m = lev.meta[s]; idx = (Int(s) - 1) * lev.stride + (j + 2) * lev.nd + (i + 2) + 1
    for v in 1:9
        worst[v] = max(worst[v], abs(f[v][idx] - ref.U[Int(m.origin[1]) + i + 3, Int(m.origin[2]) + j + 3, 3, v]))
    end
end
@printf("riemann=%s  steps: NestedMHD %d, BlockAMR %d\n", riemann, ref.nsteps, nst)
for (v, nm) in enumerate(("ρ", "ρvx", "ρvy", "ρvz", "E", "Bx", "By", "Bz", "ψ"))
    @printf("  max|Δ%-3s| = %.3e\n", nm, worst[v])
end
