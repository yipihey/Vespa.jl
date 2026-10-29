# BlockAMR GLM-MHD vs NestedMHD (the scheme it was ported from): single-level 2-D
# Orszag–Tang in Float64 (GLM, or CT from the same A_z), the same step schedule (NestedMHD's evened-out CFL steps;
# c_h from the CFL step; sigdims = 3 because NestedMHD's unsplit CFL counts z even
# in 2-D).  First produce the reference with bench/xcheck_nestedmhd_ref.jl, then:
#   julia --project=lib/BlockAMR/test lib/BlockAMR/bench/xcheck_nestedmhd.jl [refdir] [hlld|hll] [glm|ct]
# Expected (Oct 2026; GLM and CT, HLLD and HLL): identical step counts, max |Δ| ≲ 3e-8
# in every field (CT: also the face B) — the Float32 rounding of BlockAMR's
# level-uniform λ = dt/dx; BlockAMR CT ∇·B ~ 2e-15.
using BlockAMR, Printf, Serialization
const BA = BlockAMR
refdir = get(ARGS, 1, tempdir()); riemann = Symbol(get(ARGS, 2, "hlld")); mode = Symbol(get(ARGS, 3, "glm"))
N = 64; tend = 0.1; b0 = 1 / sqrt(4π)
ot(x) = (25/(36π), -sin(2π*x[2]), sin(2π*x[1]), 0.0, 5/(12π), -b0*sin(2π*x[2]), b0*sin(4π*x[1]), 0.0)
ref = deserialize(joinpath(refdir, "nested_ot_$(mode)_$(riemann).jls"))
h = AMRHierarchy(; nbase = (N, N), B = 16, physics = :mhd, riemann, T = Float64, sigdims = 3,
                 divb = mode)
lev = init_base_level!(h); build_level_tables!(h, 0)
if mode === :ct      # faces from the same A_z NestedMHD uses; E from the face-average B
    set_ct_faces_from_Az!(h, 0, (x, y) -> b0 * (cos(4π*x)/(4π) + cos(2π*y)/(2π)))
end
hbx = Array(lev.Bx); hby = Array(lev.By)
n = lev.cap * lev.stride; hh = [zeros(n) for _ in 1:10]; γ = h.gamma; dx = BA.level_dx(h, 0)
for s in lev.live, j in 0:15, i in 0:15
    m = lev.meta[s]; idx = (Int(s) - 1) * lev.stride + (j + 2) * lev.nd + (i + 2) + 1
    ρ, u, v, w, P, bx, by, bz = ot(((Int(m.origin[1]) + i + 0.5) * dx, (Int(m.origin[2]) + j + 0.5) * dx))
    mode === :ct && (bx = hbx[idx]; by = hby[idx])
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
    gi = Int(m.origin[1]) + i + 3; gj = Int(m.origin[2]) + j + 3
    for v in 1:size(ref.U, 4)          # CT: 8 (no ψ)
        r = ref.U[gi, gj, 3, v]
        # CT cell B: NestedMHD's saved last-column B_cc averages a periodic ghost face
        # that is refreshed only at its next ghost fill — compare to its periodic face mean
        mode === :ct && v == 6 && (r = 0.5 * (ref.bx[gi, gj, 3] + ref.bx[mod(gi - 2, N) + 3, gj, 3]))
        mode === :ct && v == 7 && (r = 0.5 * (ref.by[gi, gj, 3] + ref.by[gi, mod(gj - 2, N) + 3, 3]))
        worst[v] = max(worst[v], abs(f[v][idx] - r))
    end
end
@printf("riemann=%s  steps: NestedMHD %d, BlockAMR %d\n", riemann, ref.nsteps, nst)
for (v, nm) in enumerate(("ρ", "ρvx", "ρvy", "ρvz", "E", "Bx", "By", "Bz", "ψ"))
    (mode === :ct && v == 9) && continue
    @printf("  max|Δ%-3s| = %.3e\n", nm, worst[v])
end
if mode === :ct
    fx = Array(lev.FBx); fy = Array(lev.FBy); wf = 0.0
    for s in lev.live, j in 0:15, i in 0:15
        m = lev.meta[s]; idx = (Int(s) - 1) * lev.stride + (j + 2) * lev.nd + (i + 2) + 1
        gi = Int(m.origin[1]) + i + 3; gj = Int(m.origin[2]) + j + 3
        global wf = max(wf, abs(fx[idx] - ref.bx[gi, gj, 3]), abs(fy[idx] - ref.by[gi, gj, 3]))
    end
    @printf("  max|Δ face B| = %.3e   BlockAMR ∇·B = %.2e\n", wf, max_divb(h))
end
