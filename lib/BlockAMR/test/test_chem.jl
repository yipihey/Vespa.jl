# test_chem.jl — the analytic H+H2 chemistry hook: block-batched results match a
# direct ChemistryKernels call on the same physical cells (to f16/codec
# quantization), and Tau moves by exactly ΔGe (dual-energy consistency).
import ChemistryKernels
const CKm = ChemistryKernels

for BE in BACKENDS
@testset "chem_level! matches direct ChemistryKernels [$BE]" begin
    MH = 1.6726e-24; KB = 1.380649e-16; FH = 0.76; GAM = 5/3
    z = 30.0; nH = 1.0e-1                              # cosmic gas, physical CGS
    T0 = 100.0; xe = 2e-4; xh2 = 1e-6
    hier = AMRHierarchy(; nbase = (32, 32, 32), B = 16, backend = BE,
                        T = Float16, nsp = 2, gamma = GAM)
    lev0 = init_base_level!(hier)
    p0 = lev0.byorigin[(UInt128(0), UInt128(0), UInt128(0))]
    add_block!(hier, 1, p0, (4, 4, 4))
    build_level_tables!(hier, 0); build_level_tables!(hier, 1)
    rho = nH * MH / FH
    eint = KB * T0 / ((GAM - 1) * 1.22 * MH)
    for l in 0:1
        lev = hier.levels[l+1]
        n = lev.cap * lev.stride
        hD = fill(0.0, n); hG = fill(0.0, n); hz = zeros(n)
        for s in lev.live, c in ((Int(s)-1)*lev.stride+1):((Int(s)-1)*lev.stride+lev.nd^3)
            hD[c] = rho; hG[c] = rho * eint
        end
        BlockAMR.encode_from_host!(lev, hD, hz, hz, hz, hG, hG)  # Tau = Ge (at rest)
        hh = fill(CKm.encode_log2sp(Float32(xe * FH)), n)        # HII mass fraction ≈ xe·FH
        hm = fill(CKm.encode_log2sp(Float32(2 * xh2 * FH)), n)
        copyto!(lev.sp[1], hh); copyto!(lev.sp[2], hm)
    end
    dt = 1e13
    for l in 0:1
        chem_level!(hier, l, dt; a_value = 1/(1+z), density_units = 1.0,
                    length_units = 1.0, time_units = 1.0, fh = FH)
    end
    # direct reference on ONE identical cell
    r = [Float32(rho)]; e = [Float32(eint)]
    h = [CKm.encode_log2sp(Float32(xe * FH))]; m = [CKm.encode_log2sp(Float32(2 * xh2 * FH))]
    CKm.solve_chem_analytic_u16!(r, e, h, m; a_value = 1/(1+z), dt = dt,
        density_units = 1.0, length_units = 1.0, time_units = 1.0, fh = FH,
        backend = :cpu, precision = Float32)
    for l in 0:1
        lev = hier.levels[l+1]
        s = lev.live[1]; base = (Int(s)-1)*lev.stride
        idx = base + (lev.ng*lev.nd + lev.ng)*lev.nd + lev.ng + 1
        esc = Array(lev.Esc)[s]
        ge  = Float64(Array(lev.Ge)[idx]) * esc
        tau = Float64(Array(lev.Tau)[idx]) * esc
        @test isapprox(ge / rho, Float64(e[1]); rtol = 5e-3)      # f16 quantization
        @test isapprox(tau, ge; rtol = 5e-3)                      # ΔTau ≡ ΔGe (at rest)
        hII = CKm.decode_log2sp(Float64, Array(lev.sp[1])[idx])
        h2  = CKm.decode_log2sp(Float64, Array(lev.sp[2])[idx])
        @test isapprox(hII, CKm.decode_log2sp(Float64, h[1]); rtol = 5e-3)
        @test isapprox(h2,  CKm.decode_log2sp(Float64, m[1]); rtol = 5e-3)
    end
end
end # for BE

for BE in BACKENDS
@testset "species CMA: pulse advects with the flow [$BE]" begin
    hier = AMRHierarchy(; nbase = (32, 32, 32), B = 16, backend = BE, nsp = 2)
    lev = init_base_level!(hier); build_level_tables!(hier, 0)
    n = lev.cap * lev.stride
    hD = fill(1.0, n); hS = fill(0.5, n); hz = zeros(n)
    hG = fill(1.5, n); hT = fill(1.5 + 0.5 * 0.25, n)
    BlockAMR.encode_from_host!(lev, hD, hS, hz, hz, hT, hG)
    # X2: Gaussian pulse in x at x0 = 0.25
    hsp = fill(CKm.encode_log2sp(1.0f-6), n)
    for s in lev.live
        m = lev.meta[s]; base = (Int(s)-1)*lev.stride
        for k in 1:16, j in 1:16, i in 1:16
            x = (Float64(m.origin[1]) + i - 0.5) / 32
            X = 1e-6 + 0.01 * exp(-(mod(x - 0.25 + 0.5, 1.0) - 0.5)^2 / (2 * 0.06^2))
            hsp[base + ((lev.ng+k-1)*lev.nd + (lev.ng+j-1))*lev.nd + (lev.ng+i-1) + 1] =
                CKm.encode_log2sp(Float32(X))
        end
    end
    copyto!(lev.sp[2], hsp)
    copyto!(lev.sp[1], fill(CKm.encode_log2sp(0.76f0), n))
    t = 0.0
    for _ in 1:60
        dt = compute_dt(hier)
        hierarchy_rk2_step!(hier, dt); t += dt
    end
    # peak location tracks x0 + 0.5·t
    hs = Array(lev.sp[2]); best = 0.0; xb = 0.0
    for s in lev.live
        m = lev.meta[s]; base = (Int(s)-1)*lev.stride
        for i in 1:16
            X = CKm.decode_log2sp(Float64,
                hs[base + ((lev.ng+8-1)*lev.nd + (lev.ng+8-1))*lev.nd + (lev.ng+i-1) + 1])
            X > best && (best = X; xb = (Float64(m.origin[1]) + i - 0.5) / 32)
        end
    end
    xexp = mod(0.25 + 0.5 * t, 1.0)
    @test best > 3e-3                                   # pulse survived
    @test abs(mod(xb - xexp + 0.5, 1.0) - 0.5) < 0.08   # tracked the flow
end
end # for BE
