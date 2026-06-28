# reflux_capture.jl: extracting coarse–fine reflux registers from the recorded `frec`.
#
# Two gates, both backend-agnostic (CPU always; Metal f32 when present):
#   1. INDEX MAPPING — the KA gather `boundary_flux_register` must equal the independent
#      host-loop `boundary_flux_register_ref` key-for-key, value-for-value. A wrong
#      stride/offset in the gather kernels fails here (it would NOT crash).
#   2. SIGN + UNITS — the outer-boundary register must satisfy the conservation identity
#      ΔQ[f] = Σ_lo flux[f] − Σ_hi flux[f], where ΔQ[f] = Σ_active (U1−U0)[f]·dx³ is the
#      extensive change the step produced. This is the same identity Vespa's native
#      BoundaryFluxRegister obeys (src/reflux.jl), and is what makes the AMR substrate's
#      CorrectForRefinedFluxes restore conservation.

using PPMKernels: boundary_flux_register, boundary_flux_register_ref, BoundaryFluxSet

@testset "reflux capture (frec → boundary flux register)" begin
    ng = 3; n = 12; nb = n + 2ng; dims = (nb, nb, nb); N = nb^3
    gamma = 1.4; dx = 1.0 / n; dt = 0.008
    nx, ny, nz = dims; idx(i, j, k) = i + nx * (j - 1) + nx * ny * (k - 1)

    # smooth, fully 3-D initial state (no symmetry that could hide an axis bug)
    rho = zeros(N); vx = zeros(N); vy = zeros(N); vz = zeros(N); etot = zeros(N)
    for k in 1:nz, j in 1:ny, i in 1:nx
        x = (i - ng - 0.5) / n; y = (j - ng - 0.5) / n; z = (k - ng - 0.5) / n; q = idx(i, j, k)
        rho[q] = 1.0 + 0.3 * sinpi(2x) * cospi(2y); pr = 0.6 + 0.2 * cospi(2z)
        vx[q] = 0.2 * sinpi(2y); vy[q] = 0.15 * cospi(2z); vz[q] = -0.1 * sinpi(2x)
        etot[q] = pr / ((gamma - 1) * rho[q]) + 0.5 * (vx[q]^2 + vy[q]^2 + vz[q]^2)
    end
    U0 = (copy(rho), rho .* vx, rho .* vy, rho .* vz, rho .* etot)

    # run the KA hydro once (CPU f64), recording fluxes
    be = PPMKernels.backend(:cpu)
    D  = PPMKernels.to_device(be, U0[1], Float64); S1 = PPMKernels.to_device(be, U0[2], Float64)
    S2 = PPMKernels.to_device(be, U0[3], Float64); S3 = PPMKernels.to_device(be, U0[4], Float64)
    Tau = PPMKernels.to_device(be, U0[5], Float64)
    frec = ntuple(_ -> ntuple(_ -> PPMKernels.device_zeros(be, Float64, (N,)), 6), 3)
    PPMKernels.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, dims, ng;
                                      dt = dt, gamma = gamma, dx = dx, recon = :plm, fluxrec = frec)
    U1 = map(PPMKernels.to_host, (D, S1, S2, S3, Tau))

    reg  = boundary_flux_register(frec, dims, ng, dt, dx)        # KA gather
    refr = boundary_flux_register_ref(frec, dims, ng, dt, dx)    # host-loop oracle

    @testset "KA gather == host reference (index mapping)" begin
        @test reg.act == refr.act
        @test keys(reg.flux)     == keys(refr.flux)
        @test keys(reg.interior) == keys(refr.interior)
        worst = 0.0
        for (k, v) in reg.flux;     worst = max(worst, maximum(abs.(v .- refr.flux[k])));     end
        for (k, v) in reg.interior; worst = max(worst, maximum(abs.(v .- refr.interior[k]))); end
        @test worst < 1e-14
    end

    @testset "boundary register obeys the conservation identity (sign + units)" begin
        # ΔQ[f] = Σ_active (U1−U0)[f]·dx³  must equal  Σ_lo flux[f] − Σ_hi flux[f]
        Vcell = dx^3
        for f in 1:5
            dQ = 0.0
            for k in ng+1:nz-ng, j in ng+1:ny-ng, i in ng+1:nx-ng
                c = idx(i, j, k); dQ += (U1[f][c] - U0[f][c]) * Vcell
            end
            slo = 0.0; shi = 0.0
            for ((_, side, _), v) in reg.flux
                side === :lo ? (slo += v[f]) : (shi += v[f])
            end
            @test isapprox(dQ, slo - shi; atol = 1e-13, rtol = 1e-11)
        end
    end

    # GPU index-mapping lock: gather the SAME recorded fluxes on the device and assert the
    # device gather matches the host-loop reference (f32 tolerance). Skips with no Metal.
    if metal_ready()
        @testset "Metal f32 gather == reference" begin
            beg = PPMKernels.backend(:metal)
            Dg  = PPMKernels.to_device(beg, U0[1], Float32); S1g = PPMKernels.to_device(beg, U0[2], Float32)
            S2g = PPMKernels.to_device(beg, U0[3], Float32); S3g = PPMKernels.to_device(beg, U0[4], Float32)
            Taug = PPMKernels.to_device(beg, U0[5], Float32)
            frg = ntuple(_ -> ntuple(_ -> PPMKernels.device_zeros(beg, Float32, (N,)), 6), 3)
            PPMKernels.muscl_hancock_step_3d!(Dg, S1g, S2g, S3g, Taug, dims, ng;
                                              dt = dt, gamma = gamma, dx = dx, recon = :plm, fluxrec = frg)
            rg = boundary_flux_register(frg, dims, ng, dt, dx)       # device gather
            rr = boundary_flux_register_ref(frg, dims, ng, dt, dx)   # host loop over same device data
            @test keys(rg.flux) == keys(rr.flux) && keys(rg.interior) == keys(rr.interior)
            worst = 0.0
            for (k, v) in rg.flux;     worst = max(worst, maximum(abs.(v .- rr.flux[k])));     end
            for (k, v) in rg.interior; worst = max(worst, maximum(abs.(v .- rr.interior[k]))); end
            @test worst < 1e-5
        end
    end
end
