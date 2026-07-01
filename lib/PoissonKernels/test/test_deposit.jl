# cic_deposit! — KA Cloud-In-Cell particle→grid mass scatter.
#   * CPU f64 kernel ≡ a scalar reference loop (exact);
#   * mass conservation Σρ == Σmass (CIC = partition of unity);
#   * single-particle weights match the analytic trilinear weights at shift=−0.5;
#   * Metal f32 ≡ CPU f32 (atomic-scatter parity, skips without a GPU).

@testset "cic_deposit! — KA Cloud-In-Cell scatter" begin
    N = 16

    # reference scalar CIC (same registration: g = mod(pos+disp·v,1)·N + shift)
    function ref_cic(pos, vel, mass, N, disp, shift, ::Type{T}) where {T}
        ρ = zeros(T, N, N, N)
        @inbounds for p in 1:length(mass)
            gx = mod(T(pos[p,1]) + T(disp)*T(vel[p,1]), one(T))*N + T(shift)
            gy = mod(T(pos[p,2]) + T(disp)*T(vel[p,2]), one(T))*N + T(shift)
            gz = mod(T(pos[p,3]) + T(disp)*T(vel[p,3]), one(T))*N + T(shift)
            i0 = floor(Int,gx); fx = gx-i0; j0 = floor(Int,gy); fy = gy-j0; k0 = floor(Int,gz); fz = gz-k0
            m = T(mass[p])
            for dk in 0:1, dj in 0:1, di in 0:1
                wi = di==0 ? one(T)-fx : fx; wj = dj==0 ? one(T)-fy : fy; wk = dk==0 ? one(T)-fz : fz
                ρ[mod(i0+di,N)+1, mod(j0+dj,N)+1, mod(k0+dk,N)+1] += m*wi*wj*wk
            end
        end
        ρ
    end

    run_dep(name, ::Type{T}, pos, vel, mass; disp, shift) where {T} = begin
        be = PoissonKernels.backend(name)
        d(x) = PoissonKernels.to_device(be, x, T)
        ρ = PoissonKernels.device_zeros(be, T, (N*N*N,))
        PoissonKernels.cic_deposit!(ρ, d(pos[:,1]),d(pos[:,2]),d(pos[:,3]),
                                    d(vel[:,1]),d(vel[:,2]),d(vel[:,3]), d(mass);
                                    N=N, disp=disp, shift=shift)
        reshape(PoissonKernels.to_host(ρ), N, N, N)
    end

    np = 4000
    rng_pos = [ (sin(0.7p)*0.5+0.5, cos(1.3p)*0.5+0.5, sin(2.1p+1)*0.5+0.5)  for p in 1:np ]
    pos = reduce(vcat, ([a b c] for (a,b,c) in rng_pos))
    vel = 0.01 .* (pos .- 0.5)
    mass = fill(0.83, np)

    # (1) CPU f64 ≡ reference, with the production registration (drift + shift −0.5)
    disp, shift = 0.013, -0.5
    ρcpu = run_dep(:cpu, Float64, pos, vel, mass; disp=disp, shift=shift)
    ρref = ref_cic(pos, vel, mass, N, disp, shift, Float64)
    @test maximum(abs.(ρcpu .- ρref)) < 1e-10

    # (2) mass conservation (periodic CIC is a partition of unity)
    @test isapprox(sum(ρcpu), sum(mass); rtol=1e-12)

    # (3) single particle at a cell centre, shift=−0.5, no drift → all mass in one cell
    p1 = reshape([ (3+0.5)/N, (5+0.5)/N, (7+0.5)/N ], 1, 3)
    ρ1 = run_dep(:cpu, Float64, p1, zeros(1,3), [1.0]; disp=0.0, shift=-0.5)
    @test isapprox(ρ1[4,6,8], 1.0; atol=1e-12)        # cell-centre ↦ exactly its cell
    @test isapprox(sum(ρ1), 1.0; atol=1e-12)

    # (4) Metal f32 ≡ CPU f32 (atomic-scatter parity)
    if metal_ready()
        ρm = run_dep(:metal, Float32, pos, vel, mass; disp=disp, shift=shift)
        ρc = run_dep(:cpu,   Float32, pos, vel, mass; disp=disp, shift=shift)
        rel = maximum(abs.(ρm .- ρc)) / (maximum(abs.(ρc)) + 1f-30)
        @info "cic_deposit Metal≡CPU f32" maxrel=rel
        @test rel < 1f-4

        run_det_f16(name) = begin
            be = PoissonKernels.backend(name)
            d32(x) = PoissonKernels.to_device(be, x, Float32)
            d16(x) = PoissonKernels.to_device(be, Float16.(x), Float16)
            ρi = PoissonKernels.device_zeros(be, Int32, (N*N*N,))
            PoissonKernels.cic_deposit_det!(ρi, d32(pos[:,1]), d32(pos[:,2]), d32(pos[:,3]),
                                            d16(vel[:,1]), d16(vel[:,2]), d16(vel[:,3]), Float32(0.83);
                                            N=N, disp=disp, shift=shift, qbits=12)
            reshape(PoissonKernels.to_host(ρi), N, N, N)
        end
        @test run_det_f16(:metal) == run_det_f16(:cpu)
    else
        @test_skip "Metal not available — GPU deposit parity skipped"
    end
end
