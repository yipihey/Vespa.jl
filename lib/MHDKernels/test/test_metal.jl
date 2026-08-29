using Metal, MHDKernels, KernelAbstractions, FFTW, Test, Printf
import PoissonKernels

const KA = KernelAbstractions
const T = Float32

if !Metal.functional()
    @info "No functional Metal device — Metal tests skipped."
else
    be = backend(:metal)
    becpu = backend(:cpu)
    @info "Metal backend present"

    @testset "MHDKernels Metal" begin
        @testset "rectangular cube launch geometry" begin
            shared_bytes = sizeof(Float32) *
                (9*MHDKernels.RCNCP + 18*(MHDKernels.RCNFX +
                                           MHDKernels.RCNFY +
                                           MHDKernels.RCNFZ))
            @test shared_bytes == 23_040
            @test shared_bytes <= Int(Metal.device().maxThreadgroupMemoryLength)
            @test MHDKernels.METAL_CUBE_GS == 128
        end

        @testset "density-aware sub-ULP increments" begin
            perturbation = zeros(Float32, 1)
            rho = 1f0
            naive = 1f0
            drho = 1f-8
            for _ in 1:32
                rho = MHDKernels._cube_density_update!(perturbation, 1, rho, drho,
                                                       Val(true))
                naive += drho
            end
            @test perturbation[1] ≈ 32f0*drho rtol=2f-6
            @test rho == 1f0
            @test naive == 1f0
        end

        @testset "magnetic compensated sub-ULP increments" begin
            residual = zeros(Float32,1)
            compensated = 16f0
            naive = 16f0
            db = 1f-7
            for _ in 1:64
                compensated = MHDKernels._cube_magnetic_update!(
                    residual,1,compensated,db,Val(true))
                naive += db
            end
            @test compensated == Float32(16 + 64e-7)
            @test naive == 16f0
            @test abs(residual[1]) <= eps(16f0)
        end

        @testset "CPU ref ≈ Metal cube (one step, 32^3 f32) — $recon" for recon in (:plm, :ppm)
            N = 32
            sr = allocate_state(becpu, T, (N,N,N); dx=1/N, gamma=5/3, riemann=:hll, recon=recon)
            sc = allocate_state(be, T, (N,N,N); dx=1/N, gamma=5/3, riemann=:hll, recon=recon)
            init_turb_field!(sr); init_turb_field!(sc)
            dt, smax = compute_dt(sr; cfl=0.4)
            step!(sr, dt; ch=smax, integrator=:ref)
            step!(sc, dt; ch=smax, integrator=:cube)
            hr = fields_to_host(sr); hc = fields_to_host(sc)
            maxabs = 0.0; scale = 0.0
            for v in 1:9
                maxabs = max(maxabs, maximum(abs.(Float64.(hr[v]) .- Float64.(hc[v]))))
                scale = max(scale, maximum(abs.(Float64.(hr[v]))))
            end
            rel = maxabs / scale
            @printf("  [Metal %s] max|CPUref-cube| = %.3e  (rel %.3e)\n", recon, maxabs, rel)
            @test isfinite(rel)
            @test rel < 2e-2
        end

        @testset "cube multi-step stability + conservation (64^3)" begin
            N = 64
            s = allocate_state(be, T, (N,N,N); dx=1/N, gamma=5/3, riemann=:hll)
            init_turb_field!(s)
            t0 = conserved_totals(s)
            t, n = evolve!(s, 0.02; cfl=0.4, integrator=:cube)
            t1 = conserved_totals(s)
            h = fields_to_host(s)
            dmass = abs(t1.mass-t0.mass)/abs(t0.mass)
            denergy = abs(t1.energy-t0.energy)/abs(t0.energy)
            @printf("  [Metal cube] t=%.4f (%d steps): finite=%s Δmass=%.2e Δenergy=%.2e\n",
                    t, n, all(isfinite, h[1]), dmass, denergy)
            @test all(isfinite, h[1])
            @test minimum(h[1]) > 0
            @test dmass < 1e-4
            @test denergy < 1e-4
        end

        @testset "terminal CT CPU/Metal parity and divB" begin
            N = 20
            function seed_terminal_ct!(s)
                rho = Vector{T}(undef, N^3)
                Bx = similar(rho); By = similar(rho); Bz = similar(rho); E = similar(rho)
                for k in 1:N, j in 1:N, i in 1:N
                    c = linidx(N, N, i, j, k)
                    x = T(2pi * (i - 1) / N)
                    y = T(2pi * (j - 1) / N)
                    z = T(2pi * (k - 1) / N)
                    rho[c] = 1f0 + 0.02f0 * sin(x + y)
                    Bx[c] = 0.08f0 * sin(y)
                    By[c] = 0.08f0 * sin(z)
                    Bz[c] = 0.08f0 * sin(x)
                    E[c] = 1.5f0 + 0.5f0 * (Bx[c]^2 + By[c]^2 + Bz[c]^2)
                end
                copyto!(s.U[1], rho); copyto!(s.U[5], E)
                copyto!(s.U[6], Bx); copyto!(s.U[7], By); copyto!(s.U[8], Bz)
                KA.synchronize(s.be)
            end
            sr = allocate_state(becpu, T, (N,N,N); dx=1/N)
            sm = allocate_state(be, T, (N,N,N); dx=1/N)
            seed_terminal_ct!(sr); seed_terminal_ct!(sm)
            kr = terminal_ct_induction!(sr, 0.004; gamma_drag=4, pressure_coeff=0.2)
            km = terminal_ct_induction!(sm, 0.004; gamma_drag=4, pressure_coeff=0.2)
            hr = fields_to_host(sr); hm = fields_to_host(sm)
            rel = maximum(abs, Float64.(hm[6]) .- Float64.(hr[6])) /
                  maximum(abs, Float64.(hr[6]))
            @printf("  [Metal terminal CT] CPU/Metal rel=%.3e divBdx/B=%.3e nsub=%d/%d\n",
                    rel, Float64(max_divb(sm))*Float64(sm.dx)/0.08, kr[1], km[1])
            @test kr[1] == km[1]
            @test rel < 2e-5
            @test Float64(max_divb(sm))*Float64(sm.dx)/0.08 < 2e-5
        end

        @testset "exact drag CPU/Metal parity" begin
            N = 16
            sr = allocate_state(becpu, T, (N,N,N); dx=1/N)
            sm = allocate_state(be, T, (N,N,N); dx=1/N)
            for s in (sr, sm)
                fill!(s.U[1], 2f0); fill!(s.U[2], 3f0); fill!(s.U[3], -2f0)
                fill!(s.U[4], 1f0); fill!(s.U[5], 8f0)
                apply_linear_drag!(s, exp(-7); target_velocity=(0.25, -0.5, 0.125))
            end
            hr = fields_to_host(sr); hm = fields_to_host(sm)
            @test maximum(abs, hm[2] .- hr[2]) == 0
            @test maximum(abs, hm[5] .- hr[5]) == 0
        end

        @testset "source-aware Godunov drag CPU ref / Metal cube parity" begin
            N = 32
            sr = allocate_state(becpu, T, (N,N,N); dx=1/N, gamma=5/3,
                                riemann=:hll, recon=:ppm)
            sm = allocate_state(be, T, (N,N,N); dx=1/N, gamma=5/3,
                                riemann=:hll, recon=:ppm)
            init_turb_field!(sr); init_turb_field!(sm)
            dt, smax = compute_dt(sr; cfl=0.1)
            impulse = 2f0
            step_drag_godunov!(sr, dt; drag_impulse=impulse, ch=smax,
                               glm_cr=0.18, integrator=:ref)
            step_drag_godunov!(sm, dt; drag_impulse=impulse, ch=smax,
                               glm_cr=0.18, integrator=:cube)
            hr = fields_to_host(sr); hm = fields_to_host(sm)
            maxabs = 0.0; scale = 0.0
            for v in 1:9
                maxabs = max(maxabs, maximum(abs.(Float64.(hr[v]) .- Float64.(hm[v]))))
                scale = max(scale, maximum(abs.(Float64.(hr[v]))))
            end
            rel = maxabs / scale
            @printf("  [Metal source-aware drag] q=%.1f maxabs=%.3e rel=%.3e\n",
                    impulse, maxabs, rel)
            @test isfinite(rel)
            @test rel < 2e-2
            @test minimum(hm[1]) > 0
            @test all(all(isfinite, field) for field in hm)
        end

        @testset "spectral radiation full-MHD CPU ref / Metal cube parity" begin
            N=16
            sr=allocate_state(becpu,T,(N,N,N);dx=1/N,riemann=:hll,recon=:ppm)
            sm=allocate_state(be,T,(N,N,N);dx=1/N,riemann=:hll,recon=:ppm)
            init_turb_field!(sr)
            for v in 1:9 copyto!(sm.U[v],sr.U[v]) end
            wr=allocate_radiation_workspace(sr); wm=allocate_radiation_workspace(sm)
            rc=RadiationClosure(inertia_ratio=0.1f0,mean_free_path=0.01f0,
                                photon_speed=10f0,gas_sound_speed=1f0,
                                response_model=:angular)
            lr=RadiationEnergyLedger(); lm=RadiationEnergyLedger()
            dt,smax=compute_dt(sr;cfl=0.03)
            step_radiation_godunov!(sr,wr,dt;drag_impulse=1f0,closure=rc,
                                    ch=smax,integrator=:ref,energy_ledger=lr)
            step_radiation_godunov!(sm,wm,dt;drag_impulse=1f0,closure=rc,
                                    ch=smax,integrator=:cube,energy_ledger=lm)
            hr=fields_to_host(sr); hm=fields_to_host(sm)
            maxabs=maximum(maximum(abs,hm[v].-hr[v]) for v in 1:9)
            scale=maximum(maximum(abs,hr[v]) for v in 1:9)
            @printf("  [Metal radiation MHD] maxabs=%.3e rel=%.3e\n",maxabs,maxabs/scale)
            @test maxabs/scale < 3f-5
            @test minimum(hm[1])>0
            @test all(all(isfinite,x) for x in hm)
            @test maximum(abs,Array(wm.density_perturbation).-Array(wr.density_perturbation)) < 2f-5
            @test lm.samples == 1
            @test lm.total_change ≈ lr.total_change rtol=2f-3 atol=2f-4
            component_scale=abs(lm.kinetic_change)+abs(lm.internal_change)+
                            abs(lm.magnetic_change)
            @test abs(lm.component_residual)/component_scale < 5e-4
            @test lm.component_residual_abs/component_scale < 1e-3
        end

        @testset "radiation robust override is global minmod-PLM/HLL" begin
            N=32
            so=allocate_state(be,T,(N,N,N);dx=1/N,riemann=:hlld,recon=:ppm)
            sd=allocate_state(be,T,(N,N,N);dx=1/N,riemann=:hll,
                              recon=:plm_minmod)
            init_turb_field!(so)
            for v in 1:9 copyto!(sd.U[v],so.U[v]) end
            wo=allocate_radiation_workspace(so)
            wd=allocate_radiation_workspace(sd)
            fill!(wo.packed_psi_correction,0f0)
            fill!(wd.packed_psi_correction,0f0)
            dt,smax=compute_dt(so;cfl=0.1)
            kwargs=(ch=smax,decay=exp(-0.18f0*smax*dt/so.dx),
                    drag_impulse=0.7f0,track_density=true,
                    check_admissibility=true)
            step_cube_radiation!(so,dt;kwargs...,
                correction=wo.packed_psi_correction,robust_fallback=true)
            step_cube_radiation!(sd,dt;kwargs...,
                correction=wd.packed_psi_correction)
            ho=fields_to_host(so); hd=fields_to_host(sd)
            @test all(ho[v] == hd[v] for v in 1:9)
            @test Array(wo.density_perturbation) ==
                  Array(wd.density_perturbation)
        end

        @testset "evolved photon moments CPU ref / Metal cube parity ($model)" for model in
                (:moments, :moments_exponential)
            N=16
            sr=allocate_state(becpu,T,(N,N,N);dx=1/N,riemann=:hll,recon=:ppm)
            sm=allocate_state(be,T,(N,N,N);dx=1/N,riemann=:hll,recon=:ppm)
            init_turb_field!(sr)
            for v in 1:9 copyto!(sm.U[v],sr.U[v]) end
            wr=allocate_radiation_workspace(sr); wm=allocate_radiation_workspace(sm)
            pr=allocate_photon_moment_state(sr); pm=allocate_photon_moment_state(sm)
            initialize_photon_moments!(pr,wr,sr)
            initialize_photon_moments!(pm,wm,sm)
            rc=RadiationClosure(inertia_ratio=0.1f0,mean_free_path=0.01f0,
                                photon_speed=10f0,gas_sound_speed=1f0,
                                response_model=model)
            dt,smax=compute_dt(sr;cfl=0.02)
            step_radiation_godunov!(sr,wr,dt;drag_impulse=0.5f0,closure=rc,
                                    ch=smax,integrator=:ref,photon_state=pr)
            step_radiation_godunov!(sm,wm,dt;drag_impulse=0.5f0,closure=rc,
                                    ch=smax,integrator=:cube,photon_state=pm)
            hr=fields_to_host(sr); hm=fields_to_host(sm)
            maxabs=maximum(maximum(abs,hm[v].-hr[v]) for v in 1:9)
            scale=maximum(maximum(abs,hr[v]) for v in 1:9)
            phr=Array(pr.hat_batch); phm=to_host(pm.hat_batch)
            prel=maximum(abs,phm.-phr)/max(maximum(abs,phr),1f-12)
            @printf("  [Metal photon moments %s] gas_rel=%.3e photon_rel=%.3e\n",
                    String(model),maxabs/scale,prel)
            @test maxabs/scale < 5f-5
            @test prel < 5f-5
            @test minimum(hm[1])>0
            @test all(all(isfinite,x) for x in hm)
        end

        @testset "midpoint density predictor CPU/Metal parity" begin
            dims = (32, 8, 4)
            sr = allocate_state(becpu, T, dims; dx=1 / 32)
            sm = allocate_state(be, T, dims; dx=1 / 32)
            for c in eachindex(sr.U[1])
                sr.U[1][c] = 1f0 + 0.01f0 * sin(T(c))
                sr.U[2][c] = 0.02f0 * cos(T(c))
                sr.U[3][c] = -0.01f0 * sin(T(2c))
                sr.U[4][c] = 0.03f0 * cos(T(3c))
            end
            for field in 1:4
                copyto!(sm.U[field], sr.U[field])
            end
            predict_density_backward!(sr.scratch[1], sr, 0.002)
            predict_density_backward!(sm.scratch[1], sm, 0.002)
            KA.synchronize(be)
            @test maximum(abs, Array(sm.scratch[1]) .- Array(sr.scratch[1])) < 2f-7
        end

        @testset "gas gravity kick CPU/Metal parity" begin
            dims = (24, 12, 6)
            sr = allocate_state(becpu, T, dims; dx=1 / dims[1])
            sm = allocate_state(be, T, dims; dx=1 / dims[1])
            phi = Vector{T}(undef, prod(dims))
            for c in eachindex(phi)
                sr.U[1][c] = 1f0 + 0.02f0 * sin(T(c))
                sr.U[2][c] = 0.03f0 * cos(T(c)); sr.U[3][c] = -0.02f0
                sr.U[4][c] = 0.01f0; sr.U[5][c] = 2f0
                phi[c] = 0.04f0 * sin(T(c) / 7f0)
            end
            for field in 1:9
                copyto!(sm.U[field], sr.U[field])
            end
            phim = to_device(be, phi)
            apply_cell_center_gravity!(sr, phi, 0.003)
            apply_cell_center_gravity!(sm, phim, 0.003)
            KA.synchronize(be)
            hr = fields_to_host(sr); hm = fields_to_host(sm)
            for field in 2:5
                @test maximum(abs, hm[field] .- hr[field]) < 3f-7
            end
        end

        @testset "lattice displacement CIC CPU/Metal parity" begin
            N = 16
            np = N^3
            dxh = Vector{T}(undef, np)
            for p in 1:np
                i = (p - 1) % N
                dxh[p] = 0.05f0 * sinpi(2f0 * (T(i) + 0.5f0) / T(N)) / T(N)
            end
            dyh = zeros(T, np); dzh = zeros(T, np)
            vxh = fill(2f-7, np); vyh = zeros(T, np); vzh = zeros(T, np)
            rho_cpu = zeros(T, np)
            deposit_lattice_displacements!(rho_cpu, dxh, dyh, dzh, vxh, vyh, vzh;
                                            N=N, disp=0.125)

            dxm = to_device(be, dxh); dym = to_device(be, dyh); dzm = to_device(be, dzh)
            vxm = to_device(be, vxh); vym = to_device(be, vyh); vzm = to_device(be, vzh)
            rho_metal = device_zeros(be, T, (np,))
            deposit_lattice_displacements!(rho_metal, dxm, dym, dzm, vxm, vym, vzm;
                                            N=N, disp=0.125)
            KA.synchronize(be)
            @test maximum(abs, Array(rho_metal) .- rho_cpu) < 3f-6
            @test sum(Array(rho_metal)) ≈ T(np) rtol=3f-6

            target_delta = 3f-8
            displacement_amplitude = -target_delta / T(2pi)
            for p in 1:np
                i = (p - 1) % N
                x = (T(i) + 0.5f0) / T(N)
                dxh[p] = displacement_amplitude * sinpi(2f0 * x)
            end
            copyto!(dxm, dxh)
            delta_cpu = zeros(T, np)
            delta_metal = device_zeros(be, T, (np,))
            deposit_lattice_displacement_delta!(
                delta_cpu, dxh, dyh, dzh, vxh, vyh, vzh; N=N,
            )
            deposit_lattice_displacement_delta!(
                delta_metal, dxm, dym, dzm, vxm, vym, vzm; N=N,
            )
            KA.synchronize(be)
            delta_host = Array(delta_metal)
            @test maximum(abs, delta_host .- delta_cpu) < 5f-10
            mode = 0.0im
            for p in 1:np
                i = (p - 1) % N
                x = (Float64(i) + 0.5) / N
                mode += Float64(delta_host[p]) * cis(-2pi * x)
            end
            mode *= 2 / np
            expected_transfer = sinpi(2 / N) / (2pi / N)
            @test abs(mode) / target_delta ≈ expected_transfer rtol=2f-5
        end

        @testset "exponential force-drag CPU/Metal parity" begin
            N = 16
            sr = allocate_state(becpu, T, (N,N,N); dx=1/N)
            sm = allocate_state(be, T, (N,N,N); dx=1/N)
            for s in (sr, sm)
                fill!(s.scratch[1], 2f0)
                fill!(s.scratch[2], 1.6f0); fill!(s.scratch[3], -0.6f0)
                fill!(s.scratch[4], 0.4f0)
                fill!(s.U[1], 2.5f0)
                fill!(s.U[2], 3f0); fill!(s.U[3], -1.25f0); fill!(s.U[4], 0.75f0)
                fill!(s.U[6], 0.2f0); fill!(s.U[7], -0.1f0); fill!(s.U[8], 0.05f0)
                fill!(s.U[5], 5f0)
                apply_exponential_drag_increment!(s, s.scratch, 10;
                    target_velocity=(0.1, -0.05, 0.02))
            end
            hr = fields_to_host(sr); hm = fields_to_host(sm)
            @test maximum(abs, hm[2] .- hr[2]) < 2f-7
            @test maximum(abs, hm[3] .- hr[3]) < 2f-7
            @test maximum(abs, hm[5] .- hr[5]) < 2f-7
        end

        @testset "terminal pressure CT CPU-FFTW/Metal-MPS parity" begin
            N = 16
            function fftw_exponential!(dst, state, source, coeff_cells2, dt)
                state_k = rfft(Array(state))
                source_k = rfft(Array(source))
                for k in axes(state_k, 3), j in axes(state_k, 2), i in axes(state_k, 1)
                    kx = i - 1
                    ky0 = j - 1; ky = ky0 <= N ÷ 2 ? ky0 : ky0 - N
                    kz0 = k - 1; kz = kz0 <= N ÷ 2 ? kz0 : kz0 - N
                    q2 = sin(2pi * kx / N)^2 + sin(2pi * ky / N)^2 +
                         sin(2pi * kz / N)^2
                    x = coeff_cells2 * q2
                    decay = exp(-x)
                    sourcefac = x > 1e-6 ? dt * (1 - decay) / x :
                                dt * (1 - 0.5x + x^2 / 6)
                    state_k[i, j, k] = state_k[i, j, k] * decay +
                                       source_k[i, j, k] * sourcefac
                end
                copyto!(dst, irfft(state_k, N))
                return dst
            end
            function mps_exponential!(dst, state, source, coeff_cells2, dt)
                PoissonKernels.rfft_exponential_source!(dst, state, source;
                                                        coeff_cells2=coeff_cells2, dt=dt)
            end
            sr = allocate_state(becpu, T, (N, N, N); dx=1 / N)
            sm = allocate_state(be, T, (N, N, N); dx=1 / N)
            init_field_loop!(sr; amplitude=0.02, radius=0.25, velocity=(0.1, 0.05, 0))
            for field in 1:9
                copyto!(sm.U[field], sr.U[field])
            end
            rr = terminal_pressure_ct_step!(fftw_exponential!, sr, 0.002;
                                            gamma_drag=5, pressure_coeff=0.2,
                                            induction_dissipation=0.05)
            rm = terminal_pressure_ct_step!(mps_exponential!, sm, 0.002;
                                            gamma_drag=5, pressure_coeff=0.2,
                                            induction_dissipation=0.05)
            hr = fields_to_host(sr); hm = fields_to_host(sm)
            rho_rel = maximum(abs, Float64.(hm[1]) .- Float64.(hr[1])) /
                      maximum(abs, Float64.(hr[1]))
            b_abs = maximum(maximum(abs, Float64.(hm[field]) .- Float64.(hr[field]))
                            for field in 6:8)
            @printf("  [Metal terminal pressure CT] rho rel=%.3e B abs=%.3e nsub=%d/%d\n",
                    rho_rel, b_abs, rr.nsub, rm.nsub)
            @test rr.nsub == rm.nsub
            @test rho_rel < 2e-5
            @test b_abs < 2e-6
            @test Float64(max_divb(sm)) * Float64(sm.dx) / 0.02 < 2e-5
        end

        @testset "throughput: cube" begin
            for N in (64, 128)
                s = allocate_state(be, T, (N,N,N); dx=1/N, gamma=5/3, riemann=:hll)
                init_turb_field!(s)
                dt, smax = compute_dt(s; cfl=0.4)
                nc = prod(s.dims)
                iters = N == 64 ? 20 : 8
                f = () -> step!(s, dt; ch=smax, integrator=:cube)
                f(); KA.synchronize(be)
                el = @elapsed begin
                    for _ in 1:iters
                        f()
                    end
                    KA.synchronize(be)
                end
                @printf("  [Metal] cube  N=%d (%5.1fM): %6.1f Mcell/s | %.2f ms/step\n",
                        N, nc/1e6, nc*iters/el/1e6, 1e3*el/iters)
            end
            @test true
        end
    end
end
