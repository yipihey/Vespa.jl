# GPU tests for MHDKernels: ref≡cube cross-check, multi-step stability/conservation,
# and cube-vs-ref throughput. Run: julia --project=test test/test_gpu.jl
# Skips cleanly (no failures) when no CUDA device is present.
using CUDA, MHDKernels, KernelAbstractions, Test, Printf
const KA = KernelAbstractions
const T = Float32

if !CUDA.functional()
    @info "No CUDA device — GPU tests skipped."
else
    be = backend(:cuda)
    @info "CUDA device: $(CUDA.name(CUDA.device()))"

    @testset "MHDKernels GPU" begin
        @testset "ref ≡ cube (one step, 64³ f32) — $(bcs[1]) BC, $recon" for
                (bcs, recon) in (((:periodic,:periodic,:periodic), :plm),
                                 ((:outflow,:outflow,:outflow), :plm),
                                 ((:reflecting,:reflecting,:reflecting), :plm),
                                 ((:periodic,:periodic,:periodic), :ppm))
            N = 64
            sr = allocate_state(be, T, (N,N,N); dx=1/N, gamma=5/3, use_hlld=true, bcs=bcs, recon=recon)
            sc = allocate_state(be, T, (N,N,N); dx=1/N, gamma=5/3, use_hlld=true, bcs=bcs, recon=recon)
            init_turb_field!(sr); init_turb_field!(sc)
            dt, smax = compute_dt(sr; cfl=0.4)
            step!(sr, dt; ch=smax, integrator=:ref)
            step!(sc, dt; ch=smax, integrator=:cube)
            hr = fields_to_host(sr); hc = fields_to_host(sc)
            maxabs = 0.0; scale = 0.0
            for v in 1:9
                maxabs = max(maxabs, maximum(abs.(Float64.(hr[v]) .- Float64.(hc[v]))))
                scale  = max(scale, maximum(abs.(Float64.(hr[v]))))
            end
            rel = maxabs / scale
            @printf("  [%s BC, %s] max|ref-cube| = %.3e  (rel %.3e)\n", bcs[1], recon, maxabs, rel)
            @test rel < 1e-4
        end

        @testset "cube multi-step stability + conservation (128³)" begin
            N = 128
            s = allocate_state(be, T, (N,N,N); dx=1/N, gamma=5/3, use_hlld=true)
            init_turb_field!(s)
            t0 = conserved_totals(s)
            t, n = evolve!(s, 0.1; cfl=0.4, integrator=:cube)
            t1 = conserved_totals(s)
            h = fields_to_host(s)
            dmass = abs(t1.mass-t0.mass)/abs(t0.mass)
            denergy = abs(t1.energy-t0.energy)/abs(t0.energy)
            @printf("  cube evolve t=%.3f (%d steps): finite=%s Δmass=%.2e Δenergy=%.2e\n",
                    t, n, all(isfinite, h[1]), dmass, denergy)
            @test all(isfinite, h[1])
            @test dmass < 1e-4
            @test denergy < 1e-4
        end

        @testset "cube raw ≡ ka (one step, 64³)" begin
            N = 64
            sk = allocate_state(be, T, (N,N,N); dx=1/N, gamma=5/3, use_hlld=true); init_turb_field!(sk)
            sr = allocate_state(be, T, (N,N,N); dx=1/N, gamma=5/3, use_hlld=true); init_turb_field!(sr)
            dt, smax = compute_dt(sk; cfl=0.4); decay = exp(-0.18f0*smax*dt/sk.dx)
            MHDKernels.step_cube!(sk, dt; ch=smax, decay=decay, impl=:ka)
            MHDKernels.step_cube!(sr, dt; ch=smax, decay=decay, impl=:raw)
            hk = fields_to_host(sk); hr = fields_to_host(sr)
            maxabs = maximum(v->maximum(abs.(Float64.(hk[v]).-Float64.(hr[v]))), 1:9)
            @printf("  raw vs ka: max|Δ| = %.3e\n", maxabs)
            @test maxabs < 1e-3      # both paths run the identical scheme
        end

        @testset "throughput: cube(raw) vs cube(ka) vs ref" begin
            for N in (128, 256, 512)
                s = allocate_state(be, T, (N,N,N); dx=1/N, gamma=5/3, use_hlld=true)
                init_turb_field!(s)
                dt, smax = compute_dt(s; cfl=0.4); decay = exp(-0.18f0*smax*dt/s.dx)
                nc = prod(s.dims); iters = 12
                for (lbl, f) in (("cube/raw", () -> MHDKernels.step_cube!(s, dt; ch=smax, decay=decay, impl=:raw)),
                                 ("cube/ka",  () -> MHDKernels.step_cube!(s, dt; ch=smax, decay=decay, impl=:ka)),
                                 ("ref",      () -> step!(s, dt; ch=smax, integrator=:ref)))
                    f(); KA.synchronize(be)
                    el = CUDA.@elapsed (for _ in 1:iters; f(); end; KA.synchronize(be))
                    @printf("  %-9s N=%d (%5.1fM): %6.1f Mcell/s | %.2f ms/step\n",
                            lbl, N, nc/1e6, nc*iters/el/1e6, 1e3*el/iters)
                end
            end
            @test true
        end
    end
end
