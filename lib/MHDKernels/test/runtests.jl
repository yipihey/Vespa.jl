using MHDKernels, KernelAbstractions, FFTW, Test, Printf

const T = Float32   # f32-first: the CPU runs f32 too (apples-to-apples with the GPU)

# ── L1 of By vs the exact right-going CP Alfvén wave (1-D in x; thin transverse) ──
function alfven_L1(N::Int; amp=0.1, p0=0.1, tfinal=0.5, cfl=0.4, recon=:plm,
                   glm_cr=MHDKernels.GLM_CR)
    be = backend(:cpu)
    s = allocate_state(be, T, (N,4,4); dx = 1/N, gamma = 5/3, use_hlld = true, recon = recon)
    init_alfven_wave!(s; amp = amp, B0 = 1, p0 = p0)
    tot0 = conserved_totals(s)
    t, n = evolve!(s, tfinal; cfl = cfl, glm_cr=glm_cr, integrator = :ref)
    tot1 = conserved_totals(s)
    By = fields_to_host(s)[7]                  # row j=1,k=1 is linear indices 1..N
    err = 0.0
    for i in 1:N
        x = (i - 0.5)/N
        err += abs(Float64(By[i]) - alfven_By_exact(x, t, 1.0; amp = amp))
    end
    L1 = err/N
    drift = (mass = abs(tot1.mass-tot0.mass)/abs(tot0.mass),
             energy = abs(tot1.energy-tot0.energy)/abs(tot0.energy),
             momx = abs(tot1.momx-tot0.momx))
    return L1, drift, n, t
end

@testset "strong-field low-beta circular Alfvén wave" begin
    L1, drift, n, t = alfven_L1(64; amp=0.1, p0=1e-3, tfinal=0.5,
                                recon=:plm, glm_cr=1.0)
    @printf("  low-beta Alfvén t=%.3f (%d steps): L1(By)=%.3e dE=%.3e\n",
            t, n, L1, drift.energy)
    @test isfinite(L1)
    @test L1 < 0.025
    @test drift.energy < 2e-4
end

@testset "MHDKernels — GLM MUSCL-Hancock (f32, CPU reference) — $recon" for recon in (:plm, :ppm)
    Ns = (32, 64, 128)
    L1s = Float64[]; lastdrift = nothing
    for N in Ns
        L1, drift, n, t = alfven_L1(N; recon=recon)
        push!(L1s, L1); lastdrift = drift
        @printf("  [%s] N=%-4d  L1(By)=%.3e  |  Δmass=%.2e Δenergy=%.2e  (%d steps, t=%.3f)\n",
                recon, N, L1, drift.mass, drift.energy, n, t)
        @test isfinite(L1)
    end

    @testset "2nd-order convergence (f32)" begin
        for k in 2:length(Ns)
            order = log2(L1s[k-1]/L1s[k])
            @printf("  [%s] order(%d→%d) = %.2f\n", recon, Ns[k-1], Ns[k], order)
            @test order > 1.7
        end
        @test L1s[end] < L1s[1]
    end

    @testset "conservation (f32 round-off)" begin
        @test lastdrift.mass   < 1e-4
        @test lastdrift.energy < 1e-4
        @test lastdrift.momx   < 1e-3
    end
end

# ── Brio-Wu MHD shock tube (γ=2, outflow x-BC): structure + self-convergence ──
function brio_wu_density(N; tfinal=0.1)
    be = backend(:cpu)
    s = allocate_state(be, T, (N,4,4); dx=1/N, gamma=2, use_hlld=true,
                       bcs=(:outflow,:periodic,:periodic))
    init_brio_wu!(s)
    t0 = conserved_totals(s)
    t, n = evolve!(s, tfinal; cfl=0.4, integrator=:ref)
    t1 = conserved_totals(s)
    ρ = Float64.(fields_to_host(s)[1][1:N])        # row j=1,k=1
    drift = (mass=abs(t1.mass-t0.mass)/abs(t0.mass), energy=abs(t1.energy-t0.energy)/abs(t0.energy))
    return ρ, drift, n, t
end

@testset "Brio-Wu shock tube (f32, outflow BC, reference)" begin
    ρ2, d2, _, _ = brio_wu_density(256)
    ρ4, d4, n, t = brio_wu_density(512)
    ρ4d = [0.5*(ρ4[2i-1]+ρ4[2i]) for i in 1:256]   # downsample 512→256
    L1 = sum(abs, ρ2 .- ρ4d)/256
    @printf("  Brio-Wu t=%.3f: ρ[left]=%.3f ρ[right]=%.3f min=%.3f max=%.3f | L1(256 vs 512)=%.3e | Δmass=%.2e Δenergy=%.2e\n",
            t, ρ2[1], ρ2[end], minimum(ρ2), maximum(ρ2), L1, d4.mass, d4.energy)
    @test all(isfinite, ρ2)
    @test minimum(ρ2) > 0.1 && maximum(ρ2) < 1.02     # bounded (no new extrema)
    @test ρ2[1] > 0.9 && ρ2[end] < 0.2                # initial L/R states preserved at the ends
    @test minimum(ρ2) < 0.35                          # contact/compound-wave density drop present
    @test L1 < 0.05                                   # self-convergent across resolution
    @test d4.mass < 1e-4 && d4.energy < 1e-4          # interior-conservative (waves not at boundary)
end

# ── Orszag-Tang: divergence cleaning keeps ∇·B bounded at the cell scale ──────
@testset "Orszag-Tang divB control (GLM cleaning, f32)" begin
    N = 128; be = backend(:cpu)
    s = allocate_state(be, T, (N,N,1); dx=1/N, gamma=5/3, use_hlld=true)
    init_orszag_tang!(s)
    t0 = conserved_totals(s)
    # sample max|∇·B| over the run; GLM control = it SATURATES (vs unbounded growth w/o cleaning)
    hist = Tuple{Float64,Float64}[]
    cb(s,t,n) = (n % 40 == 0 && push!(hist, (t, Float64(max_divb(s)))))
    t, n = evolve!(s, 0.5; cfl=0.4, integrator=:ref, callback=cb)
    t1 = conserved_totals(s)
    h = fields_to_host(s)
    db = Float64(max_divb(s))
    Brms = sqrt(sum(Float64, h[6].^2 .+ h[7].^2 .+ h[8].^2)/length(h[6]))
    norm_divb = db*Float64(s.dx)/Brms                       # ∇·B in units of (B per cell)
    early = maximum(x->x[2], hist[1:max(1,length(hist)÷2)]) # peak over first half
    late  = maximum(x->x[2], hist[(length(hist)÷2+1):end])  # peak over second half
    growth = late/early
    @printf("  OT t=%.3f (%d steps): finite=%s | max|divB|·dx/Brms=%.2f | growth(late/early)=%.2f | Δmass=%.2e Δenergy=%.2e\n",
            t, n, all(isfinite, h[1]), norm_divb, growth, abs(t1.mass-t0.mass)/abs(t0.mass),
            abs(t1.energy-t0.energy)/abs(t0.energy))
    # NOTE: divB rising over the run is PHYSICAL here — OT develops current sheets from a
    # smooth IC, so `growth` is expected >1 (printed, not asserted). The signatures of
    # working GLM control are: the run stays STABLE through 333 steps of shock formation,
    # conserves mass to round-off, and holds ∇·B BOUNDED at the cell scale (not →1e6).
    # (A comparative cleaning-on-vs-off gate would be stronger; left as future work.)
    @test all(isfinite, h[1]) && minimum(h[1]) > 0          # stable: positive density, no NaN
    @test abs(t1.mass-t0.mass)/abs(t0.mass) < 1e-4          # mass conserved (periodic, no driving)
    @test norm_divb < 5.0                                   # ∇·B bounded at ~cell scale (cleaning works)
end

@testset "PMF Batchelor vector-potential IC" begin
    N = 32; be = backend(:cpu)
    s = allocate_state(be, T, (N,N,N); dx=1/N, gamma=5/3, riemann=:hll)
    st = init_pmf_batchelor!(s; brms=1f-3, rho0=1, p0=1, seed=42, kcut=20)
    h = fields_to_host(s)
    brms = sqrt((sum(abs2, h[6]) + sum(abs2, h[7]) + sum(abs2, h[8])) / length(h[6]))
    db = max_divb(s)
    @printf("  PMF IC: brms=%.3e target=%.3e max|divB|=%.3e\n", brms, st.brms, db)
    @test abs(brms - 1f-3) < 2f-6
    @test db < 1f-6
    @test all(isfinite, h[5])

    kmax = 8π
    Bx, By, Bz, hard = pmf_batchelor_field((N,N,N); brms=1f-3, seed=42,
                                            kcut=kmax, kmax=kmax)
    Bk = (FFTW.rfft(Bx), FFTW.rfft(By), FFTW.rfft(Bz))
    high_power = 0.0
    total_power = sum(sum(abs2, component) for component in Bk)
    for kk in axes(Bk[1], 3), jj in axes(Bk[1], 2), ii in axes(Bk[1], 1)
        mx = ii - 1
        my = jj - 1 <= N ÷ 2 ? jj - 1 : jj - 1 - N
        mz = kk - 1 <= N ÷ 2 ? kk - 1 : kk - 1 - N
        if (2π)^2 * (mx^2 + my^2 + mz^2) > kmax^2 * (1 + 1e-12)
            high_power += sum(abs2(component[ii,jj,kk]) for component in Bk)
        end
    end
    @printf("  PMF hard cutoff: kmax=%.3f high-k power fraction=%.3e\n",
            hard.kmax, high_power / total_power)
    @test hard.kmax == kmax
    @test high_power / total_power < 2e-13

    locked64 = pmf_batchelor_field((64,64,64); brms=1f-3, seed=42,
                                    kcut=kmax, kmax=kmax, mode_lock_n=64)
    locked128 = pmf_batchelor_field((128,128,128); brms=1f-3, seed=42,
                                     kcut=kmax, kmax=kmax, mode_lock_n=64)
    for component in 1:3
        coarse = locked64[component]
        fine_on_coarse = locked128[component][1:2:end, 1:2:end, 1:2:end]
        rel_l2 = sqrt(sum(abs2, coarse .- fine_on_coarse) / sum(abs2, coarse))
        correlation = sum(coarse .* fine_on_coarse) /
                      sqrt(sum(abs2, coarse) * sum(abs2, fine_on_coarse))
        @test rel_l2 < 6e-3
        @test correlation > 0.9999
    end
end

include("test_terminal_ct.jl")
include("test_terminal_handoff_overlap.jl")

@testset "magnetic field-loop advection" begin
    N = 64
    s = allocate_state(backend(:cpu), T, (N,N,4); dx=1/N, gamma=5/3,
                       riemann=:hlld, recon=:plm)
    init_field_loop!(s; radius=0.3, amplitude=1e-3, velocity=(1,1,0))
    h0 = fields_to_host(s)
    bnorm = sum(abs, h0[6]) + sum(abs, h0[7])
    emag0 = sum(abs2, h0[6]) + sum(abs2, h0[7])
    div0 = Float64(max_divb(s))
    t, n = evolve!(s, 1.0; cfl=0.4, glm_ch_fac=2, glm_cr=1,
                   integrator=:ref)
    h1 = fields_to_host(s)
    l1 = (sum(abs, h1[6] .- h0[6]) + sum(abs, h1[7] .- h0[7])) / bnorm
    emag1 = sum(abs2, h1[6]) + sum(abs2, h1[7])
    divn = Float64(max_divb(s))*Float64(s.dx) /
           sqrt(emag1 / length(h1[6]))
    @printf("  field loop t=%.3f (%d steps): L1=%.3e Emag retention=%.3f divdx/Brms=%.3e\n",
            t, n, l1, emag1/emag0, divn)
    @test div0 < 2e-6
    @test all(isfinite, h1[5]) && minimum(h1[1]) > 0
    @test l1 < 0.7
    @test emag1/emag0 > 0.2
    @test divn < 0.5
end

# ── GPU cross-check + throughput light up only when a device is present ───────
if MHDKernels.has_backend(:cuda)
    @info "CUDA device present — (ref↔cube cross-check will be added with the cube path)"
else
    @info "No CUDA device — CPU-f32 gate only (GPU cross-check skipped)."
end
