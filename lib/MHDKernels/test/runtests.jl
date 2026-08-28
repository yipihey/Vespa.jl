using MHDKernels, KernelAbstractions, FFTW, Test, Printf, Statistics

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

    @test_throws ErrorException pmf_batchelor_field((32,32,32); box=1,
        brms=1f-3, seed=42, kcut=8π, kmax=8π,
        minimum_cells_per_wavelength=16)
    resolved = pmf_batchelor_field((128,128,128); box=1, brms=1f-3,
        seed=42, kcut=8π, kmax=8π, minimum_cells_per_wavelength=16)
    @test resolved[4].actual_cells_per_wavelength == 32
    @test resolved[4].resolved_kmax == 16π

    default_spectrum = PMFSpectrumSpec(box=1, reference_brms=1f-3, seed=42,
        kcut=16π, reference_kmax=4π, minimum_cells_per_wavelength=16,
        mode_lock_n=32)
    @test default_spectrum.curl_symbol === :centered
    @test default_spectrum.normalization_kind === :reference_band_rms

    spectrum = PMFSpectrumSpec(box=1, reference_brms=1f-3, seed=42,
        kcut=16π, reference_kmax=4π, minimum_cells_per_wavelength=16,
        mode_lock_n=32, curl_symbol=:continuum)
    manifest = pmf_multiband_manifest(spectrum, (0, 4π, 8π, 16π))
    @test getfield.(manifest, :required_n) == [32, 64, 128]
    @test all(getfield.(manifest, :cells_per_wavelength) .== 16)

    low_band = pmf_batchelor_band(spectrum, (64,64,64); kmin=0, kmax=4π)
    high_band = pmf_batchelor_band(spectrum, (64,64,64); kmin=4π, kmax=8π)
    combined = pmf_batchelor_band(spectrum, (64,64,64); kmin=0, kmax=8π)
    for component in 1:3
        rel_l2 = sqrt(sum(abs2, low_band[component] .+ high_band[component] .-
                               combined[component]) / sum(abs2, combined[component]))
        @test rel_l2 < 2e-6
    end
    @test abs(low_band[4].brms_measured / spectrum.reference_brms - 1) < 2e-6
    @test high_band[4].kmin == 4π

    old_kspace_threads = get(ENV, "MHD_PMF_KSPACE_THREADS", nothing)
    try
        ENV["MHD_PMF_KSPACE_THREADS"] = "1"
        threaded_a = pmf_batchelor_field((N,N,N); brms=1f-3, seed=42,
                                          kcut=kmax, kmax=kmax)
        threaded_b = pmf_batchelor_field((N,N,N); brms=1f-3, seed=42,
                                          kcut=kmax, kmax=kmax)
        @test all(threaded_a[component] == threaded_b[component] for component in 1:3)
    finally
        old_kspace_threads === nothing ? delete!(ENV, "MHD_PMF_KSPACE_THREADS") :
            (ENV["MHD_PMF_KSPACE_THREADS"] = old_kspace_threads)
    end

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

    function shell_power(field_tuple, n, max_mode)
        power = zeros(Float64, max_mode)
        counts = zeros(Int, max_mode)
        for field in field_tuple[1:3]
            coeff = FFTW.rfft(field)
            @inbounds for kk in axes(coeff, 3), jj in axes(coeff, 2), ii in axes(coeff, 1)
                mx = ii - 1
                my = jj - 1 <= n ÷ 2 ? jj - 1 : jj - 1 - n
                mz = kk - 1 <= n ÷ 2 ? kk - 1 : kk - 1 - n
                mode = sqrt(Float64(mx*mx + my*my + mz*mz))
                mode <= 0 && continue
                shell = floor(Int, mode)
                (shell < 1 || shell > max_mode) && continue
                multiplicity = (mx == 0 || mx == n ÷ 2) ? 1 : 2
                power[shell] += multiplicity * abs2(coeff[ii,jj,kk])
                field === field_tuple[1] && (counts[shell] += multiplicity)
            end
        end
        scale = 1 / Float64(n)^6
        return power .* scale ./ counts
    end
    continuum64 = pmf_batchelor_field((64,64,64); brms=1f-3, seed=42,
                                      kcut=kmax, kmax=kmax, mode_lock_n=64,
                                      curl_symbol=:continuum)
    continuum128 = pmf_batchelor_field((128,128,128); brms=1f-3, seed=42,
                                       kcut=kmax, kmax=kmax, mode_lock_n=64,
                                       curl_symbol=:continuum)
    p64 = shell_power(continuum64, 64, 4)
    p128 = shell_power(continuum128, 128, 4)
    @test maximum(abs.(p64 ./ p128 .- 1)) < 2e-5

    extended = pmf_batchelor_field((128,128,128); brms=1f-3, seed=42,
                                    kcut=64π, kmax=64π, mode_lock_n=64,
                                    preserve_low_kmax=kmax,
                                    preserve_low_kcut=kmax)
    refk = (FFTW.rfft(locked128[1]), FFTW.rfft(locked128[2]), FFTW.rfft(locked128[3]))
    extk = (FFTW.rfft(extended[1]), FFTW.rfft(extended[2]), FFTW.rfft(extended[3]))
    low_err = 0.0
    low_pow = 0.0
    high_pow = 0.0
    for kk in axes(refk[1], 3), jj in axes(refk[1], 2), ii in axes(refk[1], 1)
        mx = ii - 1
        my = jj - 1 <= 64 ? jj - 1 : jj - 1 - 128
        mz = kk - 1 <= 64 ? kk - 1 : kk - 1 - 128
        k2mode = mx^2 + my^2 + mz^2
        if (2π)^2 * k2mode <= kmax^2 * (1 + 1e-12)
            low_err += sum(abs2, extk[c][ii,jj,kk] - refk[c][ii,jj,kk] for c in 1:3)
            low_pow += sum(abs2, refk[c][ii,jj,kk] for c in 1:3)
        else
            high_pow += sum(abs2, extk[c][ii,jj,kk] for c in 1:3)
        end
    end
    @printf("  PMF preserved low-k: rel low-k error=%.3e high/low power=%.3e brms=%.3e\n",
            sqrt(low_err / low_pow), high_pow / low_pow, extended[4].brms_measured)
    @test sqrt(low_err / low_pow) < 2e-5
    @test isapprox(extended[4].preserve_high_source_scale, sqrt(8); rtol=2e-7)
    @test high_pow > low_pow
    @test extended[4].brms_measured > locked128[4].brms_measured
    @test extended[4].normalization_kind == "reference_band_rms"
    @test isapprox(extended[4].reference_band_brms_measured, 1f-3; rtol=2e-6)
    @test extended[4].total_brms_measured == extended[4].brms_measured
    @test isapprox(extended[4].normalization_brms_measured, 1f-3; rtol=2e-6)

    fixed_total = pmf_batchelor_field((128,128,128); brms=1f-3, seed=42,
        kcut=64π, kmax=64π, mode_lock_n=64,
        preserve_low_kmax=kmax, preserve_low_kcut=kmax,
        normalization_kind=:total_band_rms)
    @test fixed_total[4].normalization_kind == "total_band_rms"
    @test isapprox(fixed_total[4].total_brms_measured, 1f-3; rtol=2e-6)
    @test fixed_total[4].reference_band_brms_measured < 1f-3
    @test isapprox(fixed_total[4].normalization_brms_measured, 1f-3; rtol=2e-6)
    @test_throws ErrorException pmf_batchelor_field((32,32,32); brms=1f-3,
        seed=42, kcut=8π, kmax=8π,
        normalization_kind=:reference_band_rms)

    continuous = pmf_batchelor_field((128,128,128); brms=1f-3, seed=42,
        kcut=32π, kmax=64π, mode_lock_n=64,
        preserve_low_kmax=8π, preserve_low_kcut=32π,
        curl_symbol=:continuum)
    continuous_k = FFTW.rfft.(continuous[1:3])
    function compensated_mode_power(mode_min, mode_max)
        total = 0.0
        count = 0
        for kk in axes(continuous_k[1], 3), jj in axes(continuous_k[1], 2),
            ii in axes(continuous_k[1], 1)
            mx = ii - 1
            my = jj - 1 <= 64 ? jj - 1 : jj - 1 - 128
            mz = kk - 1 <= 64 ? kk - 1 : kk - 1 - 128
            mode2 = mx^2 + my^2 + mz^2
            mode = sqrt(Float64(mode2))
            mode_min < mode <= mode_max || continue
            k = 2π * mode
            envelope = exp(-(k / (32π))^2)
            multiplicity = (mx == 0 || mx == 64) ? 1 : 2
            total += multiplicity *
                sum(abs2(continuous_k[c][ii,jj,kk]) for c in 1:3) /
                (k^2 * envelope)
            count += multiplicity
        end
        return total / count
    end
    compensated_low = compensated_mode_power(0, 4)
    compensated_high = compensated_mode_power(4, 12)
    compensated_ratio = compensated_high / compensated_low
    @printf("  PMF multiband continuity: compensated high/low power=%.3f\n",
            compensated_ratio)
    @test 0.6 < compensated_ratio < 1.6

    fixed_power = pmf_batchelor_field((32,32,32); box=1, brms=1f-3, seed=42,
        kcut=8π, kmax=12π, curl_symbol=:centered,
        fixed_mode_power=true)
    fixed_k = FFTW.rfft.(fixed_power[1:3])
    fixed_coefficients = Float64[]
    for kk in axes(fixed_k[1], 3), jj in axes(fixed_k[1], 2),
        ii in axes(fixed_k[1], 1)
        mx = ii - 1
        my = jj - 1 <= 16 ? jj - 1 : jj - 1 - 32
        mz = kk - 1 <= 16 ? kk - 1 : kk - 1 - 32
        k = 2π * sqrt(Float64(mx^2 + my^2 + mz^2))
        0 < k <= 8π || continue
        shape = (k / (8π))^2 * exp(-(k / (8π))^2)
        push!(fixed_coefficients,
              sum(abs2(fixed_k[c][ii,jj,kk]) for c in 1:3) / shape)
    end
    fixed_scatter = std(fixed_coefficients) / mean(fixed_coefficients)
    @printf("  PMF fixed-mode compensated scatter=%.3e\n", fixed_scatter)
    @test fixed_power[4].fixed_mode_power
    @test fixed_scatter < 2e-5

    physical_l1 = pmf_batchelor_field((32,32,32); box=1, brms=1f-3, seed=42,
        kcut=8π, kmax=12π, mode_lock_n=32,
        curl_symbol=:centered, curl_reference_n=32,
        fixed_mode_power=true, mode_realization=:physical_hash,
        realization_box=1)
    physical_lhalf = pmf_batchelor_field((32,32,32); box=0.5, brms=1f-3,
        seed=42, kcut=8π, kmax=12π, mode_lock_n=16,
        curl_symbol=:centered, curl_reference_n=16,
        fixed_mode_power=true, mode_realization=:physical_hash,
        realization_box=1)
    physical_k1 = FFTW.rfft.(physical_l1[1:3])
    physical_khalf = FFTW.rfft.(physical_lhalf[1:3])
    hash_a1 = MHDKernels._physical_hash_vector_potential_k((32,32,32), 42,
                                                            1.0, 1.0)
    hash_ahalf = MHDKernels._physical_hash_vector_potential_k((16,16,16), 42,
                                                               0.5, 1.0)
    zero_phase_a = MHDKernels._physical_hash_vector_potential_k(
        (16,16,16), 42, 1.0, 1.0; zero_phase=true)
    hash_errors = Float64[]
    common_mode_errors = Float64[]
    for mz in -2:2, my in -2:2, mx in 0:2
        mx == 0 && my == 0 && mz == 0 && continue
        4π * sqrt(Float64(mx^2 + my^2 + mz^2)) <= 12π || continue
        ihalf = mx + 1
        jhalf = my >= 0 ? my + 1 : 32 + my + 1
        khalf = mz >= 0 ? mz + 1 : 32 + mz + 1
        i1 = 2mx + 1
        j1 = my >= 0 ? 2my + 1 : 32 + 2my + 1
        k1 = mz >= 0 ? 2mz + 1 : 32 + 2mz + 1
        jahalf = my >= 0 ? my + 1 : 16 + my + 1
        kahalf = mz >= 0 ? mz + 1 : 16 + mz + 1
        for component in 1:3
            push!(hash_errors, abs(hash_ahalf[component][ihalf,jahalf,kahalf] -
                                   hash_a1[component][i1,j1,k1]))
        end
        coarse = ComplexF64[physical_khalf[c][ihalf,jhalf,khalf] for c in 1:3]
        reference = ComplexF64[physical_k1[c][i1,j1,k1] for c in 1:3]
        coarse_norm = sqrt(sum(abs2, coarse))
        reference_norm = sqrt(sum(abs2, reference))
        coarse_norm > 0 && reference_norm > 0 || continue
        coarse ./= coarse_norm
        reference ./= reference_norm
        push!(common_mode_errors, sqrt(sum(abs2, coarse - reference)))
    end
    @test maximum(hash_errors) == 0
    @test all(iszero, imag.(zero_phase_a[1]))
    @test all(iszero, imag.(zero_phase_a[2]))
    @test all(iszero, imag.(zero_phase_a[3]))
    @test !isempty(common_mode_errors)
    # The physical vector potential is exact above. The field differs slightly
    # because each target grid projects it onto its own centered-difference
    # divergence-free subspace; that difference must converge with k*dx.
    @test maximum(common_mode_errors) < 0.035
    @test physical_l1[4].mode_realization == "physical_hash"
    @test physical_lhalf[4].realization_box == 1

    zero_phase_field = pmf_batchelor_field((16,16,16); box=1, brms=1f-3,
        seed=42, kcut=8π, kmax=2π,
        mode_realization=:physical_hash_zero_phase, realization_box=1)
    @test zero_phase_field[4].mode_realization == "physical_hash_zero_phase"

    # The physical fundamental is represented on a Float32 k grid. A decimal
    # cutoff rounded infinitesimally below 2π/L must still include that shell.
    campaign_box = 1.03
    rounded_fundamental = 6.100179909883
    boundary_field = pmf_batchelor_field((16,16,16); box=campaign_box,
        brms=1f-3, seed=42, kcut=20π / campaign_box,
        kmax=8π / campaign_box, preserve_low_kmax=rounded_fundamental,
        normalization_kind=:reference_band_rms, fixed_mode_power=true,
        mode_realization=:physical_hash, realization_box=campaign_box)
    @test boundary_field[4].reference_band_brms_measured ≈ 1f-3 rtol=2e-6

    nested32 = pmf_batchelor_field((32,32,32); brms=1f-3, seed=42,
        kcut=8π, kmax=16π, mode_lock_n=16, high_mode_lock_n=32,
        preserve_low_kmax=4π, preserve_low_kcut=8π,
        curl_symbol=:continuum)
    nested64 = pmf_batchelor_field((64,64,64); brms=1f-3, seed=42,
        kcut=8π, kmax=16π, mode_lock_n=16, high_mode_lock_n=32,
        preserve_low_kmax=4π, preserve_low_kcut=8π,
        curl_symbol=:continuum)
    for component in 1:3
        coarse = nested32[component]
        fine_on_coarse = nested64[component][1:2:end, 1:2:end, 1:2:end]
        rel_l2 = sqrt(sum(abs2, coarse .- fine_on_coarse) / sum(abs2, coarse))
        correlation = sum(coarse .* fine_on_coarse) /
                      sqrt(sum(abs2, coarse) * sum(abs2, fine_on_coarse))
        @test rel_l2 < 2e-5
        @test correlation > 1 - 2e-6
    end
    @test nested32[4].high_mode_lock_n == 32
    @test nested64[4].high_mode_lock_n == 32
    @test isapprox(nested32[4].preserve_high_source_scale, sqrt(8); rtol=2e-7)
    @test nested32[4].preserve_high_source_scale ==
          nested64[4].preserve_high_source_scale

    matched32 = pmf_batchelor_field((32,32,32); brms=1f-3, seed=42,
        kcut=8π, kmax=16π, mode_lock_n=16, high_mode_lock_n=32,
        preserve_low_kmax=4π, preserve_low_kcut=8π,
        curl_symbol=:centered, curl_reference_n=32)
    matched64 = pmf_batchelor_field((64,64,64); brms=1f-3, seed=42,
        kcut=8π, kmax=16π, mode_lock_n=16, high_mode_lock_n=32,
        preserve_low_kmax=4π, preserve_low_kcut=8π,
        curl_symbol=:centered, curl_reference_n=32)
    @test abs(matched64[4].brms_measured / matched32[4].brms_measured - 1) < 5e-4
    @test matched64[4].curl_reference_n == 32
    for component in 1:3
        coarse = matched32[component]
        fine_on_coarse = matched64[component][1:2:end, 1:2:end, 1:2:end]
        correlation = sum(coarse .* fine_on_coarse) /
                      sqrt(sum(abs2, coarse) * sum(abs2, fine_on_coarse))
        @test correlation > 0.9995
    end

    tile_ref = pmf_batchelor_field((8,8,8); box=1, brms=1f-3, seed=42,
                                   kcut=4π, kmax=4π, mode_lock_n=8)
    s_tile = allocate_state(be, T, (32,32,32); dx=4/32, gamma=5/3, riemann=:hll)
    tiled = init_pmf_batchelor_tiled!(s_tile; brms=1f-3, rho0=1, p0=1,
                                      box=4, tile_box=1, tile_n=8, seed=42,
                                      kcut=4π, kmax=4π, mode_lock_n=8)
    ht = fields_to_host(s_tile)
    for component in 1:3
        tiled_component = ht[5 + component]
        reference = vec(tile_ref[component])
        for k in 1:32, j in 1:32, i in 1:32
            ref_idx = mod1(i, 8) + (mod1(j, 8)-1)*8 + (mod1(k, 8)-1)*64
            idx = i + (j-1)*32 + (k-1)*32*32
            @test tiled_component[idx] == reference[ref_idx]
        end
    end
    @test tiled.tile_box == 1
    @test tiled.tile_n == 8
end

include("test_terminal_ct.jl")
include("test_drag_godunov.jl")
include("test_radiation_coupling.jl")
include("test_recombination_transport.jl")
include("test_passive_transport.jl")
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
