using Metal, MHDKernels, KernelAbstractions, Printf, Test

function run_pmf_hlld_stability(be, integrator; N=32, steps=300, brms=0.08f0,
                                recon=:plm, cfl=0.4, glm_ch_fac=2.0, glm_cr=0.5)
    s = allocate_state(be, Float32, (N,N,N); dx=1/N, gamma=5/3,
                       riemann=:hlld, recon=recon)
    init_pmf_batchelor!(s; brms=brms, rho0=1, p0=1, seed=42,
                        kcut=0.3125f0*pi*N)
    for n in 1:steps
        dt, smax = compute_dt(s; cfl=cfl)
        dtm = dt / glm_ch_fac
        step!(s, dtm; ch=glm_ch_fac*smax, glm_cr=glm_cr, integrator=integrator)
        if n % 25 == 0
            h = fields_to_host(s)
            finite = all(isfinite, h[1]) && all(isfinite, h[5]) &&
                     minimum(h[1]) > 0 && minimum(h[5]) > 0
            @printf("  [%s/%s] n=%d rho=[%.4e,%.4e] E=[%.4e,%.4e] div=%.3e\n",
                    string(be), integrator, n, extrema(h[1])..., extrema(h[5])...,
                    Float64(max_divb(s))*Float64(s.dx)/brms)
            finite || return s, false
        end
    end
    h = fields_to_host(s)
    br = sqrt((sum(abs2, h[6]) + sum(abs2, h[7]) + sum(abs2, h[8])) / length(h[6]))
    @printf("  [%s/%s] final Brms=%.6e retention=%.6f\n",
            string(be), integrator, br, br/brms)
    return s, true
end

@testset "random PMF HLLD long-step stability" begin
    chfac = parse(Float64, get(ENV, "MHD_TEST_GLM_CH_FAC", "2"))
    cr = parse(Float64, get(ENV, "MHD_TEST_GLM_CR", "1.0"))
    steps = parse(Int, get(ENV, "MHD_TEST_STEPS", "300"))
    _, cpu_ok = run_pmf_hlld_stability(backend(:cpu), :ref;
                                       steps=steps, glm_ch_fac=chfac, glm_cr=cr)
    @test cpu_ok
    if Metal.functional() && get(ENV, "MHD_TEST_METAL", "1") != "0"
        _, metal_ok = run_pmf_hlld_stability(backend(:metal), :cube;
                                             steps=steps, glm_ch_fac=chfac, glm_cr=cr)
        @test metal_ok
    end
end
