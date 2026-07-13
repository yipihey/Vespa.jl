using MHDKernels, KernelAbstractions, Printf

const KA = KernelAbstractions
const T = Float32
const BACKEND_NAME = Symbol(lowercase(get(ENV, "MHD_BACKEND", "cpu")))

if BACKEND_NAME === :metal
    using Metal
elseif BACKEND_NAME === :cuda
    using CUDA
end

function _bool_env(name::String, default::Bool)
    v = lowercase(get(ENV, name, default ? "1" : "0"))
    v in ("1", "true", "yes", "on")
end

function _backend_from_env()
    name = BACKEND_NAME
    return backend(name), name
end

function _brms(s)
    h = fields_to_host(s)
    sqrt((sum(abs2, h[6]) + sum(abs2, h[7]) + sum(abs2, h[8])) / length(h[6]))
end

function _rho_minmax(s)
    ρ = fields_to_host(s)[1]
    minimum(ρ), maximum(ρ)
end

function run_case(be, bname, N::Int; chfac::Real, cr::Real, tfinal::Real,
                  brms::Real, riemann::Symbol, recon::Symbol, cfl::Real,
                  seed::Integer, kcut::Real)
    s = allocate_state(be, T, (N,N,N); dx=1/N, gamma=5/3, riemann=riemann, recon=recon)
    st = init_pmf_batchelor!(s; brms=brms, rho0=1, p0=1, seed=seed, kcut=kcut)
    b0 = _brms(s)
    db0 = Float64(max_divb(s))
    nc = prod(s.dims)
    t_ref = Ref(0.0)
    n_ref = Ref(0)
    elapsed = @elapsed begin
        t, n = evolve!(s, tfinal; cfl=cfl, glm_ch_fac=chfac, glm_cr=cr,
                       integrator=(bname === :cpu ? :ref : :cube))
        KA.synchronize(be)
        t_ref[] = t
        n_ref[] = n
    end
    b1 = _brms(s)
    db1 = Float64(max_divb(s))
    ρmin, ρmax = _rho_minmax(s)
    norm0 = db0 * Float64(s.dx) / Float64(b0)
    norm1 = db1 * Float64(s.dx) / Float64(b1)
    mcells = nc * n_ref[] / elapsed / 1e6
    @printf("%-5s N=%-4d recon=%-3s riemann=%-4s chfac=%4.1f cr=%5.2f steps=%4d t=%.4f  divdxB %.3e -> %.3e  Brms %.3e -> %.3e  rho=[%.6g, %.6g]  %.1f Mcell/s\n",
            String(bname), N, String(recon), String(riemann), chfac, cr, n_ref[], t_ref[],
            norm0, norm1, b0, b1, ρmin, ρmax, mcells)
    return (backend=bname, N=N, recon=recon, riemann=riemann, chfac=Float64(chfac), cr=Float64(cr),
            steps=n_ref[], t=t_ref[], divdxB0=norm0, divdxB1=norm1,
            brms0=Float64(b0), brms1=Float64(b1), rhomin=Float64(ρmin),
            rhomax=Float64(ρmax), mcells=mcells)
end

function main()
    be, bname = _backend_from_env()
    N = parse(Int, get(ENV, "MHD_PMF_N", bname === :cpu ? "32" : "64"))
    tfinal = parse(Float64, get(ENV, "MHD_PMF_TFINAL", "0.02"))
    brms = parse(Float64, get(ENV, "MHD_PMF_BRMS", "1e-3"))
    cfl = parse(Float64, get(ENV, "MHD_CFL", "0.4"))
    seed = parse(Int, get(ENV, "MHD_PMF_SEED", "42"))
    kcut = parse(Float64, get(ENV, "MHD_PMF_KCUT", string(0.625π*N)))
    riemann = Symbol(lowercase(get(ENV, "MHD_RIEMANN", "hlld")))
    recon = Symbol(lowercase(get(ENV, "MHD_RECON", "plm")))
    chfacs = parse.(Float64, split(get(ENV, "MHD_GLM_CHFACS", "1,1.5,2,3,4"), ","))
    crs = parse.(Float64, split(get(ENV, "MHD_GLM_CRS", "0.18,0.5,1,2"), ","))
    run_all = _bool_env("MHD_GLM_SCAN_ALL", true)

    @printf("PMF GLM scan: backend=%s N=%d tfinal=%.4f brms=%.3e kcut=%.3f cfl=%.2f recon=%s riemann=%s\n",
            String(bname), N, tfinal, brms, kcut, cfl, String(recon), String(riemann))
    results = NamedTuple[]
    for chfac in chfacs, cr in crs
        push!(results, run_case(be, bname, N; chfac=chfac, cr=cr, tfinal=tfinal,
                                brms=brms, riemann=riemann, recon=recon, cfl=cfl,
                                seed=seed, kcut=kcut))
        run_all || break
    end
    best = argmin(r -> (r.divdxB1, -r.brms1), results)
    @printf("BEST chfac=%.2f cr=%.2f divdxB=%.3e Brms=%.3e steps=%d throughput=%.1f Mcell/s\n",
            best.chfac, best.cr, best.divdxB1, best.brms1, best.steps, best.mcells)
end

main()
