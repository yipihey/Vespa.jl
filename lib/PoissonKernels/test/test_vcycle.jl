# End-to-end V-cycle:
#  (1) bit-tight composition — a fixed-cycle KA V-cycle vs the SAME V-cycle built
#      from the live Fortran *_ref oracles (f64-vs-f64). Running a fixed cycle
#      count removes the convergence-test's reduction-order sensitivity, so the
#      fields must agree to round-off if every kernel composes identically.
#  (2) convergence — the production `vcycle_solve!` must drive the residual down.

# Fixed-count V-cycle driven by the live Fortran oracles (host Float64 arrays).
function _ref_vcycle_fixed(sol0, rhs0, ncyc; pre = 2, post = 3)
    dims = PoissonKernels.mg_dims_schedule(size(sol0))
    nlev = length(dims)
    Sol = Vector{Array{Float64,3}}(undef, nlev)
    RHS = Vector{Array{Float64,3}}(undef, nlev)
    Sol[1] = copy(sol0); RHS[1] = copy(rhs0)
    for L in 2:nlev
        Sol[L] = zeros(Float64, dims[L]); RHS[L] = zeros(Float64, dims[L])
    end
    for _ in 1:ncyc
        for L in 1:(nlev - 1)
            for _ in 1:pre; Sol[L] = EnzoLib.mg_relax_ref(Sol[L], RHS[L]); end
            def, _ = EnzoLib.mg_calc_defect_ref(Sol[L], RHS[L])
            RHS[L+1] = EnzoLib.mg_restrict_ref(def, dims[L+1])
            fill!(Sol[L+1], 0.0)
        end
        for _ in 1:(3 * pre); Sol[nlev] = EnzoLib.mg_relax_ref(Sol[nlev], RHS[nlev]); end
        for L in (nlev - 1):-1:1
            corr = EnzoLib.mg_prolong_ref(Sol[L+1], dims[L])
            Sol[L] .+= corr
            for _ in 1:post; Sol[L] = EnzoLib.mg_relax_ref(Sol[L], RHS[L]); end
        end
    end
    return Sol[1]
end

# Same fixed-count V-cycle built from the KA kernels (CPU Float64).
function _ka_vcycle_fixed(sol0, rhs0, ncyc; pre = 2, post = 3)
    be = PoissonKernels.backend(:cpu)
    dims = PoissonKernels.mg_dims_schedule(size(sol0))
    nlev = length(dims)
    Sol = Vector{Array{Float64,3}}(undef, nlev)
    RHS = Vector{Array{Float64,3}}(undef, nlev)
    Def = Vector{Array{Float64,3}}(undef, nlev)
    Sol[1] = PoissonKernels.to_device(be, sol0, Float64)
    RHS[1] = PoissonKernels.to_device(be, rhs0, Float64)
    Def[1] = PoissonKernels.device_zeros(be, Float64, dims[1])
    for L in 2:nlev
        Sol[L] = PoissonKernels.device_zeros(be, Float64, dims[L])
        RHS[L] = PoissonKernels.device_zeros(be, Float64, dims[L])
        Def[L] = PoissonKernels.device_zeros(be, Float64, dims[L])
    end
    for _ in 1:ncyc
        for L in 1:(nlev - 1)
            for _ in 1:pre; PoissonKernels.mg_relax!(Sol[L], RHS[L]); end
            PoissonKernels.mg_calc_defect!(Def[L], Sol[L], RHS[L])
            PoissonKernels.mg_restrict!(RHS[L+1], Def[L])
            fill!(Sol[L+1], 0.0)
        end
        for _ in 1:(3 * pre); PoissonKernels.mg_relax!(Sol[nlev], RHS[nlev]); end
        for L in (nlev - 1):-1:1
            PoissonKernels.mg_prolong!(Def[L], Sol[L+1])
            Sol[L] .+= Def[L]
            for _ in 1:post; PoissonKernels.mg_relax!(Sol[L], RHS[L]); end
        end
    end
    return PoissonKernels.to_host(Sol[1])
end

@testset "V-cycle — composition + convergence" begin
    dims = (17, 17, 17)
    sol0 = zeros(Float64, dims)                       # zero guess + zero Dirichlet bndry
    rhs  = poisson_field(dims; amp = 1.0, phase = 0.5)
    rhs .-= sum(rhs) / length(rhs)                    # zero-mean RHS

    @testset "bit-tight composition vs Fortran-composed V-cycle" begin
        if !EnzoLib.grid_available()
            @test_skip "grid dylib not built"
        else
            ref = _ref_vcycle_fixed(sol0, rhs, 2)
            got = _ka_vcycle_fixed(sol0, rhs, 2)
            layerA!("vcycle.compose", got, ref)
        end
    end

    @testset "production vcycle_solve! drives the residual down" begin
        be = PoissonKernels.backend(:cpu)
        # initial residual norm (sol = 0)
        s0 = PoissonKernels.to_device(be, sol0, Float64)
        r0 = PoissonKernels.to_device(be, rhs, Float64)
        d0 = PoissonKernels.device_zeros(be, Float64, dims)
        init_norm = PoissonKernels.mg_calc_defect!(d0, s0, r0)

        s = PoissonKernels.to_device(be, sol0, Float64)
        r = PoissonKernels.to_device(be, rhs, Float64)
        _, final_norm, tol_check = PoissonKernels.vcycle_solve!(s, r; rtol = 1e-6, maxcycles = 50)

        @test isfinite(final_norm)
        @test all(isfinite, PoissonKernels.to_host(s))
        @test final_norm < init_norm / 100        # residual dropped by ≥ 2 orders
        @info "vcycle_solve!" init_norm final_norm tol_check
    end

    @testset "non-zero Dirichlet boundary (dirichlet=true) — physical-units subgrid solve" begin
        # The composite-gravity subgrid solve: a quadratic φ has an EXACT 2nd-order
        # discrete Laplacian, so with rhs=(d-1)·S²·∇²φ and the analytic boundary the
        # solver must recover φ to round-off — but ONLY with dirichlet=true. Without
        # it, mg_prolong! (which writes the full grid) lets the prolong-add drift the
        # non-zero boundary, leaving a ~1e-4 floor. Guards both the (d-1)·S² physical
        # normalization and the boundary re-imposition fix.
        be = PoissonKernels.backend(:cpu); T = Float64; d = 33; S = 0.5; x0 = 0.25
        h = S / (d - 1); coord(i) = x0 + (i - 1) * h
        φa = T[coord(i)^2 + coord(j)^2 + coord(k)^2 for i in 1:d, j in 1:d, k in 1:d]
        sol = zeros(T, d, d, d)
        sol[1,:,:] .= φa[1,:,:]; sol[d,:,:] .= φa[d,:,:]; sol[:,1,:] .= φa[:,1,:]
        sol[:,d,:] .= φa[:,d,:]; sol[:,:,1] .= φa[:,:,1]; sol[:,:,d] .= φa[:,:,d]
        rhs = fill(T((d - 1) * S^2 * 6.0), d, d, d)        # ∇²(x²+y²+z²) = 6
        s = PoissonKernels.to_device(be, sol, T); r = PoissonKernels.to_device(be, rhs, T)
        PoissonKernels.vcycle_solve!(s, r; cycle = :W, rtol = 1e-12, maxcycles = 200, dirichlet = true)
        got = PoissonKernels.to_host(s); I = 2:d-1
        rel = sqrt(sum(abs2, got[I,I,I] .- φa[I,I,I]) / sum(abs2, φa[I,I,I]))
        @test rel < 1e-8                                   # round-off recovery

        # control: WITHOUT re-imposition the non-zero boundary drifts ⇒ no round-off
        s2 = PoissonKernels.to_device(be, sol, T); r2 = PoissonKernels.to_device(be, rhs, T)
        PoissonKernels.vcycle_solve!(s2, r2; cycle = :W, rtol = 1e-12, maxcycles = 200)
        got2 = PoissonKernels.to_host(s2)
        rel2 = sqrt(sum(abs2, got2[I,I,I] .- φa[I,I,I]) / sum(abs2, φa[I,I,I]))
        @test rel2 > 10 * rel                              # the fix matters
        @info "dirichlet subgrid solve" rel rel2
    end
end
