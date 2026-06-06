# Differential test + PERFORMANCE comparison of the Metal multigrid gravity solver
# on the SWIFT 256³ Santa Barbara Cluster ICs (DM-only: 256³ = 16,777,216 dark
# matter particles, 32 Mpc/h box; gas is split from DM at runtime in SWIFT, so the
# IC's gravitating mass is the DM particles).
#
# The SWIFT IC is gadget-style HDF5. We do NOT read it here: HDF5.jl and EnzoLib's
# grid dylib each load a DIFFERENT libhdf5 (HDF5_jll vs Homebrew), and two libhdf5
# in one process abort. So `sb256_deposit.jl` reads the SWIFT file + CIC-deposits in
# a SEPARATE process and writes δ as a raw binary; this process reads that binary and
# never loads HDF5.jl. Bit-tight certification comes from the live Enzo Fortran
# kernels via the mg_*_ref bridge wrappers (plain-array, no session required).
#
#   1. (sb256_deposit.jl) SWIFT 16.7M DM particles → periodic CIC → 256³ δ binary.
#   2. (here) solve ∇²φ = δ with PoissonKernels multigrid on CPU (f64) + Metal (f32);
#      certify KA-f64 ≡ Fortran-composed multigrid bit-for-bit; report convergence
#      and CPU-vs-Metal solve timing — the 256³ point in the 32³→128³→256³ scan.
#
# Run (deposit first):
#   <julia> --project=lib/PoissonKernels/test lib/PoissonKernels/examples/sb256_deposit.jl
#   <julia> --project=lib/PoissonKernels/test lib/PoissonKernels/examples/sb256_gravity.jl

using PoissonKernels
using EnzoLib
using Printf
try
    @eval using Metal
catch err
    @info "Metal not loadable; GPU layer will be skipped" err
end

const PK = PoissonKernels
const DELTA_BIN = "/tmp/sb256_delta.bin"        # written by sb256_deposit.jl

function read_delta(path)
    open(path, "r") do io
        N = Int(read(io, Int64))
        d = Vector{Float64}(undef, N^3)
        read!(io, d)
        return reshape(d, N, N, N)
    end
end

function ref_vcycle_fixed(sol0, rhs0, ncyc; pre = 2, post = 3)
    dims = PK.mg_dims_schedule(size(sol0)); nlev = length(dims)
    Sol = Vector{Array{Float64,3}}(undef, nlev); RHS = Vector{Array{Float64,3}}(undef, nlev)
    Sol[1] = copy(sol0); RHS[1] = copy(rhs0)
    for L in 2:nlev; Sol[L] = zeros(dims[L]); RHS[L] = zeros(dims[L]); end
    for _ in 1:ncyc
        for L in 1:(nlev - 1)
            for _ in 1:pre; Sol[L] = EnzoLib.mg_relax_ref(Sol[L], RHS[L]); end
            def, _ = EnzoLib.mg_calc_defect_ref(Sol[L], RHS[L])
            RHS[L+1] = EnzoLib.mg_restrict_ref(def, dims[L+1]); fill!(Sol[L+1], 0.0)
        end
        for _ in 1:(3 * pre); Sol[nlev] = EnzoLib.mg_relax_ref(Sol[nlev], RHS[nlev]); end
        for L in (nlev - 1):-1:1
            Sol[L] .+= EnzoLib.mg_prolong_ref(Sol[L+1], dims[L])
            for _ in 1:post; Sol[L] = EnzoLib.mg_relax_ref(Sol[L], RHS[L]); end
        end
    end
    return Sol[1]
end

function ka_vcycle_fixed(be, sol0, rhs0, ncyc, ::Type{T}; pre = 2, post = 3) where {T}
    dims = PK.mg_dims_schedule(size(sol0)); nlev = length(dims)
    Sol = Vector{Any}(undef, nlev); RHS = Vector{Any}(undef, nlev); Def = Vector{Any}(undef, nlev)
    Sol[1] = PK.to_device(be, sol0, T); RHS[1] = PK.to_device(be, rhs0, T); Def[1] = PK.device_zeros(be, T, dims[1])
    for L in 2:nlev
        Sol[L] = PK.device_zeros(be, T, dims[L]); RHS[L] = PK.device_zeros(be, T, dims[L]); Def[L] = PK.device_zeros(be, T, dims[L])
    end
    for _ in 1:ncyc
        for L in 1:(nlev - 1)
            for _ in 1:pre; PK.mg_relax!(Sol[L], RHS[L]); end
            PK.mg_calc_defect!(Def[L], Sol[L], RHS[L]); PK.mg_restrict!(RHS[L+1], Def[L]); fill!(Sol[L+1], zero(T))
        end
        for _ in 1:(3 * pre); PK.mg_relax!(Sol[nlev], RHS[nlev]); end
        for L in (nlev - 1):-1:1
            PK.mg_prolong!(Def[L], Sol[L+1]); Sol[L] .+= Def[L]
            for _ in 1:post; PK.mg_relax!(Sol[L], RHS[L]); end
        end
    end
    return PK.to_host(Sol[1])
end

maxabs(a, b) = maximum(abs.(vec(a) .- vec(b)))
relerr(a, b) = maxabs(a, b) / max(maximum(abs, b), eps())

function main()
    EnzoLib.grid_available() || error("grid dylib not built")
    isfile(DELTA_BIN) || error("δ binary not found: $DELTA_BIN — run sb256_deposit.jl first")

    # ── 1+2. load the precomputed 256³ overdensity δ (SWIFT particles → CIC) ──
    delta = read_delta(DELTA_BIN)
    N = size(delta, 1)
    @printf("loaded δ from %s: %d³  δ∈[%.3f, %.3f]  Σδ=%.2e\n",
            DELTA_BIN, N, minimum(delta), maximum(delta), sum(delta))
    rhs = copy(delta); sol0 = zeros(Float64, N, N, N)

    becpu = PK.backend(:cpu)
    # ── 3a. CERTIFICATION on real 256³ data: KA-f64 ≡ Fortran-composed multigrid ──
    ncyc = 2
    @printf("\nrunning %d fixed V-cycles on %d³ (Fortran-composed reference; this is the slow step)...\n", ncyc, N)
    t_ref = @elapsed (ref_f64 = ref_vcycle_fixed(sol0, rhs, ncyc))
    ka_f64 = ka_vcycle_fixed(becpu, sol0, rhs, ncyc, Float64)
    err_bt = maxabs(ka_f64, ref_f64)
    @printf("[A] KA-f64 vs Fortran-composed V-cycle on REAL δ (%d cycles, ref %.1fs): maxabs = %.3e %s\n",
            ncyc, t_ref, err_bt, err_bt < 1e-10 ? "✓ bit-tight" : "✗")

    # ── 3b. production convergence (CPU f64) ──
    s = PK.to_device(becpu, sol0, Float64); r = PK.to_device(becpu, rhs, Float64)
    d0 = PK.device_zeros(becpu, Float64, size(rhs)); init_norm = PK.mg_calc_defect!(d0, s, r)
    _, final_norm, tol_check = PK.vcycle_solve!(s, r; rtol = 1e-6, maxcycles = 50)
    @printf("[B] vcycle_solve! CPU f64: init_norm=%.3e → final_norm=%.3e (drop %.1e×), tol_check=%.2e\n",
            init_norm, final_norm, init_norm / final_norm, tol_check)
    phi = PK.to_host(s)

    # ── 4. Metal: f32 parity + solve timing ──
    if PK.has_backend(:metal)
        bem = PK.backend(:metal)
        ka_cpu32 = ka_vcycle_fixed(becpu, sol0, rhs, ncyc, Float32)
        ka_mtl32 = ka_vcycle_fixed(bem,  sol0, rhs, ncyc, Float32)
        @printf("\n[D] cpu-f32 ≡ metal-f32 V-cycle on real δ: maxrel = %.3e %s\n",
                relerr(ka_mtl32, ka_cpu32), relerr(ka_mtl32, ka_cpu32) < 1e-3 ? "✓" : "✗")
        timesolve(be, T) = begin
            best = Inf
            for _ in 1:3
                PK.vcycle_solve!(PK.to_device(be, sol0, T), PK.to_device(be, rhs, T); rtol = 1e-6, maxcycles = 50)  # warm
                t = @elapsed PK.vcycle_solve!(PK.to_device(be, sol0, T), PK.to_device(be, rhs, T); rtol = 1e-6, maxcycles = 50)
                best = min(best, t)
            end
            best
        end
        t_cpu = timesolve(becpu, Float64); t_mtl = timesolve(bem, Float32)
        @printf("[E] solve wall-time (min of 3) on %d³: CPU f64 %.1f ms   Metal f32 %.1f ms   (%.2f× %s)\n",
                N, t_cpu * 1e3, t_mtl * 1e3, t_cpu / t_mtl, t_cpu > t_mtl ? "Metal FASTER" : "CPU faster")
    else
        println("\n[D/E] Metal backend not present — GPU parity/timing skipped.")
    end

    ok = err_bt < 1e-10 && final_norm < init_norm / 100 && all(isfinite, phi)
    println("\n", ok ? "SB-256 differential gravity test: PASS" : "SB-256 differential gravity test: FAIL")
    return ok
end

ok = main()
exit(ok ? 0 : 1)
