# Differential test of the Metal multigrid gravity solver on REAL dark-matter-only
# cosmology data (the Enzo `dm_only` simulation: 32³ DM particles, 32 Mpc/h box,
# z=50 ICs from MUSIC).
#
# We don't yet have the root-grid FFT (Phase B) or the particle-deposit/comoving/AMR
# composite (Phase C), so this is a DIFFERENTIAL test, not a full gravity-slot swap:
#   1. init the live Enzo dm_only hierarchy via the bridge; read the real 32768
#      particle positions.
#   2. CIC-deposit them (periodic) → the genuine dm_only overdensity field δ.
#   3. solve ∇²φ = δ with our PoissonKernels multigrid V-cycle on CPU (f64) and
#      Metal (f32), and — the certification — compare the KA solve bit-for-bit
#      against the SAME V-cycle composed from the live Enzo Fortran kernels
#      (mg_*_ref), now on REAL cosmological input rather than a synthetic field.
#   4. compute g = -∇φ and report convergence + Metal timing.
#
# Run:
#   <julia> --project=lib/PoissonKernels/test lib/PoissonKernels/examples/dm_only_gravity.jl
# (the test project carries PoissonKernels + EnzoLib + Metal; needs the grid dylib.)

using PoissonKernels
using EnzoLib
using Printf
try
    @eval using Metal
catch err
    @info "Metal not loadable; GPU layer will be skipped" err
end

const PK = PoissonKernels
const DM_DIR = "/Users/tabel/Projects/enzo-dev/run/CosmologySimulation/dm_only"
const PARAMFILE = joinpath(DM_DIR, "dm_only.enzo")
const N = 32                                  # root-grid resolution (matches the sim)

# ── periodic Cloud-In-Cell deposit: particles (code units [0,1)³) → N³ density ──
function cic_deposit(pos::AbstractMatrix{Float64}, N::Int)
    rho = zeros(Float64, N, N, N)
    Np = size(pos, 1)
    @inbounds for p in 1:Np
        gx = mod(pos[p, 1], 1.0) * N; gy = mod(pos[p, 2], 1.0) * N; gz = mod(pos[p, 3], 1.0) * N
        i = floor(Int, gx); fx = gx - i
        j = floor(Int, gy); fy = gy - j
        k = floor(Int, gz); fz = gz - k
        i0 = mod(i, N) + 1; i1 = mod(i + 1, N) + 1
        j0 = mod(j, N) + 1; j1 = mod(j + 1, N) + 1
        k0 = mod(k, N) + 1; k1 = mod(k + 1, N) + 1
        rho[i0, j0, k0] += (1 - fx) * (1 - fy) * (1 - fz)
        rho[i1, j0, k0] += fx * (1 - fy) * (1 - fz)
        rho[i0, j1, k0] += (1 - fx) * fy * (1 - fz)
        rho[i1, j1, k0] += fx * fy * (1 - fz)
        rho[i0, j0, k1] += (1 - fx) * (1 - fy) * fz
        rho[i1, j0, k1] += fx * (1 - fy) * fz
        rho[i0, j1, k1] += (1 - fx) * fy * fz
        rho[i1, j1, k1] += fx * fy * fz
    end
    return rho
end

# ── fixed-count V-cycles (no early exit) so KA-vs-Fortran is a clean bit-compare ──
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
    Sol[1] = PK.to_device(be, sol0, T); RHS[1] = PK.to_device(be, rhs0, T)
    Def[1] = PK.device_zeros(be, T, dims[1])
    for L in 2:nlev
        Sol[L] = PK.device_zeros(be, T, dims[L]); RHS[L] = PK.device_zeros(be, T, dims[L]); Def[L] = PK.device_zeros(be, T, dims[L])
    end
    for _ in 1:ncyc
        for L in 1:(nlev - 1)
            for _ in 1:pre; PK.mg_relax!(Sol[L], RHS[L]); end
            PK.mg_calc_defect!(Def[L], Sol[L], RHS[L])
            PK.mg_restrict!(RHS[L+1], Def[L]); fill!(Sol[L+1], zero(T))
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
    EnzoLib.grid_available() || error("grid dylib not built — bash EnzoModules/deps/build_grid_darwin.sh")

    # ── 1. live dm_only hierarchy → real particle positions ──
    pos = cd(DM_DIR) do
        h = EnzoLib.session_init(PARAMFILE)
        try
            EnzoLib.read_particles(h)
        finally
            EnzoLib.free_problem(h)
        end
    end
    Np = size(pos, 1)
    @printf("dm_only: %d DM particles read from the live hierarchy (rank %d)\n", Np, size(pos, 2))

    # ── 2. CIC deposit → overdensity δ (zero-mean periodic Poisson source) ──
    rho = cic_deposit(pos, N)
    rhobar = sum(rho) / length(rho)
    delta = rho ./ rhobar .- 1.0
    @printf("CIC %d³ grid: ρ̄=%.4f  δ∈[%.4f, %.4f]  Σδ=%.2e (zero-mean)\n",
            N, rhobar, minimum(delta), maximum(delta), sum(delta))
    rhs = copy(delta)                          # ∇²φ = δ  (G=1, comoving scale folded out)
    sol0 = zeros(Float64, N, N, N)

    # ── 3a. CERTIFICATION on real data: KA-f64 ≡ Fortran-composed multigrid, bit-tight ──
    becpu = PK.backend(:cpu)
    ncyc = 4
    ref_f64 = ref_vcycle_fixed(sol0, rhs, ncyc)
    ka_f64  = ka_vcycle_fixed(becpu, sol0, rhs, ncyc, Float64)
    err_bittight = maxabs(ka_f64, ref_f64)
    @printf("\n[A] KA-f64 vs Fortran-composed V-cycle on REAL δ  (%d cycles): maxabs = %.3e %s\n",
            ncyc, err_bittight, err_bittight < 1e-10 ? "✓ bit-tight" : "✗")

    # ── 3b. production convergence on real data (CPU f64) ──
    s = PK.to_device(becpu, sol0, Float64); r = PK.to_device(becpu, rhs, Float64)
    d0 = PK.device_zeros(becpu, Float64, size(rhs)); init_norm = PK.mg_calc_defect!(d0, s, r)
    _, final_norm, tol_check = PK.vcycle_solve!(s, r; rtol = 1e-6, maxcycles = 50)
    @printf("[B] vcycle_solve! CPU f64: init_norm=%.3e → final_norm=%.3e (drop %.1e×), tol_check=%.2e\n",
            init_norm, final_norm, init_norm / final_norm, tol_check)
    phi = PK.to_host(s)

    # ── 3c. g = -∇φ (physical sanity) ──
    ng = 1; sd = (N, N, N)                     # accel on the same grid; iflag=1 symmetric
    # (interior-only finite difference; here start=0 with periodic-ish φ for a magnitude check)
    a1 = PK.device_zeros(becpu, Float64, (N - 2, N - 2, N - 2))
    a2 = PK.device_zeros(becpu, Float64, (N - 2, N - 2, N - 2))
    a3 = PK.device_zeros(becpu, Float64, (N - 2, N - 2, N - 2))
    PK.comp_accel!(a1, a2, a3, PK.to_device(becpu, phi, Float64); iflag = 1, start = (1, 1, 1), del = (1.0 / N, 1.0 / N, 1.0 / N))
    gmag = sqrt.(PK.to_host(a1) .^ 2 .+ PK.to_host(a2) .^ 2 .+ PK.to_host(a3) .^ 2)
    @printf("[C] g=-∇φ: |g|∈[%.3e, %.3e], all finite=%s\n",
            minimum(gmag), maximum(gmag), all(isfinite, gmag))

    # ── 4. Metal: f32 parity + solve timing ──
    if PK.has_backend(:metal)
        bem = PK.backend(:metal)
        # parity: cpu-f32 vs metal-f32 (fixed cycles)
        ka_cpu32 = ka_vcycle_fixed(becpu, sol0, rhs, ncyc, Float32)
        ka_mtl32 = ka_vcycle_fixed(bem,  sol0, rhs, ncyc, Float32)
        # full-solve f32 parity is looser than a single kernel: FP rounding/FMA order
        # differs CPU↔GPU and accumulates over the multigrid cycles.
        @printf("\n[D] cpu-f32 ≡ metal-f32 V-cycle on real δ: maxrel = %.3e %s\n",
                relerr(ka_mtl32, ka_cpu32), relerr(ka_mtl32, ka_cpu32) < 1e-3 ? "✓" : "✗")
        # timing: min over reps of one full vcycle_solve!
        timesolve(be, T) = begin
            best = Inf
            for _ in 1:5
                ss = PK.to_device(be, sol0, T); rr = PK.to_device(be, rhs, T)
                PK.vcycle_solve!(ss, rr; rtol = 1e-6, maxcycles = 50)     # warm + timed below
                t = @elapsed (PK.vcycle_solve!(PK.to_device(be, sol0, T), PK.to_device(be, rr, T); rtol = 1e-6, maxcycles = 50))
                best = min(best, t)
            end
            best
        end
        t_cpu = timesolve(becpu, Float64)
        t_mtl = timesolve(bem, Float32)
        @printf("[E] solve wall-time (min of 5): CPU f64 %.2f ms   Metal f32 %.2f ms   (%.2f×)\n",
                t_cpu * 1e3, t_mtl * 1e3, t_cpu / t_mtl)
    else
        println("\n[D/E] Metal backend not present — GPU parity/timing skipped.")
    end

    # ── verdict ──
    ok = err_bittight < 1e-10 && final_norm < init_norm / 100 && all(isfinite, phi)
    println("\n", ok ? "dm_only differential gravity test: PASS" : "dm_only differential gravity test: FAIL")
    return ok
end

ok = main()
exit(ok ? 0 : 1)
