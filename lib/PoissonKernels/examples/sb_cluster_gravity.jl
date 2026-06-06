# Differential test of the Metal multigrid gravity solver on the SANTA BARBARA
# CLUSTER comparison ICs (Frenk et al. 1999): 128³ top grid, Ω_b=0.1 gas +
# Ω_CDM=0.9 dark matter (2,097,152 = 128³ particles), 32 Mpc/h box, z=63.
#
# As with the dm_only test (Phase B root-FFT + Phase C deposit/AMR not built yet),
# this is a DIFFERENTIAL test, not a full gravity-slot swap:
#   1. init the live Enzo SB-cluster hierarchy via the bridge; read the real GAS
#      density field AND the 2.1M DM particle positions.
#   2. build the total gravitating overdensity δ = (ρ_gas + ρ_DM)/ρ̄ − 1, with ρ_DM
#      from a periodic CIC deposit of the real particles.
#   3. solve ∇²φ = δ with PoissonKernels multigrid on CPU (f64) and Metal (f32),
#      certify the KA solve bit-for-bit against the SAME V-cycle composed from the
#      live Enzo Fortran kernels (mg_*_ref), and report convergence + Metal timing.
#
# The 128³ grid (2.1M cells, 64× the dm_only test) is large enough to actually
# amortize GPU kernel-launch latency, so [E] is the meaningful Metal perf datapoint.
#
# Run:
#   <julia> --project=lib/PoissonKernels/test lib/PoissonKernels/examples/sb_cluster_gravity.jl

using PoissonKernels
using EnzoLib
using Printf
try
    @eval using Metal
catch err
    @info "Metal not loadable; GPU layer will be skipped" err
end

const PK = PoissonKernels
const SB_DIR = "/Users/tabel/Projects/enzo-dev/run/CosmologySimulation/SantaBarbaraCluster"
const PARAMFILE = joinpath(SB_DIR, "SantaBarbaraCluster.enzo")
const N = 128                                  # top-grid resolution
const OMEGA_B = 0.1                            # CosmologySimulationOmegaBaryonNow
const OMEGA_CDM = 0.9                          # CosmologySimulationOmegaCDMNow

# periodic Cloud-In-Cell deposit: particles (code units [0,1)³) → N³ density (mean 1).
function cic_deposit(pos::AbstractMatrix{Float64}, N::Int)
    rho = zeros(Float64, N, N, N)
    @inbounds for p in 1:size(pos, 1)
        gx = mod(pos[p, 1], 1.0) * N; gy = mod(pos[p, 2], 1.0) * N; gz = mod(pos[p, 3], 1.0) * N
        i = floor(Int, gx); fx = gx - i
        j = floor(Int, gy); fy = gy - j
        k = floor(Int, gz); fz = gz - k
        i0 = mod(i, N) + 1; i1 = mod(i + 1, N) + 1
        j0 = mod(j, N) + 1; j1 = mod(j + 1, N) + 1
        k0 = mod(k, N) + 1; k1 = mod(k + 1, N) + 1
        rho[i0, j0, k0] += (1 - fx) * (1 - fy) * (1 - fz); rho[i1, j0, k0] += fx * (1 - fy) * (1 - fz)
        rho[i0, j1, k0] += (1 - fx) * fy * (1 - fz);       rho[i1, j1, k0] += fx * fy * (1 - fz)
        rho[i0, j0, k1] += (1 - fx) * (1 - fy) * fz;       rho[i1, j0, k1] += fx * (1 - fy) * fz
        rho[i0, j1, k1] += (1 - fx) * fy * fz;             rho[i1, j1, k1] += fx * fy * fz
    end
    return rho
end

# fixed-count V-cycles (no early exit) for a clean KA-vs-Fortran bit-compare.
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

    # ── 1. live SB hierarchy → real gas density (active) + DM particle positions ──
    gas_active, pos = cd(SB_DIR) do
        h = EnzoLib.session_init(PARAMFILE)
        try
            gd = EnzoLib.problem_grid_dims(h, 0)           # ghosted dims e.g. [134,134,134]
            ng = (gd[1] - N) ÷ 2                            # ghost zones each side
            densflat = EnzoLib.read_density(h; grid = 0)   # FieldType 0 = gas Density
            dens = reshape(densflat, gd[1], gd[2], gd[3])  # column-major, matches Enzo
            a = ng + 1; b = ng + N
            gas = Array(dens[a:b, a:b, a:b])               # active N³
            (gas, EnzoLib.read_particles(h))
        finally
            EnzoLib.free_problem(h)
        end
    end
    Np = size(pos, 1)
    @printf("SB cluster: gas density %s (mean %.4f), %d DM particles\n",
            string(size(gas_active)), sum(gas_active) / length(gas_active), Np)

    # ── 2. total gravitating overdensity δ = (ρ_gas + ρ_DM)/ρ̄ − 1 ──
    dm = cic_deposit(pos, N)
    dm .*= OMEGA_CDM / (sum(dm) / length(dm))      # normalize DM mean → Ω_CDM
    gas = gas_active .* (OMEGA_B / (sum(gas_active) / length(gas_active)))  # gas mean → Ω_b
    rho_tot = gas .+ dm
    rhobar = sum(rho_tot) / length(rho_tot)
    delta = rho_tot ./ rhobar .- 1.0
    @printf("total δ on %d³: ρ̄=%.4f  δ∈[%.3f, %.3f]  Σδ=%.2e\n",
            N, rhobar, minimum(delta), maximum(delta), sum(delta))
    rhs = copy(delta); sol0 = zeros(Float64, N, N, N)

    becpu = PK.backend(:cpu)
    # ── 3a. CERTIFICATION on real cluster data: KA-f64 ≡ Fortran-composed multigrid ──
    ncyc = 2
    @printf("\nrunning %d fixed V-cycles on %d³ (Fortran-composed reference)...\n", ncyc, N)
    ref_f64 = ref_vcycle_fixed(sol0, rhs, ncyc)
    ka_f64  = ka_vcycle_fixed(becpu, sol0, rhs, ncyc, Float64)
    err_bt = maxabs(ka_f64, ref_f64)
    @printf("[A] KA-f64 vs Fortran-composed V-cycle on REAL δ (%d cycles): maxabs = %.3e %s\n",
            ncyc, err_bt, err_bt < 1e-10 ? "✓ bit-tight" : "✗")

    # ── 3b. production convergence (CPU f64) ──
    s = PK.to_device(becpu, sol0, Float64); r = PK.to_device(becpu, rhs, Float64)
    d0 = PK.device_zeros(becpu, Float64, size(rhs)); init_norm = PK.mg_calc_defect!(d0, s, r)
    _, final_norm, tol_check = PK.vcycle_solve!(s, r; rtol = 1e-6, maxcycles = 50)
    @printf("[B] vcycle_solve! CPU f64: init_norm=%.3e → final_norm=%.3e (drop %.1e×), tol_check=%.2e\n",
            init_norm, final_norm, init_norm / final_norm, tol_check)
    phi = PK.to_host(s)

    # ── 4. Metal: f32 parity + solve timing (the meaningful perf point at 128³) ──
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
                N, t_cpu * 1e3, t_mtl * 1e3, t_cpu / t_mtl, t_cpu > t_mtl ? "Metal faster" : "CPU faster")
    else
        println("\n[D/E] Metal backend not present — GPU parity/timing skipped.")
    end

    ok = err_bt < 1e-10 && final_norm < init_norm / 100 && all(isfinite, phi)
    println("\n", ok ? "SB cluster differential gravity test: PASS" : "SB cluster differential gravity test: FAIL")
    return ok
end

ok = main()
exit(ok ? 0 : 1)
