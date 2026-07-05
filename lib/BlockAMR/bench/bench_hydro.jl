# bench_hydro.jl — RK2 vs tiled-CTU hydro substep throughput on a uniform
# level-0 grid (the baseline config: 256³, B=16 → 4096 blocks / B=64 → 64).
# Times the FULL substep as the W-cycle runs it: ghost fill + kernel(s).
#   julia --project=lib/BlockAMR/test lib/BlockAMR/bench/bench_hydro.jl [cuda|cpu]
using BlockAMR, Printf
const BA = BlockAMR
BE = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :cuda
BE === :cuda && (import CUDA)

function setup(B; nbase = 256, scheme = :rk2)
    hier = AMRHierarchy(; nbase = (nbase, nbase, nbase), B, backend = BE,
                        T = Float16, nsp = 2, scheme)
    lev = init_base_level!(hier); build_level_tables!(hier, 0)
    σ = 8.0 / nbase
    n = lev.cap * lev.stride
    h = Dict(f => zeros(Float64, n) for f in (:D, :S1, :S2, :S3, :Tau, :Ge))
    γ = hier.gamma
    for s in lev.live
        m = lev.meta[s]; base = (Int(s) - 1) * lev.stride
        for k in 1:lev.B, j in 1:lev.B, i in 1:lev.B
            x = ntuple(d -> (Float64((m.origin[d])) + (i, j, k)[d] - 0.5) / nbase, 3)
            r2 = sum(abs2, x .- 0.5)
            ρ = 1.0 + 0.5 * exp(-r2 / (2σ^2))
            P = 0.6 + 0.4 * exp(-r2 / (2σ^2))
            idx = base + ((lev.ng+k-1) * lev.nd + (lev.ng+j-1)) * lev.nd + (lev.ng+i-1) + 1
            ge = P / (γ - 1)
            h[:D][idx] = ρ; h[:S1][idx] = 0.3ρ; h[:S2][idx] = -0.2ρ; h[:S3][idx] = 0.1ρ
            h[:Ge][idx] = ge; h[:Tau][idx] = ge + 0.5ρ * (0.3^2 + 0.2^2 + 0.1^2)
        end
    end
    BA.encode_from_host!(lev, h[:D], h[:S1], h[:S2], h[:S3], h[:Tau], h[:Ge])
    fill!(lev.sp[1], BA.ChemistryKernels.encode_log2sp(1.0f-4))
    fill!(lev.sp[2], BA.ChemistryKernels.encode_log2sp(3.0f-6))
    update_scales!(hier, 0)
    return hier
end

function substep!(hier, λ)
    lev = hier.levels[1]
    if hier.scheme === :ctu
        BA.fill_ghosts!(hier, 0; θ = 0.0f0, buf = :R)
        ctu_level!(hier, 0, λ)
        BA.swap_buffers!(lev)
    else
        BA.fill_ghosts!(hier, 0; θ = 0.0f0, buf = :R)
        stage_level!(hier, 0, λ; w = 0.0f0, IN = :R, OUT = :O)
        BA.fill_ghosts!(hier, 0; θ = 0.0f0, buf = :O)
        stage_level!(hier, 0, λ; w = 0.5f0, IN = :O, OUT = :R)
    end
    return nothing
end

sync() = BE === :cuda ? CUDA.synchronize() : nothing

for B in (16, 64), scheme in (:rk2, :ctu)
    hier = setup(B; scheme)
    λ = 0.1f0
    for _ in 1:3                                       # warmup + compile
        substep!(hier, λ)
    end
    sync()
    nrep = 20
    t0 = time_ns()
    for _ in 1:nrep
        substep!(hier, λ)
    end
    sync()
    dt = (time_ns() - t0) / 1e9
    ncell = length(hier.levels[1].live) * B^3
    @printf("%s  B=%2d  %8.1f Mcell/s   (%d blocks, %d substeps, %.3f s)\n",
            scheme, B, ncell * nrep / dt / 1e6, length(hier.levels[1].live), nrep, dt)
end
