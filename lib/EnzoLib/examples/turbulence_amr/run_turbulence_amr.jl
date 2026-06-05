# Decaying turbulence with OVERDENSITY AMR, driven through the live-Enzo bridge.
#
# This uses Enzo's ACTUAL AMR infrastructure to flag → refine → evolve:
#   • session_rebuild(h, 0)  = Enzo's RebuildHierarchy: SetFlaggingField (overdensity,
#     CellFlaggingMethod=2) → Berger-Rigoutsos clustering → create subgrids → conserva-
#     tive interpolation of the fine grids.
#   • evolve_level!(h, 0, 0)  = Enzo's recursive EvolveLevel: per-grid HD_RK solve,
#     subcycle the finer levels, UpdateFromFinerGrids (flux correction + projection).
# The base level is a UNIFORM-density solenoidal-turbulence box injected from Julia
# (problem_set_field) — the same IC as the PPMKernels bench — so we control exactly
# where overdensities form and watch refinement trigger as the supersonic turbulence
# decays.
#
# Run:  <juliaup-julia> --project=test \
#         lib/EnzoLib/examples/turbulence_amr/run_turbulence_amr.jl [mach] [overdensity] [maxlevel] [cycles]

using EnzoLib, Random, LinearAlgebra, Printf
const NG  = 3                                                   # HD_RK ghost zones
const PFSRC = joinpath(@__DIR__, "decaying_turbulence_amr.enzo")

# Inject a uniform-density solenoidal turbulence IC into the live top grid (overwrites
# Density / Velocity1-3 / TotalEnergy / InternalEnergy). Σ_k A_k ê⊥(k) cos(2π k·x+φ),
# A_k ∝ |k|^(-specidx/2), normalized to the target RMS Mach over the active interior.
function inject_turbulence!(h; mach, gamma, cs = 1.0, seed = 271, kmin = 2, kmax = 3, specidx = 4.0)
    nx, ny, nz = EnzoLib.problem_grid_dims(h, 0); N = nx * ny * nz
    nax = nx - 2NG; dxn = 1.0 / nax
    X(i) = (i - NG + 0.5) * dxn                                 # periodic cell-centre coord
    Random.seed!(seed)
    vx = zeros(N); vy = zeros(N); vz = zeros(N)
    modes = [(kx, ky, kz) for kx in -kmax:kmax, ky in -kmax:kmax, kz in -kmax:kmax
             if kmin^2 <= kx^2 + ky^2 + kz^2 <= kmax^2]
    for (kx, ky, kz) in modes
        kk = sqrt(kx^2 + ky^2 + kz^2); amp = kk^(-specidx / 2); kh = (kx, ky, kz) ./ kk
        a = randn(3); a .-= dot(a, collect(kh)) .* collect(kh); na = norm(a)
        na < 1e-12 && continue; a ./= na
        φ = 2π * rand(); a1, a2, a3 = amp .* a
        @inbounds for k in 0:nz-1, j in 0:ny-1, i in 0:nx-1
            s = cos(2π * (kx * X(i) + ky * X(j) + kz * X(k)) + φ)
            q = i + nx * j + nx * ny * k + 1
            vx[q] += a1 * s; vy[q] += a2 * s; vz[q] += a3 * s
        end
    end
    s2 = 0.0; nc = 0
    @inbounds for k in NG:nz-NG-1, j in NG:ny-NG-1, i in NG:nx-NG-1
        q = i + nx * j + nx * ny * k + 1; s2 += vx[q]^2 + vy[q]^2 + vz[q]^2; nc += 1
    end
    f = mach * cs / sqrt(s2 / nc); vx .*= f; vy .*= f; vz .*= f
    eint0 = (cs^2 / gamma) / (gamma - 1)
    D = ones(N); TE = eint0 .+ 0.5 .* (vx .^ 2 .+ vy .^ 2 .+ vz .^ 2); IE = fill(eint0, N)
    EnzoLib.problem_set_field(h, 0, D); EnzoLib.problem_set_field(h, 1, vx)
    EnzoLib.problem_set_field(h, 2, vy); EnzoLib.problem_set_field(h, 3, vz)
    EnzoLib.problem_set_field(h, 4, TE); EnzoLib.problem_set_field(h, 5, IE)
    return nothing
end

# total fine cells across a level (refined-volume diagnostic). Grids on a level are
# 0-indexed: problem_grid_index_on_level(h, level, i) for i ∈ 0 … N−1 (out-of-range
# returns −1, which problem_grid_size would dereference — so iterate 0-based).
fine_cells(h, level) = sum(EnzoLib.problem_grid_size(h, EnzoLib.problem_grid_index_on_level(h, level, i))
                           for i in 0:EnzoLib.session_num_grids_on_level(h, level)-1; init = 0)

function main()
    mach   = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 5.0
    overd  = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 2.0
    maxlvl = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 2
    maxcyc = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 120
    EnzoLib.grid_available() || error("grid dylib not built (EnzoModules/deps/libenzomodules_grid.dylib)")

    # stage the param file in a temp workdir, patching the overdensity + max level
    work = mktempdir(); pf = joinpath(work, "decaying_turbulence_amr.enzo")
    txt = read(PFSRC, String)
    txt = replace(txt, r"MinimumOverDensityForRefinement = .*" => "MinimumOverDensityForRefinement = $overd")
    txt = replace(txt, r"MaximumRefinementLevel     = .*" => "MaximumRefinementLevel     = $maxlvl")
    write(pf, txt)

    cd(work) do
        h = EnzoLib.session_init(pf); h == C_NULL && error("session_init failed")
        try
            inject_turbulence!(h; mach = mach, gamma = 1.4)
            ρ = EnzoLib.problem_get_field(h, 0, 0)
            @printf("\nDecaying turbulence + overdensity AMR (Enzo HD_RK): Mach0=%.1f  ρ>%.1f flags  maxlevel=%d\n",
                    mach, overd, maxlvl)
            @printf("injected uniform box: ρ mean=%.4f min=%.4f max=%.4f\n",
                    sum(ρ)/length(ρ), minimum(ρ), maximum(ρ))
            EnzoLib.session_rebuild(h, 0)
            ngl() = Int[EnzoLib.session_num_grids_on_level(h, l) for l in 0:maxlvl]
            m0 = EnzoLib.session_global_field_integral(h, 0)
            @printf("%-5s %-9s %-16s %-7s %-9s %-10s\n", "cyc", "t", "grids/level", "ρmax", "fine%", "Δmass/M")
            @printf("init                  %-16s\n", ngl())
            cyc = 0
            while EnzoLib.session_time(h) < EnzoLib.session_stop_time(h) && cyc < maxcyc
                EnzoLib.evolve_level!(h, 0, 0.0; regrid = true)
                EnzoLib.session_rebuild(h, 0)
                ρ = EnzoLib.problem_get_field(h, 0, 0)
                m = EnzoLib.session_global_field_integral(h, 0)
                finepct = 100 * sum(fine_cells(h, l) for l in 1:maxlvl; init = 0) / EnzoLib.problem_grid_size(h, 0)
                @printf("%-5d %-9.4f %-16s %-7.2f %-9.2f %-10.1e\n",
                        cyc, EnzoLib.session_time(h), ngl(), maximum(ρ), finepct, abs(m - m0) / m0)
                cyc += 1
            end
        finally
            EnzoLib.free_problem(h)
        end
    end
end

main()
