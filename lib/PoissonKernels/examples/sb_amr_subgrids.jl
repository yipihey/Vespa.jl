# Do the multigrid mods (W-cycle, relative-residual stop) help in the AMR case?
#
# Enzo uses multigrid ONLY on refined SUBGRIDS (the root uses FFT). Those subgrids
# are small, irregularly shaped, and — in a real run — WARM-STARTED from the
# parent potential interpolated to the subgrid. This script extracts real Santa
# Barbara cluster subgrids (after session_rebuild), builds each one's Poisson
# source from its gas density, and measures how many V-cycles vs W-cycles it takes
# to converge under a COLD start (φ=0) vs a WARM start (parent-interpolated guess,
# mimicked as the converged solution with its fine-scale detail removed).
#
# Run:  <julia> --project=lib/PoissonKernels/test lib/PoissonKernels/examples/sb_amr_subgrids.jl

using PoissonKernels, EnzoLib, Printf
const PK = PoissonKernels
const PF = "/Users/tabel/Projects/enzo-dev/run/CosmologySimulation/SantaBarbaraCluster/SB_amr.enzo"

# Count μ-cycles to converge (relative residual ≤ rtol or stagnation), CPU f64.
function converge(delta::Array{Float64,3}, warm::Array{Float64,3}, cyc::Symbol;
                  pre=2, post=3, rtol=1e-6, maxc=60, stag=0.985)
    be = PK.backend(:cpu); T = Float64; d0 = size(delta)
    dims = PK.mg_dims_schedule(d0); nlev = length(dims); mu = cyc === :W ? 2 : 1
    Sol = Vector{Any}(undef,nlev); RHS = Vector{Any}(undef,nlev); Def = Vector{Any}(undef,nlev)
    Sol[1] = PK.to_device(be, warm, T); RHS[1] = PK.to_device(be, delta, T); Def[1] = PK.device_zeros(be,T,d0)
    for L in 2:nlev
        Sol[L]=PK.device_zeros(be,T,dims[L]); RHS[L]=PK.device_zeros(be,T,dims[L]); Def[L]=PK.device_zeros(be,T,dims[L])
    end
    norm0 = PK.mg_calc_defect!(Def[1], Sol[1], RHS[1])
    norm0 == 0 && return (0, 0.0, nlev)
    norm = norm0; prev = Inf; it = 0
    while it < maxc
        PK._mu_cycle!(Sol, RHS, Def, 1, nlev, pre, post, mu)
        norm = PK.mg_calc_defect!(Def[1], Sol[1], RHS[1]); it += 1
        (norm ≤ rtol*norm0 || norm ≥ stag*prev) && break
        prev = norm
    end
    return (it, (norm/norm0)^(1/max(it,1)), nlev)
end

# A representative structured Poisson source on a real subgrid shape: a smooth
# large-scale mode + a localized central peak (a forming halo). At z=63 the real
# SB field is still ~uniform (δ≈0, no structure → zero source), so we test the
# solver's convergence on the real AMR *geometry* (small, irregular subgrids) with
# a non-degenerate overdensity representative of when multigrid actually does work.
function source_field(d::NTuple{3,Int})
    a = Array{Float64,3}(undef, d)
    @inbounds for k in 1:d[3], j in 1:d[2], i in 1:d[1]
        x=(i-1)/max(d[1]-1,1); y=(j-1)/max(d[2]-1,1); z=(k-1)/max(d[3]-1,1)
        a[i,j,k] = sinpi(2x)*sinpi(2y)*sinpi(2z) +
                   3.0*exp(-(((x-0.5)^2+(y-0.5)^2+(z-0.5)^2))/0.02)
    end
    a .-= sum(a)/length(a)
    return a
end

# parent-interpolated warm start ≈ the converged φ with its finest-scale detail
# removed (restrict to the coarse parent resolution, prolong back).
function warm_guess(delta)
    be = PK.backend(:cpu); d0 = size(delta)
    sol = PK.device_zeros(be, Float64, d0); rhs = PK.to_device(be, delta, Float64)
    PK.vcycle_solve!(sol, rhs; cycle=:W, rtol=1e-10, maxcycles=40)   # true φ
    phi = PK.to_host(sol)
    cd = ((d0[1]+1)÷2, (d0[2]+1)÷2, (d0[3]+1)÷2)
    minimum(cd) < 3 && return zeros(Float64, d0)
    coarse = PK.device_zeros(be, Float64, cd); back = PK.device_zeros(be, Float64, d0)
    PK.mg_restrict!(coarse, PK.to_device(be, phi, Float64))
    PK.mg_prolong!(back, coarse)
    return PK.to_host(back)
end

cd(dirname(PF)) do
    h = EnzoLib.session_init(PF)
    try
        EnzoLib.session_set_boundary(h, 0); EnzoLib.session_rebuild(h, 0)
        # gather level-1 subgrids, pick representatives by active size
        idxs = [EnzoLib.problem_grid_index_on_level(h,1,i) for i in 0:EnzoLib.session_num_grids_on_level(h,1)-1]
        sized = sort([(EnzoLib.problem_grid_size(h,gi), gi) for gi in idxs])
        picks = [sized[1], sized[length(sized)÷4], sized[length(sized)÷2],
                 sized[3*length(sized)÷4], sized[end]]
        @printf("\nSB cluster AMR: %d level-1 subgrids. Multigrid convergence on real subgrids (CPU f64, rtol=1e-6):\n\n", length(idxs))
        @printf("%-16s %-5s | %-22s | %-22s\n", "subgrid dims", "lvls", "COLD start (φ=0)", "WARM start (parent-interp)")
        @printf("%-16s %-5s | %-10s %-10s | %-10s %-10s\n", "", "", "V cyc(fac)", "W cyc(fac)", "V cyc(fac)", "W cyc(fac)")
        println("-"^78)
        for (sz, gi) in picks
            d = EnzoLib.problem_grid_dims(h, gi)                      # real refined-subgrid shape
            delta = source_field((d[1], d[2], d[3]))                  # representative structured source
            cold = zeros(Float64, d[1], d[2], d[3])
            warm = warm_guess(delta)
            (cV,fV,nl) = converge(delta, cold, :V); (cW,fW,_) = converge(delta, cold, :W)
            (wV,gV,_)  = converge(delta, warm, :V); (wW,gW,_) = converge(delta, warm, :W)
            @printf("%-16s %-5d | %-2d (%.2f)   %-2d (%.2f)   | %-2d (%.2f)   %-2d (%.2f)\n",
                    string((d[1],d[2],d[3])), nl, cV,fV, cW,fW, wV,gV, wW,gW)
        end
    finally
        EnzoLib.free_problem(h)
    end
end
