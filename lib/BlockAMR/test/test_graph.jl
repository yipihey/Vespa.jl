# test_graph.jl — CUDA-graph replay of root steps (advance_hierarchy_graph!) is
# bitwise the plain W-cycle at the same (ladder-rounded) λ, through graph reuse,
# R/O parity alternation and a mid-run regrid (re-record).  CUDA only.
if :cuda in BACKENDS
@testset "CUDA graph root steps ≡ plain launches" begin
    mk() = (h = static2d(; nbase = (64, 32), kids = KIDS_CENTER, backend = :cuda, physics = :mhd,
                         divb = :ct, bc = (:outflow, :periodic));
            setall!(h, x -> (1 + 0.3sin(2π * x[1]) * cos(2π * x[2]), 0.5, 0.1, 0.0, 1.0)); h)
    hA = mk(); hB = mk()
    for n in 1:14
        advance_hierarchy_graph!(hA)
        λ = BA.compute_lambda!(hB); k = floor(Int, log2(Float64(λ)) * 16); λq = Float32(exp2(k / 16))
        hB.λ = λq
        BA.advance_level_w!(hB, 0, λq; φ = nothing, chem = nothing, selfgrav = nothing, cooling = nothing)
        if n == 7                      # a regrid changes topology → fresh record
            pol = BlockRefinementPolicy(; dthresh = 1.2, nbuf = 2, every = 1, lmax = 1)
            regrid!(hA, pol); regrid!(hB, pol)
        end
    end
    st = BA.graph_stats()
    @test st.replays > 0
    @test hA.nstep[1] == hB.nstep[1]
    for l in 1:length(hA.levels), f in (:D, :S1, :S2, :Tau, :Ge, :Bx, :By, :FBx, :FBy)
        @test Array(getfield(hA.levels[l], f)) == Array(getfield(hB.levels[l], f))
    end
end
end
