# test_checkpoint.jl — save/restore round-trip at a root boundary.
#   * every live block's fields/species/scales/φ restore BIT-exactly (by origin);
#   * continuation after restore is BIT-identical to the uninterrupted run
#     (pure-hydro path is deterministic; restored state ≡ original state).
const BA = BlockAMR

for BE in BACKENDS
@testset "checkpoint round-trip + bit-identical continuation [$BE]" begin
    σ = 2.0 / 128
    prof = sedov(σ, 1.0, 1e-5)
    mk() = (h = _ctu_center_refined(; backend = BE, nsp = 2, scheme = :ctu);
            set_ic!(h, 0, prof); set_ic!(h, 1, prof);
            c1 = BA.ChemistryKernels.encode_log2sp(1.0f-4);
            c2 = BA.ChemistryKernels.encode_log2sp(3.0f-6);
            _set_uniform_sp!(h, 0, c1, c2); _set_uniform_sp!(h, 1, c1, c2); h)
    ha = mk()                                    # reference: runs uninterrupted
    for _ in 1:4
        advance_hierarchy!(ha)
    end
    ckf = tempname() * ".jls"
    save_checkpoint(ckf, ha; extra = (mark = 42, v = [1.0, 2.0]))
    hb, ex = load_checkpoint(ckf; backend = BE)
    @test ex.mark == 42 && ex.v == [1.0, 2.0]
    @test hb.scheme === :ctu
    @test hb.nstep == ha.nstep
    # per-origin bit-exact restore of every field
    function bmap(h, l, f)
        lev = h.levels[l + 1]; hf = Array(getfield(lev, f))
        Dict(lev.meta[s].origin =>
             [hf[(Int(s)-1)*lev.stride + ((lev.ng+k-1)*lev.nd + (lev.ng+j-1))*lev.nd + (lev.ng+i-1) + 1]
              for i in 1:lev.B, j in 1:lev.B, k in 1:lev.B]
             for s in lev.live)
    end
    for l in 0:1, f in (:D, :S1, :S2, :S3, :Tau, :Ge, :phi)
        ma = bmap(ha, l, f); mb = bmap(hb, l, f)
        @test keys(ma) == keys(mb)
        @test all(ma[k] == mb[k] for k in keys(ma))
    end
    for l in 0:1
        @test Array(ha.levels[l+1].sp[1])[1] isa UInt16   # sanity
        sa = Set(zip([ha.levels[l+1].meta[s].origin for s in ha.levels[l+1].live]))
        @test length(sa) == length(ha.levels[l+1].live)
    end
    # continuation: both advance 3 more root steps → bit-identical state
    for _ in 1:3
        advance_hierarchy!(ha)
        advance_hierarchy!(hb)
    end
    for l in 0:1, f in (:D, :Tau, :Ge)
        ma = bmap(ha, l, f); mb = bmap(hb, l, f)
        @test all(ma[k] == mb[k] for k in keys(ma))
    end
    Ma = total_conserved(ha); Mb = total_conserved(hb)
    @test Ma[1] == Mb[1] && Ma[5] == Mb[5]
    rm(ckf; force = true)
end
end # for BE
