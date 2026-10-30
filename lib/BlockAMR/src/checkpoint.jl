# checkpoint.jl — save/restore the full hierarchy at a root-step boundary.
#
# Saved state is exactly what a root step consumes: topology (origins /
# parent / offset per live block, parents as LIVE-ORDER indices so slots can
# renumber freely on restore), ACTIVE cells of the R gas fields + u16 species
# + φ (warm start), per-block scales, and the substep clocks.  Ghosts, O
# twins, rhs/dm pools and all tables are derived state — rebuilt on restore
# (fill_ghosts!/solves recreate them before anything reads them), so a
# restored hierarchy continues BIT-IDENTICALLY on the same backend for
# deterministic paths.  Serialization format: plain arrays/tuples only (no
# custom structs), one `Serialization.serialize` blob.

import Serialization

"Host-side snapshot of `hier` (+ a caller NamedTuple `extra`, e.g. particles/clock)."
function checkpoint_state(hier::AMRHierarchy; extra = NamedTuple())
    levels = Vector{Any}()
    for l in 0:length(hier.levels)-1
        lev = hier.levels[l + 1]
        nl = length(lev.live)
        T = eltype(lev.D); B = lev.B; ng = lev.ng; nd = lev.nd
        pidx = Int32[]
        if l >= 1
            plev = hier.levels[l]
            pmap = Dict{Int32,Int32}(s => Int32(i) for (i, s) in enumerate(plev.live))
            pidx = Int32[pmap[lev.meta[s].parent] for s in lev.live]
        end
        origins = [lev.meta[s].origin for s in lev.live]
        offsets = [lev.meta[s].offset for s in lev.live]
        na = _nact(lev); Bz = _Bax(lev, 3); kg = _ngax(lev, 3)
        nf = ismhd(lev) ? 10 : 6                    # gas + (Bx, By, Bz, ψ)
        gf = ntuple(_ -> Array{T}(undef, na, nl), nf)
        fsp = [Array{UInt16}(undef, na, nl) for _ in 1:lev.nsp]
        fphi = Array{Float32}(undef, na, nl)
        hostf = map(Array, (gasfields(lev)..., (ismhd(lev) ? mfields(lev) : ())...))
        hsp = map(Array, lev.sp)
        hphi = Array(lev.phi)
        live = lev.live
        # per-block gather into compacted [B³,nl] columns — disjoint output columns
        # per block ⇒ parallel over blocks (bounded by -t auto; serial otherwise).
        Threads.@threads for i in eachindex(live)
            s = live[i]
            base = (Int(s) - 1) * lev.stride
            q = 0
            for k in 1:Bz, j in 1:B, ii in 1:B
                idx = base + ((kg+k-1) * nd + (ng+j-1)) * nd + (ng+ii-1) + 1
                q += 1
                for f in 1:nf
                    @inbounds gf[f][q, i] = hostf[f][idx]
                end
                for t in 1:lev.nsp
                    @inbounds fsp[t][q, i] = hsp[t][idx]
                end
                @inbounds fphi[q, i] = hphi[idx]
            end
        end
        li = Int.(lev.live)
        push!(levels, (; origins, parents = pidx, offsets,
                       D = gf[1], S1 = gf[2], S2 = gf[3], S3 = gf[4],
                       Tau = gf[5], Ge = gf[6], sp = fsp, phi = fphi,
                       M = nf == 10 ? gf[7:10] : (),
                       # CT faces: whole slots (B+1 faces per normal row)
                       C = isct(lev) ? map(a -> reshape(Array(a)[1:lev.cap*lev.stride], lev.stride, lev.cap)[:, li],
                                           cfields(lev)) : (),
                       Dsc = Array(lev.Dsc)[li], Ssc = Array(lev.Ssc)[li],
                       Esc = Array(lev.Esc)[li], Gsc = Array(lev.Gsc)[li]))
    end
    return (; nbase = hier.nbase, B = hier.B, ng = hier.ng, box = hier.box,
            Lcap = hier.Lcap, gamma = hier.gamma, cfl = hier.cfl,
            scheme = hier.scheme, T = eltype(hier.levels[1].D),
            nsp = hier.levels[1].nsp, nstep = copy(hier.nstep), levels, extra,
            dim = hier.dim, bc = hier.bc, physics = hier.physics, mhd = hier.mhd,
            inflow = Dict((d, s == 1 ? :lo : :hi) => hier.inflow[d][s]
                          for d in 1:3, s in 1:2 if hier.bc[d][s] === :inflow))
end

"""
    save_checkpoint(path, hier; extra = NamedTuple())

Serialize the full hierarchy state (see `checkpoint_state`) to `path`.  Call at
a root-step boundary only.  `extra` carries driver state (particles, `a`, …).
"""
function save_checkpoint(path::AbstractString, hier::AMRHierarchy;
                         extra = NamedTuple())
    tmp = path * ".tmp"
    Serialization.serialize(tmp, checkpoint_state(hier; extra))
    mv(tmp, path; force = true)                # crash-safe: never a torn file
    return nothing
end

"""
    load_checkpoint(path; backend = :cpu, Lcap = nothing) -> (hier, extra)

Rebuild an `AMRHierarchy` from a checkpoint: topology re-added block-by-block
(slots renumber freely; parents resolve through the saved live order), fields,
species, φ and scales restored to the ACTIVE cells, all tables and C/F
registers rebuilt.  `Lcap` may EXCEED the checkpoint's (deepen-later restarts).
"""
function load_checkpoint(path::AbstractString; backend::Symbol = :cpu,
                         Lcap = nothing, T = nothing)
    dbg = get(ENV, "BAM_LOADDBG", "0") == "1"
    _t() = time()
    dbg && (@info "load: deserialize start"; flush(stderr))
    t0 = _t()
    ck = Serialization.deserialize(path)
    dbg && (@printf("load: deserialized in %.1fs\n", _t()-t0); flush(stdout))
    # T override: widen an f16 checkpoint into an f32 hierarchy (the per-cell host
    # scatter converts on assignment).  Restored scales stay valid (stored=phys/scale).
    Tp = T === nothing ? ck.T : T
    # pre-2-D checkpoints carry no dim/bc: 3-D, fully periodic
    dim = hasproperty(ck, :dim) ? ck.dim : 3
    hier = AMRHierarchy(; nbase = dim == 3 ? ck.nbase : (ck.nbase[1], ck.nbase[2]),
                        B = ck.B, ng = ck.ng, box = ck.box,
                        backend, T = Tp, nsp = ck.nsp,
                        Lcap = Lcap === nothing ? ck.Lcap : Lcap,
                        gamma = ck.gamma, cfl = ck.cfl, scheme = ck.scheme,
                        bc = hasproperty(ck, :bc) ? ck.bc[1:dim] : :periodic,
                        inflow = hasproperty(ck, :inflow) ? ck.inflow :
                                 Dict{Tuple{Int,Symbol},NTuple{5,Float64}}(),
                        (hasproperty(ck, :physics) && ck.physics === :mhd ?
                         (physics = :mhd, riemann = ck.mhd.riemann,
                          divb = get(ck.mhd, :ct, false) ? :ct : :glm,
                          glm_ch_scale = ck.mhd.glm_ch_scale, glm_cp_coef = ck.mhd.glm_cp_coef,
                          smallr = ck.mhd.smallr, smallc = ck.mhd.smallc,
                          dual_energy = get(ck.mhd, :de, false),
                          de_eta1 = get(ck.mhd, :eta1, 1e-3), de_eta2 = get(ck.mhd, :eta2, 0.1),
                          shock_switch = get(ck.mhd, :sw, false),
                          shock_eps = get(ck.mhd, :seps, 5.0)) : (;))...)
    init_base_level!(hier)
    slotmaps = Vector{Vector{Int32}}()
    push!(slotmaps, Int32[hier.levels[1].byorigin[org] for org in ck.levels[1].origins])
    tg = _t()
    for l in 1:length(ck.levels)-1
        ensure_level!(hier, l)                 # checkpoints may carry EMPTY levels
        ckl = ck.levels[l + 1]
        reserve!(hier.levels[l + 1], length(ckl.origins) + 8)  # one alloc, no grow cascade
        sm = Int32[]
        for (i, org) in enumerate(ckl.origins)
            s = add_block!(hier, l, slotmaps[l][ckl.parents[i]], Int.(ckl.offsets[i]))
            @assert hier.levels[l + 1].meta[s].origin == org "restored origin mismatch"
            push!(sm, s)
        end
        push!(slotmaps, sm)
        dbg && (@printf("load: grew L%d to %d blocks in %.1fs\n", l, length(sm), _t()-tg);
                flush(stdout); tg = _t())
    end
    for l in 0:length(ck.levels)-1
        lev = hier.levels[l + 1]; ckl = ck.levels[l + 1]
        T = eltype(lev.D); B = lev.B; ng = lev.ng; nd = lev.nd
        Bz = _Bax(lev, 3); kg = _ngax(lev, 3)
        n = lev.cap * lev.stride
        nf = ismhd(lev) ? 10 : 6
        hosts = ntuple(_ -> zeros(T, n), nf)
        hsp = [zeros(UInt16, n) for _ in 1:lev.nsp]
        hphi = zeros(Float32, n)
        hDsc = ones(Float32, lev.cap); hSsc = ones(Float32, lev.cap)
        hEsc = ones(Float32, lev.cap); hGsc = ones(Float32, lev.cap)
        ckf = (ckl.D, ckl.S1, ckl.S2, ckl.S3, ckl.Tau, ckl.Ge, (nf == 10 ? ckl.M : ())...)
        # backward-compat: pre-split checkpoints have no Gsc — Ge was encoded on
        # the shared Esc, so Gsc = Esc reproduces the old physical Ge exactly.
        ckGsc = hasproperty(ckl, :Gsc) ? ckl.Gsc : ckl.Esc
        ts = _t()
        cksp = ckl.sp; ckphi = ckl.phi; smap = slotmaps[l + 1]
        # per-block interior scatter — each block owns a disjoint `base` region of
        # every host pool, so this is embarrassingly parallel (bounded by -t auto;
        # serial when single-threaded).  The dominant single-thread restore cost.
        Threads.@threads for i in eachindex(smap)
            s = smap[i]
            base = (Int(s) - 1) * lev.stride
            hDsc[s] = ckl.Dsc[i]; hSsc[s] = ckl.Ssc[i]; hEsc[s] = ckl.Esc[i]; hGsc[s] = ckGsc[i]
            q = 0
            for k in 1:Bz, j in 1:B, ii in 1:B
                idx = base + ((kg+k-1) * nd + (ng+j-1)) * nd + (ng+ii-1) + 1
                q += 1
                for f in 1:nf
                    @inbounds hosts[f][idx] = ckf[f][q, i]
                end
                for t in 1:lev.nsp
                    @inbounds hsp[t][idx] = cksp[t][q, i]
                end
                @inbounds hphi[idx] = ckphi[q, i]
            end
            lev.meta[s].flags &= ~FLAG_NEW
        end
        dbg && (@printf("load: scattered L%d (%d blocks) in %.1fs\n", l, length(smap), _t()-ts); flush(stdout))
        tc = _t()
        for (dev, hh) in zip((gasfields(lev)..., (nf == 10 ? mfields(lev) : ())...), hosts)
            copyto!(dev, hh)
        end
        for (t, sp) in enumerate(lev.sp)
            copyto!(sp, hsp[t])
        end
        copyto!(lev.phi, hphi)
        if isct(lev) && hasproperty(ckl, :C) && !isempty(ckl.C)
            for (dev, cols) in zip(cfields(lev), ckl.C)
                h = zeros(T, lev.cap * lev.stride)
                for (i, s) in enumerate(smap)
                    h[(Int(s)-1)*lev.stride+1:Int(s)*lev.stride] .= cols[:, i]
                end
                copyto!(dev, h)
            end
        end
        copyto!(lev.Dsc, hDsc); copyto!(lev.Ssc, hSsc); copyto!(lev.Esc, hEsc); copyto!(lev.Gsc, hGsc)
        dbg && (@printf("load: uploaded L%d fields in %.1fs\n", l, _t()-tc); flush(stdout))
    end
    copyto!(hier.nstep, ck.nstep)
    tb = _t()
    grav_only = isempty(hier.levels[1].Do)   # DMO: no hydro flux → no reflux register
    for l in 0:length(hier.levels)-1
        build_level_tables!(hier, l)
        (l >= 1 && !grav_only) && build_cf_register!(hier, l)
        dbg && (@printf("load: built tables L%d in %.1fs\n", l, _t()-tb); flush(stdout); tb = _t())
    end
    return hier, ck.extra
end
