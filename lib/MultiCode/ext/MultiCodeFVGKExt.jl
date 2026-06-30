# ── FiniteVolumeGodunovKA as a PatchGrid hydro solver (solver=:fvgk) ───────────
#
# Activates under `using MultiCode, FiniteVolumeGodunovKA`. Overrides the
# `MultiCode._fvgk_patch_hydro!` hook so `patch_step!(pg, dt; solver=:fvgk)` runs
# FVGK's Godunov scheme instead of PPMKernels.
#
# FVGK is an UNSPLIT single-grid solver (one CTU pass does all three directions at
# once, and it owns its periodic boundary) — so it does NOT fit the per-patch split
# sweep + per-axis ghost-exchange contract PPMKernels uses. Instead, because every
# patch is co-resident on one GPU, we ASSEMBLE the patch interiors into a single
# global FVGK grid (ncell³), step it once (the global domain IS periodic), and
# disperse the result back. This reproduces the undecomposed reference EXACTLY for
# any `np` (the decomposition-invariance criterion), reuses the fast transpiled
# `Grid3DCuMarch`, and is far simpler than a split per-axis external-ghost sweep.
#
# Layout match: `(D,S1,S2,S3,Tau)` = `(ρ,ρu,ρv,ρw,E)` with E energy-per-volume, so
# the only work is a device-side octant gather/scatter (strip ghosts → the patch's
# octant in the global var-major buffer, and back) — the same reshape-view broadcast
# `exchange_ghosts_axis!` uses, no host round-trip. Float32 / CUDA only for now.
#
# Dual energy: FVGK is single-energy. After the step we re-derive `Ge` (gas energy
# density) from the conserved state, `Ge = Tau − ½(S1²+S2²+S3²)/D`, so downstream
# chemistry sees a consistent internal energy. (No Enzo dual-energy switch.)

module MultiCodeFVGKExt

using MultiCode
using FiniteVolumeGodunovKA
import ChemistryKernels

const FV = FiniteVolumeGodunovKA
module _FVGKMetalRuntime
    using FiniteVolumeGodunovKA, Metal
    include(joinpath(pkgdir(FiniteVolumeGodunovKA), "metal", "metal.jl"))
end

@inline _fvgk_mtl_runtime() = _FVGKMetalRuntime

# number of passive species (colours) carried by the patches.
@inline _nspecies(pg::MultiCode.PatchGrid) = length(pg.patches[1].species)

# one persistent global FVGK grid, sized to the GLOBAL active grid `ncell` (decomposition-agnostic).
# With nsp>0 species the grid is an `EulerColors{nsp}` system: the species ride the hydro mass flux as
# extra passive conserved vars (slots 6..5+nsp) — ΣX=1 / uniform-X preserved by construction.
function _build_fvgk_global(pg::MultiCode.PatchGrid)
    (pg.besym === :cuda || pg.besym === :metal) ||
        error("solver=:fvgk: supports :cuda and :metal backends (got :$(pg.besym))")
    pg.T === Float32   || error("solver=:fvgk: needs Float32 patches (got $(pg.T))")
    nc  = pg.ncell; nsp = _nspecies(pg); γ = Float32(pg.gamma)
    riem = Symbol(get(ENV, "CIC_FVGK_RIEMANN", "llf"))
    rec  = Symbol(get(ENV, "CIC_FVGK_RECON",   "plm"))
    # CIC_FVGK_F16=1 → all-f16 DUAL-ENERGY system (EulerDE[Colors]): carries Ge in slot 5 (evolved, not
    # re-derived), so the COLD gas runs in f16 without the E−KE cancellation NaN.  Else the single-energy
    # f32 Euler/EulerColors path (Ge re-derived post-step).
    if get(ENV, "CIC_FVGK_F16", "0") == "1"
        if nsp == 0
            sys = FV.EulerDE(γ = γ);             z = (1f0, 0f0, 0f0, 0f0, 1f-6, 5f-7)
        else
            sys = FV.EulerDEColors{nsp}(γ = γ);  z = (1f0, 0f0, 0f0, 0f0, 1f-6, 5f-7, ntuple(_ -> 0f0, nsp)...)
        end
        U0 = [z for _ in 1:nc[1], _ in 1:nc[2], _ in 1:nc[3]]
        # GE_SCALE lifts the COLD eint into f16's normal range: CICASS Ge=ρ·eint spans ~1e-8 (z=1000)
        # down to ~4e-11 (z=20), BELOW the f16 subnormal floor (~6e-8) → raw f16 flushes to 0 → NaN.
        # 1e7 maps that range to f16-normal [~4e-4, 0.12] (and E≲1e-4 → ≲1e3, no overflow).
        gesc = parse(Float32, get(ENV, "CIC_FVGK_GE_SCALE", "1e7"))
        if pg.besym === :metal
            store = Symbol(get(ENV, "CIC_FVGK_STORE", "f16"))
            store === :f16 || error("solver=:fvgk on Metal requires CIC_FVGK_STORE=f16 (got :$store)")
            recobj = rec === :plm ? FV.PLM() : rec === :pcm ? FV.PCM() : error("CIC_FVGK_RECON=$rec")
            riemobj = riem === :llf ? FV.LLF() : riem === :hllc ? FV.HLLC() : error("CIC_FVGK_RIEMANN=$riem")
            mtl = _fvgk_mtl_runtime()
            return mtl.Grid3DMtlDE16(sys, U0; dx = Float32(pg.dx), dy = Float32(pg.dx), dz = Float32(pg.dx),
                                     recon = recobj, rsol = riemobj, ge_scale = gesc,
                                     store = :f16, de_prec = :f16)
        end
        # CIC_FVGK_STORE=f16 (default) makes g.R/g.O __half → HALVES the grid buffer (the biggest persistent
        # alloc); the gather/scatter GE_SCALE-lift the energy slots so the cold eint survives f16 storage.
        store = Symbol(get(ENV, "CIC_FVGK_STORE", "f16"))
        return Grid3DCuMarch(sys, U0; dx = Float32(pg.dx), riemann = riem, recon = rec,
                             de_prec = :f16, ge_scale = gesc, store = store)
    end
    pg.besym === :metal && error("solver=:fvgk on Metal currently requires CIC_FVGK_F16=1")
    if nsp == 0
        sys = Euler(γ = γ);                 z = (1f0, 0f0, 0f0, 0f0, 1f0)
    else
        sys = EulerColors{nsp}(γ = γ);      z = (1f0, 0f0, 0f0, 0f0, 1f0, ntuple(_ -> 0f0, nsp)...)
    end
    U0 = [z for _ in 1:nc[1], _ in 1:nc[2], _ in 1:nc[3]]        # concrete Array{NTuple{5+nsp,Float32},3}
    return Grid3DCuMarch(sys, U0; dx = Float32(pg.dx), riemann = riem, recon = rec)
end

# var c of the global FVGK buffer, as a (ncell...) view (column-major, var-major flat).
@inline _gblock(g, nc, GVOL, c) = reshape(view(g.R, (c-1)*GVOL+1 : c*GVOL), nc[1], nc[2], nc[3])
@inline _metal_de16(g) = hasproperty(g, :U) && hasproperty(g, :gs) && ndims(g.U) == 4 && eltype(g.U) === Float16
@inline _gridblock(g, nc, GVOL, c) = _metal_de16(g) ? view(g.U, :, :, :, c) : _gblock(g, nc, GVOL, c)

# dual-energy grid? EulerDE[Colors] carries Ge (nconserved = 6+nsp); single-energy Euler[Colors] is 5+nsp.
@inline _de(g, pg) = FV.nconserved(g.sys) - _nspecies(pg) == 6
# f16-storage grid (g.R is __half)? then the energy slots (5=E,6=Ge) are stored GE_SCALE-lifted.
@inline _f16store(g) = (hasproperty(g, :R) ? eltype(g.R) : eltype(g.U)) === Float16
@inline _gesc() = parse(Float32, get(ENV, "CIC_FVGK_GE_SCALE", "1e7"))

# Conserved fields map 1:1 onto the global var-major slots: (D,S1,S2,S3,Tau) then the species in slots
# 6..5+nsp as LINEAR ρ·xᵢ (the EulerColors conserved colours). Ge is NOT carried (re-derived post-step).
# With packed_species the patch stores the UInt16 mass FRACTION Xᵢ, so the boundary converts:
# gather ρXᵢ = unpack(Xᵢ)·ρ_pre, scatter Xᵢ = pack(ρXᵢ / ρ_post) — ρ_post is the stepped global density.

# gather every patch's interior into its octant of the global grid (strip ghosts).
function _fvgk_gather!(g, pg::MultiCode.PatchGrid)
    li, lj, lk = MultiCode._interior(pg); nc = pg.ncell; GVOL = prod(nc)
    de = _de(g, pg); hoff = de ? 6 : 5   # DE adds Ge as slot 6; species follow at hoff+q
    f16s = _f16store(g); gesc = f16s ? _gesc() : 1f0
    for p in pg.patches
        gi, gj, gk = MultiCode._octant(pg, p)
        for (c, f) in enumerate(de ? (p.D, p.S1, p.S2, p.S3, p.Tau, p.Ge) : (p.D, p.S1, p.S2, p.S3, p.Tau))
            Gblk = _gridblock(g, nc, GVOL, c); src = MultiCode._r3(f, pg.nd)
            if f16s && c >= 5                                    # E (5), Ge (6): lift into f16's normal range
                @views Gblk[gi, gj, gk] .= src[li, lj, lk] .* gesc
            else
                @views Gblk[gi, gj, gk] .= src[li, lj, lk]
            end
        end
        Dsrc = MultiCode._r3(p.D, pg.nd)
        for (q, sf) in enumerate(p.species)
            Gblk = _gridblock(g, nc, GVOL, hoff + q); src = MultiCode._r3(sf, pg.nd)
            if pg.packed
                @views Gblk[gi, gj, gk] .= ChemistryKernels.decode_log2sp.(Float32, src[li, lj, lk]) .* Dsrc[li, lj, lk]
            else
                @views Gblk[gi, gj, gk] .= src[li, lj, lk]          # already ρ·xᵢ
            end
        end
    end
    return nothing
end

# disperse the global grid back into the patch interiors (ghost shells left as-is).
function _fvgk_scatter!(g, pg::MultiCode.PatchGrid)
    li, lj, lk = MultiCode._interior(pg); nc = pg.ncell; GVOL = prod(nc)
    de = _de(g, pg); hoff = de ? 6 : 5
    f16s = _f16store(g); igesc = f16s ? 1f0/_gesc() : 1f0
    GD = _gridblock(g, nc, GVOL, 1)                                  # post-step global density ρ_post
    for p in pg.patches
        gi, gj, gk = MultiCode._octant(pg, p)
        for (c, f) in enumerate(de ? (p.D, p.S1, p.S2, p.S3, p.Tau, p.Ge) : (p.D, p.S1, p.S2, p.S3, p.Tau))
            Gblk = _gridblock(g, nc, GVOL, c); dst = MultiCode._r3(f, pg.nd)
            if f16s && c >= 5                                       # un-lift E, Ge back to physical f32
                @views dst[li, lj, lk] .= Float32.(Gblk[gi, gj, gk]) .* igesc
            else
                @views dst[li, lj, lk] .= Gblk[gi, gj, gk]
            end
        end
        for (q, sf) in enumerate(p.species)
            Gblk = _gridblock(g, nc, GVOL, hoff + q); dst = MultiCode._r3(sf, pg.nd)
            if pg.packed
                @views dst[li, lj, lk] .= ChemistryKernels.encode_log2sp.(Float32.(Gblk[gi, gj, gk]) ./ Float32.(GD[gi, gj, gk]))
            else
                @views dst[li, lj, lk] .= Gblk[gi, gj, gk]
            end
        end
    end
    return nothing
end

# re-derive the gas (internal) energy density from the conserved state, per patch interior.
function _fvgk_sync_ge!(pg::MultiCode.PatchGrid)
    li, lj, lk = MultiCode._interior(pg)
    for p in pg.patches
        D = MultiCode._r3(p.D, pg.nd); S1 = MultiCode._r3(p.S1, pg.nd); S2 = MultiCode._r3(p.S2, pg.nd)
        S3 = MultiCode._r3(p.S3, pg.nd); Tau = MultiCode._r3(p.Tau, pg.nd); Ge = MultiCode._r3(p.Ge, pg.nd)
        @views Ge[li,lj,lk] .= Tau[li,lj,lk] .-
            0.5f0 .* (S1[li,lj,lk].^2 .+ S2[li,lj,lk].^2 .+ S3[li,lj,lk].^2) ./ D[li,lj,lk]
    end
    return nothing
end

function MultiCode._fvgk_patch_hydro!(pg::MultiCode.PatchGrid, dt::Real)
    pg.fvgk === nothing && (pg.fvgk = _build_fvgk_global(pg))
    g = pg.fvgk; dtf = Float32(dt)
    _fvgk_gather!(g, pg)
    # one 2nd-order CTU step; sub-cycle if the driver dt exceeds FVGK's CTU CFL. With species the
    # colours are primitives in the kernel, so use the f32 path (run_ctu!) — the f16-tiled run_ctus!
    # would underflow trace species (X~1e-30 → __half 0); pure hydro keeps the fast f16 tiled kernel.
    de = _de(g, pg)
    # CIC_FVGK_F16=1 → all-f16 DUAL-ENERGY run_ctus_de16! (Ge evolved in-grid; cold gas, no NaN).  Else
    # single-energy f32: run_ctu! (CTU, accurate) or run_rk2! (CIC_FVGK_INTEGRATOR=rk2; ≈CTU cost), and
    # pure hydro uses the fast f16-tiled run_ctus!.
    if _metal_de16(g) && de
        mtl = _fvgk_mtl_runtime()
        c = mtl.mde16_max_wavespeed(g)
        dtmax = 0.45f0 * min(g.dx, g.dy, g.dz) / c
        nsub = max(1, ceil(Int, dtf / dtmax))
        dts = dtf / nsub
        for n in 0:(nsub-1)
            mtl.mde16_step!(g, dts; rev = isodd(n))
        end
    elseif de
        nsub = max(1, ceil(Int, dtf / dt_cfl(g; cfl = 0.45f0)))
        run_ctus_de16!(g, dtf / nsub, nsub)
    elseif _nspecies(pg) == 0
        nsub = max(1, ceil(Int, dtf / dt_cfl(g; cfl = 0.45f0)))
        run_ctus!(g, dtf / nsub, nsub)
    elseif get(ENV, "CIC_FVGK_INTEGRATOR", "ctu") == "rk2"
        nsub = max(1, ceil(Int, dtf / dt_cfl(g; cfl = 0.45f0)))
        run_rk2!(g, dtf / nsub, nsub)
    else
        nsub = max(1, ceil(Int, dtf / dt_cfl(g; cfl = 0.45f0)))
        run_ctu!(g, dtf / nsub, nsub)
    end
    _fvgk_scatter!(g, pg)
    de || _fvgk_sync_ge!(pg)        # DE evolves Ge in-grid; single-energy re-derives Ge = Tau − KE
    return nothing
end

end # module
