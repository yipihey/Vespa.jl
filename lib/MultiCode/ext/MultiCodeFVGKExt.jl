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

const FV = FiniteVolumeGodunovKA

# one persistent global FVGK grid, sized to the GLOBAL active grid `ncell` (decomposition-agnostic).
function _build_fvgk_global(pg::MultiCode.PatchGrid)
    pg.besym === :cuda || error("solver=:fvgk: first target is the :cuda backend (got :$(pg.besym))")
    pg.T === Float32   || error("solver=:fvgk: needs Float32 patches (got $(pg.T))")
    nc  = pg.ncell
    sys = Euler(γ = Float32(pg.gamma))
    U0  = fill((1f0, 0f0, 0f0, 0f0, 1f0), nc[1], nc[2], nc[3])   # placeholder; overwritten by the first gather
    return Grid3DCuMarch(sys, U0; dx = Float32(pg.dx))
end

# var c of the global FVGK buffer, as a (ncell...) view (column-major, var-major flat).
@inline _gblock(g, nc, GVOL, c) = reshape(view(g.R, (c-1)*GVOL+1 : c*GVOL), nc[1], nc[2], nc[3])

# gather every patch's interior into its octant of the global grid (strip ghosts).
function _fvgk_gather!(g, pg::MultiCode.PatchGrid)
    li, lj, lk = MultiCode._interior(pg); nc = pg.ncell; GVOL = prod(nc)
    for p in pg.patches
        gi, gj, gk = MultiCode._octant(pg, p)
        for (c, f) in enumerate((p.D, p.S1, p.S2, p.S3, p.Tau))
            Gblk = _gblock(g, nc, GVOL, c); src = MultiCode._r3(f, pg.nd)
            @views Gblk[gi, gj, gk] .= src[li, lj, lk]
        end
    end
    return nothing
end

# disperse the global grid back into the patch interiors (ghost shells left as-is).
function _fvgk_scatter!(g, pg::MultiCode.PatchGrid)
    li, lj, lk = MultiCode._interior(pg); nc = pg.ncell; GVOL = prod(nc)
    for p in pg.patches
        gi, gj, gk = MultiCode._octant(pg, p)
        for (c, f) in enumerate((p.D, p.S1, p.S2, p.S3, p.Tau))
            Gblk = _gblock(g, nc, GVOL, c); dst = MultiCode._r3(f, pg.nd)
            @views dst[li, lj, lk] .= Gblk[gi, gj, gk]
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
    # one 2nd-order tiled-CTU step; sub-cycle if the driver dt exceeds FVGK's CTU CFL
    nsub = max(1, ceil(Int, dtf / dt_cfl(g; cfl = 0.45f0)))
    run_ctus!(g, dtf / nsub, nsub)
    _fvgk_scatter!(g, pg)
    _fvgk_sync_ge!(pg)
    return nothing
end

end # module
