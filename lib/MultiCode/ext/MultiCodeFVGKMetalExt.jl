# ── Metal-specific FVGK grid constructor (solver=:fvgk on the Apple GPU) ───────────────────────
#
# Split out of MultiCodeFVGKExt so the COMMON FVGK path (CUDA/CPU) carries no Metal reference —
# Metal.jl is Apple-only and cannot even load on Linux, so a single [FiniteVolumeGodunovKA, Metal]
# extension breaks the CUDA platform.  This extension (trigger [FiniteVolumeGodunovKA, Metal])
# activates only under `using MultiCode, FiniteVolumeGodunovKA, Metal`, and overrides the
# `MultiCode._fvgk_build_metal_grid` hook that MultiCodeFVGKExt calls on the `:metal` backend.

module MultiCodeFVGKMetalExt

using MultiCode
using FiniteVolumeGodunovKA
using Metal

const FV = FiniteVolumeGodunovKA

# FVGK's Metal runtime (Grid3DMtlDE16 etc. — the Apple-only kernels), isolated in a submodule.
module _FVGKMetalRuntime
    using FiniteVolumeGodunovKA, Metal
    include(joinpath(pkgdir(FiniteVolumeGodunovKA), "metal", "metal.jl"))
end

# Build FVGK's Metal all-f16 dual-energy grid (called by MultiCodeFVGKExt._build_fvgk_global on :metal).
function MultiCode._fvgk_build_metal_grid(pg, sys, nc, rec, riem, gesc)
    store = Symbol(get(ENV, "CIC_FVGK_STORE", "f16"))
    store === :f16 || error("solver=:fvgk on Metal requires CIC_FVGK_STORE=f16 (got :$store)")
    recobj  = rec  === :plm ? FV.PLM() : rec  === :pcm ? FV.PCM() : error("CIC_FVGK_RECON=$rec")
    riemobj = riem === :llf ? FV.LLF() : riem === :hllc ? FV.HLLC() : error("CIC_FVGK_RIEMANN=$riem")
    return _FVGKMetalRuntime.Grid3DMtlDE16(sys, nc; dx = Float32(pg.dx), dy = Float32(pg.dx),
                                           dz = Float32(pg.dx), recon = recobj, rsol = riemobj,
                                           ge_scale = gesc, store = :f16, de_prec = :f16)
end

end # module
