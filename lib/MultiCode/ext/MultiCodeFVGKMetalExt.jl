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
@inline _msc() = parse(Float32, get(ENV, "CIC_FVGK_MOM_SCALE", "1"))

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
    speed_scratch = get(ENV, "CIC_FVGK_SPEED_SCRATCH", "0") == "1"
    return _FVGKMetalRuntime.Grid3DMtlDE16(sys, nc; dx = Float32(pg.dx), dy = Float32(pg.dx),
                                           dz = Float32(pg.dx), recon = recobj, rsol = riemobj,
                                           ge_scale = gesc, mom_scale = _msc(),
                                           dens_base = Float32(pg.dens_base),
                                           dens_scale = Float32(pg.dens_scale),
                                           store = :f16, de_prec = :f16, speed_scratch = speed_scratch)
end

function MultiCode._fvgk_metal_step!(g::_FVGKMetalRuntime.Grid3DMtlDE16,
                                     dtf::Float32, sigspeed, dx::Float32)
    c = max(sigspeed === nothing ? _FVGKMetalRuntime.mde16_max_wavespeed(g) :
            Float32(sigspeed), eps(Float32))
    dtmax = 0.45f0 * min(g.dx, g.dy, g.dz) / c
    nsub = max(1, ceil(Int, dtf / dtmax))
    dts = dtf / nsub
    for n in 0:(nsub - 1)
        _FVGKMetalRuntime.mde16_step!(g, dts; rev = isodd(n))
    end
    return nsub
end

function MultiCode._fvgk_metal_grav_kick!(g::_FVGKMetalRuntime.Grid3DMtlDE16,
                                          accel; dx, halfdt)
    _FVGKMetalRuntime.mde16_grav_kick_global_potential!(
        g, accel; dx = Float32(dx), halfdt = Float32(halfdt))
    return true
end

function MultiCode._fvgk_metal_synchronize!(g::_FVGKMetalRuntime.Grid3DMtlDE16)
    Metal.synchronize()
    return nothing
end

end # module
