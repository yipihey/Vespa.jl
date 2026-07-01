# mem_probe.jl — fast GPU memory probe for the CICASS f16-DE FVGK stack, WITHOUT the
# slow CICASS C IC-gen.  Builds the exact allocation path (patch fields + FVGK global
# grid g.R/g.O + DM particle SoA + gravity scratch) at a synthetic uniform IC and reports
# the live GPU footprint + a per-component byte breakdown, so we can (a) find the real OOM
# wall and (b) see exactly where the per-cell budget goes.
#
#   BACKEND=cuda CIC_NPROBE="256,384,512,640,704,720,736" CIC_NP=2 \
#     julia --project=lib/MultiCode/test lib/MultiCode/examples/mem_probe.jl
#
# Knobs mirror patch_cicass.jl: BACKEND, CIC_NP, CIC_PACKED, CIC_FVGK_F16
# (default 1), CIC_FVGK_STORE (default f16), CIC_FVGK_DEDUP, CIC_VEL16,
# CIC_GRAV1BUF, CIC_FVGK_RIEMANN/RECON, CIC_PIDS. Solver is always :fvgk here.

using MultiCode, Printf
import PoissonKernels
import ChemistryKernels
using FiniteVolumeGodunovKA

const BE   = Symbol(get(ENV, "BACKEND", "cuda"))
BE === :cuda && @eval import CUDA
BE === :metal && @eval import Metal
const T    = Float32
const NG   = parse(Int, get(ENV, "CIC_NG", "2"))   # patch_cicass FVGK default; dedup later drops to ng=0
const NP   = parse(Int, get(ENV, "CIC_NP", "2"))
const PACKED = get(ENV, "CIC_PACKED", "0") == "1"
const PIDS   = get(ENV, "CIC_PIDS", "1") == "1"
const DEDUP  = get(ENV, "CIC_FVGK_DEDUP", "0") == "1"
const VEL16  = get(ENV, "CIC_VEL16",  "0") == "1"   # f16 particle velocities (positions stay f32)
const GRAV1BUF = get(ENV, "CIC_GRAV1BUF", "0") == "1" # ρ/φ share one buffer (in-place solve)
const NSPEC  = 1                              # analytic chem carries one colour (HII)
const GAMMA  = 5/3
ENV["CIC_FVGK_F16"]  = get(ENV, "CIC_FVGK_F16",  "1")
ENV["CIC_FVGK_STORE"] = get(ENV, "CIC_FVGK_STORE", "f16")

gib(b) = b / 2^30
devbytes(a) = a isa AbstractArray ? length(a) * sizeof(eltype(a)) : 0
sync_backend() = BE === :cuda ? CUDA.synchronize() :
                 BE === :metal ? Metal.synchronize() : nothing
function reclaim_backend()
    sync_backend()
    GC.gc()
    BE === :cuda && CUDA.reclaim()
    return nothing
end
live_backend() = BE === :cuda ? gib(CUDA.total_memory() - CUDA.available_memory()) :
                 BE === :metal ? gib(max(0, Metal.alloc_stats.alloc_bytes - Metal.alloc_stats.free_bytes)) :
                 0.0
total_backend() = BE === :cuda ? gib(CUDA.total_memory()) : NaN
function oom_error(e)
    e isa OutOfMemoryError && return true
    return BE === :cuda && isdefined(@__MODULE__, :CUDA) && e isa CUDA.OutOfGPUMemoryError
end

# sum sizeof over array fields of an FVGK grid object (g.R/g.O or Metal g.U, plus scratch)
function fvgk_bytes(g)
    b = 0
    for f in fieldnames(typeof(g))
        v = getfield(g, f)
        b += devbytes(v)
    end
    b
end

# uniform cold-IGM IC direct on the backend. Avoid host N³ Float64 fields, or the
# probe itself becomes the OOM source at 1024³.
function scatter_uniform!(pg)
    fb = T(0.157); eint = T(1.0e-6); xhii = T(2.0e-4)
    spacked = ChemistryKernels.encode_log2sp(xhii)
    for p in pg.patches
        fill!(p.D, fb)
        fill!(p.S1, zero(T)); fill!(p.S2, zero(T)); fill!(p.S3, zero(T))
        fill!(p.Ge, fb * eint); fill!(p.Tau, fb * eint)
        for s in p.species
            eltype(s) === UInt16 ? fill!(s, spacked) : fill!(s, fb * xhii)
        end
    end
    return pg
end

function probe(N)
    ncell = (N, N, N); np = (NP, NP, NP)
    N % NP == 0 || (return @sprintf("%4d³  SKIP (not divisible by np=%d)", N, NP))
    reclaim_backend()
    pg = build_patchgrid(; ng=NG, ncell=ncell, np=np, dx=1.0/N, gamma=GAMMA, nspecies=NSPEC,
                         besym=BE, T=T, du=1.0, lu=1.0, tu=1.0, deut=false, packed_species=PACKED)
    scatter_uniform!(pg)

    # Build/dedup before DM particles, matching patch_cicass.jl. This keeps the transition
    # peak at patches + g.R/O rather than patches + particles + g.R/O.
    patch_step!(pg, 0.0; a_value=1.0e-3, order=(1,2,3), accel=nothing, chem=false,
                solver=:fvgk, do_hydro=true, do_chem=false, chemmode=:analytic)
    sync_backend()
    if DEDUP && prod(np) == 1
        MultiCode._fvgk_dedup!(pg)
        sync_backend()
    end

    Npart = N^3; VT = VEL16 ? Float16 : T
    dzero(::Type{S}) where {S} = PoissonKernels.device_zeros(pg.backend, S, (Npart,))
    parts = (px=dzero(T), py=dzero(T), pz=dzero(T),
             vx=dzero(VT), vy=dzero(VT), vz=dzero(VT),
             mass=T(0.843))
    PIDS && (parts = (; parts..., id=dzero(Int32)))

    ρd = PoissonKernels.device_zeros(pg.backend, T, ncell)
    φd = GRAV1BUF ? ρd : PoissonKernels.device_zeros(pg.backend, T, ncell)  # share ⇒ one nc³ field

    # per-component byte breakdown (device only)
    patch_b = pg.dedup ? 0 : sum(p -> sum(devbytes, MultiCode._allfields(p)), pg.patches)
    fvgk_b  = pg.fvgk === nothing ? 0 : sum(fvgk_bytes, pg.fvgk isa AbstractVector ? pg.fvgk : (pg.fvgk,))
    part_b  = sum(devbytes, values(parts))
    grav_b  = GRAV1BUF ? devbytes(ρd) : devbytes(ρd) + devbytes(φd)   # shared ⇒ count once

    reclaim_backend()
    live = live_backend()
    Mcell = N^3 / 1e6
    bcell(b) = b / N^3
    fld_b = patch_b + fvgk_b + part_b + grav_b                  # RELIABLE total (sum of actual array bytes)
    s = @sprintf("%4d³ (%6.1f Mcell) fields=%6.2f GiB (live≈%5.2f) | patch=%6.2f(%4.0f B/c) fvgk=%6.2f(%4.0f) part=%6.2f(%4.0f) grav=%5.2f(%3.0f) | sum=%5.0f B/c",
        N, Mcell, gib(fld_b), live,
        gib(patch_b), bcell(patch_b), gib(fvgk_b), bcell(fvgk_b),
        gib(part_b), bcell(part_b), gib(grav_b), bcell(grav_b),
        bcell(fld_b))
    pg = nothing; parts = nothing; ρd = nothing; φd = nothing
    reclaim_backend()
    return s
end

function main()
    Ns = [parse(Int, s) for s in split(get(ENV, "CIC_NPROBE", "256,384,512"), ",")]
    total = total_backend()
    total_s = isnan(total) ? "total=n/a" : @sprintf("total=%.1f GiB", total)
    @printf("# mem probe backend=%s np=%d packed=%s f16=%s store=%s dedup=%s vel16=%s grav1buf=%s pids=%s (%s)\n",
            String(BE), NP, PACKED, ENV["CIC_FVGK_F16"], ENV["CIC_FVGK_STORE"], DEDUP, VEL16, GRAV1BUF, PIDS, total_s)
    for N in Ns
        local line
        try
            line = probe(N)
        catch e
            if oom_error(e)
                line = @sprintf("%4d³  *** OUT OF GPU MEMORY ***", N)
            else
                line = @sprintf("%4d³  ERROR: %s", N, sprint(showerror, e))
            end
            reclaim_backend()
        end
        println(line); flush(stdout)
    end
end

main()
