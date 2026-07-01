# mem_probe.jl — fast GPU memory probe for the CICASS f16-DE FVGK stack, WITHOUT the
# slow CICASS C IC-gen.  Builds the exact allocation path (patch fields + FVGK global
# grid g.R/g.O + DM particle SoA + gravity scratch) at a synthetic uniform IC and reports
# the live GPU footprint + a per-component byte breakdown, so we can (a) find the real OOM
# wall on the A6000 and (b) see exactly where the per-cell budget goes.
#
#   BACKEND=cuda CIC_NPROBE="256,384,512,640,704,720,736" CIC_NP=2 \
#     julia --project=lib/MultiCode/test lib/MultiCode/examples/mem_probe.jl
#
# Knobs mirror patch_cicass.jl: CIC_NP, CIC_PACKED, CIC_FVGK_F16 (default 1), CIC_FVGK_STORE
# (default f16), CIC_FVGK_RIEMANN/RECON, CIC_PIDS (default 1).  Solver is always :fvgk here.

using MultiCode, Printf
import PoissonKernels
using FiniteVolumeGodunovKA
import CUDA

const BE   = :cuda
const T    = Float32
const NG   = parse(Int, get(ENV, "CIC_NG", "4"))   # patch ghost depth; FVGK uses interior-only ⇒ ng=0 valid
const NP   = parse(Int, get(ENV, "CIC_NP", "2"))
const PACKED = get(ENV, "CIC_PACKED", "0") == "1"
const PIDS   = get(ENV, "CIC_PIDS", "1") == "1"
const VEL16  = get(ENV, "CIC_VEL16",  "0") == "1"   # f16 particle velocities (positions stay f32)
const GRAV1BUF = get(ENV, "CIC_GRAV1BUF", "0") == "1" # ρ/φ share one buffer (in-place solve)
const NSPEC  = 1                              # analytic chem carries one colour (HII)
const GAMMA  = 5/3
ENV["CIC_FVGK_F16"]  = get(ENV, "CIC_FVGK_F16",  "1")
ENV["CIC_FVGK_STORE"] = get(ENV, "CIC_FVGK_STORE", "f16")

gib(b) = b / 2^30
devbytes(a) = a isa CUDA.CuArray ? sizeof(a) : 0

# sum sizeof over the CuArray fields of an FVGK grid object (g.R, g.O, any device scratch)
function fvgk_bytes(g)
    b = 0
    for f in fieldnames(typeof(g))
        v = getfield(g, f)
        b += devbytes(v)
    end
    b
end

# uniform cold-IGM IC (so the f16-DE FVGK step doesn't NaN): ρ=fb, v=0, cold eint.
function uniform_ic(ncell)
    fb = 0.157; eint = 1.0e-6
    z = zeros(Float64, ncell)
    D = fill(fb, ncell)
    Ge = fill(fb*eint, ncell); Tau = copy(Ge)
    species = [fill(fb*2.0e-4, ncell)]        # HII colour
    (D=D, S1=copy(z), S2=copy(z), S3=copy(z), Tau=Tau, Ge=Ge, species=species)
end

function probe(N)
    ncell = (N, N, N); np = (NP, NP, NP)
    N % NP == 0 || (return @sprintf("%4d³  SKIP (not divisible by np=%d)", N, NP))
    GC.gc(); CUDA.reclaim()
    free0 = CUDA.available_memory()
    pg = build_patchgrid(; ng=NG, ncell=ncell, np=np, dx=1.0/N, gamma=GAMMA, nspecies=NSPEC,
                         besym=BE, T=T, du=1.0, lu=1.0, tu=1.0, deut=false, packed_species=PACKED)
    scatter_global!(pg, uniform_ic(ncell))

    Npart = N^3; VT = VEL16 ? Float16 : T
    dev(v) = PoissonKernels.to_device(pg.backend, v, T)
    rnd() = dev(rand(T, Npart))
    parts = (px=rnd(), py=rnd(), pz=rnd(),
             vx=PoissonKernels.to_device(pg.backend, zeros(VT, Npart), VT),
             vy=PoissonKernels.to_device(pg.backend, zeros(VT, Npart), VT),
             vz=PoissonKernels.to_device(pg.backend, zeros(VT, Npart), VT),
             mass=T(0.843))
    PIDS && (parts = (; parts..., id=PoissonKernels.to_device(pg.backend, collect(Int32, 0:Npart-1))))

    ρd = PoissonKernels.device_zeros(pg.backend, T, ncell)
    φd = GRAV1BUF ? ρd : PoissonKernels.device_zeros(pg.backend, T, ncell)  # share ⇒ one nc³ field

    # build + run the FVGK grid once (allocates pg.fvgk = g.R/g.O) — global_gravity_gpu not
    # needed for the footprint; one hydro step is enough to realize the persistent buffers.
    patch_step!(pg, 0.0; a_value=1.0e-3, order=(1,2,3), accel=nothing, chem=false,
                solver=:fvgk, do_hydro=true, do_chem=false, chemmode=:analytic)   # dt=0 ⇒ nsub=1, no dt_cfl/spd
    CUDA.synchronize()

    # per-component byte breakdown (device only)
    patch_b = sum(p -> sum(devbytes, MultiCode._allfields(p)), pg.patches)
    fvgk_b  = pg.fvgk === nothing ? 0 : sum(fvgk_bytes, pg.fvgk isa AbstractVector ? pg.fvgk : (pg.fvgk,))
    part_b  = sum(devbytes, values(parts))
    grav_b  = GRAV1BUF ? devbytes(ρd) : devbytes(ρd) + devbytes(φd)   # shared ⇒ count once

    GC.gc(); CUDA.reclaim()
    live = gib(CUDA.total_memory() - CUDA.available_memory())   # NOTE: under-reports at large N (pool quirk)
    Mcell = N^3 / 1e6
    bcell(b) = b / N^3
    fld_b = patch_b + fvgk_b + part_b + grav_b                  # RELIABLE total (sum of actual array bytes)
    s = @sprintf("%4d³ (%6.1f Mcell) fields=%6.2f GiB (live≈%5.2f) | patch=%6.2f(%4.0f B/c) fvgk=%6.2f(%4.0f) part=%6.2f(%4.0f) grav=%5.2f(%3.0f) | sum=%5.0f B/c",
        N, Mcell, gib(fld_b), live,
        gib(patch_b), bcell(patch_b), gib(fvgk_b), bcell(fvgk_b),
        gib(part_b), bcell(part_b), gib(grav_b), bcell(grav_b),
        bcell(fld_b))
    pg = nothing; parts = nothing; ρd = nothing; φd = nothing
    GC.gc(); CUDA.reclaim()
    return s
end

function main()
    Ns = [parse(Int, s) for s in split(get(ENV, "CIC_NPROBE", "256,384,512"), ",")]
    @printf("# mem probe  np=%d  packed=%s  f16=%s store=%s  pids=%s  (A6000 %.1f GiB total)\n",
            NP, PACKED, ENV["CIC_FVGK_F16"], ENV["CIC_FVGK_STORE"], PIDS, gib(CUDA.total_memory()))
    for N in Ns
        local line
        try
            line = probe(N)
        catch e
            if e isa CUDA.OutOfGPUMemoryError || (e isa OutOfMemoryError)
                line = @sprintf("%4d³  *** OUT OF GPU MEMORY ***", N)
            else
                line = @sprintf("%4d³  ERROR: %s", N, sprint(showerror, e))
            end
            GC.gc(); CUDA.reclaim()
        end
        println(line); flush(stdout)
    end
end

main()
