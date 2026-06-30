# patch_cicass.jl — FULL CICASS smooth-baryon cosmology run on the in-process topgrid
# decomposition (z=1000→20), GPU hydro+chem per patch + global CPU-FFT gravity.
#
# This is the patch-decomposition analogue of cicass_highz_pk.jl (Enzo path) and
# cicass_ramses_pk.jl (RAMSES path): the SAME CICASS streaming-velocity realization is
# loaded into a PatchGrid (gas + species) plus a dark-matter particle SoA, then evolved
# in RAMSES super-comoving variables (patch_cosmo.jl) — plain leapfrog hydro/particles,
# Poisson source 1.5·Ωm·a·δ, per-a chemistry units — with NO MPI and NO host code: each
# patch's hydro (PPMKernels) + chemistry (ChemistryKernels) runs on the GPU, the top-grid
# gravity is one global threaded FFTW solve (PoissonKernels) gathered over all patches.
#
# Outputs (REPORTS/patch_cellcmp_z*.bin) use the SAME layout as the Enzo/RAMSES drivers
# (N, ρb, x_HII, f_H2, f_HD, T[K]) so the existing compare_cellcmp / plot scripts read
# them directly — the calibration of growth vs the RAMSES reference.
#
# Run (GPU 128³ → 8×64³):
#   BACKEND=cuda CIC_NGRID=128 CIC_NP=2 \
#     julia --project=lib/MultiCode/test lib/MultiCode/examples/patch_cicass.jl
#   (CPU-f64 reference: BACKEND=cpu; supply an existing IC with CIC_SNAP=/tmp/.../cic_ramses.cicass)

using MultiCode, CICASSLib, Printf, Statistics
using HDF5
import PoissonKernels
import PPMKernels
import ChemistryKernels
import EmissionKernels

const BE = Symbol(get(ENV, "BACKEND", "cuda"))
BE === :cuda && @eval import CUDA              # register the :cuda kernels backends
const T  = BE === :cpu ? Float64 : Float32
const NG = 4

# ── parameters (defaults match the cicass cross-code comparison) ──
const NGRID  = parse(Int, get(ENV, "CIC_NGRID", "128"))
const NPX    = parse(Int, get(ENV, "CIC_NP",    "2"))
const ZSTART = parse(Float64, get(ENV, "CIC_ZSTART", "1000.0"))
const ZEND   = parse(Float64, get(ENV, "CIC_ZEND",   "20.0"))
const VBC    = parse(Float64, get(ENV, "CIC_VBC",    "30.0"))
const BOXMPCH= parse(Float64, get(ENV, "CIC_BOX",    "0.2"))      # Mpc/h (CICASS default)
const GAMMA  = 5/3
const MAXEXP = parse(Float64, get(ENV, "CIC_MAXEXP",  "0.1"))     # = RAMSES da/a cap
const COURANT= parse(Float64, get(ENV, "CIC_COURANT", "0.8"))     # = RAMSES courant_factor
const PCOUR  = parse(Float64, get(ENV, "CIC_PCOURANT","0.8"))
const NOUT   = parse(Int, get(ENV, "CIC_NOUT", "6"))
const MAXCYC = parse(Int, get(ENV, "CIC_MAXCYC", "200000"))
const DODRAG = get(ENV, "CIC_COMPTON_DRAG", "1") == "1"
# CIC_SOLVER = ppm (PPMKernels split sweeps, default) | fvgk (FiniteVolumeGodunovKA unsplit CTU;
# advects the 3 species as EulerColors passive scalars). :fvgk loads FVGK to activate MultiCodeFVGKExt.
const SOLVER = Symbol(get(ENV, "CIC_SOLVER", "ppm"))
SOLVER === :fvgk && @eval using FiniteVolumeGodunovKA
# CIC_PACKED=1: store the 3 species as UInt16 log₂-packed mass fractions (2 B/cell vs 4 B
# f32) — works with both :fvgk and (now) :ppm.  The PPM sweep decodes Xᵢ·ρ→ρXᵢ into f32
# scratch per axis, advects, re-encodes ρXᵢ/ρ→Xᵢ; the chem path already solves in UInt16.
const PACKED = get(ENV, "CIC_PACKED", "0") == "1"
# CIC_OVERLAP=1: overlap the host top-grid gravity with the GPU hydro+chem.  The
# patch_step!/push_particles! GPU kernels are launched ASYNC, then the CPU computes
# the NEXT step's accel (FFT + scatter) while the GPU runs this step; the result is
# applied as the gravity kick at the next step (one-step-lagged, operator-split).
const OVERLAP = get(ENV, "CIC_OVERLAP", "0") == "1"
# CIC_CHEM=analytic: fast closed-form chemistry for the post-recombination IGM — ONLY HII Case-B
# recombination + Compton heat/cool off the CMB (the processes that touch the energy eq here), no
# stiff subcycler.  CIC_CHEM=full (default) runs the general ChemistryKernels network.
const CHEMMODE = Symbol(get(ENV, "CIC_CHEM", "full"))
# CIC_CHEM_NSUB: analytic sub-steps per hydro step (default 1 = fastest).  >1 tracks the evolving
# T so k2(T) follows the Compton heating within the step → tighter match to the stiff network.
const CHEMNSUB = parse(Int, get(ENV, "CIC_CHEM_NSUB", "1"))
# analytic chem only evolves HII, so carry a SINGLE color (HII) — H2I/HDI are unused tracers.
# Fewer colors = less advection AND a normal-valued color the FVGK f16-tiled path can carry.
const NSPEC = CHEMMODE === :analytic ? 1 : 3
const DEUT  = CHEMMODE !== :analytic            # HDI/deuterium only in the full network
# CIC_PHASE_TIMING=1: CUDA-synced per-phase split (gravity | hydro | chem | particles).  Adds
# barriers that serialize the GPU, so it INFLATES the wall a bit — use it only for the breakdown,
# never to quote production throughput (the uninstrumented sec/cyc is the real number).
const PHASE = get(ENV, "CIC_PHASE_TIMING", "0") == "1"
# CIC_PSORT=K: Morton-sort the DM particle SoA every K steps to keep the CIC deposit/force-gather
# coalesced as the DM clusters (bit-identical — deposit+push are order-independent).  0 = off.
const PSORT = parse(Int, get(ENV, "CIC_PSORT", "0"))
# CIC_PIDS=1 (auto-on under PSORT): carry a Lagrangian particle index `id` (permuted with the sort)
# so the particle→Lagrangian-grid map survives reordering — needed to rebuild the phase-space sheet.
const PIDS  = PSORT > 0 || get(ENV, "CIC_PIDS", "0") == "1"
# CIC_PK=1: measure anisotropic P(k,μ) ON DEVICE at every output redshift (gas δ, DM δ,
# gas velocity) straight from the resident GPU fields — tiny "<ckpref>_pkmu.h5" tables
# instead of multi-GB full-state dumps.  μ=|k_axis|/|k| (the v_bc stream is ∥ CIC_PKAXIS).
const PKMEAS = get(ENV, "CIC_PK",     "0") == "1"
const PKMU   = parse(Int, get(ENV, "CIC_PKMU",   "4"))
const PKAXIS = parse(Int, get(ENV, "CIC_PKAXIS", "1"))
const PKNB   = parse(Int, get(ENV, "CIC_PKNB",   "0"))    # k-bins (0 ⇒ ncell÷2)
const REPORTS= joinpath(@__DIR__, "..", "..", "..", "reports", "multicode")
const TAG    = get(ENV, "CIC_TAG", "")
const XH     = 0.76

# ── load (or generate) the CICASS realization ──
function load_snapshot()
    path = get(ENV, "CIC_SNAP", "")
    if !isempty(path)
        @printf("loading CICASS snapshot: %s\n", path); flush(stdout)
        return CICASSLib.read_snapshot(path)
    end
    @printf("generating CICASS realization: %d³ box=%.3f Mpc/h vbc=%.1f z=%.0f\n",
            NGRID, BOXMPCH, VBC, ZSTART); flush(stdout)
    r = MultiCode.run_cicass_streaming(; vbc=VBC, boxlength=BOXMPCH, zstart=ZSTART, ngrid=NGRID)
    return CICASSLib.read_snapshot(r.output)
end

# ── build the gas IC (host ncell³ arrays) in RAMSES super-comoving code units ──
# ρ_code = f_b·(1+δ_b)  (mean f_b);  S = ρ·v_code, v_code = v_kms·1e5 / scale_v(a_i);
# Ge = ρ·eint, eint = T_gas / ((γ−1)·μ·scale_T2(a_i)), μ=1.22 (neutral);  species ρ·x.
function gas_ic(snap, c::Cosmo, a_i, u_i)
    N = snap.n; nc = (N, N, N)
    s = CICASSLib.thermal_state(ZSTART)
    xHII0 = s.x_e; Tg = s.T_gas
    μ = 1.22
    eint = Tg / ((GAMMA - 1) * μ * u_i.T2)               # specific internal energy, code
    vconv = 1.0e5 / u_i.v                                # km/s → code velocity
    δb = CICASSLib.grid3d(snap, snap.gas_delta)
    gv = snap.gas_vel                                    # (N³×3) km/s, lattice m = i+jN+kN²
    D  = Array{Float64,3}(undef, nc)
    S1 = similar(D); S2 = similar(D); S3 = similar(D); Ge = similar(D); Tau = similar(D)
    HII = similar(D); H2I = similar(D); HDI = similar(D)
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        m = i + (j-1)*N + (k-1)*N^2                       # grafic lattice index
        ρ = c.fb * (1 + δb[i,j,k])
        v1 = gv[m,1]*vconv; v2 = gv[m,2]*vconv; v3 = gv[m,3]*vconv
        D[i,j,k]  = ρ
        S1[i,j,k] = ρ*v1; S2[i,j,k] = ρ*v2; S3[i,j,k] = ρ*v3
        Ge[i,j,k] = ρ*eint
        Tau[i,j,k]= ρ*(eint + 0.5*(v1^2 + v2^2 + v3^2))
        HII[i,j,k]= ρ*xHII0; H2I[i,j,k] = ρ*1e-6; HDI[i,j,k] = ρ*6.8e-5*xHII0
    end
    @printf("gas IC: f_b=%.4f  T_gas=%.1f K (eint=%.3e code)  x_HII0=%.3e  v→code=%.4e\n",
            c.fb, Tg, eint, xHII0, vconv); flush(stdout)
    species = NSPEC == 1 ? [HII] : [HII, H2I, HDI]    # analytic: HII-only color
    return (D=Float64.(D), S1=S1, S2=S2, S3=S3, Tau=Tau, Ge=Ge, species=species)
end

# ── dark-matter particle SoA (global box-normalized [0,1) positions, code velocities) ──
# mass_per = (1−f_b): N³ particles ⇒ CIC mean DM density = 1−f_b, so gas+DM mean = 1.
function dm_ic(snap, c::Cosmo, u_i, backend)
    pos = snap.dm_pos; vel = snap.dm_vel; Npart = size(pos, 1)
    vconv = 1.0e5 / u_i.v
    dev(v) = PPMKernels.to_device(backend, v, T)
    mk(col, conv) = dev([T(conv*col[p]) for p in 1:Npart])
    px = dev([T(mod(pos[p,1], 1.0)) for p in 1:Npart])
    py = dev([T(mod(pos[p,2], 1.0)) for p in 1:Npart])
    pz = dev([T(mod(pos[p,3], 1.0)) for p in 1:Npart])
    vx = mk(@view(vel[:,1]), vconv); vy = mk(@view(vel[:,2]), vconv); vz = mk(@view(vel[:,3]), vconv)
    mass = T(1 - c.fb)                                   # SCALAR: equal-mass DM ⇒ no N³ mass array
    @printf("DM IC: %d particles, mass_per=%.4f (1−f_b), v→code=%.4e%s\n",
            Npart, 1-c.fb, vconv, PIDS ? "  (+Lagrangian id)" : ""); flush(stdout)
    parts = (px=px, py=py, pz=pz, vx=vx, vy=vy, vz=vz, mass=mass)
    # Lagrangian index id = i+jN+kN² (the IC grid order) → unravel to (i,j,k) for the sheet
    PIDS && (parts = (; parts..., id=PPMKernels.to_device(backend, collect(Int32, 0:Npart-1))))
    return parts
end

# ── per-patch signal speed (code units): max(|vx|+|vy|+|vz|+3·cs) over all patches ──
function max_signal(pg)
    smax = 0.0
    li = (pg.ng+1):(pg.ng+pg.pdim[1]); lj = (pg.ng+1):(pg.ng+pg.pdim[2]); lk = (pg.ng+1):(pg.ng+pg.pdim[3])
    iv(f) = @view reshape(f, pg.nd)[li, lj, lk]      # INTERIOR view (exclude ghosts) so the
    for p in pg.patches                              # CFL timestep depends only on the physical
        D = iv(p.D)                                  # state — makes checkpoint/restart bit-exact
        cs = sqrt.(max.(GAMMA*(GAMMA-1) .* (iv(p.Ge) ./ D), 0))   # (and avoids D=0 ghost NaNs)
        sig = abs.(iv(p.S1) ./ D) .+ abs.(iv(p.S2) ./ D) .+ abs.(iv(p.S3) ./ D) .+ 3 .* cs
        smax = max(smax, Float64(maximum(sig)))
    end
    return smax
end
max_pvel(parts) = max(Float64(maximum(abs.(parts.vx))),
                      Float64(maximum(abs.(parts.vy))),
                      Float64(maximum(abs.(parts.vz))))

# ── write a cellcmp dump (same layout as enzo_cellcmp / ramses cellcmp) ──
function write_cellcmp(pg, c::Cosmo, u, a, z)
    g = gather_global(pg)
    ρb  = Float64.(vec(g.D))
    HII = Float64.(vec(g.species[1]))
    H2I = length(g.species) >= 2 ? Float64.(vec(g.species[2])) : zero(HII)
    HDI = length(g.species) >= 3 ? Float64.(vec(g.species[3])) : zero(HII)
    eint = Float64.(vec(g.Ge)) ./ ρb
    xHIIv = HII ./ ρb ./ XH; fH2v = H2I ./ ρb ./ XH; fHDv = HDI ./ ρb
    μv = 1.0 ./ ((XH + (1-XH)/4) .+ XH .* (xHIIv .- 0.5 .* fH2v))
    Tcell = eint .* ((GAMMA-1) .* μv .* u.T2)                  # K
    mkpath(REPORTS)
    open(joinpath(REPORTS, "patch_cellcmp_$(TAG)_z$(round(Int,z)).bin"), "w") do io
        write(io, Int64(pg.ncell[1]))
        write(io, ρb); write(io, xHIIv); write(io, fH2v); write(io, fHDv); write(io, Tcell)
    end
    return (ρmean=mean(ρb), δrms=std(ρb)/mean(ρb), xHII=mean(xHIIv), T=mean(Tcell))
end

# ── flat-HDF5 PatchGrid checkpoint/restart ────────────────────────────────────
# Full restart state: the 6 conserved fields + 3 species (interior, via gather_global),
# the DM particle SoA, and the scalars (a, cycle) + grid/units/cosmology metadata.
# No HG, no cell-path bookkeeping — a uniform Cartesian dump that scatter_global!
# consumes directly.  Round-trip is Float32-exact (gather/scatter strip/restore
# interiors; ghosts are re-derived by scatter_global!'s exchange_ghosts!).
const _CKF = ("D","S1","S2","S3","Tau","Ge")
const _CKS = ("HII","H2I","HDI")
const _CKP = ("px","py","pz","vx","vy","vz")          # mass is a scalar attribute (equal-mass DM)

function save_checkpoint(path, pg, parts, c, a, cyc)
    g = gather_global(pg)
    h5open(path, "w") do f
        for (nm, arr) in zip(_CKF, (g.D, g.S1, g.S2, g.S3, g.Tau, g.Ge)); f["fields/"*nm] = arr; end
        for (nm, arr) in zip(_CKS, g.species); f["species/"*nm] = arr; end
        for nm in _CKP; f["particles/"*nm] = Array(getfield(parts, Symbol(nm))); end
        haskey(parts, :id) && (f["particles/id"] = Array(parts.id))   # Lagrangian index for the sheet
        A = attrs(f)
        A["pmass"] = Float64(parts.mass)                  # scalar equal-mass DM
        A["dims"] = collect(pg.ncell); A["a"] = a; A["z"] = a_to_z(a); A["cyc"] = cyc
        A["dx"] = pg.dx; A["gamma"] = pg.gamma; A["du"] = pg.du; A["lu"] = pg.lu; A["tu"] = pg.tu
        A["Om"] = c.Om; A["OL"] = c.OL; A["Or"] = c.Or; A["h0"] = c.h0; A["box"] = c.box
        A["Ob"] = c.fb*c.Om; A["XH"] = c.XH
    end
    @printf("  ✔ checkpoint z=%.2f a=%.5f cyc=%d → %s\n", a_to_z(a), a, cyc, path); flush(stdout)
end

function load_checkpoint(path)
    h5open(path, "r") do f
        fields = (D=read(f,"fields/D"), S1=read(f,"fields/S1"), S2=read(f,"fields/S2"),
                  S3=read(f,"fields/S3"), Tau=read(f,"fields/Tau"), Ge=read(f,"fields/Ge"),
                  species=[read(f,"species/"*nm) for nm in _CKS])
        parts = NamedTuple{Symbol.(_CKP)}(Tuple(read(f,"particles/"*nm) for nm in _CKP))
        haskey(f, "particles/id") && (parts = (; parts..., id=read(f,"particles/id")))
        A = attrs(f)
        return (; fields, parts, pmass=Float64(A["pmass"]), ncell=Tuple(Int.(A["dims"])), a=Float64(A["a"]), cyc=Int(A["cyc"]),
                dx=Float64(A["dx"]), du=Float64(A["du"]), lu=Float64(A["lu"]), tu=Float64(A["tu"]),
                Om=Float64(A["Om"]), OL=Float64(A["OL"]), Or=Float64(A["Or"]),
                h0=Float64(A["h0"]), box=Float64(A["box"]), Ob=Float64(A["Ob"]), XH=Float64(A["XH"]))
    end
end

function main()
    if haskey(ENV, "CIC_RESTART")
        ck = load_checkpoint(ENV["CIC_RESTART"])
        N = ck.ncell[1]; ncell = ck.ncell; np = (NPX, NPX, NPX)
        c = Cosmo(; Om=ck.Om, OL=ck.OL, h0=ck.h0, box=ck.box, Ob=ck.Ob, XH=ck.XH, Or=ck.Or)
        a_start = ck.a; a_end = z_to_a(ZEND); u_i = cosmo_units(c, z_to_a(ZSTART)); dx = ck.dx
        cyc_start = ck.cyc
        pg = build_patchgrid(; ng=NG, ncell=ncell, np=np, dx=dx, gamma=GAMMA, nspecies=3,
                             besym=BE, T=T, du=ck.du, lu=ck.lu, tu=ck.tu, deut=true, packed_species=PACKED)
        scatter_global!(pg, ck.fields)
        parts = merge(NamedTuple{Symbol.(_CKP)}(Tuple(PPMKernels.to_device(pg.backend, getfield(ck.parts, Symbol(nm)), T) for nm in _CKP)),
                      (mass = T(ck.pmass),))
        haskey(ck.parts, :id) && (parts = (; parts..., id=PPMKernels.to_device(pg.backend, ck.parts.id)))
        @printf("RESTART %s: %d³ at z=%.2f a=%.5f cyc=%d  (→ z=%.0f)\n",
                ENV["CIC_RESTART"], N, a_to_z(a_start), a_start, cyc_start, ZEND); flush(stdout)
        return run_evolution(c, N, ncell, np, a_start, a_end, u_i, dx, pg, parts, cyc_start)
    end
    snap = load_snapshot()
    N = snap.n
    ncell = (N, N, N); np = (NPX, NPX, NPX)
    c = Cosmo(; Om=snap.omega_m, OL=snap.omega_l, h0=snap.hconst*100, box=snap.box, Ob=snap.omega_b)
    a_start = z_to_a(ZSTART); a_end = z_to_a(ZEND)
    u_i = cosmo_units(c, a_start)
    @printf("CICASS patch run: %d³ → %d patches of %d³, box=%.4f Mpc/h, Ωm=%.3f Ωb=%.4f ΩΛ=%.3f h=%.3f\n",
            N, prod(np), N÷NPX, c.box, c.Om, c.fb*c.Om, c.OL, c.h0/100)
    @printf("  z=%.0f→%.0f  a=%.3e→%.3e  scale_v(a_i)=%.4e cm/s  D(a_i)=%.4e\n",
            ZSTART, ZEND, a_start, a_end, u_i.v, growth_D(c, a_start)); flush(stdout)

    # build the decomposition (dx=1/ncell: super-comoving box=1, a absorbed into units)
    dx = 1.0 / N
    pg = build_patchgrid(; ng=NG, ncell=ncell, np=np, dx=dx, gamma=GAMMA, nspecies=NSPEC,
                         besym=BE, T=T, du=u_i.d, lu=u_i.l, tu=u_i.t, deut=DEUT, packed_species=PACKED)
    scatter_global!(pg, gas_ic(snap, c, a_start, u_i))
    parts = dm_ic(snap, c, u_i, pg.backend)
    return run_evolution(c, N, ncell, np, a_start, a_end, u_i, dx, pg, parts, 0)
end

# Shared gravity setup + cosmological evolution loop (IC start with cyc_start=0, or
# resumed from a checkpoint at cyc_start).
function run_evolution(c, N, ncell, np, a_start, a_end, u_i, dx, pg, parts, cyc_start)
    # parallel CPU FFT for the top-grid gravity: :fftw (FFTW threads) or :ka
    # (KernelAbstractions radix-2 on the CPU backend, parallel over Julia threads).
    fftsolver = Symbol(get(ENV, "CIC_FFT", "ka"))
    # CIC_ACCEL = cpu (host segment-copy scatter, default) | gpu (device comp_accel + gather)
    accelmode = Symbol(get(ENV, "CIC_ACCEL", "cpu"))
    # CIC_GRAVITY = cpu (host FFT + scatter, default) | gpu (FULL on-GPU gravity: device
    #   assemble + device FFT + device scatter — pairs with CIC_CHEM_BACKEND=cpu, the role flip).
    gravmode = Symbol(get(ENV, "CIC_GRAVITY", "cpu"))
    # CIC_CHEM_BACKEND = backend (default = BE, chem on GPU) | cpu (stiff chem on the host CPU —
    #   faster, no warp divergence; overlaps GPU gravity in the flip).
    chembk = Symbol(get(ENV, "CIC_CHEM_BACKEND", string(BE)))
    nthr = parse(Int, get(ENV, "CIC_FFT_THREADS", string(min(8, Sys.CPU_THREADS))))
    PoissonKernels.fft_set_num_threads!(nthr)
    # CIC_CHEM_TABLES=1 (default): log–log rate table for the chemistry hot path (~2.4× the
    # stiff network on GPU, <1e-5 vs the analytic fits); =0 falls back to the analytic fits.
    usetab  = get(ENV, "CIC_CHEM_TABLES", "1") == "1" && CHEMMODE !== :analytic   # analytic uses no tables
    tabbe   = chembk === :cpu ? :cpu : chembk
    ratetab = usetab ? ChemistryKernels.build_rate_tables(; precision=Float64, backend=tabbe) : nothing
    cooltab = usetab ? EmissionKernels.build_cooling_tables(; precision=Float64, backend=tabbe) : nothing
    @printf("  gravity = %s (FFT=%s, scatter=%s)  chem = %s  (FFTW threads=%d, Julia threads=%d)\n",
            gravmode, fftsolver, accelmode, chembk, nthr, Threads.nthreads()); flush(stdout)
    ρg = zeros(Float64, ncell); φg = zeros(Float64, ncell)
    # full-GPU gravity scratch in pg.T (Float32) — the mean is the known Ω-fixed constant
    # (subtracted in assemble_global_density_gpu!), so no f64 reduction / no f64 arrays needed.
    ρd = gravmode === :gpu ? PPMKernels.device_zeros(pg.backend, T, ncell) : nothing
    φd = gravmode === :gpu ? PPMKernels.device_zeros(pg.backend, T, ncell) : nothing
    pscratch = nothing
    grav_t = Ref(0.0); fft_t = Ref(0.0); ngrav = Ref(0)
    asm_t = Ref(0.0); pacc_t = Ref(0.0); pfld_t = Ref(0.0)
    gph_t = Ref(0.0); hyd_t = Ref(0.0); chm_t = Ref(0.0); prt_t = Ref(0.0); nph = Ref(0)  # CIC_PHASE_TIMING

    # output checkpoints in a: explicit CIC_ZOUT="z1,z2,…" (the cross-code list) or log-spaced.
    # Anchor the log-spaced schedule to ZSTART (NOT a_start) so it is IDENTICAL on a restart —
    # otherwise the dτ_out clamps land on different redshifts and the timestep sequence (hence
    # the whole evolution) diverges from the uninterrupted run.
    a_outs = haskey(ENV, "CIC_ZOUT") ?
        sort([z_to_a(parse(Float64, s)) for s in split(ENV["CIC_ZOUT"], ",")]) :
        exp.(range(log(z_to_a(ZSTART)), log(a_end), length=NOUT))
    NOUTS = length(a_outs)
    # on a restart, skip outputs already crossed; growth is always normalised to ZSTART
    ai = searchsortedfirst(a_outs, a_start - 1e-12); D_start = growth_D(c, z_to_a(ZSTART))
    pk_log = NamedTuple[]
    # CIC_ZCHECKPOINT="z1,z2,…": write a full flat-HDF5 PatchGrid checkpoint when a first
    # crosses each z (named <CIC_CKPT_PREFIX>_z<z>.h5, default prefix "cicass_ckpt").
    a_chks = haskey(ENV, "CIC_ZCHECKPOINT") ?
        [z_to_a(parse(Float64, s)) for s in split(ENV["CIC_ZCHECKPOINT"], ",")] : Float64[]
    z_chks = [parse(Float64, s) for s in (haskey(ENV,"CIC_ZCHECKPOINT") ? split(ENV["CIC_ZCHECKPOINT"], ",") : String[])]
    chk_done = falses(length(a_chks))
    ckpref = get(ENV, "CIC_CKPT_PREFIX", "cicass_ckpt")

    # the top-grid gravity, split so the GPU step can be launched between the host
    # density snapshot and the (overlappable) CPU FFT + accel scatter.
    snapshot!(dt_, a_) = assemble_global_density!(ρg, pg; particles=parts, dt=dt_, a=1.0)
    function solve_accel!(a_)                       # ρg already filled; returns accel tuple
        tf = time(); solve_global_poisson!(φg, ρg; G=1.5*c.Om*a_, a=1.0, boxsize=1.0, solver=fftsolver); fft_t[] += time()-tf
        tp = time()
        ga = accelmode === :gpu ? patch_accel_gpu(pg, φg; dx=dx) : patch_accel(pg, φg; dx=dx)
        BE === :cuda && accelmode === :gpu && CUDA.synchronize()    # so the timer captures the device gather
        pacc_t[] += time()-tp
        tq = time(); φpad, gle, gcs = particle_accel_field(pg, φg); pfld_t[] += time()-tq
        ngrav[] += 1
        return (gas=ga, phi=φpad, le=gle, cs=gcs)
    end
    # full gravity for scale factor a_ (and particle half-drift dt_): CPU (host FFT) path
    # needs the host density snapshot first; GPU path assembles + solves entirely on device.
    function gravity!(a_, dt_)
        if gravmode === :gpu
            g = global_gravity_gpu(pg; G=1.5*c.Om*a_, a=1.0, boxsize=1.0, particles=parts, dt=dt_, ρd=ρd, φd=φd)
            BE === :cuda && CUDA.synchronize(); ngrav[] += 1
            return g
        else
            tg = time(); snapshot!(dt_, a_); asm_t[] += time()-tg
            return solve_accel!(a_)
        end
    end

    a = a_start; m0 = total_mass(pg)
    # lag-free overlap: seed the accel from the IC density (used by cycle-0 hydro)
    acc = nothing
    if OVERLAP
        sig0 = max_signal(pg); dτ0 = min(COURANT*dx/max(sig0,1e-30), dtau_for_dlna(c, a, MAXEXP))
        acc = gravity!(a, dτ0)
    end
    @printf("%-5s %-9s %-9s %-9s %-9s %-7s\n", "cyc", "a", "z", "δb_rms", "ρmax", "sec")
    for cyc in cyc_start:MAXCYC-1
        t0 = time()
        z = a_to_z(a); u = cosmo_units(c, a)

        # ── Morton-resort the DM SoA every PSORT steps (keeps deposit/gather coalesced; bit-identical) ──
        if PSORT > 0 && cyc % PSORT == 0
            morton_sort_particles!(parts; N=N)
        end

        # ── outputs at crossed checkpoints (checked BEFORE stepping so a lands exactly) ──
        while ai <= NOUTS && a >= a_outs[ai] - 1e-12
            zo = a_to_z(a_outs[ai]); uo = cosmo_units(c, a)
            st = write_cellcmp(pg, c, uo, a, zo)
            g2 = (growth_D(c, a)/D_start)^2
            push!(pk_log, (z=zo, a=a, δrms=st.δrms, g2=g2, xHII=st.xHII, T=st.T))
            @printf("  ● output z=%.2f a=%.5f  δb_rms=%.3e  D²/D₀²=%.3e  <x_HII>=%.3e  <T>=%.1f K\n",
                    zo, a, st.δrms, g2, st.xHII, st.T); flush(stdout)
            if PKMEAS                                     # on-device anisotropic P(k,μ); tiny table, no full dump
                P  = patch_power_spectra(pg, parts; box=c.box, nmu=PKMU, nbins=PKNB, axis=PKAXIS, scale_v=uo.v/1e5)
                pf = @sprintf("%s_pkmu.h5", ckpref)
                h5open(pf, isfile(pf) ? "r+" : "w") do f
                    g = create_group(f, @sprintf("z%05.1f", zo))
                    g["k"] = collect(P.k); g["gas_delta"] = P.gas_delta; g["dm_delta"] = P.dm_delta
                    P.gas_vel === nothing || (g["gas_vel"] = P.gas_vel)
                    P.dm_vel  === nothing || (g["dm_vel"]  = P.dm_vel)
                    g["Nmodes"] = P.Nmodes
                    A = attrs(g); A["z"]=zo; A["a"]=a; A["box"]=c.box; A["axis"]=PKAXIS; A["nmu"]=PKMU
                end
                @printf("    ↳ P(k,μ) on device → %s [z%05.1f]\n", pf, zo); flush(stdout)
            end
            ai += 1
            BE === :cuda && CUDA.reclaim()
        end

        # ── full-state checkpoint when a first crosses a requested z (top of cycle) ──
        for k in eachindex(a_chks)
            if !chk_done[k] && a >= a_chks[k] - 1e-12
                save_checkpoint(@sprintf("%s_z%d.h5", ckpref, round(Int, z_chks[k])), pg, parts, c, a, cyc)
                chk_done[k] = true
            end
        end
        a >= a_end && break

        # ── timestep: min(hydro CFL, particle CFL, Δln a cap, next-output) in super-conf τ ──
        sig = max_signal(pg); vp = max_pvel(parts)
        dτ_h = COURANT * dx / max(sig, 1e-30)
        dτ_p = PCOUR   * dx / max(vp, 1e-30)
        dτ_e = dtau_for_dlna(c, a, MAXEXP)
        dτ   = min(dτ_h, dτ_p, dτ_e)
        if ai <= NOUTS && a_outs[ai] > a
            dτ_out = dtau_for_dlna(c, a, log(a_outs[ai]/a))
            dτ = min(dτ, dτ_out)
        end

        # ── advance a over dτ (RK2 on da/dτ); needed now for the next-step gravity a ──
        k1 = dadtau(c, a); amid = a + 0.5*k1*dτ; k2 = dadtau(c, amid)
        a_new = a + k2*dτ; dlna = log(a_new/a)

        # ── top-grid gravity ⊕ hydro/chem ──
        # super-comoving Poisson: ∇²φ = 1.5·Ωm·a·δ  ⇒  G/a_solver = 1.5·Ωm·a, a_solver=1.
        # `acc` holds accel(ρ_now) (built last cycle from the post-hydro density — chem doesn't
        # change ρ, so it's exact, lag-free).  Run the GPU HYDRO phase, then overlap this step's
        # CHEMISTRY with the NEXT step's GRAVITY — which device each lands on depends on the mode:
        #   gravmode=:cpu → CPU gravity (main) ∥ GPU chem (spawn)   [host FFT]
        #   gravmode=:gpu → GPU gravity (spawn) ∥ CPU chem (main)   [the role FLIP]
        order = isodd(cyc) ? (3,2,1) : (1,2,3)
        tg = time()
        if OVERLAP
            patch_step!(pg, dτ; a_value=a, order=order, accel=acc.gas, chem=true, solver=SOLVER,
                        du=u.d, lu=u.l, tu=u.t, do_hydro=true, do_chem=false,
                        chemmode=CHEMMODE, chemnsub=CHEMNSUB, cosmo_h0=c.h0, cosmo_Om=c.Om, cosmo_OL=c.OL)
            pscratch = push_particles!(parts, acc.phi, acc.le, acc.cs, dτ; scratch=pscratch)
            BE === :cuda && CUDA.synchronize()            # hydro+push done ⇒ ρ_next density final
            if gravmode === :gpu
                gpu = Threads.@spawn begin                # NEXT-step gravity, fully on GPU
                    g = global_gravity_gpu(pg; G=1.5*c.Om*a_new, a=1.0, boxsize=1.0,
                                           particles=parts, dt=dτ, ρd=ρd, φd=φd)
                    BE === :cuda && CUDA.synchronize(); g
                end
                patch_step!(pg, dτ; a_value=a, order=order, chem=true, du=u.d, lu=u.l, tu=u.t,
                            do_hydro=false, do_chem=true, chem_backend=:cpu, rate_tables=ratetab, cool_tables=cooltab,
                            chemmode=CHEMMODE, chemnsub=CHEMNSUB, cosmo_h0=c.h0, cosmo_Om=c.Om, cosmo_OL=c.OL)   # CPU chem on main thread
                acc = fetch(gpu); ngrav[] += 1
            else
                ta = time(); snapshot!(dτ, a_new); asm_t[] += time()-ta        # host ρ_next for the CPU FFT
                gpu = Threads.@spawn begin                # this step's chemistry on the GPU
                    patch_step!(pg, dτ; a_value=a, order=order, chem=true, du=u.d, lu=u.l, tu=u.t,
                                do_hydro=false, do_chem=true, rate_tables=ratetab, cool_tables=cooltab,
                                chemmode=CHEMMODE, chemnsub=CHEMNSUB, cosmo_h0=c.h0, cosmo_Om=c.Om, cosmo_OL=c.OL)
                    BE === :cuda && CUDA.synchronize()
                end
                acc = solve_accel!(a_new)                 # next-step accel on the CPU, overlaps chem
                wait(gpu)
            end
        elseif PHASE
            # per-phase split (CUDA-synced): gravity | hydro | chem | particles
            sync() = (BE === :cuda && CUDA.synchronize())
            sync(); tt = time(); acc = gravity!(a, dτ); sync(); gph_t[] += time()-tt
            tt = time()
            patch_step!(pg, dτ; a_value=a, order=order, accel=acc.gas, chem=true, solver=SOLVER,
                        du=u.d, lu=u.l, tu=u.t, do_hydro=true, do_chem=false,
                        chemmode=CHEMMODE, chemnsub=CHEMNSUB, cosmo_h0=c.h0, cosmo_Om=c.Om, cosmo_OL=c.OL)
            sync(); hyd_t[] += time()-tt; tt = time()
            patch_step!(pg, dτ; a_value=a, order=order, chem=true, solver=SOLVER,
                        du=u.d, lu=u.l, tu=u.t, do_hydro=false, do_chem=true, chem_backend=chembk,
                        rate_tables=ratetab, cool_tables=cooltab,
                        chemmode=CHEMMODE, chemnsub=CHEMNSUB, cosmo_h0=c.h0, cosmo_Om=c.Om, cosmo_OL=c.OL)
            sync(); chm_t[] += time()-tt; tt = time()
            pscratch = push_particles!(parts, acc.phi, acc.le, acc.cs, dτ; scratch=pscratch)
            sync(); prt_t[] += time()-tt; nph[] += 1
        else
            acc = gravity!(a, dτ)
            patch_step!(pg, dτ; a_value=a, order=order, accel=acc.gas, chem=true, solver=SOLVER,
                        du=u.d, lu=u.l, tu=u.t, chem_backend=chembk, rate_tables=ratetab, cool_tables=cooltab,
                        chemmode=CHEMMODE, chemnsub=CHEMNSUB, cosmo_h0=c.h0, cosmo_Om=c.Om, cosmo_OL=c.OL)
            pscratch = push_particles!(parts, acc.phi, acc.le, acc.cs, dτ; scratch=pscratch)
        end
        grav_t[] += time() - tg

        # ── Compton momentum drag on the baryons over Δln a ──
        if DODRAG
            zmid = 0.5*(z + a_to_z(a_new)); xe = CICASSLib.thermal_state(zmid).x_e
            γH = compton_drag_over_H(c, zmid, xe)
            f = exp(-γH * dlna)
            f < 0.999999 && compton_drag_patches!(pg, f)
        end
        a = a_new

        sec = time() - t0
        if cyc % 10 == 0
            ρmax = maximum(Float64(maximum(p.D)) for p in pg.patches)
            δrms = density_contrast_rms(pg)            # on-device (no full-grid gather)
            @printf("%-5d %-9.5f %-9.3f %-9.3e %-9.3f %-7.2f\n", cyc, a, a_to_z(a), δrms, ρmax, sec)
            flush(stdout)
        end
    end

    if PHASE && nph[] > 0
        let n = nph[], tot = (gph_t[]+hyd_t[]+chm_t[]+prt_t[])/nph[]
            @printf("\nPER-PHASE (CUDA-synced, %d cyc, %s/%s): gravity %.4f | hydro %.4f | chem %.4f | particles %.4f  → %.4f s/cyc\n",
                    n, SOLVER, CHEMMODE, gph_t[]/n, hyd_t[]/n, chm_t[]/n, prt_t[]/n, tot)
            @printf("  shares: gravity %.0f%% | hydro %.0f%% | chem %.0f%% | particles %.0f%%\n",
                    100gph_t[]/n/tot, 100hyd_t[]/n/tot, 100chm_t[]/n/tot, 100prt_t[]/n/tot)
        end
    end
    @printf("\nmass drift over run: %.3e\n", abs(total_mass(pg)-m0)/m0)
    let n=max(ngrav[],1)
        @printf("top-grid gravity (%s): %.4f s/solve  [assemble %.4f | FFT %.4f | patch_accel %.4f | part_field %.4f] over %d solves\n",
                fftsolver, grav_t[]/n, asm_t[]/n, fft_t[]/n, pacc_t[]/n, pfld_t[]/n, ngrav[])
    end
    @printf("\n%-8s %-10s %-12s %-12s %-12s\n", "z", "δb_rms", "D²/D₀²", "<x_HII>", "<T>[K]")
    for r in pk_log
        @printf("%-8.2f %-10.3e %-12.3e %-12.3e %-12.1f\n", r.z, r.δrms, r.g2, r.xHII, r.T)
    end
    flush(stdout)
end

main()
