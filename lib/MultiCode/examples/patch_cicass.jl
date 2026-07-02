# patch_cicass.jl — FULL CICASS smooth-baryon cosmology run on the in-process topgrid
# decomposition (z=1000→20), GPU hydro+chem per patch + global gravity.
#
# This is the patch-decomposition analogue of cicass_highz_pk.jl (Enzo path) and
# cicass_ramses_pk.jl (RAMSES path): the SAME CICASS streaming-velocity realization is
# loaded into a PatchGrid (gas + species) plus a dark-matter particle SoA, then evolved
# in RAMSES super-comoving variables (patch_cosmo.jl) — plain leapfrog hydro/particles,
# Poisson source 1.5·Ωm·a·δ, per-a chemistry units — with NO MPI and NO host code: each
# patch's hydro (PPMKernels) + chemistry (ChemistryKernels) runs on the GPU, the top-grid
# gravity is one global solve (threaded FFTW on the host or rFFT/irFFT on the GPU).
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

const KA = PPMKernels.KA

const BE = Symbol(get(ENV, "BACKEND", "cuda"))
BE === :cuda && @eval using CUDA               # register the :cuda kernel backend
BE === :metal && @eval using Metal             # register the :metal kernel backend
const T  = BE === :cpu ? Float64 : Float32
const IC_REAL_BYTES = parse(Int, get(ENV, "CICASS_REAL_BYTES", BE === :cpu ? "8" : "4"))
IC_REAL_BYTES in (4, 8) || error("CICASS_REAL_BYTES must be 4 or 8 (got $IC_REAL_BYTES)")
# Patch ghost depth.  The :fvgk solver owns periodicity internally and reads patch interiors
# only (gather/scatter/chem/density all interior); the sole ghost users are the gravity gas
# kick (∂φ central diff ⇒ ng≥1) and the particle force interp (CIC+diff ⇒ ng2≥2), so ng=2 is
# the safe floor.  :ppm needs ng≥3 (parabolic+flattening stencil); keep 4.  CIC_NG overrides.
const NG = parse(Int, get(ENV, "CIC_NG", get(ENV, "CIC_SOLVER", "ppm") == "fvgk" ? "2" : "4"))

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
const PRINT_EVERY = parse(Int, get(ENV, "CIC_PRINT_EVERY", "10"))
const MEMPROBE_CYC = parse(Int, get(ENV, "CIC_MEMPROBE_CYC", "30"))
const MEMPROBE_CONT = get(ENV, "CIC_MEMPROBE_CONT", "0") == "1"
const DODRAG = get(ENV, "CIC_COMPTON_DRAG", "1") == "1"
# CIC_SOLVER = ppm (PPMKernels split sweeps, default) | fvgk (FiniteVolumeGodunovKA unsplit CTU;
# advects the 3 species as EulerColors passive scalars). :fvgk loads FVGK to activate MultiCodeFVGKExt.
const SOLVER = Symbol(get(ENV, "CIC_SOLVER", "ppm"))
SOLVER === :fvgk && @eval using FiniteVolumeGodunovKA
# CIC_PACKED=1: store the 3 species as UInt16 log₂-packed mass fractions (2 B/cell vs 4 B
# f32) — works with both :fvgk and (now) :ppm.  The PPM sweep decodes Xᵢ·ρ→ρXᵢ into f32
# scratch per axis, advects, re-encodes ρXᵢ/ρ→Xᵢ; the chem path already solves in UInt16.
const PACKED = get(ENV, "CIC_PACKED", "0") == "1"
# CIC_OVERLAP=1: overlap the host top-grid gravity with the GPU hydro+chem. The
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
# CIC_PHASE_TIMING=1: GPU-synced per-phase split (gravity | hydro | chem | particles).  Adds
# barriers that serialize the GPU, so it INFLATES the wall a bit — use it only for the breakdown,
# never to quote production throughput (the uninstrumented sec/cyc is the real number).
const PHASE = get(ENV, "CIC_PHASE_TIMING", "0") == "1"
const GRAV_DETAIL = get(ENV, "CIC_GRAV_DETAIL", "0") == "1"
# CIC_PSORT=K: Morton-sort the DM particle SoA every K steps to keep the CIC deposit/force-gather
# coalesced as the DM clusters.  0 = off.  On Metal, CIC_PSORT_MODE=auto uses a low-memory
# coarse-bucket Morton reorder (CIC_PSORT_BUCKET, power-of-two, default ≤256) because Metal has no
# native GPU sortperm; CUDA/CPU keep the stable full sort unless CIC_PSORT_MODE=bucket is forced.
const PSORT = parse(Int, get(ENV, "CIC_PSORT", "0"))
const PSORT_START = parse(Int, get(ENV, "CIC_PSORT_START", "0"))
const SORT_TIMING = get(ENV, "CIC_SORT_TIMING", "0") == "1"

const MAX_SIGNAL_GROUP = 256
const _MAX_SIGNAL_CACHE = Dict{Any,Any}()
const _CELL_SUMMARY_CACHE = Dict{Any,Any}()
# CIC_PIDS=1: carry a Lagrangian particle index `id` (permuted with the sort) so the particle→
# Lagrangian-grid map survives reordering — needed to rebuild the phase-space sheet.  Defaults ON
# under PSORT (Morton) but CIC_PIDS=0 overrides to drop it (−4 B/cell of particle storage) when the
# sheet isn't needed — Morton still works, it just doesn't permute a (nonexistent) id.
const PIDS  = get(ENV, "CIC_PIDS", PSORT > 0 ? "1" : "0") == "1"
# CIC_PK=1: measure anisotropic P(k,μ) ON DEVICE at every output redshift (gas δ, DM δ,
# gas velocity) straight from the resident GPU fields — tiny "<ckpref>_pkmu.h5" tables
# instead of multi-GB full-state dumps.  μ=|k_axis|/|k| (the v_bc stream is ∥ CIC_PKAXIS).
const PKMEAS = get(ENV, "CIC_PK",     "0") == "1"
# CIC_NODUMP=1: skip the multi-GB per-output cellcmp .bin (still compute+print the δrms/xHII/T
# diagnostic).  For P(k)-only analysis runs (CIC_PK=1) that don't need the full-state field dumps.
const NODUMP = get(ENV, "CIC_NODUMP", "0") == "1"
# CIC_VEL16=1: store DM particle VELOCITIES as Float16 (positions stay f32) — 24→18 B/cell of
# particle storage.  The kick/drift math is promoted to f32 (particle_kick!/drift! compute in
# ≥f32), so only the between-step velocity storage is f16.  Small P(k)/growth cost — validate.
const VEL16  = get(ENV, "CIC_VEL16",  "0") == "1"
# CIC_GRAV1BUF=1 (gpu gravity): consolidate the gravity buffers — (a) share ONE device field for
# both ρ and φ (the rfft solve is in-place), and (b) particles read that GLOBAL φ with periodic
# wrap instead of a padded (ncell+2ng2)³ copy.  Drops ρd+φpad → ~4 B/cell persistent + a lower
# per-solve peak.  Bit-identical (in-place FFT + wrap == padded ghost fill).
const GRAV1BUF = get(ENV, "CIC_GRAV1BUF", "0") == "1"
GRAV1BUF && OVERLAP && error("CIC_GRAV1BUF shares the ρ/φ buffer; the async CIC_OVERLAP gravity would overwrite the live φ before the push reads it. Set CIC_OVERLAP=0.")
# global_push (particles read the GLOBAL φ with periodic wrap — the validated force path) was tied to
# GRAV1BUF, but the two-grid gravity NEEDS separate ρ/φ buffers (GRAV1BUF=0) while still wanting global_push.
# Decouple: use global_push whenever GRAV1BUF OR the two-grid solve is on (both keep φ as the global field).
const GRAV2GRID = get(ENV, "CIC_GRAV_2GRID", "0") == "1"
const GLOBALPUSH = GRAV1BUF || GRAV2GRID
# CIC_GRAV_HOST32=1: CPU-gravity host density/potential arrays are Float32. This is the
# Metal hero default; CPU-f64 reference runs can leave it off.
const GRAV_HOST32 = get(ENV, "CIC_GRAV_HOST32", "0") == "1"
const PKMU   = parse(Int, get(ENV, "CIC_PKMU",   "4"))
const PKAXIS = parse(Int, get(ENV, "CIC_PKAXIS", "1"))
const PKNB   = parse(Int, get(ENV, "CIC_PKNB",   "0"))    # k-bins (0 ⇒ ncell÷2)
const PKVEL  = get(ENV, "CIC_PK_VEL", "1") == "1"
const CELLDUMP = get(ENV, "CIC_CELL_DUMP", PKMEAS ? "0" : "1") == "1"
const REPORTS= joinpath(@__DIR__, "..", "..", "..", "reports", "multicode")
const TAG    = get(ENV, "CIC_TAG", "")
const XH     = 0.76

# ── load (or generate) the CICASS realization ──
function load_snapshot()
    path = get(ENV, "CIC_SNAP", "")
    # CIC_SYNTH_IC=1: build an in-memory synthetic snapshot at CIC_NGRID (Lagrangian grid + a tiny
    # single-mode Zel'dovich displacement) — for MEMORY/ceiling tests at grid sizes whose real IC
    # won't fit on disk.  Same array shapes as a real snapshot ⇒ identical GPU footprint.
    if get(ENV, "CIC_SYNTH_IC", "0") == "1"
        N = NGRID; Np = N^3; box = BOXMPCH
        @printf("SYNTHETIC IC: %d³ (%d particles) box=%.3f — memory/ceiling probe\n", N, Np, box); flush(stdout)
        A = 0.1                                            # displacement amplitude (cells) — tiny, valid
        dmp = Matrix{Float64}(undef, Np, 3); dmv = zeros(Float64, Np, 3)
        Threads.@threads for p in 0:Np-1
            i = p % N; j = (p ÷ N) % N; k = p ÷ (N*N)
            x = (i+0.5)/N; y = (j+0.5)/N; z = (k+0.5)/N
            @inbounds dmp[p+1,1] = mod(x + (A/N)*sinpi(2x), 1.0); dmp[p+1,2] = y; dmp[p+1,3] = z
            @inbounds dmv[p+1,1] = 10.0*sinpi(2x)          # small km/s Zel'dovich velocity
        end
        gd = zeros(Float64, Np); gv = zeros(Float64, Np, 3); gt = fill(2727.6, Np)
        return CICASSLib.CICASSSnapshot(N, 1, box, ZSTART, 0.27, 0.046, 0.73, 0.71,
                                        1.0, 1.0, VBC, 2727.6, dmp, dmv, gd, gv, gt)
    end
    if !isempty(path)
        @printf("loading CICASS snapshot: %s\n", path); flush(stdout)
        return CICASSLib.read_snapshot(path)
    end
    # CIC_FIXAMP=1 → Angulo & Pontzen (2016) fixed-amplitude ICs (|δ_k|=√P(k) exactly, random phases):
    # realized P(k) matches the input mode-by-mode, ~4× smaller box-to-box scatter.  CIC_FLIPPHASE=1 → the
    # paired run (δ→−δ); average the {0,1} pair to cancel leading non-Gaussian variance.  CIC_ICSEED sets the seed.
    fixamp = get(ENV, "CIC_FIXAMP", "0") == "1"; flipph = get(ENV, "CIC_FLIPPHASE", "0") == "1"
    icseed = parse(Int, get(ENV, "CIC_ICSEED", "113334"))
    @printf("generating CICASS realization: %d³ box=%.3f Mpc/h vbc=%.1f z=%.0f  seed=%d%s%s\n",
            NGRID, BOXMPCH, VBC, ZSTART, icseed, fixamp ? " FIXED-AMPLITUDE" : "",
            flipph ? " PHASE-FLIPPED(paired)" : ""); flush(stdout)
    r = MultiCode.run_cicass_streaming(; vbc=VBC, boxlength=BOXMPCH, zstart=ZSTART,
                                       ngrid=NGRID, real_bytes=IC_REAL_BYTES, seed=icseed,
                                       fix_amplitude=fixamp, flip_phase=flipph)
    return CICASSLib.read_snapshot(r.output)
end

const _CICASS_HEADER_BYTES = 8 + 4 + 4 + 10*sizeof(Float64)

function cicass_header(path::AbstractString)
    open(path, "r") do io
        magic = read(io, 8)
        magic_s = String(magic)
        real_bytes = magic_s == "CICASS01" ? 8 :
                     (magic_s == "CICASS02" || magic_s == "CICASSF4") ? 4 :
                     error("not a CICASS snapshot (bad magic $(repr(magic_s))): $path")
        n = Int(read(io, Int32)); nsp = Int(read(io, Int32))
        hd = Vector{Float64}(undef, 10); read!(io, hd)
        box, zinit, omm, omb, oml, h, mdm, mgas, vbc, tavg = hd
        return (; n, nsp, box, zinit, omega_m=omm, omega_b=omb, omega_l=oml,
                hconst=h, m_dm=mdm, m_gas=mgas, vbc, tavg, real_bytes,
                field_type = real_bytes == 4 ? Float32 : Float64)
    end
end

@inline _cicass_field_offset(N3::Integer, field::Integer, real_bytes::Integer) =
    _CICASS_HEADER_BYTES + (field - 1) * N3 * real_bytes

function read_cicass_field!(io, buf::Vector, h, field::Integer)
    seek(io, _cicass_field_offset(h.n^3, field, h.real_bytes))
    read!(io, buf)
    return buf
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
    VT = VEL16 ? Float16 : T                             # positions stay f32; velocities opt-in f16
    mkv(col, conv) = PPMKernels.to_device(backend, [VT(conv*col[p]) for p in 1:Npart], VT)
    px = dev([T(mod(pos[p,1], 1.0)) for p in 1:Npart])
    py = dev([T(mod(pos[p,2], 1.0)) for p in 1:Npart])
    pz = dev([T(mod(pos[p,3], 1.0)) for p in 1:Npart])
    vx = mkv(@view(vel[:,1]), vconv); vy = mkv(@view(vel[:,2]), vconv); vz = mkv(@view(vel[:,3]), vconv)
    mass = T(1 - c.fb)                                   # SCALAR: equal-mass DM ⇒ no N³ mass array
    @printf("DM IC: %d particles, mass_per=%.4f (1−f_b), v→code=%.4e%s\n",
            Npart, 1-c.fb, vconv, PIDS ? "  (+Lagrangian id)" : ""); flush(stdout)
    parts = (px=px, py=py, pz=pz, vx=vx, vy=vy, vz=vz, mass=mass)
    # Lagrangian index id = i+jN+kN² (the IC grid order) → unravel to (i,j,k) for the sheet
    PIDS && (parts = (; parts..., id=PPMKernels.to_device(backend, collect(Int32, 0:Npart-1))))
    return parts
end

function scatter_cicass_gas_stream!(pg, path::AbstractString, h, c::Cosmo, a_i, u_i)
    N = h.n; N3 = N^3; T = pg.T; li, lj, lk = MultiCode._interior(pg)
    s = CICASSLib.thermal_state(ZSTART)
    xHII0 = s.x_e; Tg = s.T_gas
    μ = 1.22
    eint = Tg / ((GAMMA - 1) * μ * u_i.T2)
    vconv = 1.0e5 / u_i.v
    @printf("gas IC: f_b=%.4f  T_gas=%.1f K (eint=%.3e code)  x_HII0=%.3e  v→code=%.4e\n",
            c.fb, Tg, eint, xHII0, vconv); flush(stdout)

    raw = Vector{h.field_type}(undef, N3)
    host = zeros(T, pg.nd)
    open(path, "r") do io
        read_cicass_field!(io, raw, h, 7) # gas delta
        δb = reshape(raw, N, N, N)
        for p in pg.patches
            gi, gj, gk = MultiCode._octant(pg, p)
            fill!(host, zero(T))
            @views host[li, lj, lk] .= T(c.fb) .* (one(T) .+ T.(δb[gi, gj, gk]))
            copyto!(p.D, vec(host))
            fill!(host, zero(T))
            @views host[li, lj, lk] .= T(c.fb * eint) .* (one(T) .+ T.(δb[gi, gj, gk]))
            copyto!(p.Ge, vec(host))
            if pg.packed
                fill!(p.species[1], ChemistryKernels.encode_log2sp(T(xHII0)))
            else
                fill!(host, zero(T))
                @views host[li, lj, lk] .= T(c.fb * xHII0) .* (one(T) .+ T.(δb[gi, gj, gk]))
                copyto!(p.species[1], vec(host))
            end
        end

        for (field, slot) in zip(8:10, (:S1, :S2, :S3))
            read_cicass_field!(io, raw, h, field)
            vel = reshape(raw, N, N, N)
            for p in pg.patches
                gi, gj, gk = MultiCode._octant(pg, p)
                fill!(host, zero(T))
                @views host[li, lj, lk] .= T(vconv) .* T.(vel[gi, gj, gk])
                copyto!(getfield(p, slot), vec(host))
            end
        end
    end
    raw = nothing; host = nothing; GC.gc()

    for p in pg.patches
        D = MultiCode._r3(p.D, pg.nd); S1 = MultiCode._r3(p.S1, pg.nd)
        S2 = MultiCode._r3(p.S2, pg.nd); S3 = MultiCode._r3(p.S3, pg.nd)
        Ge = MultiCode._r3(p.Ge, pg.nd); Tau = MultiCode._r3(p.Tau, pg.nd)
        @views begin
            S1[li, lj, lk] .*= D[li, lj, lk]
            S2[li, lj, lk] .*= D[li, lj, lk]
            S3[li, lj, lk] .*= D[li, lj, lk]
            Tau[li, lj, lk] .= Ge[li, lj, lk] .+
                T(0.5) .* (S1[li, lj, lk].^2 .+ S2[li, lj, lk].^2 .+ S3[li, lj, lk].^2) ./ D[li, lj, lk]
        end
    end
    exchange_ghosts!(pg)
    return pg
end

function direct_fvgk_ic(np)
    get(ENV, "CIC_FVGK_DIRECT_IC", "1") == "1" &&
        SOLVER === :fvgk && BE === :metal &&
        get(ENV, "CIC_FVGK_F16", "0") == "1" &&
        get(ENV, "CIC_FVGK_DEDUP", "0") == "1" &&
        prod(np) == 1
end

@inline _fvgk_store_density16(ρ::Float32, db::Float32, ds::Float32) =
    Float16(db > 0f0 ? (ρ / db - 1f0) * ds : ρ)

function _fvgk_direct_gas_params(c::Cosmo, a_i, u_i)
    s = CICASSLib.thermal_state(ZSTART)
    xHII0 = s.x_e
    Tg = s.T_gas
    μ = 1.22
    eint = Tg / ((GAMMA - 1) * μ * u_i.T2)
    vconv = 1.0e5 / u_i.v
    @printf("gas IC: f_b=%.4f  T_gas=%.1f K (eint=%.3e code)  x_HII0=%.3e  v→code=%.4e\n",
            c.fb, Tg, eint, xHII0, vconv)
    flush(stdout)
    return (; xHII0=Float32(xHII0), eint=Float32(eint), vconv=Float32(vconv))
end

function prepare_fvgk_direct_ic!(pg, c::Cosmo)
    configure_fvgk_density_storage!(pg, c)
    MultiCode._fvgk_prebuild_empty!(pg)
    BE === :metal && Metal.synchronize()
    MultiCode._fvgk_dedup!(pg)
    BE === :metal && Metal.synchronize()
    dmsg = pg.dens_base > 0 ? @sprintf(", dbase=%.6g dscale=%.0f", pg.dens_base, pg.dens_scale) : ""
    @printf("  FVGK direct IC ON: gas fields load straight into Metal g.U f16 views%s\n", dmsg)
    flush(stdout)
    return pg
end

function _copy_fvgk_slot16!(dst, buf::Vector{Float16})
    copyto!(dst, buf)
    BE === :metal && Metal.synchronize()
    return dst
end

function _load_fvgk_direct_from_fields!(pg, delta, gasvel, c::Cosmo, a_i, u_i)
    pg.dedup || error("direct FVGK gas IC requires pg.dedup=true")
    prod(pg.np) == 1 || error("direct FVGK gas IC requires np=1")
    p = pg.patches[1]
    N3 = prod(pg.ncell)
    length(delta) == N3 || error("gas delta length $(length(delta)) != ncell product $N3")
    size(gasvel, 1) == N3 && size(gasvel, 2) >= 3 || error("gas velocity has incompatible shape $(size(gasvel))")
    pars = _fvgk_direct_gas_params(c, a_i, u_i)
    ρ = Vector{Float32}(undef, N3)
    tau = Vector{Float32}(undef, N3)
    buf = Vector{Float16}(undef, N3)
    fb = Float32(c.fb); db = Float32(pg.dens_base); ds = Float32(pg.dens_scale)
    gesc = Float32(pg.gesc); msc = Float32(pg.msc)

    Threads.@threads for m in 1:N3
        @inbounds begin
            r = fb * (1f0 + Float32(delta[m]))
            ρ[m] = r
            tau[m] = r * pars.eint
            buf[m] = _fvgk_store_density16(r, db, ds)
        end
    end
    _copy_fvgk_slot16!(p.D, buf)

    Threads.@threads for m in 1:N3
        @inbounds buf[m] = Float16(ρ[m] * pars.eint * gesc)
    end
    _copy_fvgk_slot16!(p.Ge, buf)

    for (d, slot) in enumerate((:S1, :S2, :S3))
        dst = getfield(p, slot)
        Threads.@threads for m in 1:N3
            @inbounds begin
                s = ρ[m] * (Float32(gasvel[m, d]) * pars.vconv)
                tau[m] += 0.5f0 * s * s / ρ[m]
                buf[m] = Float16(s * msc)
            end
        end
        _copy_fvgk_slot16!(dst, buf)
    end

    Threads.@threads for m in 1:N3
        @inbounds buf[m] = Float16(tau[m] * gesc)
    end
    _copy_fvgk_slot16!(p.Tau, buf)
    tau = nothing
    GC.gc()

    Threads.@threads for m in 1:N3
        @inbounds buf[m] = Float16(ρ[m] * pars.xHII0)
    end
    _copy_fvgk_slot16!(p.species[1], buf)
    if length(p.species) >= 2
        Threads.@threads for m in 1:N3
            @inbounds buf[m] = Float16(ρ[m] * 1f-6)
        end
        _copy_fvgk_slot16!(p.species[2], buf)
    end
    if length(p.species) >= 3
        hdfac = Float32(6.8e-5) * pars.xHII0
        Threads.@threads for m in 1:N3
            @inbounds buf[m] = Float16(ρ[m] * hdfac)
        end
        _copy_fvgk_slot16!(p.species[3], buf)
    end
    ρ = nothing; buf = nothing; GC.gc()
    return pg
end

function load_cicass_gas_fvgk_direct!(pg, snap, c::Cosmo, a_i, u_i)
    _load_fvgk_direct_from_fields!(pg, snap.gas_delta, snap.gas_vel, c, a_i, u_i)
    @printf("  FVGK direct IC: loaded in-memory CICASS gas into resident grid\n")
    flush(stdout)
    return pg
end

function load_cicass_gas_stream_fvgk_direct!(pg, path::AbstractString, h, c::Cosmo, a_i, u_i)
    pg.dedup || error("direct FVGK gas IC requires pg.dedup=true")
    prod(pg.np) == 1 || error("direct FVGK gas IC requires np=1")
    p = pg.patches[1]
    N3 = h.n^3
    pars = _fvgk_direct_gas_params(c, a_i, u_i)
    raw = Vector{h.field_type}(undef, N3)
    ρ = Vector{Float32}(undef, N3)
    tau = Vector{Float32}(undef, N3)
    buf = Vector{Float16}(undef, N3)
    fb = Float32(c.fb); db = Float32(pg.dens_base); ds = Float32(pg.dens_scale)
    gesc = Float32(pg.gesc); msc = Float32(pg.msc)
    open(path, "r") do io
        read_cicass_field!(io, raw, h, 7)
        Threads.@threads for m in 1:N3
            @inbounds begin
                r = fb * (1f0 + Float32(raw[m]))
                ρ[m] = r
                tau[m] = r * pars.eint
                buf[m] = _fvgk_store_density16(r, db, ds)
            end
        end
        _copy_fvgk_slot16!(p.D, buf)

        Threads.@threads for m in 1:N3
            @inbounds buf[m] = Float16(ρ[m] * pars.eint * gesc)
        end
        _copy_fvgk_slot16!(p.Ge, buf)

        for (field, slot) in zip(8:10, (:S1, :S2, :S3))
            read_cicass_field!(io, raw, h, field)
            dst = getfield(p, slot)
            Threads.@threads for m in 1:N3
                @inbounds begin
                    s = ρ[m] * (Float32(raw[m]) * pars.vconv)
                    tau[m] += 0.5f0 * s * s / ρ[m]
                    buf[m] = Float16(s * msc)
                end
            end
            _copy_fvgk_slot16!(dst, buf)
        end
    end
    raw = nothing
    GC.gc()

    Threads.@threads for m in 1:N3
        @inbounds buf[m] = Float16(tau[m] * gesc)
    end
    _copy_fvgk_slot16!(p.Tau, buf)
    tau = nothing
    GC.gc()

    Threads.@threads for m in 1:N3
        @inbounds buf[m] = Float16(ρ[m] * pars.xHII0)
    end
    _copy_fvgk_slot16!(p.species[1], buf)
    if length(p.species) >= 2
        Threads.@threads for m in 1:N3
            @inbounds buf[m] = Float16(ρ[m] * 1f-6)
        end
        _copy_fvgk_slot16!(p.species[2], buf)
    end
    if length(p.species) >= 3
        hdfac = Float32(6.8e-5) * pars.xHII0
        Threads.@threads for m in 1:N3
            @inbounds buf[m] = Float16(ρ[m] * hdfac)
        end
        _copy_fvgk_slot16!(p.species[3], buf)
    end
    ρ = nothing; buf = nothing; GC.gc()
    @printf("  FVGK direct IC: streamed CICASS gas into resident grid\n")
    flush(stdout)
    return pg
end

function dm_ic_stream(path::AbstractString, h, c::Cosmo, u_i, backend)
    N3 = h.n^3; T = BE === :cpu ? Float64 : Float32
    VT = VEL16 ? Float16 : T
    raw = Vector{h.field_type}(undef, N3)
    buf = h.field_type === T ? raw : Vector{T}(undef, N3)
    vbuf = VT === h.field_type ? raw : (VT === T ? buf : Vector{VT}(undef, N3))
    vconv = 1.0e5 / u_i.v
    todev_pos() = (d = PPMKernels.to_device(backend, buf, T); BE === :metal && Metal.synchronize(); d)
    todev_vel() = (d = PPMKernels.to_device(backend, vbuf, VT); BE === :metal && Metal.synchronize(); d)
    open(path, "r") do io
        read_cicass_field!(io, raw, h, 1)
        Threads.@threads for i in eachindex(buf); @inbounds buf[i] = T(mod(raw[i], 1.0)); end
        px = todev_pos()
        read_cicass_field!(io, raw, h, 2)
        Threads.@threads for i in eachindex(buf); @inbounds buf[i] = T(mod(raw[i], 1.0)); end
        py = todev_pos()
        read_cicass_field!(io, raw, h, 3)
        Threads.@threads for i in eachindex(buf); @inbounds buf[i] = T(mod(raw[i], 1.0)); end
        pz = todev_pos()
        read_cicass_field!(io, raw, h, 4)
        Threads.@threads for i in eachindex(vbuf); @inbounds vbuf[i] = VT(raw[i] * vconv); end
        vx = todev_vel()
        read_cicass_field!(io, raw, h, 5)
        Threads.@threads for i in eachindex(vbuf); @inbounds vbuf[i] = VT(raw[i] * vconv); end
        vy = todev_vel()
        read_cicass_field!(io, raw, h, 6)
        Threads.@threads for i in eachindex(vbuf); @inbounds vbuf[i] = VT(raw[i] * vconv); end
        vz = todev_vel()
        mass = T(1 - c.fb)
        @printf("DM IC: %d particles, mass_per=%.4f (1−f_b), v→code=%.4e%s\n",
                N3, 1-c.fb, vconv, PIDS ? "  (+Lagrangian id)" : ""); flush(stdout)
        parts = (px=px, py=py, pz=pz, vx=vx, vy=vy, vz=vz, mass=mass)
        PIDS && (parts = (; parts..., id=PPMKernels.to_device(backend, collect(Int32, 0:N3-1))))
        return parts
    end
end

# ── per-patch signal speed (code units): max(|vx|+|vy|+|vz|+3·cs) over all patches ──
KA.@kernel function _max_signal_partials_k!(out, D, S1, S2, S3, Ge,
                                            nx::Int, ny::Int, ng::Int, ni::Int, nj::Int, nk::Int,
                                            gm1, three, gesc, imsc, dbase, invdscale)
    lane = KA.@index(Local, Linear)
    grp = KA.@index(Group, Linear)
    lanes = KA.@uniform prod(KA.@groupsize())
    total = ni * nj * nk
    cell = (grp - 1) * lanes + lane
    buf = KA.@localmem eltype(out) (MAX_SIGNAL_GROUP,)
    v = zero(eltype(out))
    @inbounds if cell <= total
        q = cell - 1
        ii = q % ni + ng + 1
        jj = (q ÷ ni) % nj + ng + 1
        kk = q ÷ (ni * nj) + ng + 1
        idx = ii + nx * (jj - 1) + nx * ny * (kk - 1)
        d = MultiCode._decode_density(D[idx], eltype(out)(dbase), eltype(out)(invdscale))
        if d > zero(d)
            invd = one(d) / d
            ge = eltype(out)(Ge[idx])
            cs = sqrt(max(eltype(out)(gm1) * ge * invd / eltype(out)(gesc), zero(d)))
            v = eltype(out)(imsc) * (abs(eltype(out)(S1[idx]) * invd) +     # un-lift MOM_SCALE from |v|=|S|/ρ
                abs(eltype(out)(S2[idx]) * invd) +
                abs(eltype(out)(S3[idx]) * invd)) +
                eltype(out)(three) * cs
        end
    end
    @inbounds buf[lane] = v
    KA.@synchronize
    stride = lanes >>> 1
    while stride >= 1
        if lane <= stride
            @inbounds buf[lane] = max(buf[lane], buf[lane + stride])
        end
        KA.@synchronize
        stride >>>= 1
    end
    if lane == 1
        @inbounds out[grp] = buf[1]
    end
end

function _max_signal_work(pg)
    nblocks = cld(prod(pg.pdim), MAX_SIGNAL_GROUP)
    key = (typeof(pg.patches[1].D), pg.T, pg.nd, pg.pdim, MAX_SIGNAL_GROUP)
    return get!(_MAX_SIGNAL_CACHE, key) do
        scratch = PPMKernels.device_zeros(pg.backend, pg.T, (nblocks,))
        host = Vector{pg.T}(undef, nblocks)
        (; scratch, host)
    end
end

function max_signal(pg, work = nothing)
    smax = 0.0
    work === nothing && (work = _max_signal_work(pg))
    Tloc = eltype(work.host); gm1 = Tloc(GAMMA * (GAMMA - 1)); gs = Tloc(pg.gesc); ims = Tloc(1.0/pg.msc)
    db = Tloc(pg.dens_base); ids = Tloc(1.0/pg.dens_scale)
    nblocks = cld(prod(pg.pdim), MAX_SIGNAL_GROUP)
    nx, ny, _ = pg.nd
    ni, nj, nk = pg.pdim
    for p in pg.patches                              # CFL timestep depends only on the physical
        _max_signal_partials_k!(pg.backend, MAX_SIGNAL_GROUP)(work.scratch, p.D, p.S1, p.S2, p.S3, p.Ge,
            nx, ny, pg.ng, ni, nj, nk, gm1, Tloc(3), gs, ims, db, ids; ndrange=nblocks * MAX_SIGNAL_GROUP)
        PPMKernels.KA.synchronize(pg.backend)
        copyto!(work.host, 1, work.scratch, 1, nblocks)
        smax = max(smax, Float64(maximum(@view work.host[1:nblocks])))
    end
    return smax
end
max_pvel(parts) = max(Float64(maximum(abs, parts.vx)),   # maximum(abs, ·) fuses — no |v| N-array temp
                      Float64(maximum(abs, parts.vy)),
                      Float64(maximum(abs, parts.vz)))

# ── write a cellcmp dump (same layout as enzo_cellcmp / ramses cellcmp) ──
KA.@kernel function _cell_summary_packed_k!(out, D, Ge, HII,
                                            nx::Int, ny::Int, ng::Int,
                                            ni::Int, nj::Int, nk::Int,
                                            xh, base_mu, tempfac, gesc, dbase, invdscale)
    grp = KA.@index(Group, Linear)
    lane = KA.@index(Local, Linear)
    lanes = KA.@uniform prod(KA.@groupsize())
    total = ni * nj * nk
    cell = (grp - 1) * lanes + lane
    bD = KA.@localmem eltype(out) (MAX_SIGNAL_GROUP,)
    bD2 = KA.@localmem eltype(out) (MAX_SIGNAL_GROUP,)
    bX = KA.@localmem eltype(out) (MAX_SIGNAL_GROUP,)
    bT = KA.@localmem eltype(out) (MAX_SIGNAL_GROUP,)
    sd = zero(eltype(out)); sd2 = zero(eltype(out)); sx = zero(eltype(out)); st = zero(eltype(out))
    @inbounds if cell <= total
        q = cell - 1
        ii = q % ni + ng + 1
        jj = (q ÷ ni) % nj + ng + 1
        kk = q ÷ (ni * nj) + ng + 1
        idx = ii + nx * (jj - 1) + nx * ny * (kk - 1)
        d = MultiCode._decode_density(D[idx], eltype(out)(dbase), eltype(out)(invdscale))
        ge = eltype(out)(Ge[idx])
        if d > zero(d)
            xfrac = ChemistryKernels.decode_log2sp(eltype(out), HII[idx])
            xhii = xfrac / xh
            mu = one(d) / (base_mu + xfrac)
            sd = d
            sd2 = d * d
            sx = xhii
            st = (ge / d / gesc) * tempfac * mu
        end
    end
    @inbounds begin
        bD[lane] = sd; bD2[lane] = sd2; bX[lane] = sx; bT[lane] = st
    end
    KA.@synchronize
    stride = lanes >>> 1
    while stride >= 1
        if lane <= stride
            @inbounds begin
                bD[lane] += bD[lane + stride]
                bD2[lane] += bD2[lane + stride]
                bX[lane] += bX[lane + stride]
                bT[lane] += bT[lane + stride]
            end
        end
        KA.@synchronize
        stride >>>= 1
    end
    if lane == 1
        base = 4 * (grp - 1)
        @inbounds begin
            out[base + 1] = bD[1]
            out[base + 2] = bD2[1]
            out[base + 3] = bX[1]
            out[base + 4] = bT[1]
        end
    end
end

KA.@kernel function _cell_summary_float_k!(out, D, Ge, HII,
                                           nx::Int, ny::Int, ng::Int,
                                           ni::Int, nj::Int, nk::Int,
                                           xh, base_mu, tempfac, gesc, dbase, invdscale)
    grp = KA.@index(Group, Linear)
    lane = KA.@index(Local, Linear)
    lanes = KA.@uniform prod(KA.@groupsize())
    total = ni * nj * nk
    cell = (grp - 1) * lanes + lane
    bD = KA.@localmem eltype(out) (MAX_SIGNAL_GROUP,)
    bD2 = KA.@localmem eltype(out) (MAX_SIGNAL_GROUP,)
    bX = KA.@localmem eltype(out) (MAX_SIGNAL_GROUP,)
    bT = KA.@localmem eltype(out) (MAX_SIGNAL_GROUP,)
    sd = zero(eltype(out)); sd2 = zero(eltype(out)); sx = zero(eltype(out)); st = zero(eltype(out))
    @inbounds if cell <= total
        q = cell - 1
        ii = q % ni + ng + 1
        jj = (q ÷ ni) % nj + ng + 1
        kk = q ÷ (ni * nj) + ng + 1
        idx = ii + nx * (jj - 1) + nx * ny * (kk - 1)
        d = MultiCode._decode_density(D[idx], eltype(out)(dbase), eltype(out)(invdscale))
        ge = eltype(out)(Ge[idx])
        if d > zero(d)
            xfrac = eltype(out)(HII[idx]) / d
            xhii = xfrac / xh
            mu = one(d) / (base_mu + xfrac)
            sd = d
            sd2 = d * d
            sx = xhii
            st = (ge / d / gesc) * tempfac * mu
        end
    end
    @inbounds begin
        bD[lane] = sd; bD2[lane] = sd2; bX[lane] = sx; bT[lane] = st
    end
    KA.@synchronize
    stride = lanes >>> 1
    while stride >= 1
        if lane <= stride
            @inbounds begin
                bD[lane] += bD[lane + stride]
                bD2[lane] += bD2[lane + stride]
                bX[lane] += bX[lane + stride]
                bT[lane] += bT[lane + stride]
            end
        end
        KA.@synchronize
        stride >>>= 1
    end
    if lane == 1
        base = 4 * (grp - 1)
        @inbounds begin
            out[base + 1] = bD[1]
            out[base + 2] = bD2[1]
            out[base + 3] = bX[1]
            out[base + 4] = bT[1]
        end
    end
end

function _cell_summary_work(pg)
    nblocks = cld(prod(pg.pdim), MAX_SIGNAL_GROUP)
    key = (typeof(pg.patches[1].D), typeof(pg.patches[1].species[1]), pg.T, pg.nd, pg.pdim, MAX_SIGNAL_GROUP)
    return get!(_CELL_SUMMARY_CACHE, key) do
        scratch = PPMKernels.device_zeros(pg.backend, pg.T, (4 * nblocks,))
        host = Vector{pg.T}(undef, 4 * nblocks)
        (; scratch, host)
    end
end

function cell_summary(pg, c::Cosmo, u)
    N = prod(pg.ncell)
    sD = 0.0; sD2 = 0.0; sx = 0.0; sT = 0.0
    work = _cell_summary_work(pg)
    Tloc = eltype(work.host)
    xh = Tloc(XH)
    base_mu = Tloc(XH + (1 - XH) / 4)
    tempfac = Tloc((GAMMA - 1) * u.T2)
    gs = Tloc(pg.gesc)
    db = Tloc(pg.dens_base); ids = Tloc(1 / pg.dens_scale)
    nblocks = cld(prod(pg.pdim), MAX_SIGNAL_GROUP)
    nx, ny, _ = pg.nd
    ni, nj, nk = pg.pdim
    for p in pg.patches
        if eltype(p.species[1]) === UInt16
            _cell_summary_packed_k!(pg.backend, MAX_SIGNAL_GROUP)(work.scratch, p.D, p.Ge, p.species[1],
                nx, ny, pg.ng, ni, nj, nk, xh, base_mu, tempfac, gs, db, ids;
                ndrange=nblocks * MAX_SIGNAL_GROUP)
        else
            _cell_summary_float_k!(pg.backend, MAX_SIGNAL_GROUP)(work.scratch, p.D, p.Ge, p.species[1],
                nx, ny, pg.ng, ni, nj, nk, xh, base_mu, tempfac, gs, db, ids;
                ndrange=nblocks * MAX_SIGNAL_GROUP)
        end
        PPMKernels.KA.synchronize(pg.backend)
        copyto!(work.host, 1, work.scratch, 1, 4 * nblocks)
        @inbounds for b in 1:nblocks
            q = 4 * (b - 1)
            sD  += Float64(work.host[q + 1])
            sD2 += Float64(work.host[q + 2])
            sx  += Float64(work.host[q + 3])
            sT  += Float64(work.host[q + 4])
        end
    end
    ρmean = sD / N
    var = max(sD2 / N - ρmean^2, 0.0)
    return (ρmean=ρmean, δrms=sqrt(var) / ρmean, xHII=sx/N, T=sT/N)
end

function write_cellcmp(pg, c::Cosmo, u, a, z)
    CELLDUMP || return cell_summary(pg, c, u)
    g = gather_global(pg)
    ρb  = Float64.(vec(g.D))
    HII = Float64.(vec(g.species[1]))
    H2I = length(g.species) >= 2 ? Float64.(vec(g.species[2])) : zero(HII)
    HDI = length(g.species) >= 3 ? Float64.(vec(g.species[3])) : zero(HII)
    eint = Float64.(vec(g.Ge)) ./ ρb
    xHIIv = HII ./ ρb ./ XH; fH2v = H2I ./ ρb ./ XH; fHDv = HDI ./ ρb
    μv = 1.0 ./ ((XH + (1-XH)/4) .+ XH .* (xHIIv .- 0.5 .* fH2v))
    Tcell = eint .* ((GAMMA-1) .* μv .* u.T2)                  # K
    if !NODUMP
        mkpath(REPORTS)
        open(joinpath(REPORTS, "patch_cellcmp_$(TAG)_z$(round(Int,z)).bin"), "w") do io
            write(io, Int64(pg.ncell[1]))
            write(io, ρb); write(io, xHIIv); write(io, fH2v); write(io, fHDv); write(io, Tcell)
        end
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

function prebuild_fvgk_before_particles!(pg, a, u_i)
    SOLVER === :fvgk || return pg
    if pg.fvgk === nothing
        patch_step!(pg, 0.0; a_value=a, order=(1,2,3), accel=nothing, chem=false, solver=:fvgk,
                    du=u_i.d, lu=u_i.l, tu=u_i.t, do_hydro=true, do_chem=false, chemmode=CHEMMODE)
        BE === :cuda && CUDA.synchronize()
        BE === :metal && Metal.synchronize()
    end
    # CIC_FVGK_DEDUP=1 (np=1): drop the f32 patch gas copy before DM particles are uploaded.
    # This removes the transition peak patches + particles + g.R/O; after dedup, patches are
    # just views into the FVGK grid.
    if get(ENV, "CIC_FVGK_DEDUP", "0") == "1" && prod(pg.np) == 1 && !pg.dedup
        MultiCode._fvgk_dedup!(pg)
        BE === :cuda && CUDA.synchronize()
        BE === :metal && Metal.synchronize()
        dmsg = pg.dens_base > 0 ? @sprintf(", dbase=%.6g dscale=%.0f", pg.dens_base, pg.dens_scale) : ""
        @printf("  FVGK dedup ON: patch gas → g.R views, ng=0, gesc=%.0e%s, f32 patch copy freed\n", pg.gesc, dmsg)
        flush(stdout)
    end
    return pg
end

function configure_fvgk_density_storage!(pg, c::Cosmo)
    SOLVER === :fvgk || return pg
    (BE === :metal && get(ENV, "CIC_FVGK_F16", "0") == "1") || return pg
    pg.dens_base = parse(Float64, get(ENV, "CIC_FVGK_DENS_BASE", string(c.fb)))
    pg.dens_scale = parse(Float64, get(ENV, "CIC_FVGK_DENS_SCALE", "256"))
    pg.dens_scale > 0 || error("CIC_FVGK_DENS_SCALE must be positive")
    return pg
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
        # DEFERRED particle upload (thunk) — run_evolution creates it AFTER the dedup pre-build so
        # the DM device arrays don't co-reside with the f32 patches + g.R/O (the transition peak).
        make_parts = () -> begin
            VT = VEL16 ? Float16 : T
            ptype(nm) = nm in ("vx", "vy", "vz") ? VT : T
            p = merge(NamedTuple{Symbol.(_CKP)}(Tuple(
                      PPMKernels.to_device(pg.backend, getfield(ck.parts, Symbol(nm)), ptype(nm)) for nm in _CKP)),
                      (mass = T(ck.pmass),))
            haskey(ck.parts, :id) && (p = (; p..., id=PPMKernels.to_device(pg.backend, ck.parts.id)))
            p
        end
        @printf("RESTART %s: %d³ at z=%.2f a=%.5f cyc=%d  (→ z=%.0f)\n",
                ENV["CIC_RESTART"], N, a_to_z(a_start), a_start, cyc_start, ZEND); flush(stdout)
        return run_evolution(c, N, ncell, np, a_start, a_end, u_i, dx, pg, make_parts, cyc_start)
    end
    snap_path = get(ENV, "CIC_SNAP", "")
    if !isempty(snap_path) && get(ENV, "CIC_STREAM_LOAD", "0") == "1"
        @printf("streaming CICASS snapshot: %s\n", snap_path); flush(stdout)
        h = cicass_header(snap_path)
        N = h.n
        ncell = (N, N, N); np = (NPX, NPX, NPX)
        Or = 4.15e-5 / h.hconst^2      # radiation (photons + 3 relativistic ν), matching transfer.x's
        # OmegaR — the CICASS ICs & linear theory include it, so the expansion MUST too or the DM
        # over-grows at high z (radiation is ~30% of matter at z=1000 → DM Δ² +33..43% too high).
        c = Cosmo(; Om=h.omega_m, OL=h.omega_l - Or, h0=h.hconst*100, box=h.box, Ob=h.omega_b, Or=Or)
        a_start = z_to_a(ZSTART); a_end = z_to_a(ZEND)
        u_i = cosmo_units(c, a_start)
        @printf("CICASS patch run: %d³ → %d patches of %d³, box=%.4f Mpc/h, Ωm=%.3f Ωb=%.4f ΩΛ=%.5f Ωr=%.3e h=%.3f\n",
                N, prod(np), N÷NPX, c.box, c.Om, c.fb*c.Om, c.OL, c.Or, c.h0/100)
        @printf("  z=%.0f→%.0f  a=%.3e→%.3e  scale_v(a_i)=%.4e cm/s  D(a_i)=%.4e\n",
                ZSTART, ZEND, a_start, a_end, u_i.v, growth_D(c, a_start)); flush(stdout)
        dx = 1.0 / N
        direct_ic = direct_fvgk_ic(np)
        pg = build_patchgrid(; ng=NG, ncell=ncell, np=np, dx=dx, gamma=GAMMA, nspecies=NSPEC,
                             besym=BE, T=T, du=u_i.d, lu=u_i.l, tu=u_i.t, deut=DEUT,
                             packed_species=PACKED, defer_fields=direct_ic)
        if direct_ic
            prepare_fvgk_direct_ic!(pg, c)
            load_cicass_gas_stream_fvgk_direct!(pg, snap_path, h, c, a_start, u_i)
        else
            scatter_cicass_gas_stream!(pg, snap_path, h, c, a_start, u_i)
        end
        make_parts = () -> dm_ic_stream(snap_path, h, c, u_i, pg.backend)
        return run_evolution(c, N, ncell, np, a_start, a_end, u_i, dx, pg, make_parts, 0)
    end
    snap = load_snapshot()
    snap_ref = Ref{Any}(snap)
    N = snap.n
    ncell = (N, N, N); np = (NPX, NPX, NPX)
    Or = 4.15e-5 / snap.hconst^2   # radiation (photons + 3 relativistic ν), matching transfer.x's OmegaR
    c = Cosmo(; Om=snap.omega_m, OL=snap.omega_l - Or, h0=snap.hconst*100, box=snap.box, Ob=snap.omega_b, Or=Or)
    a_start = z_to_a(ZSTART); a_end = z_to_a(ZEND)
    u_i = cosmo_units(c, a_start)
    @printf("CICASS patch run: %d³ → %d patches of %d³, box=%.4f Mpc/h, Ωm=%.3f Ωb=%.4f ΩΛ=%.5f Ωr=%.3e h=%.3f\n",
            N, prod(np), N÷NPX, c.box, c.Om, c.fb*c.Om, c.OL, c.Or, c.h0/100)
    @printf("  z=%.0f→%.0f  a=%.3e→%.3e  scale_v(a_i)=%.4e cm/s  D(a_i)=%.4e\n",
            ZSTART, ZEND, a_start, a_end, u_i.v, growth_D(c, a_start)); flush(stdout)

    # build the decomposition (dx=1/ncell: super-comoving box=1, a absorbed into units)
    dx = 1.0 / N
    direct_ic = direct_fvgk_ic(np)
    pg = build_patchgrid(; ng=NG, ncell=ncell, np=np, dx=dx, gamma=GAMMA, nspecies=NSPEC,
                         besym=BE, T=T, du=u_i.d, lu=u_i.l, tu=u_i.t, deut=DEUT,
                         packed_species=PACKED, defer_fields=direct_ic)
    if direct_ic
        prepare_fvgk_direct_ic!(pg, c)
        load_cicass_gas_fvgk_direct!(pg, snap_ref[], c, a_start, u_i)
    else
        gas = gas_ic(snap_ref[], c, a_start, u_i)
        scatter_global!(pg, gas)
        gas = nothing; GC.gc()
    end
    # DEFERRED particle upload (thunk): created AFTER the dedup pre-build inside run_evolution, so the
    # DM device arrays (~N³·18 B) don't co-reside with the f32 patches + g.R/O at the dedup transition.
    make_parts = () -> begin
        parts = dm_ic(snap_ref[], c, u_i, pg.backend)
        snap_ref[] = nothing
        GC.gc()
        parts
    end
    snap = nothing
    return run_evolution(c, N, ncell, np, a_start, a_end, u_i, dx, pg, make_parts, 0)
end

# Shared gravity setup + cosmological evolution loop (IC start with cyc_start=0, or
# resumed from a checkpoint at cyc_start).
function run_evolution(c, N, ncell, np, a_start, a_end, u_i, dx, pg, make_parts, cyc_start)
    parts = nothing   # DEFERRED: created by make_parts() AFTER the dedup pre-build below, so the DM
                      # device upload doesn't peak alongside the f32 patches + g.R/O (the transition,
                      # not the steady state, was the ceiling — see the 900³ probe).  Closures below
                      # capture this binding; it's live before any of them are actually called.
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
    # FFTW threads for the CPU gravity/KA-FFT path (unused when gravity=gpu → cuFFT).  A 512³+ CPU transform
    # wants many threads; the old min(8,·) cap badly under-parallelized it on this 64-core host.  Cap at 32
    # (FFTW's parallel efficiency for one 3D transform plateaus by ~16–32); override with CIC_FFT_THREADS.
    nthr = parse(Int, get(ENV, "CIC_FFT_THREADS", string(min(32, Sys.CPU_THREADS))))
    PoissonKernels.fft_set_num_threads!(nthr)
    # CIC_CHEM_TABLES=1 (default): log–log rate table for the chemistry hot path (~2.4× the
    # stiff network on GPU, <1e-5 vs the analytic fits); =0 falls back to the analytic fits.
    usetab  = get(ENV, "CIC_CHEM_TABLES", "1") == "1" && CHEMMODE !== :analytic   # analytic uses no tables
    tabbe   = chembk === :cpu ? :cpu : chembk
    ratetab = usetab ? ChemistryKernels.build_rate_tables(; precision=Float64, backend=tabbe) : nothing
    cooltab = usetab ? EmissionKernels.build_cooling_tables(; precision=Float64, backend=tabbe) : nothing
    GT = GRAV_HOST32 ? Float32 : Float64
    kafft = get(ENV, "CIC_GRAV_KAFFT", "0") == "1"
    kafft_c2c = get(ENV, "CIC_GRAV_KAFFT_C2C", "0") == "1"
    fftlabel = gravmode === :gpu ? (kafft ? (kafft_c2c ? "ka-c2c" : "ka-rfft") : (BE === :metal ? "mps-rfft" : "rfft")) : string(fftsolver)
    gravloc = gravmode === :gpu ? "device" : string(GT)
    @printf("  gravity = %s (FFT=%s, scatter=%s, host=%s)  chem = %s  (FFTW threads=%d, Julia threads=%d)\n",
            gravmode, fftlabel, accelmode, gravloc, chembk, nthr, Threads.nthreads()); flush(stdout)
    ρg = gravmode === :cpu ? zeros(GT, ncell) : nothing
    φg = gravmode === :cpu ? zeros(GT, ncell) : nothing
    # full-GPU gravity scratch in pg.T (Float32) — the mean is the known Ω-fixed constant
    # (subtracted in assemble_global_density_gpu!), so no f64 reduction / no f64 arrays needed.
    ρd = gravmode === :gpu ? PPMKernels.device_zeros(pg.backend, T, ncell) : nothing
    # GRAV1BUF: φ shares ρ's buffer (in-place rfft solve) — one nc³ field instead of two.
    φd = gravmode === :gpu ? (GRAV1BUF ? ρd : PPMKernels.device_zeros(pg.backend, T, ncell)) : nothing
    ρp_scratch = nothing; ρp_host = nothing; maxsig_work = nothing
    φkick_dev = nothing; φkick_host = nothing
    pscratch = nothing
    grav_t = Ref(0.0); fft_t = Ref(0.0); ngrav = Ref(0)
    asm_t = Ref(0.0); pacc_t = Ref(0.0); pfld_t = Ref(0.0)
    gd_asm_t = Ref(0.0); gd_gas_t = Ref(0.0); gd_dep_t = Ref(0.0); gd_mean_t = Ref(0.0)
    gd_fft_t = Ref(0.0); gd_pacc_t = Ref(0.0); gd_pfld_t = Ref(0.0); gd_n = Ref(0)
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
    snapshot!(dt_, a_) = assemble_global_density!(ρg, pg; particles=parts, dt=dt_, a=1.0,
        particle_density=ρp_scratch, particle_host=ρp_host)
    function solve_accel!(a_)                       # ρg already filled; returns accel tuple
        tf = time(); solve_global_poisson!(φg, ρg; G=1.5*c.Om*a_, a=1.0, boxsize=1.0, solver=fftsolver); fft_t[] += time()-tf
        tp = time()
        ga = if pg.dedup
            φkick_dev === nothing && error("dedup CPU gravity requires persistent global potential scratch")
            if φkick_host === nothing
                copyto!(φkick_dev, φg)
            else
                @. φkick_host = T(φg)
                copyto!(φkick_dev, φkick_host)
            end
            φkick_dev
        else
            accelmode === :gpu ? patch_accel_gpu(pg, φg; dx=dx) : patch_accel(pg, φg; dx=dx)
        end
        BE === :cuda && accelmode === :gpu && CUDA.synchronize()    # so the timer captures the device gather
        pacc_t[] += time()-tp
        tq = time()
        if pg.dedup
            pfld_t[] += time()-tq
            ngrav[] += 1
            return (gas=ga, phi=φkick_dev, le=0.0, cs=1.0/ncell[1], nc=ncell)
        end
        φpad, gle, gcs = particle_accel_field(pg, φg); pfld_t[] += time()-tq
        ngrav[] += 1
        return (gas=ga, phi=φpad, le=gle, cs=gcs, nc=nothing)
    end
    # full gravity for scale factor a_ (and particle half-drift dt_): CPU (host FFT) path
    # needs the host density snapshot first; GPU path assembles + solves entirely on device.
    function gravity!(a_, dt_)
        if gravmode === :gpu
            tgpu = time()
            g = global_gravity_gpu(pg; G=1.5*c.Om*a_, a=1.0, boxsize=1.0,
                                   particles=parts, dt=dt_, ρd=ρd, φd=φd,
                                   global_push=GLOBALPUSH)
            BE === :cuda && CUDA.synchronize()
            if GRAV_DETAIL && haskey(g, :timing)
                gt = g.timing
                gd_asm_t[] += gt.assemble; gd_gas_t[] += gt.gas; gd_dep_t[] += gt.deposit
                gd_mean_t[] += gt.mean; gd_fft_t[] += gt.fft; gd_pacc_t[] += gt.patch_accel
                gd_pfld_t[] += gt.particle_field; gd_n[] += 1
            end
            fft_t[] += time() - tgpu
            ngrav[] += 1
            return g
        else
            tg = time(); snapshot!(dt_, a_); asm_t[] += time()-tg
            return solve_accel!(a_)
        end
    end

    a = a_start
    configure_fvgk_density_storage!(pg, c)
    prebuild_fvgk_before_particles!(pg, a, u_i)
    BE === :cuda && (CUDA.synchronize(); GC.gc(true); CUDA.reclaim())
    BE === :metal && (Metal.synchronize(); GC.gc(true); Metal.synchronize())
    if SOLVER === :fvgk && gravmode === :gpu && ρd !== nothing
        fill!(ρd, zero(T))
        if kafft
            if kafft_c2c
                PoissonKernels.fft_poisson_root_gpu!(φd, ρd; G=1.0, a=1.0, boxsize=1.0)
            else
                PoissonKernels.fft_poisson_rfft_ka!(φd, ρd; G=1.0, a=1.0, boxsize=1.0)
            end
        else
            PoissonKernels.fft_poisson_rfft!(φd, ρd; G=1.0, a=1.0, boxsize=1.0)
        end
        BE === :cuda && CUDA.synchronize()
        BE === :metal && Metal.synchronize()
    end
    parts = make_parts()
    BE === :cuda && CUDA.synchronize()
    BE === :metal && Metal.synchronize()
    if gravmode === :cpu && parts !== nothing
        ρp_scratch = PPMKernels.device_zeros(pg.backend, T, (prod(ncell),))
        ρp_host = Vector{T}(undef, prod(ncell))
        maxsig_work = (; scratch=ρp_scratch, host=ρp_host)
    end
    dedup_cpu_phi = gravmode === :cpu && SOLVER === :fvgk &&
        get(ENV, "CIC_FVGK_DEDUP", "0") == "1" && prod(np) == 1
    φkick_dev = dedup_cpu_phi ? PPMKernels.device_zeros(pg.backend, T, ncell) : nothing
    φkick_host = dedup_cpu_phi && GT !== T ? Array{T}(undef, ncell) : nothing
    m0 = total_mass(pg)
    # lag-free overlap: seed the accel from the IC density (used by cycle-0 hydro)
    acc = nothing
    if OVERLAP
        sig0 = max_signal(pg, maxsig_work); dτ0 = min(COURANT*dx/max(sig0,1e-30), dtau_for_dlna(c, a, MAXEXP))
        acc = gravity!(a, dτ0)
    end
    @printf("%-5s %-9s %-9s %-9s %-9s %-7s\n", "cyc", "a", "z", "δb_rms", "ρmax", "sec")
    for cyc in cyc_start:MAXCYC-1
        t0 = time()
        z = a_to_z(a); u = cosmo_units(c, a)

        # ── Morton-resort the DM SoA every PSORT steps (keeps deposit/gather coalesced; bit-identical) ──
        if PSORT > 0 && cyc >= PSORT_START && (cyc - PSORT_START) % PSORT == 0
            tsort = time()
            morton_sort_particles!(parts; N=N)
            if SORT_TIMING
                @printf("  PSORT: cycle %d  %.3f s  mode=%s bucket=%s\n",
                        cyc, time() - tsort, get(ENV, "CIC_PSORT_MODE", "auto"),
                        get(ENV, "CIC_PSORT_BUCKET", "auto"))
                flush(stdout)
            end
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
                tpk = time()
                P  = patch_power_spectra(pg, parts; box=c.box, nmu=PKMU, nbins=PKNB, axis=PKAXIS,
                                         scale_v=uo.v/1e5, velocity=PKVEL)
                tpk = time() - tpk
                pf = @sprintf("%s_pkmu.h5", ckpref)
                h5open(pf, isfile(pf) ? "r+" : "w") do f
                    group_name = @sprintf("z%05.1f", zo)
                    haskey(f, group_name) && delete_object(f, group_name)
                    g = create_group(f, group_name)
                    g["k"] = collect(P.k); g["gas_delta"] = P.gas_delta; g["dm_delta"] = P.dm_delta
                    P.gas_vel === nothing || (g["gas_vel"] = P.gas_vel)
                    P.dm_vel  === nothing || (g["dm_vel"]  = P.dm_vel)
                    g["Nmodes"] = P.Nmodes
                    A = attrs(g); A["z"]=zo; A["a"]=a; A["box"]=c.box; A["axis"]=PKAXIS; A["nmu"]=PKMU
                end
                @printf("    ↳ P(k,μ) on device → %s [z%05.1f] %.2f s\n", pf, zo, tpk); flush(stdout)
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
        sig = max_signal(pg, maxsig_work); vp = max_pvel(parts)
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
            patch_step!(pg, dτ; a_value=a, order=order, accel=acc.gas, chem=true, solver=SOLVER, sigspeed=sig,
                        du=u.d, lu=u.l, tu=u.t, do_hydro=true, do_chem=false,
                        chemmode=CHEMMODE, chemnsub=CHEMNSUB, cosmo_h0=c.h0, cosmo_Om=c.Om, cosmo_OL=c.OL)
            pscratch = push_particles!(parts, acc.phi, acc.le, acc.cs, dτ;
                                       scratch=pscratch, nc=get(acc, :nc, nothing))
            BE === :cuda && CUDA.synchronize()            # hydro+push done ⇒ ρ_next density final
            if gravmode === :gpu
                gpu = Threads.@spawn begin                # NEXT-step gravity, fully on GPU
                    tgpu = time()
                    g = global_gravity_gpu(pg; G=1.5*c.Om*a_new, a=1.0, boxsize=1.0,
                                           particles=parts, dt=dτ, ρd=ρd, φd=φd,
                                           global_push=GLOBALPUSH)
                    BE === :cuda && CUDA.synchronize()
                    fft_t[] += time() - tgpu
                    g
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
            # per-phase split (GPU-synced): gravity | hydro | chem | particles
            sync() = BE === :cuda ? CUDA.synchronize() : (BE === :metal ? Metal.synchronize() : nothing)
            sync(); tt = time(); acc = gravity!(a, dτ); sync(); gph_t[] += time()-tt
            tt = time()
            patch_step!(pg, dτ; a_value=a, order=order, accel=acc.gas, chem=true, solver=SOLVER, sigspeed=sig,
                        du=u.d, lu=u.l, tu=u.t, do_hydro=true, do_chem=false,
                        chemmode=CHEMMODE, chemnsub=CHEMNSUB, cosmo_h0=c.h0, cosmo_Om=c.Om, cosmo_OL=c.OL)
            sync(); hyd_t[] += time()-tt; tt = time()
            patch_step!(pg, dτ; a_value=a, order=order, chem=true, solver=SOLVER,
                        du=u.d, lu=u.l, tu=u.t, do_hydro=false, do_chem=true, chem_backend=chembk,
                        rate_tables=ratetab, cool_tables=cooltab,
                        chemmode=CHEMMODE, chemnsub=CHEMNSUB, cosmo_h0=c.h0, cosmo_Om=c.Om, cosmo_OL=c.OL)
            sync(); chm_t[] += time()-tt; tt = time()
            pscratch = push_particles!(parts, acc.phi, acc.le, acc.cs, dτ;
                                       scratch=pscratch, nc=get(acc, :nc, nothing))
            sync(); prt_t[] += time()-tt; nph[] += 1
        else
            acc = gravity!(a, dτ)
            patch_step!(pg, dτ; a_value=a, order=order, accel=acc.gas, chem=true, solver=SOLVER, sigspeed=sig,
                        du=u.d, lu=u.l, tu=u.t, chem_backend=chembk, rate_tables=ratetab, cool_tables=cooltab,
                        chemmode=CHEMMODE, chemnsub=CHEMNSUB, cosmo_h0=c.h0, cosmo_Om=c.Om, cosmo_OL=c.OL)
            pscratch = push_particles!(parts, acc.phi, acc.le, acc.cs, dτ;
                                       scratch=pscratch, nc=get(acc, :nc, nothing))
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
        if PRINT_EVERY > 0 && cyc % PRINT_EVERY == 0
            ρmax = maximum((m = Float64(maximum(p.D)); pg.dens_base > 0 ? pg.dens_base * (1 + m / pg.dens_scale) : m) for p in pg.patches)
            δrms = density_contrast_rms(pg)            # on-device (no full-grid gather)
            @printf("%-5d %-9.5f %-9.3f %-9.3e %-9.3f %-7.2f\n", cyc, a, a_to_z(a), δrms, ρmax, sec)
            flush(stdout)
        end
        memprobe_now = get(ENV, "CIC_MEMPROBE", "") == "1" &&
            (MEMPROBE_CONT ? (MEMPROBE_CYC > 0 && cyc > 0 && cyc % MEMPROBE_CYC == 0) : cyc == MEMPROBE_CYC)
        if memprobe_now
            if BE === :cuda
                @eval import CUDA; CUDA.reclaim(); GC.gc(); CUDA.reclaim()
                used = (CUDA.total_memory() - CUDA.available_memory()) / 2^30
                @printf("  MEMPROBE: live CUDA = %.2f GiB at %d³ (%d Mcell)\n", used, N, N^3 ÷ 10^6); flush(stdout)
                MEMPROBE_CONT || break
            elseif BE === :metal
                @eval import Metal; Metal.synchronize(); GC.gc(); Metal.synchronize()
                s = Metal.alloc_stats
                used = max(0, s.alloc_bytes - s.free_bytes) / 2^30
                @printf("  MEMPROBE: live Metal = %.2f GiB at %d³ (%d Mcell)  [alloc %.2f GiB | freed %.2f GiB]\n",
                        used, N, N^3 ÷ 10^6, s.alloc_bytes/2^30, s.free_bytes/2^30); flush(stdout)
                MEMPROBE_CONT || break
            end
        end
    end

    if PHASE && nph[] > 0
        let n = nph[], tot = (gph_t[]+hyd_t[]+chm_t[]+prt_t[])/nph[]
            @printf("\nPER-PHASE (GPU-synced, %d cyc, %s/%s): gravity %.4f | hydro %.4f | chem %.4f | particles %.4f  → %.4f s/cyc\n",
                    n, SOLVER, CHEMMODE, gph_t[]/n, hyd_t[]/n, chm_t[]/n, prt_t[]/n, tot)
            @printf("  shares: gravity %.0f%% | hydro %.0f%% | chem %.0f%% | particles %.0f%%\n",
                    100gph_t[]/n/tot, 100hyd_t[]/n/tot, 100chm_t[]/n/tot, 100prt_t[]/n/tot)
        end
    end
    @printf("\nmass drift over run: %.3e\n", abs(total_mass(pg)-m0)/m0)
    let n=max(ngrav[],1)
        if gravmode === :gpu
            @printf("top-grid gravity (%s): %.4f s/solve  [device assemble+rFFT+field] over %d solves\n",
                    fftlabel, fft_t[]/n, ngrav[])
            if GRAV_DETAIL && gd_n[] > 0
                m = gd_n[]
                @printf("  GPU gravity detail: assemble %.4f [gas %.4f | deposit %.4f | mean %.4f] | FFT %.4f | patch_accel %.4f | particle_field %.4f over %d solves\n",
                        gd_asm_t[]/m, gd_gas_t[]/m, gd_dep_t[]/m, gd_mean_t[]/m,
                        gd_fft_t[]/m, gd_pacc_t[]/m, gd_pfld_t[]/m, m)
            end
        else
            gavg = (asm_t[] + fft_t[] + pacc_t[] + pfld_t[]) / n
            @printf("top-grid gravity (%s): %.4f s/solve  [assemble %.4f | FFT %.4f | patch_accel %.4f | part_field %.4f] over %d solves\n",
                    fftlabel, gavg, asm_t[]/n, fft_t[]/n, pacc_t[]/n, pfld_t[]/n, ngrav[])
        end
    end
    @printf("\n%-8s %-10s %-12s %-12s %-12s\n", "z", "δb_rms", "D²/D₀²", "<x_HII>", "<T>[K]")
    for r in pk_log
        @printf("%-8.2f %-10.3e %-12.3e %-12.3e %-12.1f\n", r.z, r.δrms, r.g2, r.xHII, r.T)
    end
    flush(stdout)
end

main()
