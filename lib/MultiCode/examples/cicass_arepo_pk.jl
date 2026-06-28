# Arepo (moving-mesh) on the SAME CICASS streaming ICs as the Enzo/RAMSES runs:
# write a double-precision Gadget-1 IC from the CICASS snapshot (gas cells on the
# 128³ grid mass-weighted by 1+δ_b, DM particles at the CICASS displaced
# positions), boot the cosmological libarepo flavor (TreePM self-gravity +
# comoving), evolve z=1000→z_end with the real Arepo step loop, run the shared
# Grackle reduced H+D chemistry on the live Voronoi cells, and measure the baryon
# (gas) and DM (particle CIC) power spectra at the same output redshifts.
#
# Run:  AREPO_LIB=$HOME/Projects/arepo/libarepo3d_cosmo.dylib \
#   DYLD_LIBRARY_PATH=/opt/homebrew/lib:$HOME/grackle_install_f32/lib BACKEND=metal \
#   <julia> --project=lib/MultiCode/test lib/MultiCode/examples/cicass_arepo_pk.jl

import ArepoLib, CICASSLib, MultiCode, PoissonKernels
using CICASSLib: CICASSSpec
using Printf, Statistics
include(joinpath(@__DIR__, "..", "deps", "gadget_ic.jl"))   # GadgetIC.write_ic
using .GadgetIC
if Symbol(get(ENV, "BACKEND", "metal")) === :metal
    using Metal
end

const BOX     = parse(Float64, get(ENV, "CIC_BOX",    "0.128"))   # Mpc/h
const ZSTART  = parse(Float64, get(ENV, "CIC_ZSTART", "1000.0"))
const ZEND    = parse(Float64, get(ENV, "CIC_ZEND",   "20.0"))
const NGRID   = parse(Int,     get(ENV, "CIC_NGRID",  "128"))
const NOUT    = parse(Int,     get(ENV, "CIC_NOUT",   "7"))
const OMEGA_M = parse(Float64, get(ENV, "CIC_OMEGAM", "0.27"))
const VBC     = parse(Float64, get(ENV, "CIC_VBC",    "30.0"))
const CHEM    = get(ENV, "CIC_CHEM", "1") == "1"
const CHEM_EVERY = parse(Int, get(ENV, "CIC_CHEM_EVERY", "1"))  # every hydro step
const CHEM_ZMAX  = parse(Float64, get(ENV, "CIC_CHEM_ZMAX", "100.0"))  # CMB-lock above this z
# chemistry engine: :grackle = C reduced lib (subprocess worker); :kernels = the
# native ChemistryKernels Julia port (in-process, GPU-capable — no worker needed).
const CHEM_ENG  = Symbol(get(ENV, "CIC_CHEM_ENGINE", "grackle"))
const CHEM_BK   = Symbol(get(ENV, "CIC_CHEM_BACKEND", "metal"))
const CHEM_PREC = CHEM_BK === :metal ? Float32 : Float64
const CHEM_INPROC = CHEM_ENG === :kernels
const BE      = Symbol(get(ENV, "BACKEND", "metal"))
const T       = Float32
const TAG     = get(ENV, "CIC_TAG", "")
const DO_DRAG = get(ENV, "CIC_COMPTON_DRAG", "0") == "1"   # Compton momentum drag on the gas
# Hard memory ceiling (MB): if process RSS exceeds this, abort the step loop
# gracefully (write what we have, finalize Arepo) rather than letting a leak run
# the machine out of RAM.  Default 50 GB — well above the ~3.5 GB a healthy run
# holds, low enough to catch a runaway long before it hurts the system.
const RSS_CEIL_MB = parse(Float64, get(ENV, "CIC_RSS_CEIL_MB", "50000"))
const REPORTS = MultiCode.run_dir("arepo")   # scratch run dir (never the repo); see lib/MultiCode/src/runout.jl
const GRACKLE_DATA = get(ENV, "GRACKLE_DATA_FILE",
    joinpath(homedir(), "Research", "codes", "grackle", "input", "CloudyData_noUVB.h5"))

# Arepo/Gadget internal units (match the cosmo_box example): kpc, 1e10 Msun, km/s.
const ULEN = 3.085678e21      # cm  (1 kpc)
const UMASS = 1.989e43        # g   (1e10 Msun)
const UVEL = 1e5              # cm/s (1 km/s)
const UDENS = UMASS/ULEN^3    # g/cm^3 (a=1, h=1 code density)
const UTIME = ULEN/UVEL       # s
const GAMMA = 5/3
const MH = 1.6726e-24; const KB = 1.380649e-16; const MU = 1.22

dev(be, a) = PoissonKernels.to_device(be, a, T)
function pk_of(grid::Array{Float64,3})
    δ = copy(grid); m = sum(δ)/length(δ); δ ./= m; δ .-= 1.0
    r = PoissonKernels.power_spectrum_gpu(dev(PoissonKernels.backend(BE), δ); boxsize=BOX, nbins=size(δ,1)÷2)
    return (k = collect(r.k), P = collect(r.P))
end
# CIC deposit (positions in box-fraction) → N³ density grid
function cic_density(xp::AbstractMatrix, w::AbstractVector, N::Integer)
    ρ = zeros(N, N, N)
    @inbounds for p in 1:size(xp,1)
        gx=mod(xp[p,1],1.0)*N; gy=mod(xp[p,2],1.0)*N; gz=mod(xp[p,3],1.0)*N
        i=floor(Int,gx);fx=gx-i;j=floor(Int,gy);fy=gy-j;k=floor(Int,gz);fz=gz-k
        i0=mod(i,N)+1;i1=mod(i+1,N)+1;j0=mod(j,N)+1;j1=mod(j+1,N)+1;k0=mod(k,N)+1;k1=mod(k+1,N)+1
        m=w[p]
        ρ[i0,j0,k0]+=m*(1-fx)*(1-fy)*(1-fz);ρ[i1,j0,k0]+=m*fx*(1-fy)*(1-fz)
        ρ[i0,j1,k0]+=m*(1-fx)*fy*(1-fz);ρ[i1,j1,k0]+=m*fx*fy*(1-fz)
        ρ[i0,j0,k1]+=m*(1-fx)*(1-fy)*fz;ρ[i1,j0,k1]+=m*fx*(1-fy)*fz
        ρ[i0,j1,k1]+=m*(1-fx)*fy*fz;ρ[i1,j1,k1]+=m*fx*fy*fz
    end
    ρ
end

# ── Memory guard ─────────────────────────────────────────────────────────────
# Each chem step (and every pk_of) allocates fresh Metal device arrays.  Their
# Julia-side handles are tiny, so Julia's GC heuristic almost never fires on its
# own — but the backing MTLBuffers are unified GPU/wired memory and are only
# freed by the MtlArray finalizers, which run on GC.  Without a forced GC the
# buffers accumulate every step until macOS jetsam SIGTERMs the process
# (signal 15 — exactly what killed the prior _c64 run at step 100).  So we drive
# GC explicitly: a cheap incremental collect every chem step (frees the step's
# device temporaries) and a full collect + memory readout periodically.
const _IS_METAL = BE === :metal
function gpu_gc!(full::Bool=false)
    _IS_METAL && Metal.synchronize()         # let in-flight kernels finish before we free their buffers
    GC.gc(full)
    return nothing
end
# current process resident set size in MB (the number jetsam watches), -1 on failure
proc_rss_mb() = try
    parse(Float64, strip(read(`ps -o rss= -p $(getpid())`, String))) / 1024
catch; -1.0 end

# ── Compton momentum drag on the gas (mirrors the Enzo/RAMSES operators) ─────
# Γ_drag/H from the CICASS recombination history (x_e); Ω_b-independent.  Damps
# the gas PECULIAR velocity toward the CMB rest frame by f = exp(-(Γ/H)·Δln a).
function compton_drag_over_H(z; hubble=0.71)
    xe = try CICASSLib.thermal_state(z).x_e catch; 0.0 end
    σT=6.6524e-25; cc=2.998e10; mH=1.673e-24; a_rad=7.5657e-15; XH=0.76; Tcmb0=2.726
    Or = 4.15e-5/hubble^2; H0 = 100.0*hubble*1e5/3.086e24            # 1/s
    Γ = (4.0/3.0)*a_rad*(Tcmb0*(1+z))^4 * xe * XH * σT / (cc*mH)     # 1/s
    H = H0*sqrt(OMEGA_M*(1+z)^3 + Or*(1+z)^4 + (1-OMEGA_M-Or))
    return Γ/H
end

# Apply the drag to the live Arepo Voronoi cells.  Arepo stores CONSERVED
# momentum (=Mass·v) and energy (=Mass·e_tot); Mass = ρ·volume.  We damp the
# peculiar velocity (v − v_bulk) toward zero, leaving the mass-weighted bulk
# (≡ the CMB/streaming frame) undamped — unit-safe (no km/s↔code conversion) and
# frame-agnostic (bulk≈0 boosted, ≈v_bc unboosted).  Internal energy (utherm) is
# unchanged; total energy absorbs the kinetic-energy change.
function arepo_compton_drag!(h, z_mid, dlna; hubble=0.71)
    f = exp(-compton_drag_over_H(z_mid; hubble=hubble) * dlna)
    f >= 0.999999 && return f                                        # negligible → skip I/O
    rho = ArepoLib.get_cell_field(h, :rho)
    vol = ArepoLib.get_cell_field(h, :volume)
    mom = ArepoLib.get_cell_field(h, :momentum)                      # n×3 (Mass·v)
    E   = ArepoLib.get_cell_field(h, :energy)                        # n   (Mass·e_tot)
    mass = max.(rho .* vol, eps())
    # mass-weighted bulk velocity — GLOBAL across MPI ranks (each rank holds only
    # its local cells, but the drag must damp toward the one global streaming frame)
    msums = Float64[sum(@view mom[:,1]), sum(@view mom[:,2]), sum(@view mom[:,3]), sum(mass)]
    ArepoLib.allreduce_sum!(msums)
    vbar = ntuple(d -> msums[d] / msums[4], 3)                       # mass-weighted bulk
    ke0  = 0.5 .* vec(sum(abs2, mom; dims=2)) ./ mass
    @inbounds for d in 1:3, i in eachindex(mass)
        v = mom[i, d] / mass[i]
        mom[i, d] = mass[i] * (vbar[d] + (v - vbar[d]) * f)          # damp peculiar part
    end
    ke1 = 0.5 .* vec(sum(abs2, mom; dims=2)) ./ mass
    E .+= ke1 .- ke0                                                 # keep e_int, swap KE
    ArepoLib.set_cell_field!(h, :momentum, mom)
    ArepoLib.set_cell_field!(h, :energy, E)
    return f
end

function build_gadget_ic(path, snap, a0, xHII0)
    N = snap.n; ng = N^3; ndm = size(snap.dm_pos, 1)
    boxk = snap.box * 1000.0                       # Mpc/h → kpc/h
    sqa = sqrt(a0)                                 # Gadget IC vel = v_pec / sqrt(a)
    ntot = ng + ndm
    pos = Array{Float64}(undef, ntot, 3); vel = similar(pos)
    mass = Array{Float64}(undef, ntot); ids = collect(1:ntot)
    u = Array{Float64}(undef, ng)
    # gas (type 0): mesh points on the uniform grid, mass-weighted by 1+δ_b
    # CIC_UNIFORM_BARYONS=1 → δ_b=0 (uniform gas density at IC), to isolate whether the
    # large-scale over-growth comes from the baryon IC structure vs Arepo's gravity/integration.
    δ = get(ENV, "CIC_UNIFORM_BARYONS", "0") == "1" ? zero(snap.gas_delta) : snap.gas_delta
    gv = snap.gas_vel; gt = snap.gas_temp
    # CIC_ZERO_BARYON_BULK=1 → BOOST to the gas (baryon/CMB) rest frame: subtract the coherent
    # gas bulk v_bc from BOTH species (Galilean; preserves the relative gas−DM streaming, which
    # carries the anisotropy + the DM growth). Arepo uses a TOTAL-energy scheme (no dual energy),
    # so with the gas streaming at v_bc~30 km/s, E_kin≈19×e_int and the per-step e_int=E_tot−E_kin
    # extraction bleeds ~hundreds of K per step → runaway over-cooling. Gas-at-rest keeps E_kin
    # small so e_int stays clean (DM, being N-body, carries the stream harmlessly).
    boost = get(ENV, "CIC_ZERO_BARYON_BULK", "0") == "1"
    gbulk = boost ? (sum(gv[:,1])/size(gv,1), sum(gv[:,2])/size(gv,1), sum(gv[:,3])/size(gv,1)) : (0.0,0.0,0.0)
    dxk = boxk / N; idx = 1
    @inbounds for k in 0:N-1, j in 0:N-1, i in 0:N-1
        c = i + N*(j + N*k) + 1                     # CICASS C-order (i fastest)
        pos[idx,1]=(i+0.5)*dxk; pos[idx,2]=(j+0.5)*dxk; pos[idx,3]=(k+0.5)*dxk
        for d in 1:3; vel[idx,d] = (gv[c,d]-gbulk[d])/sqa; end
        mass[idx] = snap.m_gas * (1.0 + δ[c])
        Tc = gt[c] > 0 ? gt[c] : snap.tavg
        u[idx] = (KB*Tc/((GAMMA-1)*MU*MH)) / UVEL^2 # specific internal energy (code)
        idx += 1
    end
    # DM (type 1): CICASS displaced positions (box-fraction → kpc/h), peculiar km/s
    @inbounds for p in 1:ndm
        for d in 1:3
            pos[ng+p,d] = mod(snap.dm_pos[p,d],1.0)*boxk
            vel[ng+p,d] = (snap.dm_vel[p,d]-gbulk[d])/sqa   # DM streams at −v_bc in the gas-rest frame
        end
        mass[ng+p] = snap.m_dm
    end
    # passive scalars in the IC: HII=x_e, H2I=1e-6, HDI=6.8e-5·x_e (mass fractions)
    pass = hcat(fill(xHII0, ng), fill(1e-6, ng), fill(6.8e-5*xHII0, ng))
    GadgetIC.write_ic(path; pos=pos, vel=vel, ids=ids, mass=mass, u=u, pass=pass, ngas=ng,
                      boxsize=boxk, a=a0, omega0=snap.omega_m, omegal=snap.omega_l,
                      hubble=snap.hconst)
    return (; ng, ndm, boxk)
end

function arepo_param(dir, snap, a0, aend)
    boxk = snap.box*1000.0
    # Gravitational softening = mean particle spacing / CIC_SOFT_DIV (default 5). Set
    # CIC_SOFT_DIV=1 to match the Eulerian codes' ~1-cell gravity resolution (the /5
    # default makes Arepo's softening 5× finer → excess small-scale clustering).
    softdiv = parse(Float64, get(ENV, "CIC_SOFT_DIV", "5"))
    soft = (boxk/snap.n)/softdiv                      # comoving kpc/h
    maxdt = parse(Float64, get(ENV, "CIC_MAXDT", "0.02"))     # MaxSizeTimestep (Δln a cap)
    erracc = parse(Float64, get(ENV, "CIC_ERRACC", "0.04"))   # ErrTolIntAccuracy (bigger → bigger dt)
    # Arepo's OutputListOn=1 makes it write a full HDF5 snapshot (4M double-precision
    # particles, SnapFormat=3) at every output redshift.  That write path allocates
    # ~12 GB per snapshot OUTSIDE Arepo's tracked arena and never frees it — with the
    # 11-output CIC_ZOUT list it runs the process out of RAM and macOS jetsam SIGTERMs
    # us (signal 15).  We don't need those snapshots: `record!` captures P(k) from the
    # LIVE Voronoi cells in-process at the same redshifts.  So default OutputListOn OFF
    # (no Arepo snapshot writes); record! still fires at every CIC_ZOUT a (≤2% overshoot,
    # bounded by MaxSizeTimestep).  Opt back in with CIC_AREPO_SNAPSHOTS=1 if you ever
    # actually want the .hdf5 dumps (and have the RAM headroom for the leak).
    olon = 0
    if haskey(ENV, "CIC_ZOUT") && get(ENV, "CIC_AREPO_SNAPSHOTS", "0") == "1"
        aouts = sort([1.0/(1.0+parse(Float64,s)) for s in split(ENV["CIC_ZOUT"], ",")])
        open(joinpath(dir, "ol.txt"), "w") do io
            for av in aouts; println(io, string(round(av; digits=10), " 1")); end
        end
        olon = 1
    end
    return """
InitCondFile         ./ics
OutputDir            ./output
SnapshotFileBase     snap
OutputListFilename   ./ol.txt
ICFormat             1
SnapFormat           3
TimeLimitCPU         200000
CpuTimeBetRestartFile 150000
ResubmitOn           0
ResubmitCommand      ./none
MaxMemSize           32000
TimeBegin            $(a0)
TimeMax              $(aend)
ComovingIntegrationOn 1
PeriodicBoundariesOn 1
CoolingOn            0
StarformationOn      0
Omega0               $(snap.omega_m)
OmegaLambda          $(snap.omega_l)
OmegaBaryon          $(snap.omega_b)
HubbleParam          $(snap.hconst)
BoxSize              $(boxk)
OutputListOn         $(olon)
TimeBetSnapshot      1.1
TimeOfFirstSnapshot  2.0
TimeBetStatistics    0.05
NumFilesPerSnapshot  1
NumFilesWrittenInParallel 1
TypeOfTimestepCriterion 0
ErrTolIntAccuracy    $(erracc)
CourantFac           0.3
MaxSizeTimestep      $(min(maxdt, 0.9*(aend-a0)))
MinSizeTimestep      1e-12
InitGasTemp          $(round(snap.tavg, digits=2))
MinGasTemp           2.7
MinimumDensityOnStartUp 1e-30
LimitUBelowThisDensity 0
LimitUBelowCertainDensityToThisValue 0
MinEgySpec           0
TypeOfOpeningCriterion 1
ErrTolTheta          0.7
ErrTolForceAcc       0.0025
MultipleDomains      4
TopNodeFactor        2.5
ActivePartFracForNewDomainDecomp 0.01
DesNumNgb            64
MaxNumNgbDeviation   4
UnitLength_in_cm     $(ULEN)
UnitMass_in_g        $(UMASS)
UnitVelocity_in_cm_per_s $(UVEL)
GravityConstantInternal 0
SofteningComovingType0 $(soft)
SofteningComovingType1 $(soft)
SofteningMaxPhysType0  $(soft)
SofteningMaxPhysType1  $(soft)
GasSoftFactor        2.5
SofteningTypeOfPartType0 0
SofteningTypeOfPartType1 1
SofteningTypeOfPartType2 1
SofteningTypeOfPartType3 1
SofteningTypeOfPartType4 1
SofteningTypeOfPartType5 1
MinimumComovingHydroSoftening $(soft/4)
AdaptiveHydroSofteningSpacing 1.2
$(get(ENV,"CIC_STATIC_MESH","0")=="1" ? "" : "CellShapingSpeed     0.5\nCellMaxAngleFactor   2.25")
"""
end

# Subsample a CICASS snapshot to a coarser Arepo load: block-average the gas grid
# by `sub` (C-order, i fastest), decimate DM particles by sub³ (a random 1/sub³
# subset of the glass is a valid lower-res particle load), masses ×sub³.  Returns
# a NamedTuple with the same fields build_gadget_ic reads.
function subsample(snap, sub::Int)
    sub == 1 && return snap
    N = snap.n; Na = N ÷ sub; s3 = sub^3
    δ = snap.gas_delta; gv = snap.gas_vel; gt = snap.gas_temp
    gδ = zeros(Na^3); ggv = zeros(Na^3, 3); ggt = zeros(Na^3)
    @inbounds for K in 0:Na-1, J in 0:Na-1, I in 0:Na-1
        C = I + Na*(J + Na*K) + 1; sδ=0.0; sv=zeros(3); st=0.0
        for dk in 0:sub-1, dj in 0:sub-1, di in 0:sub-1
            c = (I*sub+di) + N*((J*sub+dj) + N*(K*sub+dk)) + 1
            sδ += 1.0+δ[c]; sv .+= @view gv[c,:]; st += gt[c]
        end
        gδ[C] = sδ/s3 - 1.0; ggv[C,:] .= sv./s3; ggt[C] = st/s3
    end
    keep = 1:s3:size(snap.dm_pos,1)
    return (; n=Na, box=snap.box, omega_m=snap.omega_m, omega_b=snap.omega_b,
            omega_l=snap.omega_l, hconst=snap.hconst, tavg=snap.tavg,
            m_dm=snap.m_dm*s3, m_gas=snap.m_gas*s3,
            gas_delta=gδ, gas_vel=ggv, gas_temp=ggt,
            dm_pos=snap.dm_pos[keep,:], dm_vel=snap.dm_vel[keep,:])
end

# gas density on N³ grid from live Arepo gas cells (CIC of cell mass at cell pos)
function arepo_grids(h, N, boxk)
    # COLLECTIVE: gather the distributed particle state to rank 0 (every rank must
    # call). Under N MPI ranks the per-rank P[] holds only local gas+DM, and the
    # gas/DM split is per-rank, so we gather pos+mass+type globally and split by
    # TYPE (0=gas, 1=DM) — robust to the cross-rank concatenation order. Single
    # rank: the gather is a straight local copy.
    pos  = ArepoLib.gather_particle_field(h, :pos)    # global on rank0, empty elsewhere
    mass = ArepoLib.gather_particle_field(h, :mass)
    typ  = ArepoLib.gather_particle_field(h, :type)
    ArepoLib.this_task() == 0 || return nothing       # only rank 0 holds the data
    xb = pos ./ boxk                                  # → box-fraction
    isgas = typ .== 0; isdm = typ .== 1
    gas = cic_density(xb[isgas, :], Float64.(mass[isgas]), N)
    dm  = cic_density(xb[isdm,  :], Float64.(mass[isdm]),  N)
    return gas, dm
end

# Cell-by-cell ρ, x_HII, f_H2, f_HD, T on the N³ grid from the live Arepo Voronoi gas
# cells — the Arepo counterpart of the Enzo/RAMSES cellcmp dumps (plot_cicass_cellcmp.py).
# Distributed-safe with NO bridge change: each rank deposits its LOCAL gas cells (NGP by
# cell center; mass-weighted for the intensive x_HII/T/fractions) into N³ accumulators,
# then allreduce_sum! over ranks; rank 0 normalizes by the per-cell mass.  Indexing is
# x-fastest (g = ix + N·iy + N²·iz), matching the RAMSES/Enzo grids cell-for-cell (same
# physical grid → cell-by-cell aligned).  utherm is the specific internal energy
# (density-scale-invariant) → physical T directly; μ = grackle species μ (the SAME
# convention as the RAMSES/Enzo dumps so T is comparable across codes).
function arepo_cellcmp_grids(h, N, boxk)
    XH = 0.76
    cen = ArepoLib.get_cell_field(h, :center)        # n×3 cell center (kpc/h, code length)
    rho = ArepoLib.get_cell_field(h, :rho)
    vol = ArepoLib.get_cell_field(h, :volume)
    uth = ArepoLib.get_cell_field(h, :utherm)        # specific internal energy (UVEL² code)
    sc  = ArepoLib.get_cell_field(h, :scalars)       # n×(2|3): HII, H2I[, HDI] mass fractions
    nps = size(sc, 2); n = length(rho)
    mass = max.(rho .* vol, eps())
    xHII = Float64.(@view sc[:, 1]) ./ XH            # HII massfrac/X_H = ionization fraction
    fH2  = Float64.(@view sc[:, 2]) ./ XH
    fHD  = nps >= 3 ? Float64.(@view sc[:, 3]) : zeros(n)
    muv  = 1.0 ./ ((XH + (1 - XH) / 4) .+ XH .* (xHII .- 0.5 .* fH2))
    Tcell = uth .* UVEL^2 .* (GAMMA - 1) .* muv .* MH ./ KB        # K
    Wm = zeros(N^3); Wx = zeros(N^3); Wh2 = zeros(N^3); Whd = zeros(N^3); WT = zeros(N^3)
    @inbounds for i in 1:n
        ix = mod(floor(Int, mod(cen[i,1]/boxk, 1.0) * N), N)
        iy = mod(floor(Int, mod(cen[i,2]/boxk, 1.0) * N), N)
        iz = mod(floor(Int, mod(cen[i,3]/boxk, 1.0) * N), N)
        g = ix + N*(iy + N*iz) + 1; m = mass[i]
        Wm[g] += m; Wx[g] += m*xHII[i]; Wh2[g] += m*fH2[i]; Whd[g] += m*fHD[i]; WT[g] += m*Tcell[i]
    end
    for v in (Wm, Wx, Wh2, Whd, WT); ArepoLib.allreduce_sum!(v); end
    ArepoLib.this_task() == 0 || return nothing
    Wd = max.(Wm, eps())
    return (rho = Wm, xHII = Wx ./ Wd, fH2 = Wh2 ./ Wd, fHD = Whd ./ Wd, T = WT ./ Wd)
end

# ── Grackle chemistry SUBPROCESS worker (in-process Grackle co-resident with the
#    live Arepo segfaults inside solve_rate_cool, same as RAMSES — isolate it) ──
function spawn_grackle_worker(; hubble, Om, OL, a0, fh, du, lu, tu, deut, data_file)
    script = joinpath(@__DIR__, "..", "deps", "grackle_worker.jl")
    logf = joinpath(tempdir(), "grackle_worker_arepo.log")
    p = open(pipeline(`$(Base.julia_cmd()) --project=$(Base.active_project()) $script`;
                      stderr=logf), "r+")
    write(p, Float64(hubble), Float64(Om), Float64(OL), Float64(a0), Float64(fh),
          Float64(du), Float64(lu), Float64(tu))
    write(p, Int64(deut ? 1 : 0), Int64(ncodeunits(data_file))); write(p, data_file); flush(p)
    read(p, Int64) == 1 || error("grackle worker init failed (see $logf)")
    @printf("grackle worker ready (PID isolated from Arepo)\n"); flush(stdout)
    return p
end

# One reduced-chemistry step on the live Arepo gas via the worker.  Arepo :rho is
# COMOVING; Grackle (comoving_coordinates=0) wants PHYSICAL, so we send rho/a³
# (density_units fixed at the a=1, h-corrected conversion) and the species the same
# way, then convert back.  utherm (specific) is density-scale-invariant.  Subcycle
# the dt so a single giant call can't overrun Grackle's stiff high-z subcycle cap.
function arepo_worker_chem_step!(p, h; a_value, dt, deut=true, dtmax=0.02)
    # Arepo :rho is COMOVING; Grackle (comoving_coordinates=0) applies
    # co_density_units = density_units/a³ itself (calculate_temperature.c), so we
    # send the comoving rho directly with density_units = the a=1 conversion.
    rho = Float64.(ArepoLib.get_cell_field(h, :rho))
    eint  = Float64.(ArepoLib.get_cell_field(h, :utherm))
    sc    = ArepoLib.get_cell_field(h, :scalars)
    n = length(rho)
    HII = rho .* Float64.(@view sc[:,1]); H2I = rho .* Float64.(@view sc[:,2])
    HDI = deut ? rho .* Float64.(@view sc[:,3]) : nothing
    rfloor = maximum(rho)*1e-20 + eps()
    @inbounds for i in 1:n
        rho[i]  = (isfinite(rho[i]) && rho[i]>rfloor) ? rho[i] : rfloor
        eint[i] = (isfinite(eint[i]) && eint[i]>0)    ? eint[i] : eps()
        HII[i]  = clamp(isfinite(HII[i]) ? HII[i] : 0.0, 0.0, rho[i])
        H2I[i]  = clamp(isfinite(H2I[i]) ? H2I[i] : 0.0, 0.0, rho[i])
        deut && (HDI[i] = clamp(isfinite(HDI[i]) ? HDI[i] : 0.0, 0.0, rho[i]))
    end
    nsub = max(1, ceil(Int, dt/dtmax)); dts = dt/nsub
    for _ in 1:nsub
        write(p, Int64(1), Int64(n), Float64(a_value), Float64(dts))
        write(p, rho); write(p, eint); write(p, HII); write(p, H2I)
        deut && write(p, HDI); flush(p)
        read!(p, eint); read!(p, HII); read!(p, H2I); deut && read!(p, HDI)
    end
    ArepoLib.set_cell_field!(h, :utherm, eint)
    cols = deut ? hcat(HII./rho, H2I./rho, HDI./rho) : hcat(HII./rho, H2I./rho)
    ArepoLib.set_cell_field!(h, :scalars, cols)
    return (xHII1 = HII[1]/rho[1],)
end

function main()
    ArepoLib.available() || error("libarepo cosmo flavor not found (set AREPO_LIB)")
    # ── MPI bootstrap (distributed mpiexec -n N runs) ────────────────────────
    # The bridge inherits the mpiexec world; Arepo's hydro/gravity/mesh distribute
    # natively. Only the IC write, the P(k) readback and the .dat write are not
    # rank-parallel, so: rank 0 realizes the CICASS IC into a SHARED run dir and
    # broadcasts the cosmology scalars (sum-reduce of a rank-0/zeros buffer — no
    # MPI.jl needed); every rank boots from the shared IC; the readback gathers to
    # rank 0 (see arepo_grids). Non-root ranks stay silent.
    ArepoLib.mpi_init()
    rank = ArepoLib.this_task(); nrank = ArepoLib.n_tasks(); root = rank == 0
    root || redirect_stdout(devnull)
    @printf("Arepo on CICASS ICs: box=%.4f Mpc/h, %d³, z=%.0f→%.0f, chem=%s  [%d MPI rank(s)]\n",
            BOX, NGRID, ZSTART, ZEND, CHEM, nrank); flush(stdout)
    a0 = 1/(1+ZSTART); aend = 1/(1+ZEND)
    workdir = joinpath(REPORTS, "arepo_mpi_run$(TAG)")  # SHARED across ranks (one node); TAG-keyed so concurrent runs don't clobber
    scal = zeros(Float64, 9)   # hconst, omega_m, omega_l, tavg, box, n, m_dm, m_gas, x_e
    if root
        privdir = mktempdir()
        spec = CICASSSpec(boxlength=BOX, zstart=ZSTART, ngrid=NGRID, vbc=VBC,
                          Omega_m=OMEGA_M, filename="cic_arepo")
        res = CICASSLib.generate(spec; workdir=privdir)
        fs = CICASSLib.read_snapshot(res.output)
        @printf("CICASS realized: m_dm=%.4e m_gas=%.4e (1e10 Msun/h) tavg=%.1f K\n",
                fs.m_dm, fs.m_gas, fs.tavg); flush(stdout)
        fsth = try CICASSLib.thermal_state(ZSTART) catch; (x_e=0.047, T_gas=fs.tavg) end
        SUB = parse(Int, get(ENV, "CIC_SUB", "1"))      # subsample CICASS→coarser Arepo load
        fas = subsample(fs, SUB)
        SUB > 1 && @printf("subsampled CICASS %d³ → Arepo %d³ (sub=%d, m_dm=%.3e m_gas=%.3e)\n",
                           fs.n, fas.n, SUB, fas.m_dm, fas.m_gas)
        rm(workdir; force=true, recursive=true); mkpath(joinpath(workdir,"output"))
        info = build_gadget_ic(joinpath(workdir,"ics"), fas, a0, fsth.x_e)
        @printf("Gadget IC: %d gas + %d DM, box=%.2f kpc/h\n", info.ng, info.ndm, info.boxk); flush(stdout)
        write(joinpath(workdir,"param.txt"), arepo_param(workdir, fas, a0, aend))
        scal .= (fs.hconst, fs.omega_m, fs.omega_l, fs.tavg, fas.box, Float64(fas.n),
                 fas.m_dm, fas.m_gas, fsth.x_e)
    end
    ArepoLib.allreduce_sum!(scal)        # broadcast scalars (non-root contributed 0)
    ArepoLib.barrier()                   # shared IC/param now exist on disk
    snap = (; hconst=scal[1], omega_m=scal[2], omega_l=scal[3], tavg=scal[4],
              box=scal[5], n=Int(round(scal[6])), m_dm=scal[7], m_gas=scal[8])
    sth  = (; x_e=scal[9], T_gas=scal[4])
    Na = snap.n; boxk = snap.box * 1000.0

    datafile = joinpath(REPORTS, "cicass_arepo_pk$(TAG).dat")
    pk_results = NamedTuple[]
    write_tables() = root && open(datafile,"w") do io
        println(io, "# Arepo on CICASS ICs z=$ZSTART→$ZEND box=$BOX Mpc/h N=$NGRID")
        println(io, "# block: '@ z=<z> <baryon|dm>' then k[h/Mpc] P[(Mpc/h)^3]")
        for r in pk_results, tag in (:baryon,:dm)
            c = getfield(r,tag); println(io, "@ z=$(round(r.z,digits=3)) $tag")
            for i in eachindex(c.k); @printf(io,"%.6e %.6e\n", c.k[i], c.P[i]); end
        end
    end

    cd(workdir) do
        h = ArepoLib.init("param.txt")
        anow() = ArepoLib.sim_time(h); znow() = 1/anow()-1
        @printf("BOOTED: %d part (%d gas), a=%.5f z=%.1f\n",
                ArepoLib.num_part(h), ArepoLib.num_gas(h), anow(), znow()); flush(stdout)
        gw = nothing
        if CHEM
            @printf("scalars init in IC: x_HII=%.3e (HII,H2I,HDI)\n", sth.x_e)
            if CHEM_INPROC
                # density_units = UDENS·h²/a³ → PHYSICAL density from Arepo's comoving ρ
                # (solve_chem! applies no a³). Re-set per-z in the step loop; this is the IC value.
                MultiCode.chem_init!(; hubble=snap.hconst*100, Om=snap.omega_m,
                    OL=snap.omega_l, a_value=a0, fh=0.76, density_units=UDENS*snap.hconst^2/a0^3,
                    length_units=ULEN, time_units=UTIME, deuterium=true, engine=:kernels)
                @printf("chemistry engine: ChemistryKernels (:kernels, backend=%s) IN-PROCESS\n", CHEM_BK)
            else
                gw = spawn_grackle_worker(; hubble=snap.hconst*100, Om=snap.omega_m,
                    OL=snap.omega_l, a0=a0, fh=0.76, du=UDENS*snap.hconst^2,
                    lu=ULEN, tu=UTIME, deut=true, data_file=GRACKLE_DATA)
                @printf("chemistry engine: native Grackle reduced lib (:grackle, worker)\n")
            end
        end
        # CIC_ZOUT="z1,z2,…" → EXPLICIT output redshifts (consistent cross-code list).
        a_outs = haskey(ENV, "CIC_ZOUT") ?
            sort([1.0/(1.0+parse(Float64,s)) for s in split(ENV["CIC_ZOUT"], ",")]) :
            exp.(range(log(a0), log(aend), length=NOUT)); ai = 1
        function record!(z)
            res = arepo_grids(h, Na, boxk)     # COLLECTIVE gather; rank 0 gets grids, nothing else
            # COLLECTIVE (all ranks must call): cell-by-cell ρ/x_HII/f_H2/f_HD/T grids.
            cc = get(ENV, "CIC_CELLCMP", "0") == "1" ? arepo_cellcmp_grids(h, Na, boxk) : nothing
            root || return
            gas, dm = res
            push!(pk_results, (z=z, baryon=pk_of(gas), dm=pk_of(dm)))
            # full 3D baryon+DM density for the directional P(k,costh) anisotropy + cross-
            # spectra (same format as enzo_xspec/ramses_xspec: Int64 N, ρ_b[N³], ρ_d[N³],
            # Julia column-major x-fastest). Lets Arepo join plot_cicass_anisotropy/cellcmp.
            if get(ENV, "CIC_XSPEC", "0") == "1"
                MultiCode.write_grid(joinpath(REPORTS, "arepo_xspec$(TAG)_z$(round(Int,z)).bin");
                    kind="arepo_xspec", n=Na, ndim=3, columns=["rho_b"=>gas, "rho_dm"=>dm])
            end
            # cell-by-cell dump (ρ, x_HII, f_H2, f_HD, T) → plot_cicass_cellcmp.py
            if cc !== nothing
                MultiCode.write_grid(joinpath(REPORTS, "arepo_cellcmp$(TAG)_z$(round(Int,z)).bin");
                    kind="arepo_cellcmp", n=Na, ndim=3,
                    columns=["rho_b"=>cc.rho, "xHII"=>cc.xHII, "fH2"=>cc.fH2, "fHD"=>cc.fHD, "T"=>cc.T])
            end
            db = std(gas)/mean(gas); dd = std(dm)/mean(dm)
            @printf("  ● Arepo z=%.2f  δb_rms=%.3e δdm_rms=%.3e  [%d]  rss=%.0fMB\n",
                    z, db, dd, length(pk_results), proc_rss_mb())
            write_tables(); flush(stdout)
        end
        achem = a0
        chemt = 0.0; steptot = 0.0        # wall-time breakdown accumulators
        @printf("%-5s %-9s %-10s\n","step","z","sec"); flush(stdout)
        maxstep = parse(Int, get(ENV, "CIC_MAXSTEP", "200000"))   # cap steps for quick perf timing
        for step in 0:200000
            z = znow(); a = anow()
            step >= maxstep && (@printf("MAXSTEP %d reached at z=%.2f — stopping\n", maxstep, z); flush(stdout); break)
            # Hard safety net: never let a leak run the machine out of RAM.  Sys.maxrss
            # is a cheap ccall (no fork); for a monotonically-growing leak max≈current.
            rss_now = Sys.maxrss() / 2^20
            if rss_now > RSS_CEIL_MB
                @printf("ABORT: RSS %.0fMB exceeded ceiling %.0fMB at step %d z=%.2f — stopping cleanly\n",
                        rss_now, RSS_CEIL_MB, step, z); flush(stdout)
                break
            end
            while ai <= length(a_outs) && a >= a_outs[ai]-1e-12; record!(z); ai += 1; end
            z <= ZEND && break
            dbg = get(ENV,"CIC_RSS_DEBUG","0")=="1" && step < 6
            _r() = Sys.maxrss()/2^20
            dbg && @printf("    [rss step%d pre =%.0f]\n", step, _r())
            t0 = time(); st = ArepoLib.run_step!(h)
            dbg && @printf("    [rss step%d run=%.0f]\n", step, _r())
            if DO_DRAG                                              # Compton drag over the step just taken
                znew = znow(); dlna = log((1.0+z)/(1.0+znew))
                if dlna > 1e-12
                    fdr = arepo_compton_drag!(h, 0.5*(z+znew), dlna; hubble=snap.hconst)
                    (step % 20 == 0) && @printf("    [compton drag z=%.1f Γ/H=%.2f v×%.4f]\n",
                        0.5*(z+znew), compton_drag_over_H(0.5*(z+znew); hubble=snap.hconst), fdr)
                end
            end
            dbg && @printf("    [rss step%d drag=%.0f]\n", step, _r())
            if CHEM && step % CHEM_EVERY == 0 && step > 0
                anew = anow(); znew = znow()
                dphys = _dt_phys(achem, anew, snap.hconst, snap.omega_m, snap.omega_l)/UTIME
                achem = anew
                if CHEM_INPROC
                    # native ChemistryKernels engine: the implicit-Compton subcycle
                    # handles the stiff high-z CMB coupling, so run the FULL chemistry
                    # at ALL z every hydro step (no CMB-lock) — species react + advect
                    # consistently per step.
                    # PER-Z DENSITY UNITS (bug fix): solve_chem! uses ρ_cgs = ρ·density_units
                    # with NO a³ factor, but Arepo's ρ is COMOVING. The IC chem_init! set the
                    # a=1 units (UDENS·h²), so at high z the chem saw the density ~(1+z)³ too
                    # LOW → recombination far too slow → x_HII FROZEN at the IC value (0.0624 at
                    # all z, vs Enzo/RAMSES recombining to ~2e-4). Re-init each step with the
                    # physical density units at the current a: density_units = UDENS·h²/a³
                    # (exactly what RAMSES does via the per-z scale_d). length/time units are
                    # a-invariant here (utherm is specific; t_s = dt·UTIME).
                    MultiCode.chem_init!(; hubble=snap.hconst*100, Om=snap.omega_m,
                        OL=snap.omega_l, a_value=anew, fh=0.76,
                        density_units=UDENS*snap.hconst^2/anew^3, length_units=ULEN,
                        time_units=UTIME, deuterium=true, engine=:kernels)
                    _ct = time()
                    MultiCode.arepo_chem_step!(h; a_value=anew, dt=dphys,
                        engine=:kernels, backend=CHEM_BK, precision=CHEM_PREC)
                    chemt += time() - _ct
                elseif znew > CHEM_ZMAX
                    # grackle (C lib) path only: the explicit Compton term is too stiff
                    # to subcycle at high z, so lock the gas to the CMB thermal state.
                    Tg = try CICASSLib.thermal_state(znew).T_gas catch; 2.73*(1+znew) end
                    u_cmb = (KB*Tg/((GAMMA-1)*MU*MH))/UVEL^2
                    ArepoLib.set_cell_field!(h, :utherm, fill(u_cmb, ArepoLib.num_gas(h)))
                else
                    ok = false
                    for _ in 1:3
                        try; arepo_worker_chem_step!(gw, h; a_value=anew, dt=dphys, deut=true)
                            ok = true; break
                        catch e
                            e isa EOFError || rethrow()
                            try; close(gw); catch; end
                            gw = spawn_grackle_worker(; hubble=snap.hconst*100, Om=snap.omega_m,
                                OL=snap.omega_l, a0=a0, fh=0.76, du=UDENS*snap.hconst^2,
                                lu=ULEN, tu=UTIME, deut=true, data_file=GRACKLE_DATA)
                        end
                    end
                    ok || (@printf("    [chem skipped z=%.1f]\n", znew); flush(stdout))
                end
                step % (5*CHEM_EVERY) == 0 && (@printf("    [chem z=%.1f done]\n", znew); flush(stdout))
            end
            dbg && (@printf("    [rss step%d chem=%.0f]\n", step, _r()); flush(stdout))
            sec = time()-t0; steptot += sec
            # Per-step RSS (Sys.maxrss = cheap getrusage ccall, no fork, no alloc) for
            # visibility.  NOTE: we deliberately do NOT force GC/Metal.synchronize here —
            # the proven-good cmp_arepo_drag run (245 steps, no OOM) let Metal recycle its
            # command buffers on its own schedule; interleaving a forced GC every step
            # pinned ~1 GB/step instead.  The ceiling below is the safety net.
            rss = Sys.maxrss() / 2^20
            if step % 20 == 0
                @printf("%-5d z=%-8.3f %6.2fs  rss=%.0fMB (ps=%.0fMB)\n",
                        step, z, sec, rss, proc_rss_mb()); flush(stdout)
            elseif step < 16 || step % 5 == 0
                @printf("%-5d z=%-8.3f %6.2fs  rss=%.0fMB\n", step, z, sec, rss); flush(stdout)
            end
            st == :continue || (@printf("run_step returned %s at z=%.2f\n", st, z); break)
        end
        ai <= length(a_outs) && record!(znow())
        gw === nothing || (try; write(gw, Int64(0)); flush(gw); close(gw); catch; end)
        @printf("TIMING Arepo: step_loop=%.1fs  chem=%.1fs (%.1f%%)  host(hydro+grav+mesh)=%.1fs\n",
                steptot, chemt, 100*chemt/max(steptot,eps()), steptot-chemt); flush(stdout)
        ArepoLib.finalize(h)
    end
    write_tables()
    @printf("\nwrote %d Arepo outputs → %s\n", length(pk_results), datafile)
    ArepoLib.barrier()           # rank 0 finishes the .dat before anyone tears MPI down
    ArepoLib.mpi_finalize()      # clean shutdown on all ranks → mpiexec EXIT 0
end

# physical elapsed time between a1,a2 in seconds:  ∫ da/(a H(a)),  H0=100h km/s/Mpc.
# MUST include RADIATION to match Arepo's own hubble_function (darkenergy.c, Omega_r=4.15e-5/h²
# taken out of Lambda) — else the chem dt is ~14% too large at high z (no-rad H is too low),
# over-recombining/over-cooling the gas.  Keeps flat: Om + (OL-Or) + Or = 1 (Ok absorbed in OL).
function _dt_phys(a1, a2, h, Om, OL)
    H0 = 100*h * 1e5 / 3.0857e24                      # 1/s
    Or = 4.15e-5 / h^2
    f(a) = 1.0/(a*H0*sqrt(Om/a^3 + Or/a^4 + (OL - Or)))
    n=64; s=0.0; da=(a2-a1)/n
    for i in 1:n; a=a1+(i-0.5)*da; s+=f(a)*da; end
    return s
end

main()
