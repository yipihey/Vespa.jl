# CICASS high-z cosmology on the Vespa GPU path, with H+D reduced chemistry.
#
# Sets up a CICASS streaming-velocity box (default 128 kpc/h, 128³ cells+particles)
# at z=1000, injects the baryon overdensity + bulk velocity + DM particles into a
# live Enzo hierarchy, and evolves it WITHOUT refinement to z=20 with every solver
# stage on the Metal GPU:
#   hydro    → PPMKernels  (Metal f32)
#   gravity  → PoissonKernels device radix-2 FFT Poisson + comp_accel (Metal f32)
#   cooling  → Grackle, v2026 reduced H+D network (MultiSpecies=3,
#              equilibrium_deuterium=1, neutral_helium=1, equilibrium_h2_intermediates=1,
#              cmb_recombination + cmb_dissociation) — only HII,H2I,HDI advected.
#
# At logarithmic intervals in scale factor it measures the baryon (gas density) and
# dark-matter (CIC of the live particles) power spectra with the GPU FFT
# (`power_spectrum_gpu`), and overlays the CICASS linear-theory prediction: the IC
# field power spectrum grown by the ΛCDM linear growth factor D(a)² to each output
# redshift.  Writes P(k) tables to reports/multicode/ and renders a comparison plot.
#
# Run (GPU):
#   BACKEND=metal ENZOMODULES_GRID_LIB=.../f32/libenzomodules_grid_f32.dylib \
#     DYLD_LIBRARY_PATH=$HOME/grackle_install_f32/lib:/opt/homebrew/opt/hdf5/lib \
#     <julia> --project=lib/MultiCode/test lib/MultiCode/examples/cicass_highz_pk.jl

using EnzoLib, MultiCode, CICASSLib, PoissonKernels, Printf, Statistics
include(joinpath(@__DIR__, "..", "..", "EnzoLib", "examples", "sb_metal_amr.jl"))  # hydro!/helpers (defines BE)
# CUDA pool reclaim: the GPU FFT/gravity allocate fresh N³ scratch each cycle and the
# CUDA.jl pool never shrinks → footprint ratchets up (11→20 GB at 256³).  Periodic
# CUDA.reclaim() returns the cached-but-free pool memory to the driver, bounding it to
# the working set.  Interval via CIC_RECLAIM_EVERY (0 disables; default every cycle).
BE === :cuda && @eval import CUDA
const RECLAIM_EVERY = parse(Int, get(ENV, "CIC_RECLAIM_EVERY", "1"))
# Optional :hydro slot = FiniteVolumeGodunovKA (CIC_HYDRO_SOLVER=fvgk): an UNSPLIT 2nd-order
# CTU Godunov solver (CUDA f32).  Loaded only when selected (it transpiles+builds a CUDA lib).
const _USE_FVGK = get(ENV, "CIC_HYDRO_SOLVER", "ppm") == "fvgk"
_USE_FVGK && @eval import FiniteVolumeGodunovKA as FV

# ── parameters ──
const BOX    = parse(Float64, get(ENV, "CIC_BOX",   "0.128"))   # Mpc/h  (128 kpc/h)
const ZSTART = parse(Float64, get(ENV, "CIC_ZSTART","1000.0"))
const ZEND   = parse(Float64, get(ENV, "CIC_ZEND",  "20.0"))
const NGRID  = envint("CIC_NGRID", 128)
const NOUT   = envint("CIC_NOUT", 6)                            # log-a output count
const OMEGA_M = parse(Float64, get(ENV, "CIC_OMEGAM", "0.27"))
const VBC    = parse(Float64, get(ENV, "CIC_VBC",    "30.0"))   # streaming velocity (= RAMSES driver default)
const MAXCYC = envint("CIC_MAXCYC", 100000)
const GRACKLE_DATA = get(ENV, "GRACKLE_DATA_FILE",
    joinpath(homedir(), "Research", "codes", "grackle", "input", "CloudyData_noUVB.h5"))
const RECFAST = joinpath(CICASSLib.cicass_root(), "vbc_transfer", "recfast", "xeTrecfast.out")
const REPORTS = joinpath(@__DIR__, "..", "..", "..", "reports", "multicode")

# ── ΛCDM linear growth factor D(a) (growing mode), normalized D(1)=1 ──
# matter+Λ; at z≥20 this is EdS (D∝a) to <0.1%, but compute the exact integral.
function growth_D(a; Om=OMEGA_M, Ol=1-OMEGA_M)
    E(x) = sqrt(Om / x^3 + Ol)
    # D(a) ∝ E(a) ∫₀^a da'/(a' E(a'))³  (Heath 1977)
    f(x) = 1.0 / (x * E(x))^3
    n = 2000; h = a / n; s = 0.0
    @inbounds for i in 1:n
        x0 = (i-1)*h + 1e-12; x1 = i*h
        s += 0.5*(f(x0)+f(x1))*h
    end
    return E(a) * s
end

# CICASS RECFAST thermal IC (T_gas[K], x_e=n_e/n_H) at redshift z
function cicass_thermal(z)
    zs=Float64[]; xe=Float64[]; tg=Float64[]
    for (i,line) in enumerate(eachline(RECFAST))
        i==1 && continue
        t=split(line); length(t)>=3 || continue
        push!(zs,parse(Float64,t[1])); push!(xe,parse(Float64,t[2])); push!(tg,parse(Float64,t[3]))
    end
    p=sortperm(zs); zs,xe,tg=zs[p],xe[p],tg[p]
    interp(v)=(z<=zs[1] ? v[1] : z>=zs[end] ? v[end] :
               (j=searchsortedfirst(zs,z); w=(z-zs[j-1])/(zs[j]-zs[j-1]); v[j-1]*(1-w)+v[j]*w))
    return (T_gas=interp(tg), x_e=interp(xe), T_cmb=2.73*(1+z))
end

# Compton momentum-drag rate on the baryons / Hubble rate, Γ_drag/H, at redshift z.
# Γ_drag = (4/3)(ργ/ρb)·x_e·n_H·σT·c; the ρb in (ργ/ρb) cancels the ρb in n_H, so it
# reduces to (4/3)·a_rad·T_cmb⁴·x_e·X_H·σT/(c·m_H) — INDEPENDENT of Ω_b (cgs throughout).
# This is the drag the codes (and CICASS below z=1000) omit; we add it back as an
# operator.  Γ/H = 5.3 @z=990, 1 @z≈886, <0.01 below z≈440 (see compton_drag_vs_hubble).
function compton_drag_over_H(z; hubble=0.71)
    xe = cicass_thermal(z).x_e
    σT=6.6524e-25; cc=2.998e10; mH=1.673e-24; a_rad=7.5657e-15; XH=0.76; Tcmb0=2.726
    Or = 4.15e-5/hubble^2; H0 = 100.0*hubble*1e5/3.086e24            # 1/s
    Γ = (4.0/3.0)*a_rad*(Tcmb0*(1+z))^4 * xe * XH * σT / (cc*mH)     # 1/s
    H = H0*sqrt(OMEGA_M*(1+z)^3 + Or*(1+z)^4 + (1-OMEGA_M-Or))
    return Γ/H
end

# Apply the Compton momentum drag to the gas over a step of Δln a at redshift z_mid:
# damp the peculiar velocity toward the CMB rest frame (= 0 in the baryon rest frame)
# by exp(−(Γ/H)·Δln a), and rebuild the total energy from the (unchanged) internal
# energy + new kinetic energy (the dissipated bulk KE goes to the CMB, not to heat).
# Damps the peculiar velocity toward `v_cmb` (the CMB-frame gas velocity, code units):
# v → v_cmb + (v − v_cmb)·f.  v_cmb = 0 in the boosted (baryon rest) frame, = the streaming
# bulk in the unboosted frame — so the operator is Galilean-frame-correct.  Returns f.
function compton_drag!(h, z_mid, dlna; v_cmb=nothing)
    γH = compton_drag_over_H(z_mid)
    f  = exp(-γH * dlna)
    f >= 0.999999 && return f                       # negligible → skip the field I/O
    iD  = EnzoLib.field_index(h, 0; grid=0)                         # density (for mass-weighting)
    iV1 = EnzoLib.field_index(h, 4; grid=0); iV2 = EnzoLib.field_index(h, 5; grid=0)
    iV3 = EnzoLib.field_index(h, 6; grid=0); iTE = EnzoLib.field_index(h, 1; grid=0)
    gpos = findfirst(==(2), EnzoLib.problem_field_types(h, 0))   # internal-energy field (dual energy)
    ρ  = EnzoLib.problem_get_field(h, iD, 0)
    v1 = EnzoLib.problem_get_field(h, iV1, 0); v2 = EnzoLib.problem_get_field(h, iV2, 0)
    v3 = EnzoLib.problem_get_field(h, iV3, 0)
    # Damp the PECULIAR velocity toward the mass-weighted bulk (≡ CMB/streaming frame):
    # frame-agnostic, never damps the bulk — same prescription as the RAMSES/Arepo drags.
    M = sum(ρ); vb1 = sum(ρ.*v1)/M; vb2 = sum(ρ.*v2)/M; vb3 = sum(ρ.*v3)/M
    v1 .= vb1 .+ (v1 .- vb1).*f; v2 .= vb2 .+ (v2 .- vb2).*f; v3 .= vb3 .+ (v3 .- vb3).*f
    EnzoLib.problem_set_field(h, iV1, v1; grid=0); EnzoLib.problem_set_field(h, iV2, v2; grid=0)
    EnzoLib.problem_set_field(h, iV3, v3; grid=0)
    if gpos !== nothing
        ge = EnzoLib.problem_get_field(h, gpos-1, 0)
        EnzoLib.problem_set_field(h, iTE, ge .+ 0.5 .* (v1.^2 .+ v2.^2 .+ v3.^2); grid=0)
    end
    return f
end

# ── :gravity slot, fully on the GPU: device radix-2 FFT Poisson + device comp_accel
#    (the root-grid solve mirrors sb_metal_amr.gravity! but swaps the CPU FFTW root
#    solve for `fft_poisson_root_gpu!` so no stage touches the host) ──
const _OMEGA_B_CIC = parse(Float64, get(ENV, "CIC_OMEGAB", "0.046"))   # CICASS Ω_b
const _Ob_cic = OMEGA_M > 0 ? _OMEGA_B_CIC/OMEGA_M : 0.17              # f_b = Ω_b/Ω_m = 0.17
# Root Poisson on the GPU from ENZO'S OWN GravitatingMassField (cosmology-correct).
# `session_prepare_density(0)` deposits the GMF, which already carries the full
# cosmological normalization (1/a and the (3/2)Ωm amplitude that grows with a)
# — a hand-built δ has neither (it was ~10× too weak with the wrong a-dependence,
# the cause of the DM growth deficit).  Calibration (calib_gpu_gravity.jl):
# ∇²φ = 1.0·(GMF−mean) on the unit box reproduces Enzo's certified root potential
# to corr=1.00000, scale=1.0000.  The padded GMF/Potential field is N+2·ng per
# side (deposit buffer); the periodic FFT acts on the ACTIVE N³ (the true period),
# and we write φ back with the buffer filled by the PERIODIC continuation (correct
# for a periodic top grid).  Enzo then differences φ into baryon AND particle
# accelerations (`session_gravity_post`).  level>0 → certified parent-Dirichlet W-cycle.
# per-sub-step gravity timers (to localize the GPU-gravity cost)
const _GT = Dict(:prep=>Ref(0.0), :gmf=>Ref(0.0), :fft=>Ref(0.0),
                 :cont=>Ref(0.0), :set=>Ref(0.0), :post=>Ref(0.0))
# CIC_GRAV_RECON=1 (default): build the GMF Poisson source on the GPU and SKIP Enzo's
# 535ms PrepareDensityField. The particle CIC deposit is only ~50ms of that 535ms; the
# rest (DepositBaryons + periodic AddOverlappingParticleMassField + ComovingGravity-
# SourceTerm over the 140³ buffer) is what we avoid. ComovingGravitySourceTerm is just
# GMF=ρ−1 (no a-factor, no G), so the source rebuilds exactly as
#   GMF_centerN = CIC(drifted particles, shift −0.5)  +  baryon_active  −  mean
# verified corr=1.0, slope=1.0 vs problem_get_gravitating_mass on CPU AND Metal. The
# −0.5 shift maps edge→cell-centre registration (Enzo's GMF mesh); drift = ½·dt/a is
# PrepareDensityField's When=0.5 leapfrog drift. gravity_post differences only the
# PotentialField (never the GMF), so one prepare_density at init allocates the buffers
# and we never call it again.
const _RECON_GRAV = get(ENV, "CIC_GRAV_RECON", "1") == "1"
const _GRAV_DIAG  = get(ENV, "CIC_GRAV_DIAG", "0") == "1"   # legacy path: isolate baryon vs Enzo GMF
# CIC_GRAVPOST=fields (default): field-only gravity-post when the GPU push owns particles
# (skips Enzo's redundant ParticleAcceleration interp). =full forces the full post (for A/B).
const _GRAVPOST_FIELDS = get(ENV, "CIC_GRAVPOST", "fields") == "fields"
const _diagcyc = Ref(0)
const _gstate = Dict{Symbol,Int}()                  # one-time mesh-geometry capture (M, N, ng)
# Build the GPU reconstruction (DM CIC shift−0.5 drifted + baryon) on the ACTIVE N³ mesh
# — shared by the bypass path and the diagnostic. Returns (dm, bar) host arrays (mean-zero off).
function _recon_source(h, g, bep, Nact, dt, acosmo; respart=nothing)
    if respart === nothing
        px = dev(bep, EnzoLib.problem_get_particle_pos(h, 0, g))
        py = dev(bep, EnzoLib.problem_get_particle_pos(h, 1, g))
        pz = dev(bep, EnzoLib.problem_get_particle_pos(h, 2, g))
        vx = dev(bep, EnzoLib.problem_get_particle_vel(h, 0, g))
        vy = dev(bep, EnzoLib.problem_get_particle_vel(h, 1, g))
        vz = dev(bep, EnzoLib.problem_get_particle_vel(h, 2, g))
        m  = dev(bep, EnzoLib.problem_get_particle_mass(h, g))
    else
        # deposit straight from the GPU-resident particles — no bridge read this cycle
        px, py, pz = respart.px, respart.py, respart.pz
        vx, vy, vz = respart.vx, respart.vy, respart.vz
        m = respart.mass
    end
    gd = Tuple(Int.(EnzoLib.problem_grid_dims(h, g)))
    NGb = (gd[1] - Nact) ÷ 2; Rb = (NGb+1):(NGb+Nact)
    barA = Array(reshape(Float64.(EnzoLib.problem_get_field(h, 0, g)), gd...)[Rb, Rb, Rb])
    ρ = PoissonKernels.device_zeros(bep, T, (Nact^3,))
    PoissonKernels.cic_deposit!(ρ, px, py, pz, vx, vy, vz, m;
                                N=Nact, disp=0.5*dt/acosmo, shift=-0.5)
    return reshape(ρ, Nact, Nact, Nact), barA
end
function gravity_gpu!(h, level, dt; respart=nothing)
    level == 0 || return SUBGRID_GRAV!(h, level, dt)
    bep = PoissonKernels.backend(BE)
    acosmo = EnzoLib.session_cosmology(h)[1]    # Enzo internal a (=1 at z_init)
    _diagcyc[] += 1
    if !_RECON_GRAV
        # ── legacy path: Enzo builds the GMF (the 535ms), we solve it on the GPU ──
        _GT[:prep][] += @elapsed EnzoLib.session_prepare_density(h, 0)   # Enzo CIC particle deposit
        n = EnzoLib.session_num_grids_on_level(h, 0)
        for i in 0:n-1
            g = EnzoLib.problem_grid_index_on_level(h, 0, i)
            local src, M, ng, Nact
            _GT[:gmf][] += @elapsed begin
                gmff = EnzoLib.problem_get_gravitating_mass(h, g)
                M = round(Int, cbrt(length(gmff))); gmf = reshape(Float64.(gmff), M, M, M)
                Nact = round(Int, cbrt(EnzoLib.problem_num_particles(h, g)))   # active root = N³ (period)
                ng = (M - Nact) ÷ 2                              # GMF deposit-buffer per side
                rng = ntuple(d -> (ng+1):(ng+Nact), 3)
                src = Array(@view gmf[rng...]); src .-= sum(src)/length(src)   # mean-zero periodic source
            end
            # ── diagnostic: isolate the BARYON term vs Enzo's freshly-built GMF as the
            #    box evolves (corr=1.0 at z=1000 only proves DM; baryon is uniform there) ──
            if _GRAV_DIAG && i == 0 && _diagcyc[] in (1,5,10,20,30,40,60,80,100)
                dm, barA = _recon_source(h, g, bep, Nact, dt, acosmo)
                gA = @view gmf[rng...]
                gz = vec(Float64.(gA)); gz .-= sum(gz)/length(gz)
                dmz = vec(Float64.(PoissonKernels.to_host(dm))); dmz .-= sum(dmz)/length(dmz)
                barz = vec(Float64.(barA)); barz .-= sum(barz)/length(barz)
                recon = dmz .+ barz
                resid = gz .- dmz                       # = Enzo's isolated baryon contribution
                cr(a,b) = sum(a.*b)/(sqrt(sum(a.^2)*sum(b.^2))+1e-300)
                sl(a,b) = sum(a.*b)/(sum(a.^2)+1e-300)
                z = EnzoLib.session_cosmology(h)[2]
                @printf("  [grav-diag c%-3d z=%.1f] full corr=%.5f | DM-only corr=%.5f | BARYON resid-vs-bar corr=%.5f slope=%.4f | δb_rms=%.3e δdm_rms=%.3e\n",
                        _diagcyc[], z, cr(recon,gz), cr(dmz,gz), cr(resid,barz), sl(resid,barz),
                        sqrt(sum(barz.^2)/length(barz)), sqrt(sum(dmz.^2)/length(dmz)))
            end
            φh = nothing
            _GT[:fft][] += @elapsed begin
                φ = PoissonKernels.device_zeros(bep, T, (Nact, Nact, Nact))
                PoissonKernels.fft_poisson_root_gpu!(φ, dev(bep, src); G=1.0, a=acosmo, boxsize=1.0)
                φh = Float64.(PoissonKernels.to_host(φ))
            end
            local full
            _GT[:cont][] += @elapsed begin
                pidx = [mod(ii-ng-1, Nact) + 1 for ii in 1:M]
                full = φh[pidx, pidx, pidx]
            end
            _GT[:set][] += @elapsed EnzoLib.problem_set_potential(h, vec(full), g)
        end
        _GT[:post][] += @elapsed ((respart === nothing || !_GRAVPOST_FIELDS) ? EnzoLib.session_gravity_post(h, 0) :
                                                        EnzoLib.session_gravity_post_fields(h, 0))
        return nothing
    end
    # ── GPU source build: bypass PrepareDensityField ─────────────────────────────
    if !haskey(_gstate, :M)                          # once: allocate Enzo gravity buffers + capture mesh
        EnzoLib.session_prepare_density(h, 0)
        gmff = EnzoLib.problem_get_gravitating_mass(h, 0)
        Mx = round(Int, cbrt(length(gmff)))
        Nx = round(Int, cbrt(EnzoLib.problem_num_particles(h, 0)))
        _gstate[:M] = Mx; _gstate[:N] = Nx; _gstate[:ng] = (Mx - Nx) ÷ 2; _gstate[:check] = 1
    end
    M = _gstate[:M]; Nact = _gstate[:N]; ng = _gstate[:ng]
    n = EnzoLib.session_num_grids_on_level(h, 0)
    for i in 0:n-1
        g = EnzoLib.problem_grid_index_on_level(h, 0, i)
        local src3
        _GT[:prep][] += @elapsed begin
            src3, barA = _recon_source(h, g, bep, Nact, dt, acosmo; respart=respart)   # ρ_dm (device) + baryon (host)
            src3 .+= dev(bep, barA)                  # GMF = ρ_dm + baryon  (device add)
            src3 .-= sum(src3)/length(src3)          # mean-zero periodic source
        end
        if get(_gstate, :check, 0) == 1 && i == 0   # one-time correctness gate vs Enzo's own GMF
            gmf = reshape(Float64.(EnzoLib.problem_get_gravitating_mass(h, g)), M, M, M)
            R = (ng+1):(ng+Nact); gz0 = vec(gmf[R, R, R]); gz0 = gz0 .- sum(gz0)/length(gz0)
            sz = vec(Float64.(PoissonKernels.to_host(src3)))
            cc = sum(gz0 .* sz) / (sqrt(sum(gz0.^2)*sum(sz.^2)) + 1e-300)
            @printf("  [grav-recon] corr(GPU source, Enzo GMF)=%.5f  → bypassing PrepareDensityField\n", cc)
            _gstate[:check] = 0
        end
        φh = nothing
        _GT[:fft][] += @elapsed begin
            φ = PoissonKernels.device_zeros(bep, T, (Nact, Nact, Nact))
            PoissonKernels.fft_poisson_root_gpu!(φ, src3; G=1.0, a=acosmo, boxsize=1.0)
            φh = Float64.(PoissonKernels.to_host(φ))
        end
        local full
        _GT[:cont][] += @elapsed begin
            pidx = [mod(ii-ng-1, Nact) + 1 for ii in 1:M]
            full = φh[pidx, pidx, pidx]
        end
        _GT[:set][] += @elapsed EnzoLib.problem_set_potential(h, vec(full), g)
    end
    # field-only post when the GPU push owns the particle interp (skips Enzo's redundant
    # ParticleAcceleration interpolation + ±½ drift); else the full post (fills particles too)
    _GT[:post][] += @elapsed ((respart === nothing || !_GRAVPOST_FIELDS) ? EnzoLib.session_gravity_post(h, 0) :
                                                    EnzoLib.session_gravity_post_fields(h, 0))   # difference φ → accelerations
    return nothing
end

# ── alternative :hydro slot — the fast one-ghost Local PPM (Enzo HydroMethod 10) ──
# `muscl_hancock_step_3d!` with recon=:ppm_local on the CONSERVED state (dual energy),
# with the cosmology coupling the bare solver lacks added here:
#   • comoving cell width dx·a(t)   (continuity coupling — same as the PPM slot),
#   • species HII/H2I/HDI carried as passive COLOURS riding the mass flux (the new
#     `colours=` path; uniform mass fraction stays uniform, species mass conserved),
#   • gravity as an operator-split KDK kick on the GAS (½ before + ½ after the hydro),
#     using the same AccelerationField the PPM slot consumes; eint is untouched.
# Selected by CIC_HYDRO_SOLVER=localppm.  Enzo still owns ghost fill + AMR.
const _lstep = Ref(0)
@inline function _grav_kick!(S1, S2, S3, Tau, Dd, ax, ay, az, c)   # v += c·g (KE-consistent)
    dS1 = Dd .* ax .* c; dS2 = Dd .* ay .* c; dS3 = Dd .* az .* c
    Tau .+= ((S1 .* dS1 .+ S2 .* dS2 .+ S3 .* dS3) .+
             T(0.5) .* (dS1 .^ 2 .+ dS2 .^ 2 .+ dS3 .^ 2)) ./ Dd
    S1 .+= dS1; S2 .+= dS2; S3 .+= dS3
    return nothing
end
function hydro_localppm!(h, level, dt)
    bep = PPMKernels.backend(BE)
    n = EnzoLib.session_num_grids_on_level(h, level)
    order = isodd(_lstep[]) ? (3,2,1) : (1,2,3); _lstep[] += 1
    acosmo = EnzoLib.session_cosmology(h)[1]
    chalf = T(0.5 * dt)
    for i in 0:n-1
        g = EnzoLib.problem_grid_index_on_level(h, level, i)
        gd = Tuple(Int.(EnzoLib.problem_grid_dims(h, g)))
        gl, gr = EnzoLib.problem_grid_edge(h, g)
        Nactx = gd[1] - 2*NG
        dxc = acosmo * (gr[1] - gl[1]) / Nactx
        Dd = dev(bep, EnzoLib.problem_get_field(h, iD, g))
        v1 = dev(bep, EnzoLib.problem_get_field(h, iV1, g))
        v2 = dev(bep, EnzoLib.problem_get_field(h, iV2, g))
        v3 = dev(bep, EnzoLib.problem_get_field(h, iV3, g))
        TE = dev(bep, EnzoLib.problem_get_field(h, iTE, g))
        GE = dev(bep, EnzoLib.problem_get_field(h, iGE, g))
        S1 = Dd .* v1; S2 = Dd .* v2; S3 = Dd .* v3
        Tau = Dd .* TE; Ge = Dd .* GE
        ax = dev(bep, EnzoLib.problem_get_acceleration(h, 0, g))
        ay = dev(bep, EnzoLib.problem_get_acceleration(h, 1, g))
        az = dev(bep, EnzoLib.problem_get_acceleration(h, 2, g))
        sp   = _species_indices(h, g)
        cols = Tuple(dev(bep, EnzoLib.problem_get_field(h, fi, g)) for fi in sp)   # ρ_s densities
        _grav_kick!(S1, S2, S3, Tau, Dd, ax, ay, az, chalf)            # ½ kick (pre)
        PPMKernels.muscl_hancock_step_3d!(Dd, S1, S2, S3, Tau, gd, NG;
            dt=dt, gamma=GAMMA, dx=dxc, ge=Ge, colours=cols, order=order,
            recon=:ppm_local, predictor=:trace, riemann=:hll, face_periodic=(level==0))
        _grav_kick!(S1, S2, S3, Tau, Dd, ax, ay, az, chalf)            # ½ kick (post)
        wr(fi, a) = EnzoLib.problem_set_field(h, fi, Float64.(PPMKernels.to_host(a)); grid=g)
        wr(iD, Dd); wr(iV1, S1 ./ Dd); wr(iV2, S2 ./ Dd); wr(iV3, S3 ./ Dd)
        wr(iTE, Tau ./ Dd); wr(iGE, Ge ./ Dd)
        for (fi, c) in zip(sp, cols)
            wr(fi, c)
        end
    end
    return nothing
end

# ── alternative :hydro slot — FiniteVolumeGodunovKA (CIC_HYDRO_SOLVER=fvgk) ───────────
# FVGK is an UNSPLIT 2nd-order CTU Godunov solver (CUDA f32) that owns its own periodic
# boundary on a SINGLE active grid (no ghosts).  The CICASS box is single-level periodic,
# so it maps directly: gather the active region (strip Enzo's NG ghosts) into FVGK's
# var-major device buffer (D,S1,S2,S3,Tau, then species as ρ·xᵢ "colours"), step once
# (sub-cycling to FVGK's CTU CFL), and scatter back.  Cosmology is coupled exactly like
# the PPM slots: comoving cell width dx·a(t) (set per step), an operator-split ½+½ KDK
# gravity kick on the gas (same AccelerationField), and species riding the mass flux.
# Single-grid only (asserts level 0, one grid) — no AMR flux correction.  The FVGK grid is
# built once (it transpiles+compiles a CUDA kernel) and reused; dx is updated each step.
const _FVGRID = Ref{Any}(nothing)     # cached Grid3DCuMarch
const _FVNSP  = Ref(-1)               # species count it was built for
function hydro_fvgk!(h, level, dt)
    @assert level == 0 "FVGK hydro slot is single-grid; run with CIC_MAXLEVEL=0"
    ng = EnzoLib.session_num_grids_on_level(h, level)
    @assert ng == 1 "FVGK hydro slot expects one grid at level 0 (got $ng)"
    g    = EnzoLib.problem_grid_index_on_level(h, level, 0)
    gd   = Tuple(Int.(EnzoLib.problem_grid_dims(h, g)))
    Nact = gd[1] - 2*NG
    gl, gr = EnzoLib.problem_grid_edge(h, g)
    acosmo = EnzoLib.session_cosmology(h)[1]
    dxc    = Float32(acosmo * (gr[1] - gl[1]) / Nact)
    chalf  = T(0.5 * dt)
    A      = (NG+1):(NG+Nact)                                  # active slice (strip ghosts)
    act(fi) = Float32.(@view reshape(EnzoLib.problem_get_field(h, fi, g), gd)[A, A, A])
    acc(d)  = Float32.(@view reshape(EnzoLib.problem_get_acceleration(h, d, g), gd)[A, A, A])

    D = act(iD); S1 = D .* act(iV1); S2 = D .* act(iV2); S3 = D .* act(iV3); Tau = D .* act(iTE)
    ax = acc(0); ay = acc(1); az = acc(2)
    sp = _species_indices(h, g); nsp = length(sp)
    cols = [act(fi) for fi in sp]                              # ρ_s densities = ρ·xᵢ colours
    _grav_kick!(S1, S2, S3, Tau, D, ax, ay, az, chalf)        # ½ KDK (pre, pre-D)

    # build/cache the FVGK grid (Euler + nsp passive colours), sized to the active region
    if _FVGRID[] === nothing || _FVNSP[] != nsp
        γ = Float32(GAMMA)
        if nsp == 0
            sys = FV.Euler(γ = γ);             z = (1f0, 0f0, 0f0, 0f0, 1f0)
        else
            sys = FV.EulerColors{nsp}(γ = γ);  z = (1f0, 0f0, 0f0, 0f0, 1f0, ntuple(_ -> 0f0, nsp)...)
        end
        U0 = [z for _ in 1:Nact, _ in 1:Nact, _ in 1:Nact]
        _FVGRID[] = FV.Grid3DCuMarch(sys, U0; dx = dxc); _FVNSP[] = nsp
    end
    gv = _FVGRID[]; gv.dx = dxc
    VOL = Nact^3; nv = 5 + nsp
    Uh  = Vector{Float32}(undef, nv*VOL)
    put(c, arr) = (Uh[(c-1)*VOL+1 : c*VOL] = vec(arr))
    put(1, D); put(2, S1); put(3, S2); put(4, S3); put(5, Tau)
    for (q, c) in enumerate(cols); put(5+q, c); end
    copyto!(gv.R, Uh)

    nsub = max(1, ceil(Int, Float32(dt) / FV.dt_cfl(gv; cfl = 0.45f0)))
    if nsp > 0
        FV.run_ctu!(gv, Float32(dt)/nsub, nsub)               # f32 (colours are primitives)
    else
        FV.run_ctus!(gv, Float32(dt)/nsub, nsub)              # f16-tiled fast path (pure hydro)
    end

    Rh = Array(gv.R)
    getv(c) = reshape(Rh[(c-1)*VOL+1 : c*VOL], Nact, Nact, Nact)
    D = getv(1); S1 = getv(2); S2 = getv(3); S3 = getv(4); Tau = getv(5)
    cols = [getv(5+q) for q in 1:nsp]
    _grav_kick!(S1, S2, S3, Tau, D, ax, ay, az, chalf)        # ½ KDK (post, post-D)
    Ge = Tau .- 0.5f0 .* (S1.^2 .+ S2.^2 .+ S3.^2) ./ D       # re-derive gas energy (FVGK is single-energy)

    # write the active region back into the full (ghosted) field; Enzo refills ghosts.
    function wr(fi, active)
        full = reshape(copy(EnzoLib.problem_get_field(h, fi, g)), gd)
        @views full[A, A, A] .= Float64.(active)
        EnzoLib.problem_set_field(h, fi, vec(full); grid=g)
    end
    wr(iD, D); wr(iV1, S1 ./ D); wr(iV2, S2 ./ D); wr(iV3, S3 ./ D)
    wr(iTE, Tau ./ D); wr(iGE, Ge ./ D)
    for (fi, c) in zip(sp, cols); wr(fi, c); end
    return nothing
end

# CICASS analytic linear-theory input power.  Columns: k, Δ²(G1), Δ²(G5), Δ²(G3),
# Δ²(G7) (dimensionless k³P/2π²) at z_init.  Per the G[] assignments in
# makeCosICs/main.c (generateDisplacements: G[1]=DM density, G[5]=BARYON density),
# col2 is DM and col3 is BARYON — the file's "Deltak_baryons Deltak_dm" header is
# SWAPPED.  Convert to P(k) = 2π²·Δ²/k³.
function read_cicass_pk(path)
    isfile(path) || return nothing
    k=Float64[]; Pb=Float64[]; Pd=Float64[]
    for line in eachline(path)
        startswith(strip(line), "#") && continue
        t = split(line); length(t) >= 3 || continue
        kv = parse(Float64, t[1]); kv > 0 || continue
        c = 2π^2 / kv^3
        push!(k, kv); push!(Pd, c*parse(Float64,t[2])); push!(Pb, c*parse(Float64,t[3]))
    end
    return (k=k, Pb=Pb, Pd=Pd)
end

# overdensity δ = ρ/ρ̄ − 1 over the active region; power spectrum via GPU FFT
function pk_of(field_active::Array{Float64,3}; box_mpc)
    δ = copy(field_active); m = sum(δ)/length(δ); δ ./= m; δ .-= 1.0
    bep = PoissonKernels.backend(BE)
    r = PoissonKernels.power_spectrum_gpu(dev(bep, δ); boxsize=box_mpc, nbins=size(δ,1)÷2)
    return (k = collect(r.k), P = collect(r.P), N = collect(r.Nmodes))
end

# raw-field power (mean-subtracted, no /mean) — for the gravitational potential φ
function pk_field(field::Array{Float64,3})
    f = copy(field); f .-= sum(f)/length(f)
    bep = PoissonKernels.backend(BE)
    r = PoissonKernels.power_spectrum_gpu(dev(bep, f); boxsize=BOX, nbins=size(f,1)÷2)
    return (k = collect(r.k), P = collect(r.P), N = collect(r.Nmodes))
end

function main_cic()
    EnzoLib.grid_available() || error("Enzo grid bridge not built")
    s0 = cicass_thermal(ZSTART)
    # CFL/timestep parameters set EQUAL to RAMSES for an apples-to-apples comparison
    # (RAMSES newdt_fine.f90: courant_factor=0.8, da/a ≤ 0.1/hexp). Appended last so
    # they override the native param defaults (0.015 / 0.5). Override via env if needed.
    maxexp   = parse(Float64, get(ENV, "CIC_MAXEXP",   "0.1"))    # = RAMSES da/a cap (0.1/hexp)
    courant  = parse(Float64, get(ENV, "CIC_COURANT",  "0.8"))    # = RAMSES courant_factor
    pcourant = parse(Float64, get(ENV, "CIC_PCOURANT", "0.8"))    # = RAMSES courant_factor (particles)
    use_grackle = get(ENV, "CIC_USE_GRACKLE", "1") == "1" ? 1 : 0
    chem = """
    CosmologyMaxExpansionRate    = $(maxexp)
    CourantSafetyNumber          = $(courant)
    ParticleCourantSafetyNumber  = $(pcourant)
    RadiativeCooling             = $(use_grackle)
    use_grackle                  = $(use_grackle)
    with_radiative_cooling       = $(use_grackle)
    MultiSpecies                 = 3
    CaseBRecombination           = 1
    cmb_dissociation             = 1
    cmb_recombination            = 1
    equilibrium_h2_intermediates = 1
    neutral_helium               = 1
    equilibrium_deuterium        = 1
    grackle_data_file            = $(GRACKLE_DATA)
    DualEnergyFormalism          = 1
    GreensFunctionMaxNumber      = 30
    NumberOfGhostZones           = 4
    CosmologyFinalRedshift       = $(ZEND)
    CosmologySimulationInitialFractionHII = $(s0.x_e)
    """
    # NB: the gas temperature is OWNED by run_cicass_enzo (init_temperature below); it sets
    # the internal energy explicitly with μ=1.22 (neutral).  Do NOT also set
    # CosmologySimulationInitialTemperature here — Enzo's μ≈0.6 conversion would be 2× hot.
    @printf("CICASS z=%.0f: box=%.4f Mpc/h, %d³, T_gas=%.1f K, x_e=%.3e (H+D reduced chem)\n",
            ZSTART, BOX, NGRID, s0.T_gas, s0.x_e)

    # CIC_ZERO_BARYON_BULK=1 → boost to the baryon rest frame (subtract the coherent
    # gas bulk velocity from BOTH gas and DM); preserves the relative streaming offset.
    zero_bulk = get(ENV, "CIC_ZERO_BARYON_BULK", "0") == "1"
    # CIC_UNIFORM_BARYONS=1 → start baryons uniform (δb=0) at rest + Compton drag
    # (the physical recombination start; pair with CIC_COMPTON_DRAG=1 below).
    uniform_b = get(ENV, "CIC_UNIFORM_BARYONS", "0") == "1"
    # CIC_BARYON_IC = smooth (default) | particle (CIC from displaced grid) | uniform
    baryon_ic = Symbol(get(ENV, "CIC_BARYON_IC", "smooth"))
    res = MultiCode.run_cicass_enzo(; vbc=VBC, boxlength=BOX, zstart=ZSTART, ngrid=NGRID,
                                    omega_m=OMEGA_M, param_extra=chem, zero_baryon_bulk=zero_bulk,
                                    uniform_baryons=uniform_b, baryon_ic=baryon_ic,
                                    init_temperature=s0.T_gas, mu_init=1.22)
    h = res.handle; dims = res.dims; act = res.act; N = res.n; snap = res.snap
    cic_lin = read_cicass_pk(res.pk_file)           # CICASS analytic linear theory @ z_start
    pk_results = NamedTuple[]                        # one per output redshift
    mkpath(REPORTS)
    TAG = get(ENV, "CIC_TAG", "")                    # suffix so diagnostic runs don't clobber
    datafile = joinpath(REPORTS, "cicass_highz_pk$(TAG).dat")
    function write_tables()                          # rewrite the full table (cheap; ≤NOUT outputs)
        open(datafile, "w") do io
            println(io, "# CICASS z=$ZSTART→$ZEND box=$BOX Mpc/h N=$NGRID  (GPU Vespa, H+D chem)")
            println(io, "# theory_* = IC realization grown by D(a)²; theory_*_cic = CICASS analytic linear theory grown by D(a)²")
            println(io, "# block: z  comp  then  k[h/Mpc] P[(Mpc/h)^3] rows")
            for r in pk_results
                for tag in (:baryon, :dm, :phi, :theory_b, :theory_dm, :theory_b_cic, :theory_dm_cic)
                    c = getfield(r, tag); c === nothing && continue
                    println(io, "@ z=$(round(r.z,digits=3)) $tag")
                    for i in eachindex(c.k)
                        @printf(io, "%.6e %.6e\n", c.k[i], c.P[i])
                    end
                end
            end
        end
    end
    chemt = Ref(0.0); evolvet = Ref(0.0)   # wall-time accumulators (function scope)
    hydrot = Ref(0.0); gravt = Ref(0.0); ncyc = Ref(0); particlet = Ref(0.0); rebuildt = Ref(0.0)
    # CIC_PROBE=1 attaches a per-slot timer (boundary/baryon_copy/comoving_expansion/…)
    probe = get(ENV, "CIC_PROBE", "0") == "1" ? EnzoLib.SlotProbe() : nothing
    try
        # The native CICASS HDF5 ICs (run_cicass_enzo) already carry the baryon density
        # in Enzo code units — (Ωb/Ωm)·(1+δb) per cell, from the realization's OWN
        # constants — so the GravitatingMassField sums to 1 at boot and the chem species
        # (MultiSpecies=3) are initialized by Enzo relative to that correct density.  No
        # post-init renormalization of the gas field is needed or wanted.

        # CIC_CHEM_INIT_MATCH=1 → set the initial species fields to the SAME explicit
        # fractions the RAMSES driver uses (HII=ρ·x_e, H2I=ρ·1e-6, HDI=ρ·6.8e-5·x_e), so
        # both codes feed ChemistryKernels identical starting HII/H2I/HDI (field 9/14/18).
        if get(ENV, "CIC_CHEM_INIT_MATCH", "0") == "1"
            xe0 = s0.x_e
            ρf = reshape(EnzoLib.problem_get_field(h, iD, 0), dims...)
            for (ft, frac) in ((9, xe0), (14, 1e-6), (18, 6.8e-5*xe0))
                fi = try EnzoLib.field_index(h, ft; grid=0) catch; -1 end
                fi < 0 && continue
                EnzoLib.problem_set_field(h, fi, vec(ρf) .* frac; grid=0)
            end
            # (gas temperature is now owned by run_cicass_enzo: μ=1.22 explicit eint.)
            # verify the override took (read back H2I/ρ over the active region)
            chk(ft) = begin
                a = active_of(EnzoLib.problem_get_field(h, EnzoLib.field_index(h,ft;grid=0), 0), dims, N)
                sum(a)/sum(active_of(EnzoLib.problem_get_field(h, iD, 0), dims, N))
            end
            @printf("Enzo chem init match: set {HII,H2I,HDI}/ρ = {%.3e,1e-6,%.2e} + T_gas=%.0fK(μ=1.22); READBACK = {%.3e,%.3e,%.3e}\n",
                    xe0, 6.8e-5*xe0, s0.T_gas, chk(9), chk(14), chk(18))
        end

        # ── IC power spectra (z=zstart) — the linear-theory anchor ──
        a_start = 1.0 / (1.0 + ZSTART); a_end = 1.0 / (1.0 + ZEND)
        D_start = growth_D(a_start)
        ρb_ic = active_of(EnzoLib.read_density(h; grid=0), dims, N)
        ρd_ic = active_of(EnzoLib.deposit_particle_density(h; grid=0, periodic=true), dims, N)
        pk_b_ic = pk_of(ρb_ic; box_mpc=BOX)
        pk_d_ic = pk_of(ρd_ic; box_mpc=BOX)
        @printf("IC anchored: baryon kmax=%.2f h/Mpc, DM modes=%d, D(z=%.0f)=%.4e\n",
                maximum(pk_b_ic.k), sum(pk_d_ic.N), ZSTART, D_start)

        # ── log-spaced scale-factor checkpoints a_start→a_end, with CIC_NEARLY extra
        #    early-time (high-z) outputs in the first factor-4 of a (where the drag acts) ──
        a_main = exp.(range(log(a_start), log(a_end), length=NOUT))
        nearly = parse(Int, get(ENV, "CIC_NEARLY", "4"))
        # CIC_ZOUT="z1,z2,…" → EXPLICIT output redshifts (the consistent cross-code list);
        # else the default log-spaced a_main + CIC_NEARLY early extras.
        a_outs = haskey(ENV, "CIC_ZOUT") ?
            sort([1.0/(1.0+parse(Float64,s)) for s in split(ENV["CIC_ZOUT"], ",")]) :
            nearly > 0 ?
            sort(unique(vcat(exp.(range(log(1.06*a_start), log(min(4*a_start, a_end)), length=nearly)), a_main))) :
            a_main
        NOUT_eff = length(a_outs)
        ai = 1                                          # next checkpoint index

        gmode = Symbol(get(ENV, "CIC_GRAV", "julia"))   # :julia (Metal hook) or :enzo (certified)
        hmode = Symbol(get(ENV, "CIC_HYDRO", "julia"))  # :julia (PPMKernels) or :enzo (native PPM)
        # chemistry engine: :kernels (DEFAULT) = the native ChemistryKernels Julia port via
        # the cooling=:julia hook, run on the GPU — profiling showed native Grackle cooling
        # was ~47% of the per-cycle cost (1.125 s/cyc → 0.186 s/cyc on Metal, a 1.8× whole-
        # cycle speedup). :enzo = native C Grackle (cooling=:enzo) for cross-checking.
        cmode = Symbol(get(ENV, "CIC_CHEM_ENGINE", "kernels"))
        cbk   = Symbol(get(ENV, "CIC_CHEM_BACKEND", string(BE)))   # match the main backend (cpu/metal)
        hooks = Dict{Symbol,Function}()
        # CIC_HYDRO_SOLVER: :ppm (Enzo-PPM port, default) or :localppm (fast one-ghost Local PPM)
        hsolver = Symbol(get(ENV, "CIC_HYDRO_SOLVER", "ppm"))
        hydrofn = hsolver === :localppm ? hydro_localppm! :
                  hsolver === :fvgk     ? hydro_fvgk!     : hydro!
        hmode === :julia && @printf("  hydro slot = :julia / %s\n", hsolver)
        hmode === :julia && (hooks[:hydro]   = (hh_,l_,d_)->(t=time(); hydrofn(hh_,l_,d_);       hydrot[]+=time()-t))
        # `respart` (the GPU-resident particle state, set below) lets the gravity slot
        # deposit straight from the device-resident particles instead of re-reading
        # them from Enzo every cycle — captured by reference so the closure sees it.
        respart = nothing
        gmode === :julia && (hooks[:gravity] = (hh_,l_,d_)->(t=time(); gravity_gpu!(hh_,l_,d_; respart=respart); gravt[]+=time()-t))
        coolmode = cmode === :kernels ? :julia : :enzo
        if cmode === :kernels
            cprec = cbk === :metal ? Float32 : Float64
            # the hook feeds physical CGS to solve_chem! → _CHEM_CFG units = 1; the
            # cosmology (for Compton/Peebles H(z)) is the live Enzo cosmology.
            # hubble_expansion=true: the chemistry kernel now does adiabatic expansion
            # cooling (−2H·e) COUPLED with the stiff Compton term in its adaptive sub-
            # cycle (ȧ/a sourced from Enzo's own cosmology per step in enzo_chem_step!).
            # This replaces Enzo's separate operator-split energy expansion term (the
            # comoving_expansion slot below is reduced to velocity drag only) so the gas
            # stays at the true Compton–adiabatic balance even on short output steps.
            MultiCode.chem_init!(; hubble=snap.hconst*100, Om=OMEGA_M, OL=1-OMEGA_M,
                a_value=1.0/(1+ZSTART), fh=0.76, density_units=1.0, length_units=1.0,
                time_units=1.0, deuterium=true, engine=:kernels, hubble_expansion=true)
            hooks[:cooling] = (hh_, lev_, dt_) -> begin
                _t = time()
                r = MultiCode.enzo_chem_step!(hh_, lev_, dt_;
                    Om=OMEGA_M, OL=1-OMEGA_M, hub=snap.hconst, box=BOX, zri=ZSTART,
                    fh=0.76, deuterium=true, ng=4, engine=:kernels, backend=cbk, precision=cprec)
                chemt[] += time() - _t
                r
            end
            @printf("chemistry engine: ChemistryKernels (:kernels, backend=%s)\n", cbk)
        else
            @printf("chemistry engine: native Enzo Grackle (:enzo)\n")
        end
        # ── GPU-resident particle push: move the per-cycle CIC accel interpolation
        #    + leapfrog drift/kick (Enzo's session_update_particles, ~N³=2M particles
        #    on the CPU) onto the GPU via PoissonKernels, particles resident on the
        #    device, coefficients from Enzo's CosmologyComputeExpansionFactor. Synced
        #    back to Enzo each cycle so compute_dt/rebuild/diagnostics stay correct. ──
        pmode = Symbol(get(ENV, "CIC_PARTICLES", "enzo"))   # :julia (GPU resident) or :enzo
        # CIC_PARTICLE_SYNC: "output" (default for :julia) keeps particles resident and
        # writes them back to Enzo ONLY at output checkpoints — gravity deposits from the
        # resident arrays, so NO particles cross the bridge per cycle. "cycle" syncs every
        # cycle (the safe baseline: Enzo's compute_dt/rebuild always see fresh particles).
        psync = Symbol(get(ENV, "CIC_PARTICLE_SYNC", "output"))
        if pmode === :julia
            respart = MultiCode.resident_particles_init(h, PoissonKernels.backend(BE), T; grid=0, wrap=1.0)
            sync_each = psync === :cycle
            hooks[:particle_push] = (hh_,l_,d_)->(t=time(); MultiCode.particle_push_gpu!(respart, hh_, l_, d_; sync=sync_each); particlet[]+=time()-t)
            @printf("  particle push = :julia (GPU resident, backend=%s, N=%d, sync=%s)\n", BE, respart.N, psync)
        end
        # Comoving expansion. With the :kernels chemistry, the gas-energy adiabatic
        # term is now done (coupled with Compton) in the cooling slot, so this slot is
        # reduced to a velocity-only Hubble drag via a :julia hook — using ENZO's own
        # scale factor (time-centered at t−½dt, matching ComovingExpansionTerms.C's
        # VELOCITY_METHOD3), so the velocity coefficient is byte-for-byte Enzo's. The
        # cooling slot runs BEFORE session_advance_time and this slot AFTER it, so here
        # session_time = t_end and the midpoint is session_time − ½dt. Native :enzo path
        # (cmode=:enzo) keeps the full native energy+velocity term.
        coexp = :enzo
        if cmode === :kernels
            coexp = :julia
            hooks[:comoving_expansion] = (hh_, lev_, dt_) -> begin
                am, adm = EnzoLib.session_expansion_factor(hh_, EnzoLib.session_time(hh_) - 0.5*dt_)
                am <= 0 && return nothing
                C = dt_ * adm / am
                f = (1 - 0.5C) / (1 + 0.5C)          # semi-implicit Hubble drag (γ-independent)
                for ft in (4, 5, 6)                  # Velocity1/2/3 field types
                    vi = try EnzoLib.field_index(hh_, ft; grid=0) catch; -1 end
                    vi < 0 && continue
                    v = EnzoLib.problem_get_field(hh_, vi, 0); v .*= f
                    EnzoLib.problem_set_field(hh_, vi, v; grid=0)
                end
            end
        end
        eng = EnzoLib.EngineConfig(; hydro=hmode, gravity=gmode, cooling=coolmode,
                                   comoving_expansion=coexp, reflux=false,
                                   particle_push=pmode, hooks=hooks, probe=probe)
        # CIC_COPYOLD=0 skips the per-cycle OldBaryonField copy (~8s/run) but is UNSAFE:
        # measured to change baryon large-scale P(k) ~2× (the comoving-expansion/gravity
        # source reads OldBaryonField). Keep =1 for self-gravitating cosmology.
        COPYOLD = get(ENV, "CIC_COPYOLD", "1") == "1"
        @printf("hydro mode: %s  gravity mode: %s\n", hmode, gmode)
        function enzo_phi_grid()
            pf = EnzoLib.problem_get_potential(h, 0)
            M = round(Int, cbrt(length(pf))); fullp = reshape(Float64.(pf), M, M, M)
            off = (M - N) ÷ 2
            return Array(@view fullp[(off+1):(off+N), (off+1):(off+N), (off+1):(off+N)])
        end
        function record!(a, z)
            ρb = active_of(EnzoLib.read_density(h; grid=0), dims, N)
            ρd = active_of(EnzoLib.deposit_particle_density(h; grid=0, periodic=true), dims, N)
            φ  = enzo_phi_grid()
            # ── DM bulk (streaming) velocity diagnostic: the coherent mean DM peculiar
            #    velocity should redshift as 1/a (∝ 1+z) — only Hubble drag acts on the
            #    bulk (no Compton drag on DM, no net gravity on a uniform flow). ──
            if get(ENV, "CIC_DMBULK", "0") == "1"
                np = EnzoLib.problem_num_particles(h, 0)
                vku = 1.22475e7 * BOX * sqrt(OMEGA_M) * sqrt(1+ZSTART) / 1e5      # km/s per code vel
                dmv = ntuple(d -> sum(EnzoLib.problem_get_particle_vel(h, d-1, 0))/np * vku, 3)
                gxf = EnzoLib.problem_get_field(h, EnzoLib.field_index(h,4;grid=0), 0)   # gas v_x
                gasvx = (sum(@view active_of(gxf, dims, N)[:]) / N^3) * vku
                exp_vbc = VBC * (1+z) / 1001.0                                    # 1/a redshift of v_bc
                @printf("    [bulk z=%.1f: DM=(%.3f,%.3f,%.3f)|%.4f  gas_vx=%.4f km/s  expected 1/a=%.4f  DMratio=%.4f gasratio=%.4f]\n",
                        z, dmv..., sqrt(sum(abs2,dmv)), gasvx, exp_vbc, sqrt(sum(abs2,dmv))/exp_vbc, gasvx/exp_vbc); flush(stdout)
            end
            # ── mid-box (z=N/2) slices: baryon δ, DM δ, vx, vy [km/s] — one per output ──
            if get(ENV, "CIC_SLICES", "1") == "1"
                kmid = N ÷ 2 + 1
                vkms = 1.22475e7 * BOX * sqrt(OMEGA_M) * sqrt(1+ZSTART) / 1e5   # km/s per code vel
                vxf = active_of(EnzoLib.problem_get_field(h, EnzoLib.field_index(h,4;grid=0), 0), dims, N)
                vyf = active_of(EnzoLib.problem_get_field(h, EnzoLib.field_index(h,5;grid=0), 0), dims, N)
                open(joinpath(REPORTS, "enzo_slice$(TAG)_z$(round(Int,z)).bin"), "w") do io
                    write(io, Int64(N))
                    write(io, vec(ρb[:, :, kmid]));           write(io, vec(ρd[:, :, kmid]))
                    write(io, vec(vxf[:, :, kmid] .* vkms));  write(io, vec(vyf[:, :, kmid] .* vkms))
                end
            end
            # ── full 3D baryon+DM density for cross-spectra P_bc(k), r(k) ──
            if get(ENV, "CIC_XSPEC", "0") == "1"
                open(joinpath(REPORTS, "enzo_xspec$(TAG)_z$(round(Int,z)).bin"), "w") do io
                    write(io, Int64(N)); write(io, vec(ρb)); write(io, vec(ρd))
                end
            end
            # ── per-cell density + chemistry on the regular grid, for cell-by-cell vs RAMSES.
            #    x_HII=n_HII/n_H, f_H2=2n_H2/n_H, f_HD=n_HD/n_H (species fields 9,14,18; X_H=0.76). ──
            if get(ENV, "CIC_CELLCMP", "0") == "1"
                XH = 0.76
                HII = active_of(EnzoLib.problem_get_field(h, EnzoLib.field_index(h, 9;grid=0), 0), dims, N)
                H2I = active_of(EnzoLib.problem_get_field(h, EnzoLib.field_index(h,14;grid=0), 0), dims, N)
                HDI = active_of(EnzoLib.problem_get_field(h, EnzoLib.field_index(h,18;grid=0), 0), dims, N)
                geA = active_of(EnzoLib.problem_get_field(h, iGE, 0), dims, N)  # specific internal e
                vu_cgs = 1.22475e7 * BOX * sqrt(OMEGA_M) * sqrt(1+ZSTART)       # cm/s (const z_init)
                xHIIv = vec(HII) ./ vec(ρb) ./ XH; fH2v = vec(H2I) ./ vec(ρb) ./ XH
                # grackle reduced-network μ (neutral He, n_e=n_HII): μ=1/[(X_H+Y/4)+
                # X_H(x_HII−f_H2/2)] (→1.22 neutral) — consistent with calculate_temperature
                # and with the RAMSES/Arepo dumps (was a fixed μ=1.22).
                muv = 1.0 ./ ((XH+(1-XH)/4) .+ XH.*(xHIIv .- 0.5.*fH2v))
                Tcell = vec(geA) .* muv .* ((5/3-1)*1.6726e-24*vu_cgs^2/1.380649e-16)
                open(joinpath(REPORTS, "enzo_cellcmp$(TAG)_z$(round(Int,z)).bin"), "w") do io
                    write(io, Int64(N))
                    write(io, vec(ρb))                               # gas density (mean Ωb/Ωm)
                    write(io, xHIIv)                                 # x_HII
                    write(io, fH2v)                                  # f_H2
                    write(io, vec(HDI) ./ vec(ρb))                   # f_HD
                    write(io, Tcell)                                 # gas temperature [K] (grackle μ)
                end
            end
            # ── DM particle dump (array order = grafic lattice m=x+y·N+z·N²), for
            #    particle-by-particle Enzo-vs-RAMSES tracking.  pos in box-fraction
            #    [0,1), vel in km/s.  Match to RAMSES by idp / initial position. ──
            if get(ENV, "CIC_PDUMP", "0") == "1"
                vku = 1.22475e7 * BOX * sqrt(OMEGA_M) * sqrt(1+ZSTART) / 1e5   # km/s per code vel
                ppx = EnzoLib.problem_get_particle_pos(h, 0, 0); ppy = EnzoLib.problem_get_particle_pos(h, 1, 0)
                ppz = EnzoLib.problem_get_particle_pos(h, 2, 0)
                pvx = EnzoLib.problem_get_particle_vel(h, 0, 0); pvy = EnzoLib.problem_get_particle_vel(h, 1, 0)
                pvz = EnzoLib.problem_get_particle_vel(h, 2, 0)
                open(joinpath(REPORTS, "enzo_pdump$(TAG)_z$(round(Int,z)).bin"), "w") do io
                    write(io, Int64(length(ppx)))
                    write(io, ppx); write(io, ppy); write(io, ppz)
                    write(io, pvx .* vku); write(io, pvy .* vku); write(io, pvz .* vku)
                end
            end
            # gas temperature diagnostic: e_int (code) × VelocityUnits² → T [K].
            # Enzo's VelocityUnits/TemperatureUnits are CONSTANT — defined once at the
            # INITIAL redshift (CosmologyGetUnits.C uses (1+InitialRedshift), not the
            # current z).  Using sqrt(1+z) here understated Tg by (1+z)/(1+ZSTART) and
            # spuriously looked like the gas thermally decoupling at z~700.
            ge = active_of(EnzoLib.problem_get_field(h, iGE, 0), dims, N)
            vu = 1.22475e7 * BOX * sqrt(OMEGA_M) * sqrt(1+ZSTART)     # cm/s (Enzo vel unit, fixed at z_init)
            Tg = (5/3-1) * (sum(ge)/length(ge)) * 1.22*1.6726e-24 * vu^2 / 1.380649e-16
            xHII = try
                hi = active_of(EnzoLib.problem_get_field(h, EnzoLib.field_index(h, 9; grid=0), 0), dims, N)
                (sum(hi)/length(hi)) / (sum(ρb)/length(ρb)) / 0.76   # n_HII/n_H
            catch; NaN end
            @printf("    [Enzo gas: T_gas=%.1f K  T_cmb=%.1f K  Tg/Tcmb=%.3f  x_HII=%.3e  δb_rms=%.3e δdm_rms=%.3e]\n",
                    Tg, 2.73*(1+z), Tg/(2.73*(1+z)), xHII,
                    std(ρb)/(sum(ρb)/length(ρb)), std(ρd)/(sum(ρd)/length(ρd))); flush(stdout)
            open(joinpath(REPORTS, "enzo_fields$(TAG)_z$(round(Int,z)).bin"), "w") do io
                write(io, Float64.(vec(φ))); write(io, Float64.(vec(ρd)))   # φ, DM density (for x-corr)
            end
            # ── per-cell physical phase dump (ρ/ρ̄, n_H[cm⁻³], T[K], f_H2, x_HII) ──
            # for density PDF + T(ρ)/H2(ρ) phase diagrams vs RAMSES. Same μ=1.22 + γ=5/3
            # + X_H=0.76 conventions both codes use, so the physics is directly comparable.
            let mh=1.6726e-24, kB=1.380649e-16, γ=5/3, XH=0.76
                DU = 1.8788e-29*OMEGA_M*snap.hconst^2*(1+z)^3            # physical g/cm³ per code density
                ρbar = sum(ρb)/length(ρb)
                H2 = active_of(EnzoLib.problem_get_field(h, EnzoLib.field_index(h,14;grid=0), 0), dims, N)
                HI = active_of(EnzoLib.problem_get_field(h, EnzoLib.field_index(h, 9;grid=0), 0), dims, N)
                rrel = vec(ρb) ./ ρbar
                nH   = vec(ρb) .* (DU*XH/mh)
                Tcell= vec(ge) .* ((γ-1)*1.22*mh*vu^2/kB)
                fH2  = (vec(H2) ./ vec(ρb)) ./ XH                        # 2·n_H2/n_H
                xHIIc= (vec(HI) ./ vec(ρb)) ./ XH
                open(joinpath(REPORTS, "enzo_phase$(TAG)_z$(round(Int,z)).bin"), "w") do io
                    write(io, Int64(length(rrel)))
                    write(io, rrel); write(io, nH); write(io, Tcell); write(io, fH2); write(io, xHIIc)
                end
            end
            g2 = (growth_D(a)/D_start)^2
            rec = (z=z, baryon=pk_of(ρb; box_mpc=BOX), dm=pk_of(ρd; box_mpc=BOX), phi=pk_field(φ),
                   theory_b=(k=pk_b_ic.k, P=pk_b_ic.P .* g2),
                   theory_dm=(k=pk_d_ic.k, P=pk_d_ic.P .* g2),
                   theory_b_cic = cic_lin===nothing ? nothing : (k=cic_lin.k, P=cic_lin.Pb .* g2),
                   theory_dm_cic = cic_lin===nothing ? nothing : (k=cic_lin.k, P=cic_lin.Pd .* g2))
            push!(pk_results, rec)
            write_tables()                              # incremental: partial results survive a kill
            @printf("  ● output z=%.2f a=%.5f  (linear growth D²/D₀²=%.3e) [%d outputs written]\n",
                    z, a, g2, length(pk_results))
            flush(stdout)
        end

        # CIC_COMPTON_DRAG=1 → apply the baryon momentum drag each cycle.  Damps toward
        # res.v_cmb (= 0 boosted / streaming-bulk unboosted), so it is frame-correct.
        DO_DRAG = get(ENV, "CIC_COMPTON_DRAG", "0") == "1"
        DRAG_VCMB = hasproperty(res, :v_cmb) ? res.v_cmb : (0.0, 0.0, 0.0)
        # CIC_PERDIR_CFL=1 → override Enzo's sum-over-dims CFL with the per-direction formula
        # (matches RAMSES stepping; correct for Enzo's dimension-split PPM).
        do_perdir = get(ENV, "CIC_PERDIR_CFL", "0") == "1"
        dt_force_next = NaN
        if do_perdir
            tn0 = EnzoLib.session_time(h); ae0, _ = EnzoLib.session_expansion_factor(h, tn0)
            gef0 = active_of(EnzoLib.problem_get_field(h, iGE, 0), dims, N)
            v1f0 = active_of(EnzoLib.problem_get_field(h, iV1, 0), dims, N)
            v2f0 = active_of(EnzoLib.problem_get_field(h, iV2, 0), dims, N)
            v3f0 = active_of(EnzoLib.problem_get_field(h, iV3, 0), dims, N)
            cs0  = sqrt.(max.(5/3 .* (2/3) .* gef0, 0.0))
            s10  = maximum(abs.(v1f0) .+ cs0); s20 = maximum(abs.(v2f0) .+ cs0); s30 = maximum(abs.(v3f0) .+ cs0)
            dt_force_next = ae0 * courant * (1.0/N) / max(s10, s20, s30)
        end
        z_prev = ZSTART
        @printf("%-4s %-8s %-9s %-10s %-8s\n", "cyc", "a_phys", "z", "ρmax", "sec")
        for cyc in 0:MAXCYC-1
            t0 = time()
            # Cap dt so the step lands EXACTLY on the next output a — the Julia frontend
            # honoring CosmologyOutputRedshift like Enzo's own EvolveHierarchy. a_outs is
            # physical; Enzo's internal a is normalized to 1 at z_start ⇒ ×(1+ZSTART).
            mdt = Inf
            if ai <= NOUT_eff
                tn = EnzoLib.session_time(h); ae, dadt = EnzoLib.session_expansion_factor(h, tn)
                aout_e = a_outs[ai]*(1.0+ZSTART)
                (aout_e > ae && dadt > 0) && (mdt = (aout_e - ae)/dadt)
            end
            EnzoLib.evolve_level!(h, 0, 0.0; engine=eng, regrid=false, copy_old=COPYOLD, max_dt=mdt, dt_force=dt_force_next)
            sec = time() - t0; evolvet[] += sec; ncyc[] += 1
            BE === :cuda && RECLAIM_EVERY > 0 && (cyc % RECLAIM_EVERY == 0) && CUDA.reclaim()
            tr = time(); EnzoLib.session_rebuild(h, 0); rebuildt[] += time() - tr
            _, z = EnzoLib.session_cosmology(h)        # Enzo's a is normalized to 1 at z_start;
            a = 1.0 / (1.0 + z)                          # use the PHYSICAL scale factor
            if DO_DRAG                                    # Compton momentum drag over this Δln a
                dlna = log((1.0+z_prev)/(1.0+z)); zmid = 0.5*(z_prev+z)
                # v_CMB is a peculiar velocity → redshifts as 1/a (∝ 1+z), like the DM bulk.
                # (0 in the boosted/CMB-rest frame stays 0; the streaming bulk decays unboosted.)
                vc = DRAG_VCMB .* ((1.0+zmid)/(1.0+ZSTART))
                f = compton_drag!(h, zmid, dlna; v_cmb=vc)
                (cyc % 10 == 0) && @printf("    [compton drag z=%.1f: Γ/H=%.2f  v×%.4f/cyc]\n",
                                           zmid, compton_drag_over_H(zmid), f)
                EnzoLib.session_rebuild(h, 0)            # refresh derived quantities after the kick
            end
            if get(ENV, "CIC_DTDIAG", "0") == "1" || do_perdir
                tn2 = EnzoLib.session_time(h); ae2, da2 = EnzoLib.session_expansion_factor(h, tn2)
                # gas signal speed from current fields (active region only)
                gef = active_of(EnzoLib.problem_get_field(h, iGE, 0), dims, N)
                v1f = active_of(EnzoLib.problem_get_field(h, iV1, 0), dims, N)
                v2f = active_of(EnzoLib.problem_get_field(h, iV2, 0), dims, N)
                v3f = active_of(EnzoLib.problem_get_field(h, iV3, 0), dims, N)
                cs  = sqrt.(max.(5/3 .* (2/3) .* gef, 0.0))   # sound speed in code vel units
                dx  = 1.0 / N
                s1 = maximum(abs.(v1f) .+ cs); s2 = maximum(abs.(v2f) .+ cs); s3 = maximum(abs.(v3f) .+ cs)
                dt_hE = courant * dx / max(s1, s2, s3)                                       # per-dir (Enzo)
                dt_hR = courant * dx / maximum(abs.(v1f) .+ abs.(v2f) .+ abs.(v3f) .+ 3 .* cs)  # sum-dir (RAMSES)
                if do_perdir
                    dt_force_next = ae2 * dt_hE    # per-direction for next step (Enzo code-time units)
                end
                if get(ENV, "CIC_DTDIAG", "0") == "1"
                    a_prev_phys = 1.0 / (1.0 + z_prev)
                    Δaa = 100.0 * (a - a_prev_phys) / a_prev_phys
                    dt_nxt = EnzoLib.session_compute_dt(h, 0)
                    dt_exp = maxexp * ae2 / da2
                    vmax_p = max(maximum(abs.(EnzoLib.problem_get_particle_vel(h, 0, 0))),
                                 maximum(abs.(EnzoLib.problem_get_particle_vel(h, 1, 0))),
                                 maximum(abs.(EnzoLib.problem_get_particle_vel(h, 2, 0))))
                    dt_p = pcourant * dx / vmax_p
                    who = dt_hE <= dt_p && dt_hE <= dt_exp ? "HYDRO" :
                          dt_p  <= dt_hE && dt_p  <= dt_exp ? "PART" : "EXPND"
                    @printf("  [dt c%-3d z=%7.2f Δa/a=%5.2f%%  h_E=%.3e  h_R=%.3e  p=%.3e  e=%.3e  Enz=%.3e →%s]\n",
                            cyc, z, Δaa, dt_hE, dt_hR, dt_p, dt_exp, dt_nxt, who)
                    flush(stdout)
                end
            end
            z_prev = z
            ρ = EnzoLib.problem_get_field(h, iD, 0)
            if cyc % 10 == 0 || z <= ZEND
                @printf("%-4d %-8.5f %-9.3f %-10.3f %-8.2f\n", cyc, a, z, maximum(ρ), sec)
                flush(stdout)
            end
            any(isnan, ρ) && (println("  NaN — abort"); break)
            # record at each crossed log-a checkpoint. In resident mode the particles
            # live on the GPU and are NOT synced per cycle — flush them back to Enzo here
            # so record!'s deposit_particle_density (Enzo's own CIC) sees the live state.
            if ai <= NOUT_eff && a >= a_outs[ai] - 1e-12
                respart !== nothing && MultiCode.sync_to_enzo!(respart, h)
                while ai <= NOUT_eff && a >= a_outs[ai] - 1e-12
                    record!(a, z); ai += 1
                end
                # The output diagnostics (3× power_spectrum_gpu FFTs) transiently balloon
                # the CUDA.jl pool to ~20 GB; the per-cycle reclaim doesn't run here, so the
                # pool stays at that high-water.  Reclaim after each output to keep the PEAK
                # at the ~3 GB working set.  (BE/RECLAIM_EVERY gate as in the cycle loop.)
                BE === :cuda && RECLAIM_EVERY > 0 && CUDA.reclaim()
            end
            z <= ZEND && break
        end
        if ai <= NOUT_eff
            respart !== nothing && MultiCode.sync_to_enzo!(respart, h)
            _, zf = EnzoLib.session_cosmology(h); record!(1.0/(1.0+zf), zf)   # final
        end
    finally
        res.free()
    end

    # ── final table (already written incrementally per checkpoint) ──
    write_tables()
    @printf("\nwrote %d outputs → %s\n", length(pk_results), datafile)
    @printf("TIMING Enzo: evolve=%.1fs  chem=%.1fs  hydro=%.1fs  gravity=%.1fs  particles=%.1fs  other=%.1fs  | ncyc=%d (%.3fs/cyc: hydro %.3f, grav %.3f, part %.3f, chem %.3f)\n",
            evolvet[], chemt[], hydrot[], gravt[], particlet[],
            evolvet[]-chemt[]-hydrot[]-gravt[]-particlet[], ncyc[],
            evolvet[]/max(ncyc[],1), hydrot[]/max(ncyc[],1), gravt[]/max(ncyc[],1), particlet[]/max(ncyc[],1), chemt[]/max(ncyc[],1))
    @printf("  GRAVITY sub-steps (total s): prep(deposit)=%.1f  gmf_copy=%.1f  fft=%.1f  continuation=%.1f  set_potential=%.1f  post(diff)=%.1f\n",
            _GT[:prep][], _GT[:gmf][], _GT[:fft][], _GT[:cont][], _GT[:set][], _GT[:post][])
    @printf("  REBUILD (driver session_rebuild, outside evolve): total=%.1fs  %.3fs/cyc\n",
            rebuildt[], rebuildt[]/max(ncyc[],1))
    if probe !== nothing
        ps = EnzoLib.probe_summary(probe); nc = max(ncyc[],1)
        @printf("  PER-SLOT breakdown (probe, total s | s/cyc):\n")
        for sl in (:boundary, :baryon_copy, :comoving_expansion, :gravity, :hydro, :cooling, :particle_push)
            haskey(ps, sl) || continue
            @printf("    %-20s %.2fs   %.4fs/cyc  (%d calls)\n", sl, ps[sl].total_ns/1e9, ps[sl].total_ns/1e9/nc, ps[sl].calls)
        end
    end
    flush(stdout)
    return datafile
end

main_cic()
