# blockamr_cicass.jl — CICASS high-z cosmology on the BlockAMR hierarchy.
#
# The whole gas problem runs in BlockAMR (level 0 fully tiles the box; dynamic
# refinement above it), in RAMSES super-comoving units where the hydro is PLAIN
# (patch_cosmo.jl): cosmology enters only through (a) the dτ↔a stepping, (b) the
# Poisson source 1.5·Ωm·a·δ, and (c) the per-a unit scalings fed to the
# chemistry.  MultiCode supplies the CICASS ICs, the cosmology, the DM particle
# SoA and the output conventions; gravity reuses the patchgrid topgrid model:
# a global FFT solve of the composite (gas + DM-CIC) δ seeds the level-1
# Dirichlet BCs (phi_from_global!), levels ≥ 1 solve natively (red-black on
# block unions) with per-level DM particle deposits (use_dm).
#
#   BACKEND=cuda BAM_NGRID=32 BAM_LMAX=2 CIC_ZSTART=1000 CIC_ZEND=600 \
#     julia --project=lib/MultiCode/test lib/MultiCode/examples/blockamr_cicass.jl
#
# v1 notes: particle KDK uses the root-step φ for both kicks; P(k) output is
# deferred — the phase dump feeds the H2(ρ) scatter pipeline.  Compton momentum
# DRAG (CIC_COMPTON_DRAG=1, default on): peculiar velocities damp toward the
# global mass-weighted bulk by exp(−Γ/H·Δln a), with the mean x_e refreshed from
# the level-0 species every 10 steps (Compton COOLING lives in the chemistry).

using MultiCode, BlockAMR, CICASSLib, Printf
import PoissonKernels, ChemistryKernels
try; @eval using CUDA; catch; end

const BE      = Symbol(get(ENV, "BACKEND", "cpu"))
const NGRID   = parse(Int, get(ENV, "BAM_NGRID", "32"))
const BBLK    = parse(Int, get(ENV, "BAM_B", "16"))
const LMAX    = parse(Int, get(ENV, "BAM_LMAX", "2"))
const DTHRESH = parse(Float64, get(ENV, "BAM_DTHRESH", "1.5"))   # gas overdensity ρ/ρ̄_b
const REGRIDN = parse(Int, get(ENV, "BAM_REGRID", "4"))
const COMPACT = get(ENV, "BAM_COMPACT", "1") == "1"   # Morton slot compaction at regrid
const CKPTN   = parse(Int, get(ENV, "BAM_CKPT", "0"))     # checkpoint every N root steps (0=off)
const RESTART = get(ENV, "BAM_RESTART", "")               # checkpoint path to resume from
const STOPL   = parse(Int, get(ENV, "BAM_STOPLEVEL", "99"))  # stop once this level populates
const NSWEEP  = parse(Int, get(ENV, "BAM_NSWEEP", "30"))
const LFAC    = parse(Float64, get(ENV, "BAM_LFAC", "4.0"))   # threshold ×lfac per level
# Deepest level receiving DIRECT DM particle deposits.  Topgrid-mass particles
# CIC-deposited at level l are 8^l-amplified point sources (one particle per
# 2^3l fine cells) — beyond l≈1 the shot noise spikes the level Poisson solves
# until the kicks blow up (the z≈15 NaN of the first L4 run).  Deeper levels see
# DM through their Dirichlet boundaries, the correct treatment for particles far
# below their native resolution (gas dominates collapsed cores anyway).
const LDM     = parse(Int, get(ENV, "BAM_LDM", "1"))
const BOXMPCH = parse(Float64, get(ENV, "CIC_BOX", "0.128"))
const ZSTART  = parse(Float64, get(ENV, "CIC_ZSTART", "1000.0"))
const ZEND    = parse(Float64, get(ENV, "CIC_ZEND", "600.0"))
const VBC     = parse(Float64, get(ENV, "CIC_VBC", "0.0"))
const MAXEXP  = parse(Float64, get(ENV, "CIC_MAXEXP", "0.02"))
const OMEGAM  = parse(Float64, get(ENV, "CIC_OMEGAM", "0.27"))
const OMEGAB  = parse(Float64, get(ENV, "CIC_OMEGAB", "0.046"))
const HCONST  = parse(Float64, get(ENV, "CIC_H0", "71.0"))
const GAMMA   = 5.0 / 3.0
const DODRAG  = get(ENV, "CIC_COMPTON_DRAG", "1") == "1"
const TAG     = get(ENV, "CIC_TAG", "_bamr$(NGRID)")
const REPORTS = MultiCode.run_dir("bamr")

function main()
    c = MultiCode.Cosmo(; Om = OMEGAM, OL = 1 - OMEGAM, h0 = HCONST, box = BOXMPCH,
                        Ob = OMEGAB, Or = 4.15e-5 / (HCONST/100)^2)
    a  = MultiCode.z_to_a(ZSTART); a_end = MultiCode.z_to_a(ZEND)
    u0 = MultiCode.cosmo_units(c, a)
    @printf("BlockAMR CICASS: %d³ (B=%d, lmax=%d) box=%.4f Mpc/h  z=%.0f→%.0f  [%s]\n",
            NGRID, BBLK, LMAX, BOXMPCH, ZSTART, ZEND, BE); flush(stdout)

    # ── ICs: CICASS realization, or restart from a checkpoint ──
    μ = 1.22
    local hier, parts, mass_code, xHII0, Np
    if RESTART != ""
    hier, ex = BlockAMR.load_checkpoint(RESTART; backend = BE, Lcap = LMAX)
    devr = v -> BlockAMR.to_device(hier.be, v, Float32)
    parts = (px = devr(ex.px), py = devr(ex.py), pz = devr(ex.pz),
             vx = devr(ex.vx), vy = devr(ex.vy), vz = devr(ex.vz))
    Np = length(ex.px); mass_code = ex.mass_code
    a = ex.a; xHII0 = ex.xe_mean
    @printf("RESTART %s: z=%.3f  nstep=%d  blocks=%s\n", RESTART, 1/a - 1,
            Int(hier.nstep[1]),
            string([length(hier.levels[l+1].live) for l in 0:length(hier.levels)-1]))
    flush(stdout)
    else
    r = MultiCode.run_cicass_streaming(; vbc = VBC, boxlength = BOXMPCH,
                                       zstart = ZSTART, ngrid = NGRID)
    snap = CICASSLib.read_snapshot(r.output)
    ts = CICASSLib.thermal_state(ZSTART)
    xHII0 = ts.x_e; Tg0 = ts.T_gas
    eint0 = Tg0 / ((GAMMA - 1) * μ * u0.T2)
    vconv = 1.0e5 / u0.v

    hier = AMRHierarchy(; nbase = (NGRID, NGRID, NGRID), B = BBLK, backend = BE,
                        T = Float16, nsp = 2, gamma = GAMMA, cfl = 0.3,
                        Lcap = LMAX,
                        scheme = Symbol(get(ENV, "BAM_SCHEME", "rk2")))
    lev0 = init_base_level!(hier); build_level_tables!(hier, 0)

    # gas → level-0 pools (host stage in the pool layout, then encode)
    δb = CICASSLib.grid3d(snap, snap.gas_delta); gv = snap.gas_vel
    n = lev0.cap * lev0.stride
    hD = zeros(n); hS1 = zeros(n); hS2 = zeros(n); hS3 = zeros(n)
    hT = zeros(n); hG = zeros(n)
    for s in lev0.live
        m = lev0.meta[s]; base = (Int(s) - 1) * lev0.stride
        for k in 1:BBLK, j in 1:BBLK, i in 1:BBLK
            gi = Int(m.origin[1]) + i; gj = Int(m.origin[2]) + j; gk = Int(m.origin[3]) + k
            mm = gi + (gj - 1) * NGRID + (gk - 1) * NGRID^2
            ρ  = c.fb * (1 + δb[gi, gj, gk])
            v1 = gv[mm, 1] * vconv; v2 = gv[mm, 2] * vconv; v3 = gv[mm, 3] * vconv
            idx = base + ((lev0.ng+k-1) * lev0.nd + (lev0.ng+j-1)) * lev0.nd + (lev0.ng+i-1) + 1
            hD[idx] = ρ; hS1[idx] = ρ*v1; hS2[idx] = ρ*v2; hS3[idx] = ρ*v3
            hG[idx] = ρ * eint0
            hT[idx] = ρ * (eint0 + 0.5 * (v1^2 + v2^2 + v3^2))
        end
    end
    BlockAMR.encode_from_host!(lev0, hD, hS1, hS2, hS3, hT, hG)
    copyto!(lev0.sp[1], fill(ChemistryKernels.encode_log2sp(Float32(xHII0 * c.XH)), n))
    copyto!(lev0.sp[2], fill(ChemistryKernels.encode_log2sp(Float32(2e-6 * c.XH)), n))

    # DM particle SoA (device, f32) — mass (1−fb)/Np ⇒ mean DM density 1−fb
    pos = snap.dm_pos; vel = snap.dm_vel; Np = size(pos, 1)
    dev(v) = BlockAMR.to_device(hier.be, v, Float32)
    parts = (px = dev(Float32[mod(pos[p,1], 1.0) for p in 1:Np]),
             py = dev(Float32[mod(pos[p,2], 1.0) for p in 1:Np]),
             pz = dev(Float32[mod(pos[p,3], 1.0) for p in 1:Np]),
             vx = dev(Float32[vel[p,1] * vconv for p in 1:Np]),
             vy = dev(Float32[vel[p,2] * vconv for p in 1:Np]),
             vz = dev(Float32[vel[p,3] * vconv for p in 1:Np]))
    mass_code = (1 - c.fb) / Np
    @printf("gas IC: f_b=%.4f T=%.1fK x_HII=%.3e;  DM: %d particles\n",
            c.fb, Tg0, xHII0, Np); flush(stdout)
    end                                          # if RESTART

    pol = BlockRefinementPolicy(; dthresh = DTHRESH * c.fb, nbuf = 2,
                                every = REGRIDN, lmax = LMAX, lfac = LFAC)
    ρg = BlockAMR.device_zeros(hier.be, Float32, (NGRID^3,))
    φg = BlockAMR.device_zeros(hier.be, Float32, (NGRID^3,))
    pax = BlockAMR.device_zeros(hier.be, Float32, (Np,))
    pay = BlockAMR.device_zeros(hier.be, Float32, (Np,))
    paz = BlockAMR.device_zeros(hier.be, Float32, (Np,))

    nstep = Int(hier.nstep[1]); t0 = time(); xe_mean = xHII0
    while a < a_end
        # ── timestep: hydro CFL (global λ), particle CFL, expansion cap ──
        λ = compute_lambda!(hier)
        vp = max(maximum(x -> abs(Float64(x)), parts.vx),
                 maximum(x -> abs(Float64(x)), parts.vy),
                 maximum(x -> abs(Float64(x)), parts.vz))
        dx0 = BlockAMR.level_dx(hier, 0)
        dτ = min(Float64(λ) * dx0, 0.3 * dx0 / max(vp, 1e-30),
                 MultiCode.dtau_for_dlna(c, a, MAXEXP))
        hier.λ = Float32(dτ / dx0)

        # ── gravity: composite topgrid FFT (gas from level 0 + DM CIC), then
        #    the native level solves via the selfgrav slot ──
        fill!(ρg, 0.0f0)
        global_from_level0!(hier, ρg)
        PoissonKernels.cic_deposit!(ρg, parts.px, parts.py, parts.pz,
                                    parts.vx, parts.vy, parts.vz,
                                    Float32(mass_code * NGRID^3);
                                    N = NGRID, disp = 0.0f0, shift = -0.5f0)
        ρg .-= 1.0f0                                     # δ (total mean = 1)
        if get(ENV, "BAM_CHECKNAN", "0") == "1"
            hρ = Array(ρg); nbad = count(!isfinite, hρ)
            vpx = maximum(abs, Array(parts.vx)); vpz = maximum(abs, Array(parts.vz))
            if nbad > 0 || !isfinite(vpx) || !isfinite(vpz)
                @printf("SRCHUNT step %d: rhog nonfinite=%d  max|vx|=%.3e max|vz|=%.3e  rhog max=%.3e min=%.3e\n",
                        nstep, nbad, vpx, vpz, maximum(hρ), minimum(hρ)); flush(stdout)
                nbad > 0 && error("nonfinite gravity source at step $nstep")
            end
        end
        ρ3 = reshape(ρg, NGRID, NGRID, NGRID)
        φ3 = reshape(φg, NGRID, NGRID, NGRID)
        PoissonKernels.fft_poisson_rfft!(φ3, ρ3; G = 1.5 * c.Om * a, a = 1.0,
                                         boxsize = 1.0)
        phi_from_global!(hier, φg)
        if get(ENV, "BAM_CHECKNAN", "0") == "1"
            # ordered probes: particle state → (deposits) → φ-live → (gather)
            mpx = maximum(Array(parts.px)); mvx = maximum(abs, Array(parts.vx))
            (isfinite(mpx) && isfinite(mvx)) ||
                (@printf("PHUNT step %d: PARTICLE STATE nonfinite pre-deposit (px=%.3e vx=%.3e)\n",
                         nstep, mpx, mvx); flush(stdout); error("particle state"))
            for l in 0:length(hier.levels)-1
                lv = hier.levels[l+1]; isempty(lv.live) && continue
                hph = Array(lv.phi); bad = 0
                for s_ in lv.live
                    b_ = (Int(s_) - 1) * lv.stride
                    for cix in 1:97:lv.nd^3
                        isfinite(hph[b_ + cix]) || (bad += 1)
                    end
                end
                bad > 0 && (@printf("PHUNT step %d: level %d LIVE phi nonfinite (strided hits=%d) PRE-advance\n",
                                    nstep, l, bad); flush(stdout); error("phi pre-advance"))
            end
        end
        for l in 1:min(length(hier.levels) - 1, LDM)
            isempty(hier.levels[l+1].live) && continue
            deposit_particles_level!(hier, l, parts; mass_code)
        end

        # ── one root step: hydro + native gravity + chem on every level ──
        u = MultiCode.cosmo_units(c, a)
        if get(ENV, "BAM_DISSECT", "") == string(nstep + 1)
            # replicate advance_level_w!(0) stage by stage with finite probes
            λv = hier.λ
            sg = (coef = 1.5 * c.Om * a, nsweep = NSWEEP, nsweep0 = 0,
                  rho_mean = 1.0, use_dm = true)
            ch = (a_value = a, density_units = u.d, length_units = u.l,
                  time_units = u.t, hubble = c.h0, Om = c.Om, OL = c.OL,
                  fh = c.XH)
            lev0d = hier.levels[1]
            probe(tag) = begin
                for lp in 0:length(hier.levels)-1
                    lvp = hier.levels[lp + 1]
                    isempty(lvp.live) && continue
                    sm = BlockAMR.max_signal(lvp, hier.gamma)
                    nT = count(x -> !isfinite(Float32(x)), Array(lvp.Tau))
                    nD = count(x -> !isfinite(Float32(x)), Array(lvp.D))
                    (isfinite(sm) && nT == 0 && nD == 0) ||
                        @printf("DISSECT[%s]: L%d smax=%s Tau-bad=%d D-bad=%d\n",
                                tag, lp, string(sm), nT, nD)
                end
                @printf("DISSECT[%s]: done\n", tag); flush(stdout)
            end
            probe("start")
            # decompose level 1's FIRST pass: level-2 substeps inline with probes
            lev1d = hier.levels[2]; lev2d = hier.levels[3]
            dt1 = Float64(λv) * BlockAMR.level_dx(hier, 1)
            dt2 = Float64(λv) * BlockAMR.level_dx(hier, 2)
            for pass2 in 1:2
                BlockAMR.solve_gravity_level!(hier, 2; source_coef = sg.coef,
                    nsweep = NSWEEP, rho_mean = 1.0, use_dm = true, residual = false)
                probe("L2p$(pass2)-solve(phi-only)")
                BlockAMR.grav_kick_level_pool!(hier, 2, 0.5 * dt2)
                probe("L2p$(pass2)-K1")
                BlockAMR.fill_ghosts!(hier, 2; θ = 0.0f0, buf = :R)
                probe("L2p$(pass2)-ghosts")
                BlockAMR.ctu_level!(hier, 2, λv)
                BlockAMR.capture_fine!(hier, 2, :R, 0.5f0; λ = λv)
                BlockAMR.swap_buffers!(lev2d)
                probe("L2p$(pass2)-ctu")
                BlockAMR.grav_kick_level_pool!(hier, 2, 0.5 * dt2)
                probe("L2p$(pass2)-K2")
                BlockAMR.chem_level!(hier, 2, dt2; ch...)
                probe("L2p$(pass2)-chem")
                hier.nstep[3] += 1
            end
            BlockAMR.solve_gravity_level!(hier, 1; source_coef = sg.coef,
                nsweep = NSWEEP, rho_mean = 1.0, use_dm = true, residual = false)
            BlockAMR.grav_kick_level_pool!(hier, 1, 0.5 * dt1)
            probe("L1-K1")
            BlockAMR.fill_ghosts!(hier, 1; θ = 0.0f0, buf = :R)
            BlockAMR.ctu_level!(hier, 1, λv)
            BlockAMR.capture_fine!(hier, 1, :R, 0.5f0; λ = λv)
            BlockAMR.capture_coarse!(hier, 2, :R, 1.0f0; λ = λv)
            BlockAMR.swap_buffers!(lev1d)
            probe("L1-ctu")
            BlockAMR.grav_kick_level_pool!(hier, 1, 0.5 * dt1)
            BlockAMR.chem_level!(hier, 1, dt1; ch...)
            probe("L1-K2chem")
            BlockAMR.restrict_level!(hier, 2)
            BlockAMR.reflux_apply!(hier, 2, λv)
            hier.nstep[2] += 1
            probe("child-pass-1")
            BlockAMR.advance_level_w!(hier, 1, λv; φ = nothing, chem = ch, selfgrav = sg)
            probe("child-pass-2")
            dt0 = Float64(λv) * BlockAMR.level_dx(hier, 0)
            BlockAMR.solve_gravity_level!(hier, 0; source_coef = sg.coef, nsweep = 0,
                                          rho_mean = 1.0, use_dm = true, residual = false)
            BlockAMR.grav_kick_level_pool!(hier, 0, 0.5 * dt0)
            probe("K1")
            BlockAMR.fill_ghosts!(hier, 0; θ = 0.0f0, buf = :R)
            probe("ghosts")
            BlockAMR.ctu_level!(hier, 0, λv)
            hDo = Array(lev0d.Do); hTo = Array(lev0d.Tauo)
            @printf("DISSECT[ctu-O]: Do-nonfinite=%d Tauo-nonfinite=%d\n",
                    count(x -> !isfinite(Float32(x)), hDo),
                    count(x -> !isfinite(Float32(x)), hTo)); flush(stdout)
            BlockAMR.capture_coarse!(hier, 1, :R, 1.0f0; λ = λv)
            BlockAMR.swap_buffers!(lev0d)
            probe("ctu+swap")
            BlockAMR.grav_kick_level_pool!(hier, 0, 0.5 * dt0)
            probe("K2")
            BlockAMR.chem_level!(hier, 0, dt0; ch...)
            probe("chem")
            BlockAMR.restrict_level!(hier, 1)
            probe("restrict")
            BlockAMR.reflux_apply!(hier, 1, λv)
            probe("reflux")
            hier.nstep[1] += 1
            error("DISSECT complete at step $(nstep + 1)")
        end
        BlockAMR.advance_level_w!(hier, 0, hier.λ;
            selfgrav = (coef = 1.5 * c.Om * a, nsweep = NSWEEP, nsweep0 = 0,
                        rho_mean = 1.0, use_dm = true),
            chem = get(ENV, "BAM_NOCHEM", "0") == "1" ? nothing :
                   (a_value = a, density_units = u.d, length_units = u.l,
                    time_units = u.t, hubble = c.h0, Om = c.Om, OL = c.OL,
                    fh = c.XH))
        # level 0 uses the FFT φ pool directly (nsweep0=0 keeps its Dirichlet-free
        # RB pass a no-op refinement of the FFT solution)

        # ── particles: KDK from the finest covering level's φ ──
        if get(ENV, "BAM_CHECKNAN", "0") == "1"
            for l in 0:length(hier.levels)-1
                isempty(hier.levels[l+1].live) && continue
                sm2 = BlockAMR.max_signal(hier.levels[l+1], hier.gamma)
                isfinite(sm2) ||
                    (@printf("PHUNT step %d: GAS nonfinite at level %d POST-advance (smax=%s)\n",
                             nstep, l, string(sm2)); flush(stdout); error("gas post-advance"))
            end
        end
        gather_accel_particles!(hier, parts, pax, pay, paz)
        if get(ENV, "BAM_CHECKNAN", "0") == "1"
            ma = maximum(abs, Array(pax)); mb = maximum(abs, Array(pay))
            mc2 = maximum(abs, Array(paz))
            if !(isfinite(ma) && isfinite(mb) && isfinite(mc2))
                @printf("PHUNT step %d: gather accel nonfinite  |ax|=%.3e |ay|=%.3e |az|=%.3e\n",
                        nstep, ma, mb, mc2); flush(stdout)
                # locate: which level's phi is poisoned?
                for l in 0:length(hier.levels)-1
                    lv = hier.levels[l+1]; isempty(lv.live) && continue
                    nb2 = count(!isfinite, Array(lv.phi))
                    nb2 > 0 && @printf("PHUNT   level %d phi nonfinite=%d (of %d)\n",
                                       l, nb2, length(lv.phi))
                end
                error("nonfinite gather at step $nstep")
            end
        end
        particles_kick!(hier, parts, pax, pay, paz, 0.5 * dτ)
        particles_drift!(hier, parts, dτ)
        particles_kick!(hier, parts, pax, pay, paz, 0.5 * dτ)
        if get(ENV, "BAM_CHECKNAN", "0") == "1"
            mv2 = maximum(abs, Array(parts.vx))
            isfinite(mv2) || (@printf("PHUNT step %d: vx nonfinite AFTER kicks\n", nstep);
                              flush(stdout); error("nonfinite vx at step $nstep"))
        end

        # ── Compton momentum drag (frame-agnostic, exp(−Γ/H·Δln a)) ──
        if DODRAG
            dlna = MultiCode.dadtau(c, a) * dτ / a
            gam = MultiCode.compton_drag_over_H(c, 1/a - 1, xe_mean)
            compton_drag!(hier, exp(-gam * dlna); scratch = ρg)
        end

        # ── a-advance (RK2 midpoint on da/dτ) + maintenance ──
        k1 = MultiCode.dadtau(c, a); amid = a + 0.5 * k1 * dτ
        a = min(a + MultiCode.dadtau(c, amid) * dτ, a_end)
        nstep += 1
        if nstep % 10 == 0                             # refresh the mean x_e estimate
            h1s = Array(hier.levels[1].sp[1])
            acc = 0.0; cnt = 0
            for s_ in hier.levels[1].live
                b_ = (Int(s_) - 1) * hier.levels[1].stride
                for cix in 1:64:hier.levels[1].nd^3    # strided subsample
                    acc += ChemistryKernels.decode_log2sp(Float64, h1s[b_ + cix]); cnt += 1
                end
            end
            xe_mean = max(acc / cnt / c.XH, 1e-6)
        end
        for l in 0:length(hier.levels)-1
            update_scales!(hier, l)
        end
        nstep % REGRIDN == 0 && regrid!(hier, pol; compact = COMPACT)
        ckextra() = (a = a, mass_code = mass_code, xe_mean = xe_mean,
                     px = Array(parts.px), py = Array(parts.py), pz = Array(parts.pz),
                     vx = Array(parts.vx), vy = Array(parts.vy), vz = Array(parts.vz))
        if CKPTN > 0 && nstep % CKPTN == 0
            ckf = joinpath(REPORTS, "bamr_ckpt$(TAG)_" *
                           (isodd(nstep ÷ CKPTN) ? "A" : "B") * ".jls")
            BlockAMR.save_checkpoint(ckf, hier; extra = ckextra())
            @printf("ckpt @ step %d z=%.3f → %s\n", nstep, 1/a - 1, ckf); flush(stdout)
        end
        if STOPL < 99 && length(hier.levels) > STOPL &&
           !isempty(hier.levels[STOPL + 1].live)
            @printf("LEVEL %d POPULATED at z=%.4f (blocks=%s) — stopping\n", STOPL,
                    1/a - 1,
                    string([length(hier.levels[l+1].live) for l in 0:length(hier.levels)-1]))
            BlockAMR.save_checkpoint(joinpath(REPORTS, "bamr_ckpt$(TAG)_L$(STOPL).jls"),
                                     hier; extra = ckextra())
            break
        end
        if get(ENV, "BAM_CHECKNAN", "0") == "1"
            for l in 0:length(hier.levels)-1
                isempty(hier.levels[l+1].live) && continue
                sm = BlockAMR.max_signal(hier.levels[l+1], hier.gamma)
                if !isfinite(sm)
                    lev = hier.levels[l+1]
                    hDn = Array(lev.D); hGn = Array(lev.Ge); hTn = Array(lev.Tau); hSn = Array(lev.S1)
                    nb_ = count(!isfinite, Float32.(hDn)); ng_ = count(!isfinite, Float32.(hGn))
                    nt_ = count(!isfinite, Float32.(hTn)); ns_ = count(!isfinite, Float32.(hSn))
                    @printf("NANHUNT step %d z=%.3f level %d: smax=%s  nonfinite D=%d Ge=%d Tau=%d S1=%d  blocks=%d\n",
                            nstep, 1/a - 1, l, string(sm), nb_, ng_, nt_, ns_, length(lev.live))
                    # locate up to 8 offenders: block origin, local cell, C/F adjacency
                    shown = 0
                    flev = l + 2 <= length(hier.levels) ? hier.levels[l + 2] : nothing
                    for s_ in lev.live
                        shown >= 8 && break
                        m_ = lev.meta[s_]; base_ = (Int(s_) - 1) * lev.stride
                        for kk in 1:lev.B, jj in 1:lev.B, ii in 1:lev.B
                            idx_ = base_ + ((lev.ng+kk-1)*lev.nd + (lev.ng+jj-1))*lev.nd + (lev.ng+ii-1) + 1
                            if !isfinite(Float32(hGn[idx_])) || !isfinite(Float32(hTn[idx_]))
                                g_ = (Int(m_.origin[1]) + ii - 1, Int(m_.origin[2]) + jj - 1,
                                      Int(m_.origin[3]) + kk - 1)
                                cov = flev !== nothing && !isempty(flev.live) &&
                                      !isempty(overlapping_blocks(flev,
                                          (Int128(2g_[1]), Int128(2g_[2]), Int128(2g_[3])), (2,2,2)))
                                nbcf = flev !== nothing && !isempty(flev.live) &&
                                       !isempty(overlapping_blocks(flev,
                                          (Int128(2g_[1]-2), Int128(2g_[2]-2), Int128(2g_[3]-2)), (6,6,6)))
                                @printf("  cell %s D=%.4g Ge=%s Tau=%s covered=%s nearCF=%s\n",
                                        string(g_), Float32(hDn[idx_]) * Array(lev.Dsc)[s_],
                                        string(Float32(hGn[idx_])), string(Float32(hTn[idx_])),
                                        string(cov), string(nbcf))
                                shown += 1; shown >= 8 && break
                            end
                        end
                    end
                    flush(stdout)
                    error("NaN detected at level $l step $nstep")
                end
            end
        end
        if nstep % 10 == 0
            nb = [length(hier.levels[l+1].live) for l in 0:length(hier.levels)-1]
            @printf("step %4d  z=%8.2f  dτ=%.3e  blocks=%s  λ=%.3e\n",
                    nstep, 1/a - 1, dτ, string(nb), hier.λ); flush(stdout)
        end
    end
    if !isfinite(a)
        @printf("ABORTED: a went nonfinite at step %d — no dumps\n", nstep)
        return nothing
    end
    @printf("done: %d root steps to z=%.1f in %.1f s\n", nstep, 1/a - 1, time() - t0)

    # ── phase dump (composite level-0 after restriction) for the H2(ρ) pipeline ──
    lev = hier.levels[1]
    hDf = Array(lev.D); hGf = Array(lev.Ge); hsc = Array(lev.Dsc); hesc = Array(lev.Esc)
    h1 = Array(lev.sp[1]); h2 = Array(lev.sp[2])
    u = MultiCode.cosmo_units(c, a)
    NC = NGRID^3
    ρb = zeros(NC); xh = zeros(NC); fh2 = zeros(NC); Tc = zeros(NC)
    for s in lev.live
        m = lev.meta[s]; base = (Int(s) - 1) * lev.stride
        for k in 1:BBLK, j in 1:BBLK, i in 1:BBLK
            gi = Int(m.origin[1]) + i; gj = Int(m.origin[2]) + j; gk = Int(m.origin[3]) + k
            gidx = gi + (gj - 1) * NGRID + (gk - 1) * NGRID^2
            idx = base + ((lev.ng+k-1) * lev.nd + (lev.ng+j-1)) * lev.nd + (lev.ng+i-1) + 1
            ρ = Float64(hDf[idx]) * hsc[s]
            e = Float64(hGf[idx]) * hesc[s] / max(ρ, 1e-30)
            ρb[gidx] = ρ
            Tc[gidx] = e * (GAMMA - 1) * μ * u.T2
            xh[gidx]  = ChemistryKernels.decode_log2sp(Float64, h1[idx]) / c.XH
            fh2[gidx] = ChemistryKernels.decode_log2sp(Float64, h2[idx]) / c.XH
        end
    end
    zi = round(Int, 1/a - 1)
    MultiCode.write_grid(joinpath(REPORTS, "bamr_cellcmp$(TAG)_z$(zi).bin");
        kind = "bamr_cellcmp", n = NGRID, ndim = 3,
        columns = ["rho" => ρb, "xHII" => xh, "fH2" => fh2, "fHD" => zeros(NC),
                   "T" => Tc])
    @printf("z=%d: <x_HII>=%.3e <f_H2>=%.3e <T>=%.1fK  → %s\n",
            zi, sum(xh)/NC, sum(fh2)/NC, sum(Tc)/NC, REPORTS); flush(stdout)

    # ── leaf-cell dump around the density peak (ALL levels, finest covering) ──
    # Columns (Int64 n header + 8 × Float64[n]): x,y,z,dx in box units (exact
    # from the integer origins), rho (code, level-0 mean ≈ 1), x_HII, f_H2, T[K].
    # Coverage is per OCTANT (regrid children are always half-block octants).
    if get(ENV, "BAM_PROFDUMP", "1") == "1"
        Rp = parse(Float64, get(ENV, "BAM_PROFR", "0.1875"))      # box units
        pg = argmax(ρb) - 1
        pc = (((pg % NGRID) + 0.5) / NGRID, (((pg ÷ NGRID) % NGRID) + 0.5) / NGRID,
              ((pg ÷ NGRID^2) + 0.5) / NGRID)
        cols = ntuple(_ -> Float64[], 8)                # x,y,z,dx,rho,xHII,fH2,T
        for l in 0:length(hier.levels)-1
            lv = hier.levels[l + 1]
            isempty(lv.live) && continue
            Nl = NGRID * 2^l
            cov = Dict{Int32,UInt8}()                   # parent slot → covered octants
            if l + 2 <= length(hier.levels)
                for cs in hier.levels[l + 2].live
                    cm = hier.levels[l + 2].meta[cs]
                    ob = (cm.offset[1] > 0 ? 1 : 0) | ((cm.offset[2] > 0 ? 1 : 0) << 1) |
                         ((cm.offset[3] > 0 ? 1 : 0) << 2)
                    cov[cm.parent] = get(cov, cm.parent, 0x00) | (UInt8(1) << ob)
                end
            end
            hD = Array(lv.D); hG = Array(lv.Ge)
            hs1 = Array(lv.sp[1]); hs2 = Array(lv.sp[2])
            hdsc = Array(lv.Dsc); hesc = Array(lv.Esc)
            Bh = BBLK ÷ 2
            for s in lv.live
                m = lv.meta[s]
                bc = ntuple(d -> (Float64(m.origin[d]) + BBLK / 2) / Nl, 3)
                bd = sqrt(sum(d -> (mod(bc[d] - pc[d] + 0.5, 1.0) - 0.5)^2, 1:3))
                bd > Rp + BBLK * 0.87 / Nl && continue
                cb = get(cov, s, 0x00)
                base = (Int(s) - 1) * lv.stride
                for k in 1:BBLK, j in 1:BBLK, i in 1:BBLK
                    ob = (i > Bh ? 1 : 0) | ((j > Bh ? 1 : 0) << 1) | ((k > Bh ? 1 : 0) << 2)
                    (cb >> ob) & 0x01 == 0x01 && continue        # covered by a child
                    x = (Float64(m.origin[1]) + i - 0.5) / Nl
                    y = (Float64(m.origin[2]) + j - 0.5) / Nl
                    zc = (Float64(m.origin[3]) + k - 0.5) / Nl
                    # cut by the CONTAINING level-0 cell center: the leaf set
                    # is then the exact complement of the level-0 dump's r>Rp
                    # exterior (no boundary double-count/gap in the profile)
                    r0 = sqrt(sum(q -> (mod((floor((x, y, zc)[q] * NGRID) + 0.5) / NGRID -
                                            pc[q] + 0.5, 1.0) - 0.5)^2, 1:3))
                    r0 > Rp && continue
                    idx = base + ((lv.ng+k-1) * lv.nd + (lv.ng+j-1)) * lv.nd + (lv.ng+i-1) + 1
                    ρ = Float64(hD[idx]) * hdsc[s]
                    e = Float64(hG[idx]) * hesc[s] / max(ρ, 1e-30)
                    push!(cols[1], x); push!(cols[2], y); push!(cols[3], zc)
                    push!(cols[4], 1.0 / Nl); push!(cols[5], ρ)
                    push!(cols[6], ChemistryKernels.decode_log2sp(Float64, hs1[idx]) / c.XH)
                    push!(cols[7], ChemistryKernels.decode_log2sp(Float64, hs2[idx]) / c.XH)
                    push!(cols[8], e * (GAMMA - 1) * μ * u.T2)
                end
            end
        end
        open(joinpath(REPORTS, "bamr_leafprof$(TAG)_z$(zi).bin"), "w") do io
            write(io, Int64(length(cols[1])))
            for col in cols
                write(io, col)
            end
        end
        @printf("leafprof: %d leaf cells within R=%.4f of peak (%.4f,%.4f,%.4f)\n",
                length(cols[1]), Rp, pc[1], pc[2], pc[3]); flush(stdout)
    end
end

import Profile
if get(ENV, "BAM_JLPROF", "0") == "1"
    Profile.init(n = 10^7, delay = 0.001)
    Profile.@profile main()
    open("/tmp/claude-1002/bamr_jlprof.txt", "w") do io
        Profile.print(IOContext(io, :displaysize => (2000, 240));
                      format = :flat, sortedby = :count, mincount = 30)
    end
else
    main()
end
