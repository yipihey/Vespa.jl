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

    # ── ICs: CICASS realization (gas δ,v + DM positions/velocities) ──
    r = MultiCode.run_cicass_streaming(; vbc = VBC, boxlength = BOXMPCH,
                                       zstart = ZSTART, ngrid = NGRID)
    snap = CICASSLib.read_snapshot(r.output)
    ts = CICASSLib.thermal_state(ZSTART)
    xHII0 = ts.x_e; Tg0 = ts.T_gas
    μ = 1.22
    eint0 = Tg0 / ((GAMMA - 1) * μ * u0.T2)
    vconv = 1.0e5 / u0.v

    hier = AMRHierarchy(; nbase = (NGRID, NGRID, NGRID), B = BBLK, backend = BE,
                        T = Float16, nsp = 2, gamma = GAMMA, cfl = 0.3,
                        Lcap = LMAX)
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

    pol = BlockRefinementPolicy(; dthresh = DTHRESH * c.fb, nbuf = 2,
                                every = REGRIDN, lmax = LMAX, lfac = LFAC)
    ρg = BlockAMR.device_zeros(hier.be, Float32, (NGRID^3,))
    φg = BlockAMR.device_zeros(hier.be, Float32, (NGRID^3,))
    pax = BlockAMR.device_zeros(hier.be, Float32, (Np,))
    pay = BlockAMR.device_zeros(hier.be, Float32, (Np,))
    paz = BlockAMR.device_zeros(hier.be, Float32, (Np,))

    nstep = 0; t0 = time(); xe_mean = xHII0
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
        ρ3 = reshape(ρg, NGRID, NGRID, NGRID)
        φ3 = reshape(φg, NGRID, NGRID, NGRID)
        PoissonKernels.fft_poisson_rfft!(φ3, ρ3; G = 1.5 * c.Om * a, a = 1.0,
                                         boxsize = 1.0)
        phi_from_global!(hier, φg)
        for l in 1:min(length(hier.levels) - 1, LDM)
            isempty(hier.levels[l+1].live) && continue
            deposit_particles_level!(hier, l, parts; mass_code)
        end

        # ── one root step: hydro + native gravity + chem on every level ──
        u = MultiCode.cosmo_units(c, a)
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
        gather_accel_particles!(hier, parts, pax, pay, paz)
        particles_kick!(hier, parts, pax, pay, paz, 0.5 * dτ)
        particles_drift!(hier, parts, dτ)
        particles_kick!(hier, parts, pax, pay, paz, 0.5 * dτ)

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
end

main()
