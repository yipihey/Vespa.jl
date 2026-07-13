# blockamr_be.jl — Bonnor-Ebert spherical-collapse test on BlockAMR (Enzo's
# SphereType 6 / SphericalCollapse setup), scaled to PRIMORDIAL gas, run with the
# real H+H₂ chemistry and cooling (ChemistryKernels).  Unlike the BB93 rotating
# test (blockamr_bb.jl), this is SPHERICALLY SYMMETRIC and at rest: an isothermal
# Lane-Emden (Bonnor-Ebert) sphere made supercritical by an overdensity factor,
# collapsing to a SINGLE center with NO rotation, bar, or fragmentation.  That
# gives a clean, reproducible T–n phase track for comparing the chemistry paths
# (analytic reduced vs full network) in a real hydro+gravity+AMR collapse — the
# fragmentation/f16-cold-bleed noise of the BB run is gone.
#
# Density: ρ(ξ)=ρ_c·BE(ξ), ξ=r·ξ_max/R_sphere, BE = Enzo's polynomial fit to the
# isothermal Lane-Emden solution; ξ_max=6.5 (contrast ~15, near-critical).  The
# sphere radius follows the BE structure R=ξ_max·a, a=c_s/√(4πGρ_c).  Ambient =
# pressure-continuous edge density.  FOVER>1 makes it supercritical → collapse.
#
# NON-cosmological: a=1 fixed.  Units: ρ_u=ρ_c, L_u=box, t_u=1/√(4πGρ_c) ⇒ Poisson
# coef=1, gravity kick = natural code form (dv=−∇φ·dt).
#
# Run (CUDA): BACKEND=cuda BAM_NGRID=64 BAM_LMAX=8 BAM_MEM=managed BAM_ZOOMR=0 \
#   BAM_NH=100 BAM_TINIT=2000 BAM_JEANS=32 BAM_STOPLEVEL=6 BAM_BE_FOVER=1.1 \
#   julia --project=test -t auto examples/blockamr_be.jl

using Printf
import BlockAMR
using BlockAMR: AMRHierarchy, init_base_level!, build_level_tables!,
                BlockRefinementPolicy, regrid!, advance_hierarchy!, compute_lambda!,
                max_signal, update_scales!, global_from_level0!, phi_from_global!,
                device_zeros, to_device, enable_timers!, reset_timers!, timer_report
import MultiCode, PoissonKernels, ChemistryKernels
include(joinpath(@__DIR__, "jeans_profile.jl"))
get(ENV, "BACKEND", "cpu") == "cuda" && @eval using CUDA   # register the :cuda backend

const BE      = get(ENV, "BACKEND", "cpu") == "cuda" ? :cuda : :cpu
const NGRID   = parse(Int,   get(ENV, "BAM_NGRID", "64"))
const BBLK    = parse(Int,   get(ENV, "BAM_B", "16"))
const LMAX    = parse(Int,   get(ENV, "BAM_LMAX", "8"))
const MEMMODE = Symbol(get(ENV, "BAM_MEM", "device"))
const NSWEEP  = parse(Int,   get(ENV, "BAM_NSWEEP", "20"))
const REGRIDN = parse(Int,   get(ENV, "BAM_REGRID", "4"))
const JEANS   = parse(Int,   get(ENV, "BAM_JEANS", "64"))     # cells per Jeans length
const JRHO    = parse(Float64, get(ENV, "BAM_JEANS_RHO", "3.0"))  # Jeans gate: ρ>JRHO·ρ̄
const DTHRESH = parse(Float64, get(ENV, "BAM_DTHRESH", "3.0"))
const DTFF    = parse(Float64, get(ENV, "BAM_DTFF", "0.05"))  # free-fall dt safety
const STOPL   = parse(Int,   get(ENV, "BAM_STOPLEVEL", "99"))
const CKPTN   = parse(Int,   get(ENV, "BAM_CKPT", "0"))
const NROOT   = parse(Int,   get(ENV, "BAM_NROOT", "100000"))
const TEND_FF = parse(Float64, get(ENV, "BAM_TEND_FF", "2.0")) # stop at this many t_ff
const SLICEN  = parse(Int,   get(ENV, "BAM_SLICE", "20"))    # midplane density slice every N steps (0=off)
const TIMERN  = parse(Int,   get(ENV, "BAM_TIMERS", "0"))    # phase-timing breakdown every N steps (0=off)
const CHEM_ITCAP  = parse(Int,     get(ENV, "BAM_CHEM_ITCAP", "5000"))   # chem subcycle cap/cell/call
const CHEM_DTFRAC = parse(Float64, get(ENV, "BAM_CHEM_DTFRAC", "0.1"))   # chem substep target (frac change)
const CHEM_NHSW   = parse(Float64, get(ENV, "BAM_CHEM_NHSWITCH", "0.0")) # n_H [cm⁻³] above which cells use the FULL network (0=off, analytic everywhere)
const TAG     = get(ENV, "CIC_TAG", "_be")
const PREC    = get(ENV, "BAM_PREC", "f16") == "f32" ? Float32 : Float16  # gas field precision
const REPORTS = MultiCode.run_dir("beamr")

# ── physical constants (cgs) + the Bonnor-Ebert scaling ───────────────────────
const kB = 1.380649e-16; const mH = 1.6726219e-24; const Gcgs = 6.674e-8
const Msun = 1.989e33; const pc = 3.0857e18
const μ = 1.22; const XH = 0.76; const γ = 5.0/3.0
const NH0   = parse(Float64, get(ENV, "BAM_NH", "100.0"))     # cm^-3
const TINIT = parse(Float64, get(ENV, "BAM_TINIT", "2000.0")) # K
# Bonnor-Ebert sphere (Enzo SphereType 6): isothermal Lane-Emden profile, at rest,
# made supercritical by an overdensity factor to trigger a spherically-symmetric
# collapse (no rotation, no m=2 bar → single center, no fragmentation, clean T–n track).
const XIMAX  = parse(Float64, get(ENV, "BAM_BE_XIMAX",  "6.5")) # dimensionless cutoff (~critical ξ)
const FOVER  = parse(Float64, get(ENV, "BAM_BE_FOVER",  "1.1")) # overdensity factor (>1 ⇒ supercritical)
const BOXFAC = parse(Float64, get(ENV, "BAM_BE_BOXFAC", "3.0")) # box = BOXFAC · R_sphere
# Enzo's polynomial fit to the isothermal Lane-Emden (Bonnor-Ebert) density profile
BE_prof(ξ) = 4.8089e-4*ξ^5 - 1.0173e-2*ξ^4 + 7.7899e-2*ξ^3 - 2.3299e-1*ξ^2 + 1.4721e-2*ξ + 1.0008

const ρ0  = NH0 * mH / XH                                     # central density (g/cm³)
const cs2 = kB * TINIT / (μ * mH)                            # isothermal c_s² (cm²/s²)
const r0BE = sqrt(cs2 / (4π * Gcgs * ρ0))                    # BE scale length a = c_s/√(4πGρ_c) (cm)
const Rsph = XIMAX * r0BE                                     # sphere radius (cm)
const Mcl  = (4.0/3.0) * π * Rsph^3 * ρ0                      # sphere mass scale (g)
const t_u  = 1.0 / sqrt(4π * Gcgs * ρ0)                      # gravitational time unit (s)
const t_ff = sqrt(3π / (32.0 * Gcgs * ρ0))                   # free-fall time (central)
const Lbox = BOXFAC * Rsph                                    # box
const v_u  = Lbox / t_u
const T2   = v_u^2 * mH / kB                                  # K per code specific-energy
const e_init = TINIT / ((γ - 1) * μ * T2)                    # code specific internal energy
const Rcode  = Rsph / Lbox                                    # = 1/BOXFAC
const RHO_EDGE = BE_prof(XIMAX)                              # code edge density (ambient, pressure-continuous)
const cs_code = sqrt(cs2) / v_u
# chem unit conversions (code → cgs)
const U_D = ρ0; const U_L = Lbox; const U_T = t_u; const U_V = v_u
const XHII0 = 1.0e-4; const XH2_0 = 1.0e-6   # low H2 → must FORM via H⁻ channel during collapse

@printf("Bonnor-Ebert collapse: n_H=%.1f T=%.0fK  →  M=%.3e Msun  R=%.3f pc  t_ff=%.3f Myr\n",
        NH0, TINIT, Mcl/Msun, Rsph/pc, t_ff/(3.156e13))
@printf("  ξ_max=%.2f f_over=%.2f contrast=%.1f  R_code=%.3f  cs_code=%.4f  box=%.3f pc  [%s LMAX=%d]\n",
        XIMAX, FOVER, BE_prof(0.0)/RHO_EDGE, Rcode, cs_code, Lbox/pc, BE, LMAX)
flush(stdout)

# ── build hierarchy + BB initial conditions ───────────────────────────────────
# out-of-core: engage managed (unified) memory BEFORE any pool allocation, so the
# collapse — whose central free-fall time races ahead of the outer grid and piles
# up refinement memory beyond any device — streams cold pools to host RAM.  Without
# this, device_zeros falls back to plain device memory and OOMs at the card limit.
BlockAMR.set_memory_mode!(MEMMODE)
hier = AMRHierarchy(; nbase = (NGRID, NGRID, NGRID), B = BBLK, backend = BE,
                    T = PREC, nsp = 2, gamma = γ, cfl = 0.3, Lcap = LMAX,
                    scheme = Symbol(get(ENV, "BAM_SCHEME", "ctu")))
lev0 = init_base_level!(hier); build_level_tables!(hier, 0)

"BE IC on level `l`: supercritical Bonnor-Ebert sphere at rest, isothermal, primordial."
function be_ic!(hier, l)
    lev = hier.levels[l + 1]; isempty(lev.live) && return
    dx = BlockAMR.level_dx(hier, l); n = lev.cap * lev.stride
    hD = zeros(n); hS1 = zeros(n); hS2 = zeros(n); hS3 = zeros(n); hT = zeros(n); hG = zeros(n)
    for s in lev.live
        m = lev.meta[s]; base = (Int(s) - 1) * lev.stride
        for k in 1:lev.B, j in 1:lev.B, i in 1:lev.B
            gi = Int(m.origin[1]) + i; gj = Int(m.origin[2]) + j; gk = Int(m.origin[3]) + k
            x = (gi - 0.5) * dx - 0.5; y = (gj - 0.5) * dx - 0.5; z = (gk - 0.5) * dx - 0.5
            r = sqrt(x*x + y*y + z*z)
            # BE profile inside R_sphere (ξ = 0..ξ_max), pressure-continuous ambient outside;
            # supercritical overdensity FOVER triggers the collapse.  Gas at rest, isothermal.
            ξ = (r / Rcode) * XIMAX
            ρ = FOVER * (r < Rcode ? BE_prof(ξ) : RHO_EDGE)
            idx = base + ((lev.ng+k-1) * lev.nd + (lev.ng+j-1)) * lev.nd + (lev.ng+i-1) + 1
            hD[idx] = ρ; hS1[idx] = 0.0; hS2[idx] = 0.0; hS3[idx] = 0.0
            hG[idx] = ρ * e_init
            hT[idx] = ρ * e_init            # at rest ⇒ no kinetic energy
        end
    end
    BlockAMR.encode_from_host!(lev, hD, hS1, hS2, hS3, hT, hG)
    n2 = lev.cap * lev.stride
    copyto!(lev.sp[1], fill(ChemistryKernels.encode_log2sp(Float32(XHII0 * XH)), n2))
    copyto!(lev.sp[2], fill(ChemistryKernels.encode_log2sp(Float32(XH2_0 * XH)), n2))
end
be_ic!(hier, 0)

# refinement: overdensity + Jeans (super-refine the collapsing core).  Zoom confines
# levels ≥ zoom_lmin to a central sphere (the BB collapse is GLOBAL — the whole Jeans-
# unstable sphere flags — so deep refinement must be confined to the central runaway
# to bound memory).  BAM_ZOOMR=0 ⇒ off (refine everywhere).
const ZOOMR    = parse(Float64, get(ENV, "BAM_ZOOMR", "0.0"))
const ZOOMLMIN = parse(Int,     get(ENV, "BAM_ZOOMLMIN", "2"))
mkpol() = ZOOMR > 0 ?
    BlockRefinementPolicy(; dthresh = DTHRESH, nbuf = 2, every = REGRIDN, lmax = LMAX,
                          jeans_ncells = JEANS, jeans_rho_floor = JRHO,
                          zoom_center = (0.5, 0.5, 0.5), zoom_r = ZOOMR, zoom_lmin = ZOOMLMIN) :
    BlockRefinementPolicy(; dthresh = DTHRESH, nbuf = 2, every = REGRIDN, lmax = LMAX,
                          jeans_ncells = JEANS, jeans_rho_floor = JRHO)
pol = mkpol()
jeanscoef() = 8π^2 * γ * (γ - 1)                  # λ_J² = jcoef·(Ge·Esc)/ρ² ; G_code=1 ⇒ ∇²φ=δ
regrid!(hier, pol; jeans_coef = jeanscoef())
for l in 1:length(hier.levels)-1; be_ic!(hier, l); end
MEMMODE === :managed && BlockAMR.advise_all_host!(hier)   # pools born host-resident

# level-0 gravity workspace
ρg = device_zeros(hier.be, Float32, (NGRID^3,)); φg = device_zeros(hier.be, Float32, (NGRID^3,))
uu = (T2 = T2, v = U_V)
c_prof = (box = Lbox/pc, Om = 1.0, XH = XH, fb = 1.0)   # for JeansProfile (radius in pc)

# central density peak + its temperature (code): the collapse tracker
function core_state(hier)
    ρbest = 0.0; ecore = NaN; Lbest = 0; pk = (0.5,0.5,0.5)
    for l in 0:length(hier.levels)-1
        lev = hier.levels[l+1]; isempty(lev.live) && continue
        Nl = NGRID * 2^l
        hD=Array(lev.D); hG=Array(lev.Ge); hDsc=Array(lev.Dsc); hGsc=Array(lev.Gsc)
        for s in lev.live
            m=lev.meta[s]; base=(Int(s)-1)*lev.stride
            for k in 1:lev.B, j in 1:lev.B, i in 1:lev.B
                idx=base+((lev.ng+k-1)*lev.nd+(lev.ng+j-1))*lev.nd+(lev.ng+i-1)+1
                ρ=Float64(hD[idx])*hDsc[s]
                if ρ>ρbest
                    ρbest=ρ; Lbest=l; ecore=Float64(hG[idx])*hGsc[s]/max(ρ,1e-30)
                    pk=((Float64(m.origin[1])+i-0.5)/Nl,(Float64(m.origin[2])+j-0.5)/Nl,(Float64(m.origin[3])+k-0.5)/Nl)
                end
            end
        end
    end
    return ρbest, ecore*(γ-1)*μ*T2, Lbest, pk       # ρ_code, T[K], level, peak
end

# composite midplane (z=0.5) density+temperature slice over a central window, sampled
# from the FINEST covering level (coarse→fine, fine overwrites) — reveals the m=2 bar
# and fragmentation that radial averaging washes out.  Binary: Int32 Nsl, then Nsl² f32
# ρ (code), then Nsl² f32 T[K].  Visualize with numpy (reshape (Nsl,Nsl)).
function dump_slice(path, hier; Nsl=384, lo=0.30, hi=0.70)
    win = hi - lo; dpix = win / Nsl
    rho = fill(Float32(NaN), Nsl, Nsl); tmp = fill(Float32(NaN), Nsl, Nsl)
    for l in 0:length(hier.levels)-1
        lev = hier.levels[l+1]; isempty(lev.live) && continue
        Nl = NGRID * 2^l; dx = 1.0/Nl
        hD=Array(lev.D); hG=Array(lev.Ge); hsc=Array(lev.Dsc); hgs=Array(lev.Gsc)
        for s in lev.live
            m = lev.meta[s]; base = (Int(s)-1)*lev.stride
            z0 = Float64(m.origin[3])*dx; z1 = z0 + lev.B*dx
            (0.5 < z0 || 0.5 >= z1) && continue
            k = clamp(floor(Int,(0.5 - z0)/dx)+1, 1, lev.B)
            for j in 1:lev.B, i in 1:lev.B
                x=(Float64(m.origin[1])+i-0.5)*dx; y=(Float64(m.origin[2])+j-0.5)*dx
                (x<lo||x>=hi||y<lo||y>=hi) && continue
                px=floor(Int,(x-lo)/dpix)+1; py=floor(Int,(y-lo)/dpix)+1
                (px<1||px>Nsl||py<1||py>Nsl) && continue
                idx = base + ((lev.ng+k-1)*lev.nd+(lev.ng+j-1))*lev.nd+(lev.ng+i-1)+1
                ρ = Float64(hD[idx])*hsc[s]; rho[px,py]=Float32(ρ)
                tmp[px,py] = Float32(Float64(hG[idx])*hgs[s]/max(ρ,1e-30)*(γ-1)*μ*T2)
            end
        end
    end
    open(path,"w") do io; write(io, Int32(Nsl)); write(io, rho); write(io, tmp); end
end

# ── main loop: level-0 FFT gravity + W-cycle (native gravity + chem) + regrid ──
function run!(hier, pol)
    ρ3 = reshape(ρg, NGRID, NGRID, NGRID); φ3 = reshape(φg, NGRID, NGRID, NGRID)
    t = 0.0; nstep = 0; maxseen = maximum(l -> isempty(hier.levels[l+1].live) ? -1 : l, 0:length(hier.levels)-1)
    ch = (a_value = 1.0, density_units = U_D, length_units = U_L, time_units = U_T,
          hubble = 67.4, Om = 0.315, OL = 0.685, fh = XH,
          dtfrac = CHEM_DTFRAC, itcap = CHEM_ITCAP,
          rho_switch = CHEM_NHSW > 0 ? CHEM_NHSW * mH / XH : 0.0)   # n_H [cm⁻³] → CGS g/cm³
    TIMERN > 0 && (enable_timers!(true); reset_timers!())
    while nstep < NROOT && t < TEND_FF * (t_ff / t_u)
        # timestep: hydro CFL + free-fall limiter (per level, 2^l relaxed)
        λ = compute_lambda!(hier); dx0 = BlockAMR.level_dx(hier, 0)
        dτ_ff = Inf
        for l in 0:length(hier.levels)-1
            lv = hier.levels[l+1]; isempty(lv.live) && continue
            ρl = 1.0; hd = Array(lv.Dsc)
            for s in lv.live; ρl = max(ρl, 2.0*Float64(hd[s])); end
            dτ_ff = min(dτ_ff, exp2(l) * DTFF / sqrt(ρl))     # coef=1 ⇒ t_dyn ∝ 1/√ρ
        end
        dτ = min(Float64(λ)*dx0, dτ_ff); hier.λ = Float32(dτ/dx0)
        # level-0 gravity: gas density → δ → FFT Poisson (coef=1) → φ
        fill!(ρg, 0.0f0); global_from_level0!(hier, ρg)
        ρmean = sum(Array(ρg))/length(ρg); ρg .-= Float32(ρmean)
        PoissonKernels.fft_poisson_rfft!(φ3, ρ3; G = 1.0, a = 1.0, boxsize = 1.0)
        phi_from_global!(hier, φg)
        sg = (coef = 1.0, nsweep = NSWEEP, nsweep0 = 0, rho_mean = ρmean, use_dm = false)
        advance_hierarchy!(hier; φ = φg, selfgrav = sg, chem = ch)
        for l in 0:length(hier.levels)-1; update_scales!(hier, l); end
        nstep += 1; t += dτ
        if nstep % REGRIDN == 0
            regrid!(hier, pol; jeans_coef = jeanscoef())
            MEMMODE === :managed && BlockAMR.advise_all_host!(hier)   # new blocks → host home
        end
        ρc, Tc, Lc, pk = core_state(hier)
        if nstep % 10 == 0
            blocks = [length(hier.levels[l+1].live) for l in 0:length(hier.levels)-1]
            @printf("step %4d  t=%.3f t_ff  Lmax=%d  ρ_c=%.3e  T_c=%.1fK  blocks=%s\n",
                    nstep, t/(t_ff/t_u), Lc, ρc, Tc, string(blocks)); flush(stdout)
        end
        if TIMERN > 0 && nstep % TIMERN == 0                # per-window phase breakdown
            @printf("── timers @ step %d (last %d steps) ──\n", nstep, TIMERN)
            timer_report(); flush(stdout); reset_timers!()
        end
        if SLICEN > 0 && nstep % SLICEN == 0                # midplane slice → collapse morphology
            dump_slice(joinpath(REPORTS, @sprintf("slice%s_%04d.bin", TAG, nstep)), hier)
        end
        if Lc > maxseen                                    # new level → radial profile
            maxseen = Lc
            zz = 0.0
            pth = joinpath(REPORTS, "profile$(TAG)_L$(Lc).txt")
            npts, _ = JeansProfile.dump_radial_profile(pth, hier, c_prof, uu; peak = pk,
                        z = zz, level = Lc, gamma = γ, μ = μ, XH = XH, rmax = 0.02)
            @printf("  ▸ NEW LEVEL %d @ t=%.3f t_ff  ρ_c=%.3e T_c=%.1fK → %s (%d cells)\n",
                    Lc, t/(t_ff/t_u), ρc, Tc, basename(pth), npts); flush(stdout)
            STOPL < 99 && Lc >= STOPL && (@printf("LEVEL %d reached — stopping\n", STOPL); break)
        end
        !isfinite(ρc) && (@printf("NONFINITE at step %d\n", nstep); break)
    end
    @printf("DONE: %d steps, t=%.3f t_ff, final ρ_c=%.3e T_c=%.1fK Lmax=%d\n",
            nstep, t/(t_ff/t_u), core_state(hier)[1], core_state(hier)[2], core_state(hier)[3])
end
run!(hier, pol)
