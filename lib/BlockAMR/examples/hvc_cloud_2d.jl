# hvc_cloud_2d.jl — a high-velocity cloud in a hot, Mach-2 halo wind: 2-D MHD
# (constrained transport) on BlockAMR, with tabulated Grackle cooling and UV
# heating (FG2011, z = 0; thin or shielded table), in an elongated wind tunnel.
#
# The "runs two / three" of the first-Grackle-runs plan (04.05, 2026-10-28),
# reconstructed from that document (its companion IC documents 01.01 / 04.03 were
# not available): halo 1e6 K at n_tot = 1e-3 cm⁻³ (P/k = 1000 K cm⁻³), cloud
# radius 100 pc, wind Mach 2 of the halo sound speed, the cloud at the table's
# warm equilibrium on the halo isobar (density from pressure balance with the
# equilibrium mean mass, plan eq. chi3), metallicity Z = 0.3 Z☉ (reproduces the
# plan's thin warm phase, ≈ 8.6e3 K vs "near 8000 K").  B = 0, or a uniform field
# transverse to the wind at plasma β (HVC_BETA).
#
# Run (GPU):  julia --project=lib/BlockAMR/test lib/BlockAMR/examples/hvc_cloud_2d.jl
# Knobs (env): HVC_OUT (dir), HVC_TABLE (dir of an exported Cloudy table),
#   HVC_BETA (Inf | number), HVC_NX (base cells along x, default 256), HVC_LMAX (4),
#   HVC_TEND (in cloud-crushing times, 6), HVC_SNAP (snapshots per t_cc, 4),
#   HVC_Z (Z/Z☉, 0.3), HVC_BACKEND (cuda), HVC_T (Float64), HVC_DTHRESH (refine
#   above this × halo density, 3), HVC_LX, HVC_LY (box in cloud radii, 32 × 8),
#   HVC_X0 (cloud centre x in radii, 4), HVC_MAXSTEPS, HVC_WALL (hours, stop+save),
#   HVC_COOL_T (cooling arithmetic, Float32 | Float64; default Float32),
#   HVC_MOVIE (movie frames per t_cc, 0 = off), HVC_MOVIE_LEVEL (image resolution
#   level, default HVC_LMAX), HVC_MOVIE_FIELDS (comma list of MOVIE_CATALOG keys,
#   default: all that apply), HVC_SHOCKSW (1: PLUTO-style shock switch,
#   HLLD → HLL + minmod in flagged shock cells, against the carbuncle; default 0).  Movie → HVC_OUT/movie (open with the player:
#   `python3 -m http.server -d HVC_OUT/movie`, or `movie_bundle` it for sharing).
using BlockAMR, Printf, Serialization, Dates
try; @eval using CUDA; catch; end
const BA = BlockAMR

env(k, d) = get(ENV, k, string(d))
const OUT     = env("HVC_OUT", joinpath(tempdir(), "hvc2d"))
const TABLE   = env("HVC_TABLE", expanduser("~/data/grackle_tables/FG2011_z0"))
const BETA    = parse(Float64, env("HVC_BETA", "Inf"))
const NX      = parse(Int, env("HVC_NX", 256))
const LMAX    = parse(Int, env("HVC_LMAX", 4))
const TEND    = parse(Float64, env("HVC_TEND", 6.0))
const NSNAP   = parse(Float64, env("HVC_SNAP", 4.0))
const ZREL    = parse(Float64, env("HVC_Z", 0.3))
const BE      = Symbol(env("HVC_BACKEND", "cuda"))
const TT      = env("HVC_T", "Float64") == "Float32" ? Float32 : Float64
const DTHR    = parse(Float64, env("HVC_DTHRESH", 3.0))
const LX      = parse(Float64, env("HVC_LX", 32.0)); const LY = parse(Float64, env("HVC_LY", 8.0))
const X0      = parse(Float64, env("HVC_X0", 4.0))
const MAXST   = parse(Int, env("HVC_MAXSTEPS", 10^9))
const WALL    = parse(Float64, env("HVC_WALL", 1e9)) * 3600
# density floor in units of the halo density: magnetized runs need it well above
# vacuum — a positivity-floored cell keeps its (draped, amplified) field, and at
# ρ → 1e-6 ρ_h its Alfvén speed collapses the global time step (β = 100 runs, v1)
const SMALLR  = parse(Float64, env("HVC_SMALLR", 1e-6))
const NMOVIE  = parse(Float64, env("HVC_MOVIE", 0.0))
const LMOVIE  = parse(Int, env("HVC_MOVIE_LEVEL", LMAX))
const FMOVIE  = env("HVC_MOVIE_FIELDS", "default")
const SHOCKSW = env("HVC_SHOCKSW", "0") == "1"
mkpath(OUT); mkpath(joinpath(OUT, "snap"))

# ── units: length 100 pc (the cloud radius), velocity 100 km/s, density m_H cm⁻³ ──
const PC = 3.0857e18; const KB = BA._KB_CGS; const MH = BA._MH_CGS
const U = CoolingUnits(density = MH, length = 100PC, velocity = 1e7)
const TU = BA.time_unit(U); const MYR = 3.15576e13
const γ = 5 / 3
# the cooling table (and its kernel arithmetic) in Float32 by default, independent of
# the hydro precision: 22× faster on GPUs with slow Float64, final e within 2e-6
const COOLT = env("HVC_COOL_T", "Float32") == "Float64" ? Float64 : Float32
cool = load_cloudy_cooling(TABLE; be = BA.backend(BE), units = U, metallicity = ZREL, T = COOLT)

# ── the shared isobar: halo state, the cloud's warm equilibrium ────────────────
const Pk = 1000.0                         # P/k = n_tot T  [K cm⁻³]
const Th = 1e6
function state(Tg)                        # (ρ [g cm⁻³], μ) on the isobar at Tg
    μ = 0.6
    for _ in 1:60
        μn = cloudy_mu(cool, cool.XH * μ * Pk / Tg, Tg)
        abs(μn / μ - 1) < 1e-12 && (μ = μn; break); μ = μn
    end
    return μ * MH * Pk / Tg, μ
end
iso = cloudy_isobar(cool, Pk; T = 10 .^ range(1, 6.5, 5501))
# stable equilibria: ė changes from + (heating) to − (cooling) with rising T
stab = [T for (k, T) in enumerate(iso.T[1:end-1]) if iso.edot[k] > 0 && iso.edot[k+1] <= 0 && T < 1e5]
isempty(stab) && error("no warm equilibrium below 1e5 K on the isobar for this table")
const Tc = maximum(stab)
ρh, μh = state(Th); ρc, μc = state(Tc)
const χ = ρc / ρh
const P = Pk * KB                         # erg cm⁻³
csh = sqrt(γ * P / ρh); vw = 2csh         # Mach 2
code(ρ) = ρ / U.density; codeP(p) = p / (U.density * U.velocity^2); codev(v) = v / U.velocity
const RH, RC, PH, VW = code(ρh), code(ρc), codeP(P), codev(vw)
const B0 = isinf(BETA) ? 0.0 : sqrt(2PH / BETA)        # B²/2 = P/β (Heaviside–Lorentz code units)
const TCC = sqrt(χ) * 1.0 / VW                          # crushing time, code (R = 1)
@printf("halo: T=%.3g K μ=%.4f ρ=%.4g (code)   cloud: T=%.4g K μ=%.4f ρ=%.4g  χ=%.2f\n", Th, μh, RH, Tc, μc, RC, χ)
@printf("wind: c_s=%.1f km/s  v=%.1f km/s (code %.4f)   t_cc=%.3f code = %.3f Myr   B0=%.4g code (β=%s, %.3g μG)\n",
        csh / 1e5, vw / 1e5, VW, TCC, TCC * TU / MYR, B0, BETA,
        B0 * sqrt(4π * U.density * U.velocity^2) * 1e6)

# ── hierarchy: elongated wind tunnel ──────────────────────────────────────────
const NY = round(Int, NX * LY / LX)
inflow = Dict((1, :lo) => (RH, VW, 0.0, 0.0, PH, 0.0, B0, 0.0))
hier = AMRHierarchy(; nbase = (NX, NY), B = 16, box = LX, backend = BE, T = TT, cap0 = 64,
                    physics = :mhd, divb = :ct, riemann = :hlld, cfl = 0.4,
                    bc = ((:inflow, :outflow), :periodic), inflow, Lcap = LMAX,
                    smallr = SMALLR * RH, smallc = 1e-4, shock_switch = SHOCKSW)
init_base_level!(hier); build_level_tables!(hier, 0)
pol = BlockRefinementPolicy(; dthresh = DTHR * RH, nbuf = 2, every = 1, lmax = LMAX)

# smooth cloud edge (width w in radii), at rest in the wind
const W = 0.02
cloudf(x) = 0.5 * (1 - tanh((sqrt((x[1] - X0)^2 + (x[2] - LY / 2)^2) - 1) / W))
prim(x) = (f = cloudf(x); (RH + (RC - RH) * f, VW * (1 - f), 0.0, 0.0, PH))

function set_level_ic!(h, l)
    lev = h.levels[l + 1]; dx = BA.level_dx(h, l); n = lev.cap * lev.stride
    set_ct_faces_from_Az!(h, l, (x, y) -> -B0 * x)          # uniform By = B0 (∇·B = 0)
    hb = (Array(lev.Bx), Array(lev.By))
    a = [zeros(Float64, n) for _ in 1:10]
    for s in lev.live
        m = lev.meta[s]; base = (Int(s) - 1) * lev.stride
        for j in 0:lev.B-1, i in 0:lev.B-1
            x = ((Int(m.origin[1]) + i + 0.5) * dx, (Int(m.origin[2]) + j + 0.5) * dx)
            ρ, u, v, w, p = prim(x)
            idx = base + (j + lev.ng) * lev.nd + (i + lev.ng) + 1
            ge = p / (γ - 1)
            vals = (ρ, ρ*u, ρ*v, ρ*w, ge + 0.5ρ*(u^2 + v^2 + w^2) + 0.5*(hb[1][idx]^2 + hb[2][idx]^2),
                    ge, hb[1][idx], hb[2][idx], 0.0, 0.0)
            for f in 1:10; a[f][idx] = vals[f]; end
        end
    end
    BA.encode_mhd_from_host!(lev, a...)
end
setall!(h) = for l in 0:length(h.levels)-1; isempty(h.levels[l+1].live) || set_level_ic!(h, l); end

set_level_ic!(hier, 0)
for _ in 1:LMAX                                 # refine onto the IC, level by level
    regrid!(hier, pol); setall!(hier)
end
@printf("initial blocks per level: %s\n", [length(lv.live) for lv in hier.levels])

# ── diagnostics ───────────────────────────────────────────────────────────────
const TAB = Float64.(Array(cool.tab))
const ZF = ZREL * cool.Zsun
const GM1UTEM = (γ - 1) * MH / KB          # e passed in cgs (erg/g)
temp_cgs(ρc_, ec_) = BA._cloudy_T(ec_ * U.velocity^2, log10(cool.XH * ρc_ * U.density / MH), ZF,
                                  TAB, cool.nn, cool.nt, cool.n0, cool.dn, cool.t0, cool.dT,
                                  GM1UTEM, cool.Tfloor)[1]

"Per-level host copies + cell temperature."
function levdata(h, l)
    lev = h.levels[l + 1]
    D = Array(lev.D) .* 1.0; S1 = Array(lev.S1); S2 = Array(lev.S2); S3 = Array(lev.S3)
    E = Array(lev.Tau); Bx = Array(lev.Bx); By = Array(lev.By); Bz = Array(lev.Bz)
    dsc = Array(lev.Dsc); ssc = Array(lev.Ssc); esc = Array(lev.Esc)
    (; lev, D, S1, S2, S3, E, Bx, By, Bz, dsc, ssc, esc)
end
cellstate(L, s, idx) = begin
    ρ = L.D[idx] * L.dsc[s]; s1 = L.S1[idx] * L.ssc[s]; s2 = L.S2[idx] * L.ssc[s]; s3 = L.S3[idx] * L.ssc[s]
    eint = (L.E[idx] * L.esc[s] - 0.5 * (s1^2 + s2^2 + s3^2) / ρ - 0.5 * (L.Bx[idx]^2 + L.By[idx]^2 + L.Bz[idx]^2)) / ρ
    (ρ, s1 / ρ, s2 / ρ, temp_cgs(ρ, max(eint, 1e-30)))
end

# composite scan over leaf cells, on the device (BlockAMR.leaf_moments: the
# dual-energy temperature from the cooling table; 7 numbers per level to the host)
scan(h) = leaf_moments(h, cool; Tcold = 2e4, Tmix = 5e5, rho_dense = RC / 3)

# uniform map at output level Lo over a window [x0, x0+w) × [y0, y0+hgt) (code units)
function rastermap(h, Lo, x0, y0, w, hgt)
    dxo = BA.level_dx(h, Lo)
    nx = round(Int, w / dxo); ny = round(Int, hgt / dxo)
    i0 = round(Int, x0 / dxo); j0 = round(Int, y0 / dxo)
    R = fill(NaN32, nx, ny); Tm = fill(NaN32, nx, ny); By = fill(NaN32, nx, ny); Vx = fill(NaN32, nx, ny)
    for l in 0:min(Lo, length(h.levels) - 1)
        lev = h.levels[l + 1]; isempty(lev.live) && continue
        L = levdata(h, l); r = 2^(Lo - l)
        for s in lev.live
            o = lev.meta[s].origin; base = (Int(s) - 1) * lev.stride
            for j in 0:lev.B-1, i in 0:lev.B-1
                pi0 = (Int(o[1]) + i) * r - i0; pj0 = (Int(o[2]) + j) * r - j0
                (pi0 + r <= 0 || pi0 >= nx || pj0 + r <= 0 || pj0 >= ny) && continue
                idx = base + (j + lev.ng) * lev.nd + (i + lev.ng) + 1
                ρ, vx, _, Tg = cellstate(L, s, idx)
                for b in max(pj0, 0):min(pj0 + r, ny) - 1, a in max(pi0, 0):min(pi0 + r, nx) - 1
                    R[a + 1, b + 1] = ρ; Tm[a + 1, b + 1] = Tg; By[a + 1, b + 1] = L.By[idx]; Vx[a + 1, b + 1] = vx
                end
            end
        end
    end
    return (; R, T = Tm, By, Vx, x0 = i0 * dxo, y0 = j0 * dxo, dx = dxo)
end

function snapshot(h, k, t, sc)
    Lbox = min(2, LMAX)
    full = rastermap(h, Lbox, 0.0, 0.0, LX, LY)
    xc = isnan(sc.xc) ? X0 : sc.xc
    wx = 8.0; wy = min(4.0, LY)
    xw = clamp(xc - 2.5, 0.0, LX - wx); yw = clamp(LY / 2 - wy / 2, 0.0, LY - wy)
    zoom = rastermap(h, length(h.levels) - 1, xw, yw, wx, wy)
    open(joinpath(OUT, "snap", @sprintf("snap_%04d.jls", k)), "w") do io
        serialize(io, (; t, t_cc = t / TCC, t_myr = t * TU / MYR, full, zoom, scan = sc,
                       nblocks = [length(lv.live) for lv in h.levels]))
    end
end

# ── run ───────────────────────────────────────────────────────────────────────
open(joinpath(OUT, "params.toml"), "w") do io
    println(io, "# hvc_cloud_2d.jl — $(now())\ntable = \"$(TABLE)\"\ntable_source = \"$(replace(cool.source, '\n' => "; "))\"\nmetallicity_Zsun = $ZREL")
    println(io, "beta = $(isinf(BETA) ? "\"inf\"" : BETA)\nB0_code = $B0\nNX = $NX\nNY = $NY\nLMAX = $LMAX")
    println(io, "box_radii = [$LX, $LY]\ncloud_x0_radii = $X0\nT_halo = $Th\nT_cloud = $Tc\nmu_halo = $μh\nmu_cloud = $μc")
    println(io, "chi = $χ\nP_over_k = $Pk\nv_wind_kms = $(vw/1e5)\nt_cc_Myr = $(TCC*TU/MYR)\nt_end_tcc = $TEND")
    println(io, "shock_switch = $SHOCKSW\ndual_energy = $(hier.mhd.de)\ncooling_float = \"$COOLT\"")
    println(io, "smallr_halo = $SMALLR\ndthresh_halo = $DTHR\nfloat = \"$TT\"\nbackend = \"$BE\"\nunits = { length_pc = 100, velocity_kms = 100, density = \"m_H cm^-3\", time_Myr = $(TU/MYR) }")
end
ts = open(joinpath(OUT, "timeseries.csv"), "w")
println(ts, "step,t_code,t_cc,t_myr,dt_code,mass,mcold,mdense,mint,xc,vxc,divb,blocks,wall_s"); flush(ts)
sc0 = scan(hier); M0c = sc0.mcold
snapshot(hier, 0, 0.0, sc0)
# movie: whole tunnel at level LMOVIE; axes in pc, clocks t/t_cc and Myr, fields in
# physical units (Heaviside–Lorentz B: B_G = √(4π ρ_u) v_u B_code)
const BG = sqrt(4π * MH) * 1e7
mw = NMOVIE > 0 ? MovieWriter(joinpath(OUT, "movie"), hier; cooling = cool, level = LMOVIE,
        fields = FMOVIE == "default" ? :default : Symbol.(strip.(split(FMOVIE, ","))),
        length_unit = ("pc", 100.0), time_labels = [("t", "t_cc", 1 / TCC), ("t", "Myr", TU / MYR)],
        units = Dict(:rho => ("m_H cm⁻³", 1.0), :p => ("K cm⁻³", MH * 1e14 / KB),
                     :B2 => ("μG²", (BG * 1e6)^2), :absJ => ("μG/pc", BG * 1e6 / 100), :Jz => ("μG/pc", BG * 1e6 / 100),
                     :vmag => ("km/s", 100.0), :vx => ("km/s", 100.0),
                     :divv => ("Myr⁻¹", MYR / TU), :absw => ("Myr⁻¹", MYR / TU), :wz => ("Myr⁻¹", MYR / TU)),
        title = "HVC in a Mach-2 halo wind — " * basename(OUT),
        meta = Dict("χ" => round(χ; sigdigits = 4), "β" => isinf(BETA) ? "∞" : BETA, "table" => basename(TABLE),
                    "Z/Z☉" => ZREL, "base" => "$(NX)×$(NY)", "L_max" => LMAX)) : nothing
mw === nothing || write_frame!(mw, hier, 0.0)
tmnext = NMOVIE > 0 ? TCC / NMOVIE : Inf
t = 0.0; step = 0; nsnap = 1; tw0 = time(); tnext = TCC / NSNAP
const TW = Dict(:advance => 0.0, :regrid => 0.0, :diag => 0.0, :movie => 0.0)   # wall-clock per phase
get(ENV, "HVC_TIMERS", "0") == "1" && BA.enable_timers!()       # synchronized GPU phase timers
while t < TEND * TCC && step < MAXST
    global t, step, nsnap, tnext, tmnext
    # synchronize so the phase clocks charge GPU work to the phase that queued it
    # (kernel launches are asynchronous: without this, the advance's kernels drain
    # inside the regrid's first device→host copy and are billed to "regrid")
    TW[:advance] += @elapsed (dt = advance_hierarchy!(hier; cooling = cool); BA.KA.synchronize(hier.levels[1].be))
    t += dt; step += 1
    TW[:regrid] += @elapsed regrid!(hier, pol)
    if t >= tmnext
        TW[:movie] += @elapsed write_frame!(mw, hier, t)
        tmnext = (floor(t * NMOVIE / TCC) + 1) * TCC / NMOVIE
    end
    if step % 25 == 0 || t >= tnext
        td0 = time()
        sc = scan(hier)
        @printf(ts, "%d,%.6e,%.5f,%.4f,%.4e,%.8e,%.8e,%.8e,%.8e,%.5f,%.5f,%.3e,\"%s\",%.1f\n", step, t, t / TCC,
                t * TU / MYR, dt, sc.mass, sc.mcold, sc.mdense, sc.mint, sc.xc, sc.vxc, max_divb(hier),
                join([length(lv.live) for lv in hier.levels], " "), time() - tw0)
        flush(ts)
        if t >= tnext
            snapshot(hier, nsnap, t, sc)
            @printf("snap %3d  t=%.3f t_cc (%.2f Myr)  step %6d  cold %.3f  xc=%.2f vxc/vw=%.3f  blocks %s  ∇·B=%.1e  wall %.0f s\n",
                    nsnap, t / TCC, t * TU / MYR, step, sc.mcold / M0c, sc.xc, sc.vxc / VW,
                    [length(lv.live) for lv in hier.levels], max_divb(hier), time() - tw0)
            flush(stdout)
            @printf("          wall by phase: advance %.0f s, regrid %.0f s, diagnostics %.0f s, movie %.0f s (%d frames)\n",
                    TW[:advance], TW[:regrid], TW[:diag], TW[:movie], mw === nothing ? 0 : length(mw.frames))
            nsnap += 1; tnext = nsnap * TCC / NSNAP
        end
        TW[:diag] += time() - td0
        sc.xc > LX - 4 && (println("cold gas reached the outflow end — stopping"); break)
        time() - tw0 > WALL && (println("wall-clock limit — stopping"); break)
    end
end
close(ts)
save_checkpoint(joinpath(OUT, "final.ckpt"), hier; extra = (t = t, step = step))
get(ENV, "HVC_TIMERS", "0") == "1" && BA.timer_report()
@printf("done: t=%.3f t_cc (%.2f Myr), %d root steps, %.0f s wall\n", t / TCC, t * TU / MYR, step, time() - tw0)
