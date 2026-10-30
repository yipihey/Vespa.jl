# cooling.jl — tabulated radiative cooling + UV-background heating (Grackle's
# Cloudy mode, primordial_chemistry = 0) as an operator-split source term.
#
# Tables: Grackle's CloudyData_*.h5 (e.g. CloudyData_UVB=FG2011[_shielded].h5),
# exported at one redshift by tools/cloudy_table_export.py into .npy files:
# Λ/n_H², H/n_H² [erg cm³ s⁻¹] (interpolated in log₁₀) for the primordial gas and the
# metals at solar metallicity, and the primordial mean molecular weight, on a
# uniform (log₁₀ n_H, log₁₀ T) grid.  What Grackle does with them (src/clib:
# cool1d_cloudy_g.F, calc_temp1d_cloudy_g.F, solve_rate_cool_g.F), reproduced:
#
#   ė = n_H² [ 10^H_prim − 10^Λ_prim + (Z/Z☉)(10^H_met − 10^Λ_met) ]   erg cm⁻³ s⁻¹
#   n_H = X_H ρ / m_H,   X_H = 1/(1 + 0.1·3.971) ≈ 0.7158 (Grackle forces this in
#   fully tabulated mode: Cloudy's n_He/n_H = 0.1),   Z☉ = 0.01295;
#   T from e by Grackle's damped fixed point on μ(n_H, T) (μ ← ½(μ_new+μ_old),
#   1 % convergence, ≤ 20 iterations, metal particles at 16 m_H);
#   log-linear interpolation in (log₁₀ n_H, log₁₀ T), inputs clamped to the table;
#   sub-cycled explicit integration with Δt_sub = min(0.1·e/|ė/ρ|, remaining)
#   (Grackle's 10 % rule, verbatim — incl. its ±10 % oscillation about an
#   equilibrium); T is clamped at T_floor and cooling stops there (Grackle:
#   T ≤ 1.01·temstart with ė < 0), and a cell entering below e(T_floor) is
#   raised to it (energy floor); T_floor defaults to the table edge.
#
# Applied per level substep (dt_l) after the hydro update: hydro runs change Ge
# (dual energy) and Tau by the same ρΔe; MHD runs change the total E (and keep
# Ge = p/(γ−1)).  Code units → cgs through `CoolingUnits` (ρ, L, v).

const _MH_CGS = 1.6726219e-24          # Grackle mh
const _KB_CGS = 1.3806504e-16          # Grackle kboltz

"Code → cgs unit factors: density [g cm⁻³], length [cm], velocity [cm s⁻¹]."
Base.@kwdef struct CoolingUnits
    density  :: Float64
    length   :: Float64
    velocity :: Float64
end
time_unit(u::CoolingUnits) = u.length / u.velocity

"""
    CloudyCooling

A Grackle Cloudy-mode table on the device + its parameters.  Build with
`load_cloudy_cooling(dir; be, units, metallicity, …)`.
"""
struct CloudyCooling{V}
    tab      :: V                 # 5 × nn × nt: prim_cool, prim_heat, met_cool, met_heat, mmw
    nn :: Int; nt :: Int
    n0 :: Float64; dn :: Float64  # log₁₀ n_H grid
    t0 :: Float64; dT :: Float64  # log₁₀ T grid
    units      :: CoolingUnits
    Zrel       :: Float64         # Z / Z☉ (scales the metal channel)
    XH         :: Float64
    Zsun       :: Float64
    Tfloor     :: Float64
    eps        :: Float64         # sub-cycle energy fraction (Grackle: 0.1)
    maxsub     :: Int
    source     :: String
end

# minimal .npy reader (little-endian f8, C order, ≤ 2-D) — avoids an HDF5/NPZ dep
function _read_npy(path::AbstractString)
    open(path) do io
        magic = read(io, 6); magic == UInt8[0x93, UInt8('N'), UInt8('U'), UInt8('M'), UInt8('P'), UInt8('Y')] ||
            error("$path: not a .npy file")
        major = read(io, UInt8); read(io, UInt8)
        hl = major == 1 ? Int(read(io, UInt16)) : Int(read(io, UInt32))
        hdr = String(read(io, hl))
        occursin("'<f8'", hdr) || error("$path: need little-endian float64 ('<f8')")
        occursin("'fortran_order': False", hdr) || error("$path: need C order")
        m = match(r"'shape':\s*\(([^)]*)\)", hdr)
        dims = Tuple(parse(Int, s) for s in split(m.captures[1], ',') if !isempty(strip(s)))
        data = Vector{Float64}(undef, prod(dims; init = 1))
        read!(io, data)
        return length(dims) <= 1 ? data : permutedims(reshape(data, reverse(dims)), length(dims):-1:1)
    end
end

"""
    load_cloudy_cooling(dir; be, units, metallicity = 1.0, Tfloor = nothing,
                        XH = 0.7158, Zsun = 0.01295, eps = 0.1, maxsub = 100_000)

Load a table exported by `tools/cloudy_table_export.py` (directory `dir`).
`metallicity` is Z/Z☉; `Tfloor` defaults to the table's lowest temperature.
"""
function load_cloudy_cooling(dir::AbstractString; be, units::CoolingUnits,
                             metallicity::Real = 1.0, Tfloor = nothing,
                             XH::Real = 1 / (1 + 0.1 * 3.971),
                             Zsun::Real = 0.01295, eps::Real = 0.1, maxsub::Int = 100_000,
                             T::Type = Float64)
    lognH = _read_npy(joinpath(dir, "lognH.npy")); logT = _read_npy(joinpath(dir, "logT.npy"))
    # Grackle treats the grids as uniform with the spacing from the end points
    # (interpolate_*_g: dgridPar = (grid[N] − grid[1])/(N − 1)); the file's T values
    # are rounded at ~1e-5
    for g in (lognH, logT)
        d = diff(g); maximum(abs.(d .- d[1])) < 1e-3 * abs(d[1]) || error("table grid must be uniform")
    end
    nn, nt = length(lognH), length(logT)
    tabs = [_read_npy(joinpath(dir, f * ".npy")) for f in ("prim_cool", "prim_heat", "met_cool", "met_heat", "mmw")]
    for t in tabs; size(t) == (nn, nt) || error("table shape $(size(t)) ≠ ($nn, $nt)"); end
    flat = zeros(T, 5 * nn * nt)
    for (q, t) in enumerate(tabs), j in 1:nt, i in 1:nn
        # the files hold Λ/n_H², H/n_H² LINEARLY; Grackle interpolates their log₁₀
        # (initialize_cloudy_data) — mmw stays linear
        v = q <= 4 ? log10(max(t[i, j], 1e-300)) : t[i, j]
        flat[((q - 1) * nt + (j - 1)) * nn + i] = T(v)
    end
    src = isfile(joinpath(dir, "SOURCE.txt")) ? strip(read(joinpath(dir, "SOURCE.txt"), String)) : dir
    return CloudyCooling(to_device(be, flat, T), nn, nt, lognH[1], (lognH[end] - lognH[1]) / (nn - 1),
                         logT[1], (logT[end] - logT[1]) / (nt - 1), units, Float64(metallicity), Float64(XH),
                         Float64(Zsun), Tfloor === nothing ? 10.0^logT[1] : Float64(Tfloor),
                         Float64(eps), maxsub, String(src))
end

# bilinear in (log n_H, log T), inputs clamped to the table (Grackle: no extrapolation
# of the cooling rates outside the table range)
@inline function _tab2(tab, q::Int, ln::T, lt::T, nn::Int, nt::Int, n0::T, dn::T, t0::T, dT::T) where {T}
    x = clamp((ln - n0) / dn, zero(T), T(nn - 1)); y = clamp((lt - t0) / dT, zero(T), T(nt - 1))
    i = min(unsafe_trunc(Int, x), nn - 2); j = min(unsafe_trunc(Int, y), nt - 2)
    fx = x - T(i); fy = y - T(j)
    b = (q - 1) * nt * nn
    @inbounds begin
        v00 = tab[b + j * nn + i + 1];       v10 = tab[b + j * nn + i + 2]
        v01 = tab[b + (j + 1) * nn + i + 1]; v11 = tab[b + (j + 1) * nn + i + 2]
    end
    return (one(T) - fy) * ((one(T) - fx) * v00 + fx * v10) + fy * ((one(T) - fx) * v01 + fx * v11)
end

# Grackle's calc_temp1d_cloudy: T(e) with the damped μ fixed point (+ metal particles)
@inline function _cloudy_T(e::T, ln::T, zfrac::T, tab, nn::Int, nt::Int,
                           n0::T, dn::T, t0::T, dT::T, gm1utem::T, Tmin::T) where {T}
    munew = one(T); muold = one(T); Tg = one(T)
    for _ in 1:20
        muold = munew
        Tg = max(gm1utem * e * munew, Tmin)
        munew = _tab2(tab, 5, ln, log10(Tg), nn, nt, n0, dn, t0, dT)
        munew = T(0.5) * (munew + muold)
        Tg = Tg * munew / muold
        if abs(munew / muold - one(T)) <= T(1e-2)
            muold = munew
            munew = one(T) / ((one(T) - zfrac) / munew + zfrac / T(16))   # metal particles
            Tg = Tg * munew / muold
            break
        end
    end
    return Tg, munew
end

# net volumetric rate [erg cm⁻³ s⁻¹]
@inline function _cloudy_edot(nH::T, Tg::T, ln::T, Zrel::T, tab, nn::Int, nt::Int,
                              n0::T, dn::T, t0::T, dT::T) where {T}
    lt = log10(Tg)
    pc = _tab2(tab, 1, ln, lt, nn, nt, n0, dn, t0, dT); ph = _tab2(tab, 2, ln, lt, nn, nt, n0, dn, t0, dT)
    mc = _tab2(tab, 3, ln, lt, nn, nt, n0, dn, t0, dT); mh = _tab2(tab, 4, ln, lt, nn, nt, n0, dn, t0, dT)
    return nH * nH * ((exp10(ph) - exp10(pc)) + Zrel * (exp10(mh) - exp10(mc)))
end

# integrate the specific internal energy e (cgs) over dt (s); returns e_new
@inline function _cloudy_integrate(e0::T, ρ::T, dt::T, tab, nn::Int, nt::Int, n0::T, dn::T,
                                   t0::T, dT::T, XH::T, zfrac::T, Zrel::T, gm1utem::T,
                                   Tfloor::T, eps0::T, maxsub::Int) where {T}
    nH = XH * ρ / T(_MH_CGS)
    ln = log10(nH)
    # energy floor at T_floor (MinGasTemp-like): a cell arriving with e ≪ e(T_floor)
    # — e ≤ 0 clamped, from a hydro/MHD pressure floor — would otherwise climb
    # back through ~10 % heating sub-steps (thousands of them, stalling its warp)
    μf = _tab2(tab, 5, ln, log10(Tfloor), nn, nt, n0, dn, t0, dT)
    μf = one(T) / ((one(T) - zfrac) / μf + zfrac / T(16))
    e = max(e0, Tfloor / (gm1utem * μf)); t = zero(T)
    for _ in 1:maxsub
        t >= dt && break
        Tg, μ = _cloudy_T(e, ln, zfrac, tab, nn, nt, n0, dn, t0, dT, gm1utem, Tfloor)
        edot = _cloudy_edot(nH, Tg, ln, Zrel, tab, nn, nt, n0, dn, t0, dT) / ρ
        # Grackle (solve_rate_cool_g, energy-only path): at the floor cooling stops;
        # Δt_sub = min(0.1·e/|ė|, remaining) — no damping, so a cell near its
        # equilibrium oscillates within ±10 % exactly as in Grackle
        (Tg <= T(1.01) * Tfloor && edot < zero(T)) && break
        edot == zero(T) && break
        h = min(eps0 * e / abs(edot), dt - t)
        e = e + edot * h
        t += h
    end
    return max(e, floatmin(T))
end

# per active cell; MHD (ismhd): e from E − KE − ½B², write E; hydro: e from Ge, write Ge+Tau
@kernel function _cool_k!(D, S1, S2, S3, E, Ge, Bx, By, Bz, @Const(live_d),
                          @Const(Dsc), @Const(Ssc), @Const(Esc), @Const(Gsc), @Const(tab),
                          dt::T, uρ::T, uv2::T, nn::Int, nt::Int, n0::T, dn::T, t0::T, dT::T,
                          XH::T, zfrac::T, Zrel::T, gm1utem::T, Tfloor::T, epsc::T, maxsub::Int,
                          mhd::Bool, B::Int32, ng::Int32, nd::Int32, stride::Int32,
                          dimv::Val{DIM}) where {T,DIM}
    t = @index(Global)
    t0_ = Int32(t) - Int32(1)
    B3 = _nblk(B, dimv)
    bi = t0_ ÷ B3; c = t0_ % B3
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        idx = (slot - Int32(1)) * stride + _lidx(_cell_ijk(c, B, ng, dimv)..., nd)
        dsc = T(Dsc[slot]); esc = T(Esc[slot]); gsc = T(Gsc[slot]); ssc = T(Ssc[slot])
        ρ = T(D[idx]) * dsc
        if ρ > zero(T)
            ec = if mhd
                s1 = T(S1[idx]) * ssc; s2 = T(S2[idx]) * ssc; s3 = T(S3[idx]) * ssc
                bx = T(Bx[idx]); by = T(By[idx]); bz = T(Bz[idx])
                (T(E[idx]) * esc - T(0.5) * (s1*s1 + s2*s2 + s3*s3) / ρ -
                 T(0.5) * (bx*bx + by*by + bz*bz)) / ρ
            else
                T(Ge[idx]) * gsc / ρ
            end
            ec = max(ec, floatmin(T))
            en = _cloudy_integrate(ec * uv2, ρ * uρ, dt, tab, nn, nt, n0, dn, t0, dT, XH, zfrac,
                                   Zrel, gm1utem, Tfloor, epsc, maxsub) / uv2
            Δ = ρ * (en - ec)
            E[idx]  = _narrow(eltype(E), (T(E[idx]) * esc + Δ) / esc)
            Ge[idx] = _narrow(eltype(Ge), ρ * en / gsc)
        end
    end
end

"""
    cool_level!(hier, l, dt, cool::CloudyCooling)

Apply tabulated cooling/heating to every active cell of level `l` over the
code-time `dt` (operator split; R buffers).
"""
function cool_level!(hier::AMRHierarchy, l::Int, dt::Real, cool::CloudyCooling)
    lev = hier.levels[l + 1]
    nb = length(lev.live); nb == 0 && return nothing
    T = eltype(cool.tab); u = cool.units
    γ = hier.gamma
    gm1utem = T((γ - 1) * _MH_CGS / _KB_CGS)        # T = (γ−1) μ m_H e / k_B with e in cgs
    zfrac = T(cool.Zrel * cool.Zsun)
    mhd = ismhd(lev)
    Bs = mhd ? mfields(lev)[1:3] : (lev.D, lev.D, lev.D)
    _cool_k!(lev.be)(gasfields(lev)[1:6]..., Bs..., lev.live_d, lev.Dsc, lev.Ssc, lev.Esc, lev.Gsc,
                     cool.tab, T(dt * time_unit(u)), T(u.density), T(u.velocity^2),
                     cool.nn, cool.nt, T(cool.n0), T(cool.dn), T(cool.t0), T(cool.dT),
                     T(cool.XH), zfrac, T(cool.Zrel), gm1utem, T(cool.Tfloor), T(cool.eps),
                     cool.maxsub, mhd, Int32(lev.B), Int32(lev.ng), Int32(lev.nd),
                     Int32(lev.stride), Val(lev.dim); ndrange = nb * _nact(lev))
    return nothing
end

# ── host-side helpers (ICs, equilibria, diagnostics) ──────────────────────────
"Temperature [K] and μ of gas with density ρ [g cm⁻³] and specific energy e [erg g⁻¹]."
function cloudy_temperature(c::CloudyCooling, ρ::Real, e::Real; gamma::Real = 5/3)
    tab = Float64.(Array(c.tab)); nH = c.XH * ρ / _MH_CGS
    _cloudy_T(Float64(e), log10(nH), c.Zrel * c.Zsun, tab, c.nn, c.nt, c.n0, c.dn, c.t0, c.dT,
              (gamma - 1) * _MH_CGS / _KB_CGS, 1.0)
end

"μ(n_H, T) from the table (with Grackle's metal-particle correction)."
function cloudy_mu(c::CloudyCooling, nH::Real, Tg::Real)
    tab = Float64.(Array(c.tab)); zf = c.Zrel * c.Zsun
    μ = _tab2(tab, 5, log10(Float64(nH)), log10(Float64(Tg)), c.nn, c.nt, c.n0, c.dn, c.t0, c.dT)
    return 1 / ((1 - zf) / μ + zf / 16)
end

"Net heating − cooling ė [erg cm⁻³ s⁻¹] at (n_H, T)."
function cloudy_edot(c::CloudyCooling, nH::Real, Tg::Real)
    tab = Float64.(Array(c.tab))
    _cloudy_edot(Float64(nH), Float64(Tg), log10(Float64(nH)), c.Zrel, tab, c.nn, c.nt,
                 c.n0, c.dn, c.t0, c.dT)
end

"""
    cloudy_isobar(c, P_over_k; T = 10 .^ range(1, 7, 601)) -> NamedTuple

Along the isobar P/k = n_tot T: per temperature the hydrogen density, μ, net
ė, and |cooling time| (s); `Teq` lists the temperatures where ė changes sign
(stable equilibria have ė going from + to − with increasing T).
"""
function cloudy_isobar(c::CloudyCooling, Pk::Real; T = 10 .^ range(1, 7, 601), gamma::Real = 5/3)
    nH = similar(collect(Float64, T)); μs = similar(nH); ed = similar(nH); tc = similar(nH)
    for (k, Tg) in enumerate(T)
        μ = 1.0; ntot = Pk / Tg
        for _ in 1:30                       # n_tot = ρ/(μ m_H) → n_H = X_H μ n_tot
            n = c.XH * μ * ntot
            μn = cloudy_mu(c, n, Tg); abs(μn / μ - 1) < 1e-10 && (μ = μn; break); μ = μn
        end
        nH[k] = c.XH * μ * ntot; μs[k] = μ
        ed[k] = cloudy_edot(c, nH[k], Tg)
        ρ = nH[k] * _MH_CGS / c.XH
        tc[k] = (Pk * _KB_CGS / (gamma - 1)) / abs(ed[k])
    end
    Teq = Float64[]
    for k in 1:length(T)-1
        sign(ed[k]) != sign(ed[k+1]) && push!(Teq, T[k] - ed[k] * (T[k+1] - T[k]) / (ed[k+1] - ed[k]))
    end
    return (T = collect(Float64, T), nH, mu = μs, edot = ed, tcool = tc, Teq)
end
