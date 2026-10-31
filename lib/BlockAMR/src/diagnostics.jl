# diagnostics.jl — on-device leaf reductions (no field copies to the host).
#
# `leaf_moments` sums, over every LEAF cell (cells covered by a finer level are
# skipped through the cached coverage mask, chem.jl `_cov_mask!`), the total
# mass and the mass / mass-weighted position / x-velocity of gas in temperature
# bands, plus the mass above a density threshold.  Temperature comes from the
# Cloudy table exactly as the cooling sees it (the dual-energy thermal energy,
# Grackle's μ fixed point), per-cell terms in the table's precision, sums in Float64.  One kernel per level writes per-cell contributions;
# a device reduction returns 7 numbers per level.  2-D and 3-D (z ignored in the
# position moments).

@kernel function _leaf_moments_k!(out, @Const(cov), D, S1, S2, S3, E, Ge, Bx, By, Bz,
                                  @Const(live_d), @Const(org), @Const(Dsc), @Const(Ssc),
                                  @Const(Esc), @Const(Gsc), γ::T, mhd::Bool, de::Bool, eta1::T,
                                  dA::T, dx::T, @Const(tab), nn::Int, nt::Int, n0::T, dn::T,
                                  t0::T, dT::T, XH::T, zfrac::T, gm1utem::T, Tfloor::T,
                                  uρ::T, uv2::T, Tcold::T, Tmix::T, ρdense::T,
                                  B::Int32, ng::Int32, nd::Int32, stride::Int32, n::Int,
                                  dimv::Val{DIM}) where {T,DIM}
    t = @index(Global)
    t0_ = Int32(t) - Int32(1)
    B3 = _nblk(B, dimv)
    bi = t0_ ÷ B3; c = t0_ % B3
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        i, j, k = _cell_ijk(c, B, ng, dimv)
        idx = (slot - Int32(1)) * stride + _lidx(i, j, k, nd)
        m = zero(T); mc = zero(T); xm = zero(T); ym = zero(T); vm = zero(T); mi = zero(T); md = zero(T)
        if cov[idx] == 0x00
            ρ = T(D[idx]) * T(Dsc[slot])
            s1 = T(S1[idx]) * T(Ssc[slot]); s2 = T(S2[idx]) * T(Ssc[slot]); s3 = T(S3[idx]) * T(Ssc[slot])
            KE = T(0.5) * (s1*s1 + s2*s2 + s3*s3) / ρ
            eint = if mhd
                EB = T(0.5) * (T(Bx[idx])^2 + T(By[idx])^2 + T(Bz[idx])^2)
                e5 = T(E[idx]) * T(Esc[slot])
                (de ? _de_total(e5, KE, EB, T(Ge[idx]) * T(Gsc[slot]), eta1) : e5) - KE - EB
            else
                T(Ge[idx]) * T(Gsc[slot])
            end
            ec = max(eint / ρ, floatmin(T)) * uv2
            ln = log10(XH * ρ * uρ / T(_MH_CGS))
            Tg, _ = _cloudy_T(ec, ln, zfrac, tab, nn, nt, n0, dn, t0, dT, gm1utem, Tfloor)
            m = ρ * dA
            if Tg < Tcold
                mc = m
                xm = m * (T(org[DIM * bi + 1]) + T(i - ng) + T(0.5)) * dx
                ym = m * (T(org[DIM * bi + 2]) + T(j - ng) + T(0.5)) * dx
                vm = s1 * dA
            elseif Tg < Tmix
                mi = m
            end
            ρ > ρdense && (md = m)
        end
        o = Int(t)                                  # per-cell terms in T, sums in Float64
        out[o] = Float64(m); out[n + o] = Float64(mc); out[2n + o] = Float64(xm)
        out[3n + o] = Float64(ym); out[4n + o] = Float64(vm); out[5n + o] = Float64(mi)
        out[6n + o] = Float64(md)
    end
end

"""
    leaf_moments(hier, cooling; Tcold = 2e4, Tmix = 5e5, rho_dense = Inf)
        -> (; mass, mcold, xc, yc, vxc, mint, mdense)

On-device sums over all leaf cells (code units): total mass; mass, centroid and
mean x-velocity of gas below `Tcold` [K]; mass between `Tcold` and `Tmix`; mass
with ρ > `rho_dense` (code).  Temperatures from the cooling table `cooling`
(a `CloudyCooling`), with the dual-energy thermal energy.  Area/volume element
dx^DIM of each level.  Requires current level tables (the coverage mask comes
from the child's restriction table).
"""
function leaf_moments(hier::AMRHierarchy, c::CloudyCooling; Tcold::Real = 2e4, Tmix::Real = 5e5,
                      rho_dense::Real = Inf)
    T = eltype(c.tab)                    # per-cell arithmetic in the table precision
    acc = zeros(7)
    u = c.units
    for l in 0:length(hier.levels)-1
        lev = hier.levels[l + 1]; isempty(lev.live) && continue
        nb = length(lev.live); n = nb * _nact(lev)
        cov = _cov_mask!(hier, l)
        dim = lev.dim
        org = to_device(lev.be, Int32[Int32(lev.meta[s].origin[d]) for s in lev.live for d in 1:dim], Int32)
        out = device_zeros(lev.be, Float64, (7n,))
        mhd = ismhd(lev)
        Bs = mhd ? mfields(lev)[1:3] : (lev.D, lev.D, lev.D)
        dx = level_dx(hier, l)
        _leaf_moments_k!(lev.be)(out, cov, gasfields(lev)..., Bs..., lev.live_d, org,
                                 lev.Dsc, lev.Ssc, lev.Esc, lev.Gsc, T(hier.gamma), mhd,
                                 mhd && hier.mhd.de, T(mhd ? hier.mhd.eta1 : 0.0), T(dx^dim), T(dx),
                                 c.tab, c.nn, c.nt, T(c.n0), T(c.dn), T(c.t0), T(c.dT), T(c.XH),
                                 T(c.Zrel * c.Zsun), T((hier.gamma - 1) * _MH_CGS / _KB_CGS),
                                 T(c.Tfloor), T(u.density), T(u.velocity^2), T(Tcold), T(Tmix),
                                 T(rho_dense), Int32(lev.B), Int32(lev.ng), Int32(lev.nd),
                                 Int32(lev.stride), n, Val(dim); ndrange = n)
        acc .+= vec(Array(sum(reshape(out, n, 7); dims = 1)))
    end
    m, mc, xm, ym, vm, mi, md = acc
    return (mass = m, mcold = mc, xc = mc > 0 ? xm / mc : NaN, yc = mc > 0 ? ym / mc : NaN,
            vxc = mc > 0 ? vm / mc : NaN, mint = mi, mdense = md)
end
