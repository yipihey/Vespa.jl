# jeans_profile.jl — radial profiles around the baryon density maximum, dumped
# each time the collapse triggers a new refinement level.  Spherically averages
# the LEAF cells (finest covering, no double-count) within `rmax` of the peak
# into log-spaced radial bins: gas-mass-weighted ρ_gas, T, x_HII, f_H2, radial
# velocity, plus the enclosed gas mass profile.  Physical units via the per-a
# cosmo unit scalings.  Pairs with the Jeans refinement criterion (256 cells per
# Jeans length) to capture a cooling halo's runaway collapse.
module JeansProfile
using Printf
import BlockAMR, ChemistryKernels

"""
Baryon density peak of the DEEPEST populated level — i.e. the collapsing core
that is being deeply refined, NOT the global max (which may be a different,
coarsely-refined halo).  Centering the profile + zoom tracking here locks both
onto the actual collapse.  Scans from the finest level down, returns the first
(deepest) level's peak cell.
"""
function find_baryon_peak(hier)
    for l in (length(hier.levels)-1):-1:0
        lev = hier.levels[l+1]; isempty(lev.live) && continue
        Nl = size_at(hier, l)
        hD = Array(lev.D); hsc = Array(lev.Dsc)
        best = -Inf; ppos = (0.0, 0.0, 0.0)
        for s in lev.live
            m = lev.meta[s]; base = (Int(s)-1)*lev.stride
            for k in 1:lev.B, j in 1:lev.B, i in 1:lev.B
                idx = base + ((lev.ng+k-1)*lev.nd+(lev.ng+j-1))*lev.nd+(lev.ng+i-1)+1
                ρ = Float64(hD[idx])*hsc[s]
                if ρ > best
                    best = ρ
                    ppos = ((Float64(m.origin[1])+i-0.5)/Nl,
                            (Float64(m.origin[2])+j-0.5)/Nl,
                            (Float64(m.origin[3])+k-0.5)/Nl)
                end
            end
        end
        return ppos, best, l
    end
    return (0.0,0.0,0.0), 0.0, 0
end

size_at(hier, l) = hier.nbase[1] * 2^l

# periodic minimal-image separation (box units)
@inline _dwrap(a, b) = (d = a - b; d - round(d))

"""
    dump_radial_profile(io_or_path, hier, c, u; peak, z, level, gamma, μ, XH,
                        rmax, nbin) -> (npts, path)

Bin leaf cells within `rmax` (box units) of `peak` into `nbin` log-spaced radial
shells; write `# r_mpch rho_gas T_K xHII fH2 vr_kms Menc_Msun ncell` rows.
`rho_gas` code units (level-0 gas mean = f_b); radius/mass in physical Mpc/h,M⊙.
"""
function dump_radial_profile(path, hier, c, u; peak, z, level, gamma, μ, XH,
                             rmax=0.05, nbin=56)
    B = hier.B
    box_mpch = c.box                                  # Mpc/h
    rmin = 0.5 / size_at(hier, length(hier.levels)-1) # half finest cell (box units)
    lr0 = log10(rmin); lr1 = log10(rmax); dlr = (lr1 - lr0)/nbin
    Σm  = zeros(nbin); ΣmT = zeros(nbin); Σmx = zeros(nbin); Σmf = zeros(nbin)
    Σmv = zeros(nbin); Σv  = zeros(nbin); nc = zeros(Int, nbin)
    # cell mass in M⊙: ρ_code · (mean_matter_density) · dV_phys.  ρ_code is relative
    # to the total-matter mean (=1 for total; gas D has mean f_b), so ρ_code · ρ̄_m ·
    # dV gives the gas mass.  ρ̄_m = Ω_m·ρ_crit; ρ_crit = 2.775e11 h² M⊙/Mpc³.
    ρ̄m_Msun_mpch3 = c.Om * 2.775e11                    # M⊙/h / (Mpc/h)³ · h²… (h-units cancel below)
    for l in 0:length(hier.levels)-1
        lev = hier.levels[l+1]; isempty(lev.live) && continue
        Nl = size_at(hier, l); dx = 1.0/Nl            # box units
        dV_mpch3 = (dx*box_mpch)^3                     # (Mpc/h)³
        # child-octant coverage (skip cells covered by a finer block)
        cov = Dict{Int32,UInt8}()
        if l+2 <= length(hier.levels)
            for cs in hier.levels[l+2].live
                cm = hier.levels[l+2].meta[cs]
                ob = (cm.offset[1]>0 ? 1 : 0) | ((cm.offset[2]>0 ? 1 : 0)<<1) | ((cm.offset[3]>0 ? 1 : 0)<<2)
                cov[cm.parent] = get(cov, cm.parent, 0x00) | (UInt8(1)<<ob)
            end
        end
        hD=Array(lev.D); hG=Array(lev.Ge); hS1=Array(lev.S1); hS2=Array(lev.S2); hS3=Array(lev.S3)
        hsc=Array(lev.Dsc); hesc=Array(lev.Gsc); hssc=Array(lev.Ssc)  # Ge on its own scale Gsc
        h1=Array(lev.sp[1]); h2=Array(lev.sp[2]); Bh=B÷2
        for s in lev.live
            m = lev.meta[s]
            bc = ntuple(d->(Float64(m.origin[d])+B/2)/Nl, 3)
            bd = sqrt(sum(d->_dwrap(bc[d], peak[d])^2, 1:3))
            bd > rmax + B*0.87/Nl && continue          # block too far
            cb = get(cov, s, 0x00); base=(Int(s)-1)*lev.stride
            for k in 1:B, j in 1:B, i in 1:B
                ob = (i>Bh ? 1 : 0) | ((j>Bh ? 1 : 0)<<1) | ((k>Bh ? 1 : 0)<<2)
                (cb>>ob)&0x01 == 0x01 && continue      # covered by child
                x=(Float64(m.origin[1])+i-0.5)/Nl; y=(Float64(m.origin[2])+j-0.5)/Nl; zc=(Float64(m.origin[3])+k-0.5)/Nl
                r = sqrt(_dwrap(x,peak[1])^2 + _dwrap(y,peak[2])^2 + _dwrap(zc,peak[3])^2)
                (r < rmin || r >= rmax) && continue
                bin = clamp(floor(Int,(log10(r)-lr0)/dlr)+1, 1, nbin)
                idx = base + ((lev.ng+k-1)*lev.nd+(lev.ng+j-1))*lev.nd+(lev.ng+i-1)+1
                ρ = Float64(hD[idx])*hsc[s]; ρ<=0 && continue
                e = Float64(hG[idx])*hesc[s]/ρ
                T = e*(gamma-1)*μ*u.T2
                # velocity (code units) = momentum_phys/density_phys = (S·Ssc)/(D·Dsc)
                srat = Float64(hssc[s])/hsc[s]
                dcode = max(Float64(hD[idx]), 1e-30)
                v1=Float64(hS1[idx])/dcode*srat; v2=Float64(hS2[idx])/dcode*srat; v3=Float64(hS3[idx])/dcode*srat
                # radial unit vector (peak → cell), periodic; vr in km/s
                rx=_dwrap(x,peak[1]); ry=_dwrap(y,peak[2]); rz=_dwrap(zc,peak[3]); rr=max(r,1e-30)
                vr = (v1*rx+v2*ry+v3*rz)/rr * u.v/1e5
                mgas = ρ * ρ̄m_Msun_mpch3 * dV_mpch3    # M⊙/h
                xh = ChemistryKernels.decode_log2sp(Float64, h1[idx])/XH
                fh = ChemistryKernels.decode_log2sp(Float64, h2[idx])/XH
                Σm[bin]+=mgas; ΣmT[bin]+=mgas*T; Σmx[bin]+=mgas*xh; Σmf[bin]+=mgas*fh
                Σmv[bin]+=mgas*vr; Σv[bin]+=dV_mpch3; nc[bin]+=1
            end
        end
    end
    Menc = 0.0
    open(path, "w") do io
        @printf(io, "# radial profile around baryon peak (%.5f,%.5f,%.5f) z=%.3f newlevel=%d\n",
                peak[1],peak[2],peak[3],z,level)
        println(io, "# r_mpch  rho_gas_code  T_K  xHII  fH2  vr_kms  Menc_gas_Msunh  ncell")
        for b in 1:nbin
            r_mid = 10.0^(lr0 + (b-0.5)*dlr) * box_mpch    # Mpc/h
            Menc += Σm[b]
            nc[b]==0 && continue
            ρg = Σm[b]/(c.Om*2.775e11) / Σv[b]             # back to code units: mgas/(ρ̄m·dV)
            @printf(io, "%.6e %.6e %.4e %.6e %.6e %.4e %.6e %d\n",
                    r_mid, ρg, ΣmT[b]/Σm[b], Σmx[b]/Σm[b], Σmf[b]/Σm[b], Σmv[b]/Σm[b], Menc, nc[b])
        end
    end
    return sum(nc), path
end
end # module
