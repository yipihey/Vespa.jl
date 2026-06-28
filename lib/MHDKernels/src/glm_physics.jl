# ── GLM-MHD device functions (precision-generic, @fastmath) ───────────────────
# Ideal MHD + Dedner mixed-GLM divergence cleaning. Conserved U and primitive W:
#   U = (ρ, ρvx, ρvy, ρvz, E, Bx, By, Bz, ψ)
#   W = (ρ, vx, vy, vz, p,  Bx, By, Bz, ψ)
# Reconstruction MUSCL-Hancock + PLM(MonCen); Riemann HLLD (Miyoshi & Kusano 2005)
# with an LLF (Rusanov) fallback for low ρ/p; the (Bn,ψ) pair is cleaned per face
# (glm_pair). All functions are pure, allocation-free, `@inline`, and parameterised
# on `T` so they compile to CPU (T=Float64) and GPU (T=Float32). `@fastmath` lowers
# the trace's many min/max/abs to hardware intrinsics (the dominant perf factor).

@fastmath @inline function cons2prim(c::NTuple{9,T}, γ::T, smallr::T, pfl::T) where {T}
    rho = max(c[1], smallr); ir = one(T)/rho
    vx=c[2]*ir; vy=c[3]*ir; vz=c[4]*ir; bx=c[6]; by=c[7]; bz=c[8]
    ekin = T(0.5)*(c[2]*c[2]+c[3]*c[3]+c[4]*c[4])*ir
    emag = T(0.5)*(bx*bx+by*by+bz*bz)
    p = max((γ-one(T))*(c[5]-ekin-emag), pfl)
    (rho,vx,vy,vz,p,bx,by,bz,c[9])
end
@inline function prim2cons(q::NTuple{9,T}, γ::T) where {T}
    rho=q[1];vx=q[2];vy=q[3];vz=q[4];p=q[5];bx=q[6];by=q[7];bz=q[8]
    E = p/(γ-one(T))+T(0.5)*rho*(vx*vx+vy*vy+vz*vz)+T(0.5)*(bx*bx+by*by+bz*bz)
    (rho,rho*vx,rho*vy,rho*vz,E,bx,by,bz,q[9])
end
@fastmath @inline function fast_speed(q::NTuple{9,T}, γ::T, bn::T) where {T}
    rho=q[1]; c2=γ*q[5]/rho; b2=(q[6]*q[6]+q[7]*q[7]+q[8]*q[8])/rho; d2=T(0.5)*(b2+c2)
    sqrt(d2 + sqrt(max(d2*d2 - c2*bn*bn/rho, zero(T))))
end
@inline function phys_flux_x(q::NTuple{9,T}, γ::T) where {T}
    rho=q[1];vx=q[2];vy=q[3];vz=q[4];p=q[5];bx=q[6];by=q[7];bz=q[8]
    b2=bx*bx+by*by+bz*bz; ptot=p+T(0.5)*b2
    E=p/(γ-one(T))+T(0.5)*rho*(vx*vx+vy*vy+vz*vz)+T(0.5)*b2; vb=vx*bx+vy*by+vz*bz
    (rho*vx, rho*vx*vx+ptot-bx*bx, rho*vx*vy-bx*by, rho*vx*vz-bx*bz,
     (E+ptot)*vx-bx*vb, zero(T), vx*by-vy*bx, vx*bz-vz*bx, zero(T))
end
# Rotate so direction `dir` is the x-normal (cyclic v & B permutation).
@inline function rot_to(q::NTuple{9,T}, dir::Int) where {T}
    dir==1 ? q :
    dir==2 ? (q[1],q[3],q[4],q[2],q[5],q[7],q[8],q[6],q[9]) :
             (q[1],q[4],q[2],q[3],q[5],q[8],q[6],q[7],q[9])
end
@inline function rot_flux_from(f::NTuple{9,T}, dir::Int) where {T}
    dir==1 ? f :
    dir==2 ? (f[1],f[4],f[2],f[3],f[5],f[8],f[6],f[7],f[9]) :
             (f[1],f[3],f[4],f[2],f[5],f[7],f[8],f[6],f[9])
end
# MonCen slope limiter (slope_type=2); @fastmath ⇒ hardware fmin/fmax, branchless.
@fastmath @inline function moncen(dl::T, dr::T) where {T}
    dc = T(0.5)*(dl+dr); sgn = ifelse(dc>=zero(T), one(T), -one(T))
    val = sgn*min(T(2)*min(abs(dl),abs(dr)), abs(dc))
    ifelse(dl*dr <= zero(T), zero(T), val)
end
@fastmath @inline prim_slope(L::NTuple{9,T},M::NTuple{9,T},R::NTuple{9,T}) where {T} =
    ntuple(i->T(0.5)*moncen(M[i]-L[i], R[i]-M[i]), 9)
@inline padd(q::NTuple{9,T}, s::NTuple{9,T}, a::T) where {T} = ntuple(i->q[i]+a*s[i], 9)
@inline dir_flux(q::NTuple{9,T}, dir::Int, γ::T) where {T} = rot_flux_from(phys_flux_x(rot_to(q,dir),γ), dir)

# MUSCL-Hancock half-step predictor: U^{n+1/2} = U^n - ½dt/dx · Σ_d (F(W+s_d)-F(W-s_d)).
@fastmath @inline function hancock(m0::NTuple{9,T}, sx,sy,sz, dtdx::T, γ::T) where {T}
    u0 = prim2cons(m0,γ); h = T(0.5)*dtdx
    fxp=dir_flux(padd(m0,sx,one(T)),1,γ); fxm=dir_flux(padd(m0,sx,-one(T)),1,γ)
    fyp=dir_flux(padd(m0,sy,one(T)),2,γ); fym=dir_flux(padd(m0,sy,-one(T)),2,γ)
    fzp=dir_flux(padd(m0,sz,one(T)),3,γ); fzm=dir_flux(padd(m0,sz,-one(T)),3,γ)
    ntuple(i->u0[i]-h*((fxp[i]-fxm[i])+(fyp[i]-fym[i])+(fzp[i]-fzm[i])), 9)
end
# ── Reconstruction options: PLM (MonCen) and local PPM (CW84), unified as the ──
# per-variable face OFFSETS (δL,δR) from the cell mean — face⁻ = m0+δL, face⁺ = m0+δR.
# Both use only a ±1 stencil, so each fits the cube's 2-cell halo with no resize.
# PLM: δR=+½·moncen, δL=-½·moncen (recovers prim_slope/padd exactly). PPM: the
# monotonized local parabola edges (Ustyugov-style local stencil + CW84 monotonize).
@inline _ppm_edges_unlim(qm::T,q0::T,qp::T) where {T} =
    (slope=(qp-qm)*T(0.25); curve=(qm-T(2)*q0+qp)/T(12); (q0-slope+curve, q0+slope+curve))
@inline function _ppm_monotonize(qL::T,qa::T,qR::T) where {T}
    dq=qR-qL; qmid=T(0.5)*(qL+qR); dq2_6=dq*dq/T(6); diff=(qa-qmid)*dq
    if (qR-qa)*(qa-qL) <= zero(T); return (qa,qa)
    elseif diff > dq2_6;  return (T(3)*qa-T(2)*qR, qR)
    elseif diff < -dq2_6; return (qL, T(3)*qa-T(2)*qL)
    end
    (qL,qR)
end
@inline function _ppm_edges(qm::T,q0::T,qp::T) where {T}
    qL,qR=_ppm_edges_unlim(qm,q0,qp); _ppm_monotonize(qL,q0,qR)
end

const RECON_PLM = 0
const RECON_PPM = 1
recon_code(s::Symbol) = s === :plm ? RECON_PLM : s === :ppm ? RECON_PPM :
                        error("unknown recon :$s (have :plm, :ppm)")

# Per-axis face offsets (δL,δR) for the 9 primitives, given the ∓ neighbours.
# `rec` is a `Val` so the kernel specialises to ONE reconstruction — a runtime Int
# branch would keep both code paths live and spike GPU register pressure (~1.8× slower).
@inline function recon_offsets(mm::NTuple{9,T}, m0::NTuple{9,T}, mp::NTuple{9,T}, ::Val{RECON_PLM}) where {T}
    s = prim_slope(mm, m0, mp)                     # ½·moncen per var
    (ntuple(i->-s[i],9), s)
end
@inline function recon_offsets(mm::NTuple{9,T}, m0::NTuple{9,T}, mp::NTuple{9,T}, ::Val{RECON_PPM}) where {T}
    e = ntuple(i->_ppm_edges(mm[i],m0[i],mp[i]), 9)   # (qL,qR) per var
    (ntuple(i->e[i][1]-m0[i],9), ntuple(i->e[i][2]-m0[i],9))
end

# MUSCL-Hancock ½-step predictor from per-axis edge offsets (PLM δR=+s,δL=-s ⇒ ≡ hancock).
@fastmath @inline function hancock_edges(m0::NTuple{9,T}, δLx,δRx,δLy,δRy,δLz,δRz, dtdx::T, γ::T) where {T}
    u0=prim2cons(m0,γ); h=T(0.5)*dtdx
    fxp=dir_flux(ntuple(i->m0[i]+δRx[i],9),1,γ); fxm=dir_flux(ntuple(i->m0[i]+δLx[i],9),1,γ)
    fyp=dir_flux(ntuple(i->m0[i]+δRy[i],9),2,γ); fym=dir_flux(ntuple(i->m0[i]+δLy[i],9),2,γ)
    fzp=dir_flux(ntuple(i->m0[i]+δRz[i],9),3,γ); fzm=dir_flux(ntuple(i->m0[i]+δLz[i],9),3,γ)
    ntuple(i->u0[i]-h*((fxp[i]-fxm[i])+(fyp[i]-fym[i])+(fzp[i]-fzm[i])), 9)
end

# Dedner GLM pair: clean the normal field. Returns (bn*, ψ*).
@inline function glm_pair(bnL::T,bnR::T,psiL::T,psiR::T,ch::T) where {T}
    bns = T(0.5)*(bnL+bnR)-T(0.5)*(psiR-psiL)/ch
    psis = T(0.5)*(psiL+psiR)-T(0.5)*ch*(bnR-bnL)
    bns, psis
end
@fastmath @inline function llf_x(L::NTuple{9,T},R::NTuple{9,T},γ::T,ch::T,fbn::T,fpsi::T) where {T}
    fL=phys_flux_x(L,γ); fR=phys_flux_x(R,γ); uL=prim2cons(L,γ); uR=prim2cons(R,γ)
    smax=max(abs(L[2])+fast_speed(L,γ,L[6]), abs(R[2])+fast_speed(R,γ,R[6]), ch)
    f=ntuple(i->T(0.5)*(fL[i]+fR[i])-T(0.5)*smax*(uR[i]-uL[i]), 9)
    (f[1],f[2],f[3],f[4],f[5],fbn,f[7],f[8],fpsi)
end
# Miyoshi-Kusano single-star transverse state (lifted out of hlld_x so it's not a
# captured closure — closures can inflate GPU register pressure / hurt occupancy).
@fastmath @inline function _hlld_star(d::T,u::T,v::T,w::T,by::T,bz::T,S::T,bn::T,SM::T) where {T}
    denK = d*(S-u)*(S-SM) - bn*bn
    if abs(denK) < T(1e-12)
        return (v, w, by, bz)
    else
        fac = bn*(SM-u)/denK
        return (v-by*fac, w-bz*fac, by*(d*(S-u)*(S-u)-bn*bn)/denK, bz*(d*(S-u)*(S-u)-bn*bn)/denK)
    end
end
@fastmath @inline function hlld_x(L::NTuple{9,T},R::NTuple{9,T},γ::T,ch::T,bn::T,fbn::T,fpsi::T) where {T}
    dL,uL,vL,wL,pL=L[1],L[2],L[3],L[4],L[5]; byL,bzL=L[7],L[8]
    dR,uR,vR,wR,pR=R[1],R[2],R[3],R[4],R[5]; byR,bzR=R[7],R[8]
    b2L=bn*bn+byL*byL+bzL*bzL; b2R=bn*bn+byR*byR+bzR*bzR
    ptL=pL+T(0.5)*b2L; ptR=pR+T(0.5)*b2R
    EL=pL/(γ-one(T))+T(0.5)*dL*(uL*uL+vL*vL+wL*wL)+T(0.5)*b2L
    ER=pR/(γ-one(T))+T(0.5)*dR*(uR*uR+vR*vR+wR*wR)+T(0.5)*b2R
    cfL=fast_speed(L,γ,bn); cfR=fast_speed(R,γ,bn)
    SL=min(min(uL,uR)-max(cfL,cfR), zero(T)); SR=max(max(uL,uR)+max(cfL,cfR), zero(T))
    UL=(dL,dL*uL,dL*vL,dL*wL,EL,bn,byL,bzL,zero(T)); UR=(dR,dR*uR,dR*vR,dR*wR,ER,bn,byR,bzR,zero(T))
    FL=phys_flux_x((dL,uL,vL,wL,pL,bn,byL,bzL,zero(T)),γ); FR=phys_flux_x((dR,uR,vR,wR,pR,bn,byR,bzR,zero(T)),γ)
    den=(SR-uR)*dR-(SL-uL)*dL
    SM=((SR-uR)*dR*uR-(SL-uL)*dL*uL-ptR+ptL)/den
    pts=((SR-uR)*dR*ptL-(SL-uL)*dL*ptR+dL*dR*(SR-uR)*(SL-uL)*(uR-uL))/den
    dLs=max(dL*(SL-uL)/(SL-SM),T(1e-12)); dRs=max(dR*(SR-uR)/(SR-SM),T(1e-12))
    sqdLs=sqrt(dLs); sqdRs=sqrt(dRs); SLs=SM-abs(bn)/sqdLs; SRs=SM+abs(bn)/sqdRs
    vyLs,vzLs,byLs,bzLs = _hlld_star(dL,uL,vL,wL,byL,bzL,SL,bn,SM)
    vyRs,vzRs,byRs,bzRs = _hlld_star(dR,uR,vR,wR,byR,bzR,SR,bn,SM)
    vdotbL=uL*bn+vL*byL+wL*bzL; vdotbLs=SM*bn+vyLs*byLs+vzLs*bzLs
    ELs=((SL-uL)*EL-ptL*uL+pts*SM+bn*(vdotbL-vdotbLs))/(SL-SM)
    vdotbR=uR*bn+vR*byR+wR*bzR; vdotbRs=SM*bn+vyRs*byRs+vzRs*bzRs
    ERs=((SR-uR)*ER-ptR*uR+pts*SM+bn*(vdotbR-vdotbRs))/(SR-SM)
    ULs=(dLs,dLs*SM,dLs*vyLs,dLs*vzLs,ELs,bn,byLs,bzLs,zero(T))
    URs=(dRs,dRs*SM,dRs*vyRs,dRs*vzRs,ERs,bn,byRs,bzRs,zero(T))
    sgn = bn>=zero(T) ? one(T) : -one(T); invsum=one(T)/(sqdLs+sqdRs)
    vyss=(sqdLs*vyLs+sqdRs*vyRs+(byRs-byLs)*sgn)*invsum
    vzss=(sqdLs*vzLs+sqdRs*vzRs+(bzRs-bzLs)*sgn)*invsum
    byss=(sqdLs*byRs+sqdRs*byLs+sqdLs*sqdRs*(vyRs-vyLs)*sgn)*invsum
    bzss=(sqdLs*bzRs+sqdRs*bzLs+sqdLs*sqdRs*(vzRs-vzLs)*sgn)*invsum
    vdotbss=SM*bn+vyss*byss+vzss*bzss
    ELss=ELs-sqdLs*(vdotbLs-vdotbss)*sgn; ERss=ERs+sqdRs*(vdotbRs-vdotbss)*sgn
    ULss=(dLs,dLs*SM,dLs*vyss,dLs*vzss,ELss,bn,byss,bzss,zero(T))
    URss=(dRs,dRs*SM,dRs*vyss,dRs*vzss,ERss,bn,byss,bzss,zero(T))
    if SLs>=zero(T)
        f=ntuple(i->FL[i]+SL*(ULs[i]-UL[i]), 9)
    elseif SM>=zero(T)
        f=ntuple(i->FL[i]+SLs*ULss[i]-(SLs-SL)*ULs[i]-SL*UL[i], 9)
    elseif SRs>=zero(T)
        f=ntuple(i->FR[i]+SRs*URss[i]-(SRs-SR)*URs[i]-SR*UR[i], 9)
    else
        f=ntuple(i->FR[i]+SR*(URs[i]-UR[i]), 9)
    end
    (f[1],f[2],f[3],f[4],f[5],fbn,f[7],f[8],fpsi)
end
# Riemann dispatch in direction `dir`: rotate, clean GLM, HLLD (LLF fallback), rotate back.
@fastmath @inline function riemann(Lq::NTuple{9,T},Rq::NTuple{9,T},dir::Int,
        γ::T,ch::T,smallr::T,pfl::T,llf_dmin::T,llf_pmin::T,use_hlld::Bool) where {T}
    L0=rot_to(Lq,dir); R0=rot_to(Rq,dir)
    L=(max(L0[1],smallr),L0[2],L0[3],L0[4],max(L0[5],pfl),L0[6],L0[7],L0[8],L0[9])
    R=(max(R0[1],smallr),R0[2],R0[3],R0[4],max(R0[5],pfl),R0[6],R0[7],R0[8],R0[9])
    bns,psis=glm_pair(L[6],R[6],L[9],R[9],ch); fbn=psis; fpsi=ch*ch*bns
    Lc=(L[1],L[2],L[3],L[4],L[5],bns,L[7],L[8],L[9]); Rc=(R[1],R[2],R[3],R[4],R[5],bns,R[7],R[8],R[9])
    uself=(llf_dmin>zero(T) && min(L[1],R[1])<llf_dmin)||(llf_pmin>zero(T) && min(L[5],R[5])<llf_pmin)||!use_hlld
    f = uself ? llf_x(Lc,Rc,γ,ch,fbn,fpsi) : hlld_x(Lc,Rc,γ,ch,bns,fbn,fpsi)
    rot_flux_from(f,dir)
end
