# ── Cube integrator: fused shared-memory MUSCL-Hancock GLM-MHD step (GPU path) ──
# The throughput path. One KA workitem-group owns a TB^3=4^3 block of cells; it loads
# an 8^3 primitive tile (2-cell halo) into shared, traces each interior cell's
# Hancock-predicted per-FACE interface states into 6 shared face buffers, solves the
# GLM Riemann problem ONCE per face, and updates the 4^3 owned cells. Each
# reconstruction computed once, each face Riemann solved once (vs the reference
# kernel's recompute). Identical scheme to `step_ref!` ⇒ cross-checked to f32 round-off.
#
# Pure KernelAbstractions (@localmem + @synchronize, like the hydro cube), so the same
# source targets CUDA and Metal. Shared budget (f32): 9·512 + 6·9·80 = 8928 floats =
# 34.9 KB < 48 KB. (f64 would need 70 KB > 48 KB — the cube is the f32 perf path; the
# reference integrator is the portable/any-precision path.) Requires N%4==0 per axis.
# Metal selects the f16-staged 8x4x4 variant below for power-of-two production grids.
export step_cube!, step_cube_drag!, step_cube_radiation!

const CTB = 4                 # owned cells/dim/block
const CTT = CTB + 4           # 8: prim tile incl. 2-cell halo
const CNCP = CTT*CTT*CTT      # 512
const CNFX = (CTB+1)*CTB*CTB  # 80 faces per direction
const CUBE_GS = 192           # threads per group

# Metal has 32 KiB of threadgroup memory. This 4x4x2 tile occupies 23,040 B
# with Float32 staging. Early-PMF density, pressure, and velocity perturbations
# are below Float16 precision, so the former 8x4x4 half-staged tile is not a
# valid science path despite its higher arithmetic throughput.
const RCTBX = 4
const RCTBY = 4
const RCTBZ = 2
const RCTTX = RCTBX + 4
const RCTTY = RCTBY + 4
const RCTTZ = RCTBZ + 4
const RCNCP = RCTTX*RCTTY*RCTTZ
const RCNFX = (RCTBX+1)*RCTBY*RCTBZ
const RCNFY = RCTBX*(RCTBY+1)*RCTBZ
const RCNFZ = RCTBX*RCTBY*(RCTBZ+1)
const METAL_CUBE_GS = 128

@inline _ctile(pi,pj,pk) = pi + CTT*(pj + CTT*pk)
@inline _cfx(fi,fj,fk) = fi + (CTB+1)*(fj + CTB*fk)
@inline _cfy(fi,fj,fk) = fi + CTB*(fj + (CTB+1)*fk)
@inline _cfz(fi,fj,fk) = fi + CTB*(fj + CTB*fk)
@inline _rctile(pi,pj,pk) = pi + RCTTX*(pj + RCTTY*pk)
@inline _rcfx(fi,fj,fk) = fi + (RCTBX+1)*(fj + RCTBY*fk)
@inline _rcfy(fi,fj,fk) = fi + RCTBX*(fj + (RCTBY+1)*fk)
@inline _rcfz(fi,fj,fk) = fi + RCTBX*(fj + RCTBY*fk)
# SoA variable-major shared (lin contiguous ⇒ coalesced / bank-conflict-free).
@inline _sg9(S,NC,lin) = @inbounds ntuple(v -> S[(v-1)*NC+lin+1], 9)
@inline _sg9_as(::Type{T}, S, NC, lin) where {T} =
    @inbounds ntuple(v -> T(S[(v-1)*NC+lin+1]), 9)
@inline function _sp9!(S,NC,lin,q)
    @inbounds for v in 1:9; S[(v-1)*NC+lin+1] = q[v]; end
end

@inline function _cube_masked_riemann(qL,qR,dir,γ,ch,smallr,pfl,llf_dmin,
        llf_pmin,rsol,packed,stride,N,iL,jL,kL,iR,jR,kR,
        ::Val{CHECK}) where {CHECK}
    if CHECK==2
        lidx=_periodic_corr_index(N,N,N,iL,jL,kL)
        ridx=_periodic_corr_index(N,N,N,iR,jR,kR)
        if @inbounds(packed[4stride+lidx]>0 || packed[4stride+ridx]>0)
            return riemann(qL,qR,dir,γ,ch,smallr,pfl,llf_dmin,llf_pmin,
                           Val(RSOLVE_LLF))
        end
    end
    riemann(qL,qR,dir,γ,ch,smallr,pfl,llf_dmin,llf_pmin,rsol)
end

@inline _cube_density_update!(rho_perturbation, idx, rho0::T, drho::T,
                              ::Val{false}) where {T} = rho0 + drho

@inline function _cube_density_update!(rho_perturbation, idx, rho0::T, drho::T,
                                       ::Val{true}) where {T}
    delta0 = @inbounds rho_perturbation[idx]
    deltan = delta0 + drho
    @inbounds rho_perturbation[idx] = deltan
    return rho0 + drho
end

@inline function _cube_inadmissible(rh::NTuple{9,T},smallr::T,pfl::T,
        gamma::T,llf_dmin::T,::Val{CHECK}) where {T,CHECK}
    if CHECK != false
        rho=rh[1]
        rho_gate=max(smallr,llf_dmin)
        ir=inv(max(rho,smallr))
        ek=T(0.5)*(rh[2]*rh[2]+rh[3]*rh[3]+rh[4]*rh[4])*ir
        eb=T(0.5)*(rh[6]*rh[6]+rh[7]*rh[7]+rh[8]*rh[8])
        eint=rh[5]-ek-eb
        emin=pfl/(gamma-one(T))
        finite=isfinite(rho)&&isfinite(eint)&&isfinite(rh[2])&&isfinite(rh[3])&&
               isfinite(rh[4])&&isfinite(rh[5])&&isfinite(rh[6])&&
               isfinite(rh[7])&&isfinite(rh[8])&&isfinite(rh[9])
        return !finite || rho<rho_gate || eint<emin
    end
    false
end

@kernel function step_cube_rect_kernel!(o1,o2,o3,o4,o5,o6,o7,o8,o9,
        @Const(a1),@Const(a2),@Const(a3),@Const(a4),@Const(a5),@Const(a6),@Const(a7),@Const(a8),a9,
        N::Int, nbx::Int, nby::Int, cx::Int, cy::Int, cz::Int,
        shared::Val{ST}, rec::Val, per::Val, rsol::Val,
        dtdx::T, γ::T, ch::T, decay::T, smallr::T, pfl::T,
        llf_dmin::T, llf_pmin::T, drag::Val{DRAG},
        dc::DragCoefficients{T},rad::Val{RAD},track_density::Val{TRACK},
        check_admissibility::Val{CHECK}) where {T,ST,DRAG,RAD,TRACK,CHECK}
    SP = @localmem ST (9*RCNCP)
    LX = @localmem ST (9*RCNFX); RX = @localmem ST (9*RCNFX)
    LY = @localmem ST (9*RCNFY); RY = @localmem ST (9*RCNFY)
    LZ = @localmem ST (9*RCNFZ); RZ = @localmem ST (9*RCNFZ)
    @fastmath @inbounds begin
        u = (a1,a2,a3,a4,a5,a6,a7,a8,a9)
        tid = @index(Local, Linear)
        nth = @uniform prod(@groupsize())
        g0 = @index(Group, Linear) - 1
        bx = g0 % nbx; by = (g0 ÷ nbx) % nby; bz = g0 ÷ (nbx*nby)
        ox = bx*RCTBX; oy = by*RCTBY; oz = bz*RCTBZ

        t = tid
        while t <= RCNCP
            l=t-1; pi=l%RCTTX; pj=(l÷RCTTX)%RCTTY; pk=l÷(RCTTX*RCTTY)
            q=_loadP_sel(per,u,N,N,N,ox+pi-1,oy+pj-1,oz+pk-1,
                         cx,cy,cz,γ,smallr,pfl)
            _sp9!(SP,RCNCP,_rctile(pi,pj,pk),q)
            t += nth
        end
        @synchronize

        t = tid
        while t <= (RCTBX+2)*(RCTBY+2)*(RCTBZ+2)
            l=t-1; ci=l%(RCTBX+2); cj=(l÷(RCTBX+2))%(RCTBY+2)
            ck=l÷((RCTBX+2)*(RCTBY+2)); pi=ci+1; pj=cj+1; pk=ck+1
            m0=_sg9_as(T,SP,RCNCP,_rctile(pi,pj,pk))
            ri=mod(ox+pi-2,N)+1; rj=mod(oy+pj-2,N)+1; rk=mod(oz+pk-2,N)+1
            ridx=((rk-1)*N+(rj-1))*N+ri
            if CHECK==3 || (CHECK==2 && a9[4*N*N*N+ridx]>zero(T))
                z=ntuple(_->zero(T),9)
                δLx,δRx=z,z; δLy,δRy=z,z; δLz,δRz=z,z
            else
                δLx,δRx=recon_offsets(_sg9_as(T,SP,RCNCP,_rctile(pi-1,pj,pk)),m0,
                                         _sg9_as(T,SP,RCNCP,_rctile(pi+1,pj,pk)),rec)
                δLy,δRy=recon_offsets(_sg9_as(T,SP,RCNCP,_rctile(pi,pj-1,pk)),m0,
                                         _sg9_as(T,SP,RCNCP,_rctile(pi,pj+1,pk)),rec)
                δLz,δRz=recon_offsets(_sg9_as(T,SP,RCNCP,_rctile(pi,pj,pk-1)),m0,
                                         _sg9_as(T,SP,RCNCP,_rctile(pi,pj,pk+1)),rec)
            end
            uh=hancock_edges(m0,δLx,δRx,δLy,δRy,δLz,δRz,dtdx,γ)
            mh=cons2prim(uh,γ,smallr,pfl)
            inxt=(pj>=2&&pj<=RCTBY+1)&&(pk>=2&&pk<=RCTBZ+1)
            inyt=(pi>=2&&pi<=RCTBX+1)&&(pk>=2&&pk<=RCTBZ+1)
            inzt=(pi>=2&&pi<=RCTBX+1)&&(pj>=2&&pj<=RCTBY+1)
            if inxt
                if pi<=RCTBX+1
                    f0=_facep(m0,δRx); fh=_facep(mh,δRx)
                    _sp9!(LX,RCNFX,_rcfx(pi-1,pj-2,pk-2),
                        _facep_sources_coeff(f0,fh,drag,dc,rad,a9,ridx,N*N*N))
                end
                if pi>=2
                    f0=_facep(m0,δLx); fh=_facep(mh,δLx)
                    _sp9!(RX,RCNFX,_rcfx(pi-2,pj-2,pk-2),
                        _facep_sources_coeff(f0,fh,drag,dc,rad,a9,ridx,N*N*N))
                end
            end
            if inyt
                if pj<=RCTBY+1
                    f0=_facep(m0,δRy); fh=_facep(mh,δRy)
                    _sp9!(LY,RCNFY,_rcfy(pi-2,pj-1,pk-2),
                        _facep_sources_coeff(f0,fh,drag,dc,rad,a9,ridx,N*N*N))
                end
                if pj>=2
                    f0=_facep(m0,δLy); fh=_facep(mh,δLy)
                    _sp9!(RY,RCNFY,_rcfy(pi-2,pj-2,pk-2),
                        _facep_sources_coeff(f0,fh,drag,dc,rad,a9,ridx,N*N*N))
                end
            end
            if inzt
                if pk<=RCTBZ+1
                    f0=_facep(m0,δRz); fh=_facep(mh,δRz)
                    _sp9!(LZ,RCNFZ,_rcfz(pi-2,pj-2,pk-1),
                        _facep_sources_coeff(f0,fh,drag,dc,rad,a9,ridx,N*N*N))
                end
                if pk>=2
                    f0=_facep(m0,δLz); fh=_facep(mh,δLz)
                    _sp9!(RZ,RCNFZ,_rcfz(pi-2,pj-2,pk-2),
                        _facep_sources_coeff(f0,fh,drag,dc,rad,a9,ridx,N*N*N))
                end
            end
            t += nth
        end
        @synchronize

        t = tid
        while t <= RCNFX+RCNFY+RCNFZ
            if t<=RCNFX
                l=t-1; fi=l%(RCTBX+1); fj=(l÷(RCTBX+1))%RCTBY
                fk=l÷((RCTBX+1)*RCTBY); ln=_rcfx(fi,fj,fk)
                _sp9!(LX,RCNFX,ln,_cube_masked_riemann(
                    _sg9_as(T,LX,RCNFX,ln),_sg9_as(T,RX,RCNFX,ln),1,γ,ch,
                    smallr,pfl,llf_dmin,llf_pmin,rsol,a9,N*N*N,N,
                    ox+fi-1,oy+fj,oz+fk,ox+fi,oy+fj,oz+fk,
                    check_admissibility))
            elseif t<=RCNFX+RCNFY
                l=t-1-RCNFX; fi=l%RCTBX; fj=(l÷RCTBX)%(RCTBY+1)
                fk=l÷(RCTBX*(RCTBY+1)); ln=_rcfy(fi,fj,fk)
                _sp9!(LY,RCNFY,ln,_cube_masked_riemann(
                    _sg9_as(T,LY,RCNFY,ln),_sg9_as(T,RY,RCNFY,ln),2,γ,ch,
                    smallr,pfl,llf_dmin,llf_pmin,rsol,a9,N*N*N,N,
                    ox+fi,oy+fj-1,oz+fk,ox+fi,oy+fj,oz+fk,
                    check_admissibility))
            else
                l=t-1-RCNFX-RCNFY; fi=l%RCTBX; fj=(l÷RCTBX)%RCTBY
                fk=l÷(RCTBX*RCTBY); ln=_rcfz(fi,fj,fk)
                _sp9!(LZ,RCNFZ,ln,_cube_masked_riemann(
                    _sg9_as(T,LZ,RCNFZ,ln),_sg9_as(T,RZ,RCNFZ,ln),3,γ,ch,
                    smallr,pfl,llf_dmin,llf_pmin,rsol,a9,N*N*N,N,
                    ox+fi,oy+fj,oz+fk-1,ox+fi,oy+fj,oz+fk,
                    check_admissibility))
            end
            t += nth
        end
        @synchronize

        t = tid
        while t <= RCTBX*RCTBY*RCTBZ
            l=t-1; a=l%RCTBX; b=(l÷RCTBX)%RCTBY; c=l÷(RCTBX*RCTBY)
            gi=mod(ox+a,N)+1; gj=mod(oy+b,N)+1; gk=mod(oz+c,N)+1
            idx=((gk-1)*N+(gj-1))*N+gi
            Fxl=_sg9_as(T,LX,RCNFX,_rcfx(a,b,c)); Fxh=_sg9_as(T,LX,RCNFX,_rcfx(a+1,b,c))
            Fyl=_sg9_as(T,LY,RCNFY,_rcfy(a,b,c)); Fyh=_sg9_as(T,LY,RCNFY,_rcfy(a,b+1,c))
            Fzl=_sg9_as(T,LZ,RCNFZ,_rcfz(a,b,c)); Fzh=_sg9_as(T,LZ,RCNFZ,_rcfz(a,b,c+1))
            U0=(a1[idx],a2[idx],a3[idx],a4[idx],a5[idx],a6[idx],a7[idx],a8[idx],a9[idx])
            rho_new=_cube_density_update!(a9,4*N*N*N+idx,U0[1],
                dtdx*((Fxl[1]-Fxh[1])+(Fyl[1]-Fyh[1])+(Fzl[1]-Fzh[1])),track_density)
            rh=(rho_new,
                U0[2]+dtdx*((Fxl[2]-Fxh[2])+(Fyl[2]-Fyh[2])+(Fzl[2]-Fzh[2])),
                U0[3]+dtdx*((Fxl[3]-Fxh[3])+(Fyl[3]-Fyh[3])+(Fzl[3]-Fzh[3])),
                U0[4]+dtdx*((Fxl[4]-Fxh[4])+(Fyl[4]-Fyh[4])+(Fzl[4]-Fzh[4])),
                U0[5]+dtdx*((Fxl[5]-Fxh[5])+(Fyl[5]-Fyh[5])+(Fzl[5]-Fzh[5])),
                U0[6]+dtdx*((Fxl[6]-Fxh[6])+(Fyl[6]-Fyh[6])+(Fzl[6]-Fzh[6])),
                U0[7]+dtdx*((Fxl[7]-Fxh[7])+(Fyl[7]-Fyh[7])+(Fzl[7]-Fzh[7])),
                U0[8]+dtdx*((Fxl[8]-Fxh[8])+(Fyl[8]-Fyh[8])+(Fzl[8]-Fzh[8])),
                U0[9]+dtdx*((Fxl[9]-Fxh[9])+(Fyl[9]-Fyh[9])+(Fzl[9]-Fzh[9])))
            inadmissible=_cube_inadmissible(rh,smallr,pfl,γ,llf_dmin,
                                             check_admissibility)
            r=DRAG ? drag_correct_conserved(U0,rh,dc.decay,dc.phi1,
                dc.A,dc.C,dc.D,zero(T),zero(T),zero(T),smallr,T(1e-30)) : rh
            o1[idx]=inadmissible ? -max(abs(r[1]),max(smallr,llf_dmin)) : r[1]
            o2[idx]=r[2];o3[idx]=r[3];o4[idx]=r[4];o5[idx]=r[5]
            o6[idx]=r[6];o7[idx]=r[7];o8[idx]=r[8];o9[idx]=r[9]*decay
            t += nth
        end
    end
end

@kernel function step_cube_kernel!(o1,o2,o3,o4,o5,o6,o7,o8,o9,
        @Const(a1),@Const(a2),@Const(a3),@Const(a4),@Const(a5),@Const(a6),@Const(a7),@Const(a8),a9,
        N::Int, nb::Int, cx::Int, cy::Int, cz::Int, shared::Val{ST}, rec::Val, per::Val, rsol::Val, dtdx::T, γ::T, ch::T, decay::T,
        smallr::T, pfl::T, llf_dmin::T, llf_pmin::T, drag::Val{DRAG},
        dc::DragCoefficients{T},rad::Val{RAD},track_density::Val{TRACK},
        check_admissibility::Val{CHECK}) where {T,ST,DRAG,RAD,TRACK,CHECK}
    SP = @localmem ST (9*CNCP)
    LX = @localmem ST (9*CNFX); RX = @localmem ST (9*CNFX)
    LY = @localmem ST (9*CNFX); RY = @localmem ST (9*CNFX)
    LZ = @localmem ST (9*CNFX); RZ = @localmem ST (9*CNFX)
    @fastmath @inbounds begin
        u = (a1,a2,a3,a4,a5,a6,a7,a8,a9)
        tid = @index(Local, Linear)
        nth = @uniform prod(@groupsize())
        g0  = @index(Group, Linear) - 1
        bx = g0 % nb; by = (g0 ÷ nb) % nb; bz = g0 ÷ (nb*nb)
        ox = bx*CTB; oy = by*CTB; oz = bz*CTB
        # Stage 1: cons2prim -> 8^3 tile
        t = tid
        while t <= CNCP
            l=t-1; pi=l%CTT; pj=(l÷CTT)%CTT; pk=l÷(CTT*CTT)
            # tile cell pi (0..7) ↔ global 1-based logical (ox+pi-1): 2-cell halo each
            # side ⇒ owned tile 2..5 ↔ global ox+1..ox+4 (matched in Stage 4). BC
            # synthesises out-of-range halo cells (periodic/outflow/reflecting).
            lx=ox+pi-1; ly=oy+pj-1; lz=oz+pk-1
            q=_loadP_sel(per,u,N,N,N,lx,ly,lz,cx,cy,cz,γ,smallr,pfl)
            _sp9!(SP,CNCP,_ctile(pi,pj,pk),q); t+=nth
        end
        @synchronize
        # Stage 2: trace (multidim Hancock) over inner 6^3 -> per-face interface states
        t = tid
        while t <= (CTB+2)^3
            l=t-1; ci=l%(CTB+2); cj=(l÷(CTB+2))%(CTB+2); ck=l÷((CTB+2)*(CTB+2))
            pi=ci+1; pj=cj+1; pk=ck+1
            m0=_sg9_as(T,SP,CNCP,_ctile(pi,pj,pk))
            ri=mod(ox+pi-2,N)+1; rj=mod(oy+pj-2,N)+1; rk=mod(oz+pk-2,N)+1
            ridx=((rk-1)*N+(rj-1))*N+ri
            if CHECK==3 || (CHECK==2 && a9[4*N*N*N+ridx]>zero(T))
                z=ntuple(_->zero(T),9)
                δLx,δRx=z,z; δLy,δRy=z,z; δLz,δRz=z,z
            else
                δLx,δRx=recon_offsets(_sg9_as(T,SP,CNCP,_ctile(pi-1,pj,pk)),m0,_sg9_as(T,SP,CNCP,_ctile(pi+1,pj,pk)),rec)
                δLy,δRy=recon_offsets(_sg9_as(T,SP,CNCP,_ctile(pi,pj-1,pk)),m0,_sg9_as(T,SP,CNCP,_ctile(pi,pj+1,pk)),rec)
                δLz,δRz=recon_offsets(_sg9_as(T,SP,CNCP,_ctile(pi,pj,pk-1)),m0,_sg9_as(T,SP,CNCP,_ctile(pi,pj,pk+1)),rec)
            end
            uh=hancock_edges(m0,δLx,δRx,δLy,δRy,δLz,δRz,dtdx,γ); mh=cons2prim(uh,γ,smallr,pfl)
            inxt=(pj>=2&&pj<=CTB+1)&&(pk>=2&&pk<=CTB+1)
            inyt=(pi>=2&&pi<=CTB+1)&&(pk>=2&&pk<=CTB+1)
            inzt=(pi>=2&&pi<=CTB+1)&&(pj>=2&&pj<=CTB+1)
            if inxt
                if pi<=CTB+1
                    f0=_facep(m0,δRx); fh=_facep(mh,δRx)
                    _sp9!(LX,CNFX,_cfx(pi-1,pj-2,pk-2),_facep_sources_coeff(f0,fh,drag,dc,rad,a9,ridx,N*N*N))
                end
                if pi>=2
                    f0=_facep(m0,δLx); fh=_facep(mh,δLx)
                    _sp9!(RX,CNFX,_cfx(pi-2,pj-2,pk-2),_facep_sources_coeff(f0,fh,drag,dc,rad,a9,ridx,N*N*N))
                end
            end
            if inyt
                if pj<=CTB+1
                    f0=_facep(m0,δRy); fh=_facep(mh,δRy)
                    _sp9!(LY,CNFX,_cfy(pi-2,pj-1,pk-2),_facep_sources_coeff(f0,fh,drag,dc,rad,a9,ridx,N*N*N))
                end
                if pj>=2
                    f0=_facep(m0,δLy); fh=_facep(mh,δLy)
                    _sp9!(RY,CNFX,_cfy(pi-2,pj-2,pk-2),_facep_sources_coeff(f0,fh,drag,dc,rad,a9,ridx,N*N*N))
                end
            end
            if inzt
                if pk<=CTB+1
                    f0=_facep(m0,δRz); fh=_facep(mh,δRz)
                    _sp9!(LZ,CNFX,_cfz(pi-2,pj-2,pk-1),_facep_sources_coeff(f0,fh,drag,dc,rad,a9,ridx,N*N*N))
                end
                if pk>=2
                    f0=_facep(m0,δLz); fh=_facep(mh,δLz)
                    _sp9!(RZ,CNFX,_cfz(pi-2,pj-2,pk-2),_facep_sources_coeff(f0,fh,drag,dc,rad,a9,ridx,N*N*N))
                end
            end
            t+=nth
        end
        @synchronize
        # Stage 3: GLM Riemann once per face, flux written back into L*
        nfx=(CTB+1)*CTB*CTB; t=tid
        while t<=3*nfx
            if t<=nfx
                l=t-1; fi=l%(CTB+1); fj=(l÷(CTB+1))%CTB; fk=l÷((CTB+1)*CTB); ln=_cfx(fi,fj,fk)
                F=_cube_masked_riemann(_sg9_as(T,LX,CNFX,ln),
                    _sg9_as(T,RX,CNFX,ln),1,γ,ch,smallr,pfl,llf_dmin,llf_pmin,
                    rsol,a9,N*N*N,N,ox+fi-1,oy+fj,oz+fk,
                    ox+fi,oy+fj,oz+fk,check_admissibility)
                _sp9!(LX,CNFX,ln,F)
            elseif t<=2*nfx
                l=t-1-nfx; fi=l%CTB; fj=(l÷CTB)%(CTB+1); fk=l÷(CTB*(CTB+1)); ln=_cfy(fi,fj,fk)
                F=_cube_masked_riemann(_sg9_as(T,LY,CNFX,ln),
                    _sg9_as(T,RY,CNFX,ln),2,γ,ch,smallr,pfl,llf_dmin,llf_pmin,
                    rsol,a9,N*N*N,N,ox+fi,oy+fj-1,oz+fk,
                    ox+fi,oy+fj,oz+fk,check_admissibility)
                _sp9!(LY,CNFX,ln,F)
            else
                l=t-1-2*nfx; fi=l%CTB; fj=(l÷CTB)%CTB; fk=l÷(CTB*CTB); ln=_cfz(fi,fj,fk)
                F=_cube_masked_riemann(_sg9_as(T,LZ,CNFX,ln),
                    _sg9_as(T,RZ,CNFX,ln),3,γ,ch,smallr,pfl,llf_dmin,llf_pmin,
                    rsol,a9,N*N*N,N,ox+fi,oy+fj,oz+fk-1,
                    ox+fi,oy+fj,oz+fk,check_admissibility)
                _sp9!(LZ,CNFX,ln,F)
            end
            t+=nth
        end
        @synchronize
        # Stage 4: conservative update of the 4^3 owned cells (+ψ damping)
        t=tid
        while t<=CTB*CTB*CTB
            l=t-1; a=l%CTB; b=(l÷CTB)%CTB; c=l÷(CTB*CTB)
            gi=mod(ox+a+1-1,N)+1; gj=mod(oy+b+1-1,N)+1; gk=mod(oz+c+1-1,N)+1; idx=((gk-1)*N+(gj-1))*N+gi
            Fxl=_sg9_as(T,LX,CNFX,_cfx(a,b,c));   Fxh=_sg9_as(T,LX,CNFX,_cfx(a+1,b,c))
            Fyl=_sg9_as(T,LY,CNFX,_cfy(a,b,c));   Fyh=_sg9_as(T,LY,CNFX,_cfy(a,b+1,c))
            Fzl=_sg9_as(T,LZ,CNFX,_cfz(a,b,c));   Fzh=_sg9_as(T,LZ,CNFX,_cfz(a,b,c+1))
            U0=(a1[idx],a2[idx],a3[idx],a4[idx],a5[idx],a6[idx],a7[idx],a8[idx],a9[idx])
            rho_new=_cube_density_update!(a9,4*N*N*N+idx,U0[1],
                dtdx*((Fxl[1]-Fxh[1])+(Fyl[1]-Fyh[1])+(Fzl[1]-Fzh[1])),track_density)
            rh=(rho_new,
                U0[2]+dtdx*((Fxl[2]-Fxh[2])+(Fyl[2]-Fyh[2])+(Fzl[2]-Fzh[2])),
                U0[3]+dtdx*((Fxl[3]-Fxh[3])+(Fyl[3]-Fyh[3])+(Fzl[3]-Fzh[3])),
                U0[4]+dtdx*((Fxl[4]-Fxh[4])+(Fyl[4]-Fyh[4])+(Fzl[4]-Fzh[4])),
                U0[5]+dtdx*((Fxl[5]-Fxh[5])+(Fyl[5]-Fyh[5])+(Fzl[5]-Fzh[5])),
                U0[6]+dtdx*((Fxl[6]-Fxh[6])+(Fyl[6]-Fyh[6])+(Fzl[6]-Fzh[6])),
                U0[7]+dtdx*((Fxl[7]-Fxh[7])+(Fyl[7]-Fyh[7])+(Fzl[7]-Fzh[7])),
                U0[8]+dtdx*((Fxl[8]-Fxh[8])+(Fyl[8]-Fyh[8])+(Fzl[8]-Fzh[8])),
                U0[9]+dtdx*((Fxl[9]-Fxh[9])+(Fyl[9]-Fyh[9])+(Fzl[9]-Fzh[9])))
            inadmissible=_cube_inadmissible(rh,smallr,pfl,γ,llf_dmin,
                                             check_admissibility)
            r=DRAG ? drag_correct_conserved(U0,rh,dc.decay,dc.phi1,
                dc.A,dc.C,dc.D,zero(T),zero(T),zero(T),smallr,T(1e-30)) : rh
            o1[idx]=inadmissible ? -max(abs(r[1]),max(smallr,llf_dmin)) : r[1]
            o2[idx]=r[2];o3[idx]=r[3];o4[idx]=r[4];o5[idx]=r[5];o6[idx]=r[6];o7[idx]=r[7];o8[idx]=r[8]
            o9[idx]=r[9]*decay
            t+=nth
        end
    end
end

"""
    step_cube!(s, dt; ch, decay, impl=:auto)

Advance one step with the fused shared-memory cube kernel (GPU throughput path).
Requires a cubic grid with N%4==0. Same scheme as [`step_ref!`]. `impl` selects the
launch backend: `:ka` = the portable KernelAbstractions kernel (CUDA + Metal); `:raw`/
`:auto` = a raw-CUDA specialization (loaded with the CUDA extension) that recovers the
NVIDIA peak — `:auto` uses raw on a CUDA-f32 state and falls back to KA otherwise.
"""
function step_cube!(s::MHDState{T}, dt::Real; ch::Real, decay::Real, impl::Symbol = :auto) where {T}
    N = s.dims[1]
    all(==(N), s.dims) || error("step_cube! currently needs a cubic grid; got $(s.dims)")
    N % CTB == 0 || error("step_cube! needs N % $CTB == 0; got N=$N")
    _cube_launch!(Val(impl), s.be, s, dt, ch, decay)   # writes s.scratch from s.U (+sync)
    s.U, s.scratch = s.scratch, s.U
    return s
end

# Default launch: the portable KA kernel (any backend, any `impl`). The CUDA extension
# adds a more-specific method for `Val{:auto}`/`Val{:raw}` on a CUDABackend Float32 state.
function _cube_launch!(::Val, be, s::MHDState{T}, dt::Real, ch::Real, decay::Real) where {T}
    N = s.dims[1]; nb = N ÷ CTB; dtdx = T(dt)/s.dx; cx,cy,cz = bc_codes(s)
    rec = Val(recon_code_of(s)); per = Val(all_periodic(s)); rsol = Val(riemann_code_of(s))
    step_cube_kernel!(be, CUBE_GS)(s.scratch..., s.U..., N, nb, cx,cy,cz, Val(Float32), rec, per, rsol, dtdx, s.γ, T(ch), T(decay),
            s.smallr, s.pfl, s.llf_dmin, s.llf_pmin, Val(false),
            _no_drag_coefficients(T),Val(false),Val(false),Val(false);
            ndrange = nb*nb*nb*CUBE_GS)
    KA.synchronize(be)
end

function step_cube_drag!(s::MHDState{T},dt::Real; ch::Real,decay::Real,
                         drag_impulse::Real,target_velocity=(0,0,0)) where {T}
    N=s.dims[1]
    all(==(N),s.dims) || error("step_cube_drag! currently needs a cubic grid; got $(s.dims)")
    N % CTB == 0 || error("step_cube_drag! needs N % $CTB == 0; got N=$N")
    all(iszero,target_velocity) ||
        error("step_cube_drag! is specialized for Compton drag in the CMB frame; use integrator=:ref for a moving target")
    q=T(drag_impulse)
    _cube_drag_launch!(s.be,s,dt,ch,decay,q)
    s.U,s.scratch=s.scratch,s.U
    s
end

function _cube_drag_launch!(be,s::MHDState{T},dt::Real,ch::Real,decay::Real,
                            q::T) where {T}
    N=s.dims[1]; nb=N÷CTB; dtdx=T(dt)/s.dx; cx,cy,cz=bc_codes(s)
    rec=Val(recon_code_of(s)); per=Val(all_periodic(s)); rsol=Val(riemann_code_of(s))
    dc=_drag_coefficients(q)
    step_cube_kernel!(be,CUBE_GS)(s.scratch...,s.U...,N,nb,cx,cy,cz,Val(Float32),rec,per,rsol,
        dtdx,s.γ,T(ch),T(decay),s.smallr,s.pfl,s.llf_dmin,s.llf_pmin,Val(true),
        dc,Val(false),Val(false),Val(false);
        ndrange=nb*nb*nb*CUBE_GS)
    KA.synchronize(be)
end


function step_cube_radiation!(s::MHDState{T},dt::Real;ch::Real,decay::Real,
        drag_impulse::Real,correction,track_density::Bool=true,
        check_admissibility::Bool=false,fallback::Bool=false,
        global_fallback::Bool=false,robust_fallback::Bool=false) where {T}
    N=s.dims[1]
    all(==(N),s.dims) || error("step_cube_radiation! requires a cubic grid")
    N%CTB==0 || error("step_cube_radiation! needs N % $CTB == 0")
    _cube_radiation_launch!(s.be,s,dt,ch,decay,T(drag_impulse),correction,
                            Val(track_density),Val(check_admissibility),Val(fallback),
                            Val(global_fallback),Val(robust_fallback))
    s.U,s.scratch=s.scratch,s.U
    s
end

function _cube_radiation_launch!(be,s::MHDState{T},dt::Real,ch::Real,decay::Real,
        q::T,correction,track_density::Val,check::Val,
        fallback::Val{FALLBACK},global_fallback::Val{GLOBAL},
        robust_fallback::Val{ROBUST}) where {T,FALLBACK,GLOBAL,ROBUST}
    N=s.dims[1]; nb=N÷CTB; dtdx=T(dt)/s.dx; cx,cy,cz=bc_codes(s)
    rec=ROBUST ? Val(RECON_PLM_MINMOD) : Val(recon_code_of(s))
    per=Val(all_periodic(s))
    rsol=ROBUST ? Val(RSOLVE_HLL) : GLOBAL ? Val(RSOLVE_LLF) :
        Val(riemann_code_of(s))
    dc=_drag_coefficients(q)
    kernel_check=GLOBAL ? Val(3) : FALLBACK ? Val(2) : check
    step_cube_kernel!(be,CUBE_GS)(s.scratch...,s.U[1],s.U[2],s.U[3],s.U[4],s.U[5],
        s.U[6],s.U[7],s.U[8],correction,N,nb,cx,cy,cz,Val(Float32),
        rec,per,rsol,dtdx,s.γ,T(ch),T(decay),s.smallr,s.pfl,s.llf_dmin,s.llf_pmin,
        Val(true),dc,Val(true),track_density,kernel_check;
        ndrange=nb*nb*nb*CUBE_GS)
    KA.synchronize(be)
end
