# ── Reference integrator: portable per-cell MUSCL-Hancock GLM-MHD step ─────────
# One `@kernel` workitem per cell, reading a ±2 periodic stencil from global memory.
# This is the OBVIOUSLY-correct path: it runs on every backend (CPU f64 = the
# convergence/conservation oracle; GPU f32) and is the cross-check target for the
# shared-memory cube. It recomputes each cell's Hancock-predicted faces from global
# (no tiling) — simple, not fast; the cube is the throughput path.
#
# Per cell: build the 1-axis-Hancock face states of itself and its 6 face-neighbours
# (`_mh_slopes` = MonCen slopes + the multidim Hancock ½-step), solve the GLM Riemann
# problem at each of the 6 interfaces ONCE per side, sum the flux divergence, and
# write U_new = U_old + (dt/dx)·Σ_d (F_{lo}-F_{hi}); ψ gets the extra parabolic ×decay.

export step_ref!, step_ref_drag!, step_ref_radiation!

# A cell's Hancock-predicted primitive `mh` plus its per-axis face offsets (δL,δR),
# with per-axis BC synthesis at the domain edges and the chosen reconstruction `rec`.
@inline function _mh_edges(u::Tuple,Nx::Int,Ny::Int,Nz::Int,i::Int,j::Int,k::Int,
                           cx::Int,cy::Int,cz::Int,rec::Val,γ::T,dtdx::T,
                           smallr::T,pfl::T) where {T}
    m0 = _loadP_bc(u,Nx,Ny,Nz,i,j,k,cx,cy,cz,γ,smallr,pfl)
    δLx,δRx = recon_offsets(_loadP_bc(u,Nx,Ny,Nz,i-1,j,k,cx,cy,cz,γ,smallr,pfl), m0, _loadP_bc(u,Nx,Ny,Nz,i+1,j,k,cx,cy,cz,γ,smallr,pfl), rec)
    δLy,δRy = recon_offsets(_loadP_bc(u,Nx,Ny,Nz,i,j-1,k,cx,cy,cz,γ,smallr,pfl), m0, _loadP_bc(u,Nx,Ny,Nz,i,j+1,k,cx,cy,cz,γ,smallr,pfl), rec)
    δLz,δRz = recon_offsets(_loadP_bc(u,Nx,Ny,Nz,i,j,k-1,cx,cy,cz,γ,smallr,pfl), m0, _loadP_bc(u,Nx,Ny,Nz,i,j,k+1,cx,cy,cz,γ,smallr,pfl), rec)
    uh = hancock_edges(m0,δLx,δRx,δLy,δRy,δLz,δRz,dtdx,γ); mh = cons2prim(uh,γ,smallr,pfl)
    (m0, mh, δLx,δRx, δLy,δRy, δLz,δRz)
end
@inline _facep(mh::NTuple{9,T}, δ::NTuple{9,T}) where {T} = ntuple(i->mh[i]+δ[i], 9)
@inline function _facep_drag(m0::NTuple{9,T},mh::NTuple{9,T},δ::NTuple{9,T},
                             qh::T,vx0::T,vy0::T,vz0::T) where {T}
    drag_predict_face(_facep(m0,δ),_facep(mh,δ),qh,vx0,vy0,vz0)
end

@inline function _facep_sources(m0::NTuple{9,T},mh::NTuple{9,T},delta::NTuple{9,T},
        drag::Val{DRAG},qh::T,vx0::T,vy0::T,vz0::T,rad::Val{RAD},
        packed,idx::Int,stride::Int) where {T,DRAG,RAD}
    q=DRAG ? _facep_drag(m0,mh,delta,qh,vx0,vy0,vz0) : _facep(mh,delta)
    RAD ? (q[1],q[2]+packed[stride+idx],q[3]+packed[2stride+idx],
           q[4]+packed[3stride+idx],q[5],q[6],q[7],q[8],q[9]) : q
end

@inline function _facep_sources_coeff(q0::NTuple{9,T},qh::NTuple{9,T},
        drag::Val{DRAG},dc::DragCoefficients{T},rad::Val{RAD},
        packed,idx::Int,stride::Int) where {T,DRAG,RAD}
    q=DRAG ? drag_predict_face(q0,qh,dc.face_decay,dc.face_phi1,
                               zero(T),zero(T),zero(T)) : qh
    RAD ? (q[1],q[2]+packed[stride+idx],q[3]+packed[2stride+idx],
           q[4]+packed[3stride+idx],q[5],q[6],q[7],q[8],q[9]) : q
end

@inline function _periodic_corr_index(Nx::Int,Ny::Int,Nz::Int,i::Int,j::Int,k::Int)
    ii=mod(i-1,Nx)+1; jj=mod(j-1,Ny)+1; kk=mod(k-1,Nz)+1
    ((kk-1)*Ny+(jj-1))*Nx+ii
end

@kernel function step_ref_kernel!(o1,o2,o3,o4,o5,o6,o7,o8,o9,
        @Const(a1),@Const(a2),@Const(a3),@Const(a4),@Const(a5),@Const(a6),@Const(a7),@Const(a8),a9,
        Nx::Int, Ny::Int, Nz::Int, cx::Int, cy::Int, cz::Int, rec::Val, rsol::Val, dtdx::T, γ::T, ch::T, decay::T,
        smallr::T, pfl::T, llf_dmin::T, llf_pmin::T, drag::Val{DRAG},
        qh::T, qfull::T, vx0::T, vy0::T, vz0::T,
        rad::Val{RAD},track_density::Val{TRACK}) where {T,DRAG,RAD,TRACK}
    c = @index(Global, Linear)
    @inbounds begin
        u = (a1,a2,a3,a4,a5,a6,a7,a8,a9)
        i=(c-1)%Nx+1; j=((c-1)÷Nx)%Ny+1; k=(c-1)÷(Nx*Ny)+1
        (m0c,mhc,δLxc,δRxc,δLyc,δRyc,δLzc,δRzc) = _mh_edges(u,Nx,Ny,Nz,i,j,k,cx,cy,cz,rec,γ,dtdx,smallr,pfl)
        # x faces (L = +x edge of left cell, R = -x edge of right cell)
        (m0xm,mhxm,_,δRxm,_,_,_,_) = _mh_edges(u,Nx,Ny,Nz,i-1,j,k,cx,cy,cz,rec,γ,dtdx,smallr,pfl)
        (m0xp,mhxp,δLxp,_,_,_,_,_) = _mh_edges(u,Nx,Ny,Nz,i+1,j,k,cx,cy,cz,rec,γ,dtdx,smallr,pfl)
        stride=Nx*Ny*Nz; ic=_periodic_corr_index(Nx,Ny,Nz,i,j,k)
        ixm=_periodic_corr_index(Nx,Ny,Nz,i-1,j,k); ixp=_periodic_corr_index(Nx,Ny,Nz,i+1,j,k)
        Lxl=_facep_sources(m0xm,mhxm,δRxm,drag,qh,vx0,vy0,vz0,rad,a9,ixm,stride)
        Rxl=_facep_sources(m0c,mhc,δLxc,drag,qh,vx0,vy0,vz0,rad,a9,ic,stride)
        Lxh=_facep_sources(m0c,mhc,δRxc,drag,qh,vx0,vy0,vz0,rad,a9,ic,stride)
        Rxh=_facep_sources(m0xp,mhxp,δLxp,drag,qh,vx0,vy0,vz0,rad,a9,ixp,stride)
        Fxl = riemann(Lxl,Rxl,1,γ,ch,smallr,pfl,llf_dmin,llf_pmin,rsol)
        Fxh = riemann(Lxh,Rxh,1,γ,ch,smallr,pfl,llf_dmin,llf_pmin,rsol)
        # y faces
        (m0ym,mhym,_,_,_,δRym,_,_) = _mh_edges(u,Nx,Ny,Nz,i,j-1,k,cx,cy,cz,rec,γ,dtdx,smallr,pfl)
        (m0yp,mhyp,_,_,δLyp,_,_,_) = _mh_edges(u,Nx,Ny,Nz,i,j+1,k,cx,cy,cz,rec,γ,dtdx,smallr,pfl)
        iym=_periodic_corr_index(Nx,Ny,Nz,i,j-1,k); iyp=_periodic_corr_index(Nx,Ny,Nz,i,j+1,k)
        Lyl=_facep_sources(m0ym,mhym,δRym,drag,qh,vx0,vy0,vz0,rad,a9,iym,stride)
        Ryl=_facep_sources(m0c,mhc,δLyc,drag,qh,vx0,vy0,vz0,rad,a9,ic,stride)
        Lyh=_facep_sources(m0c,mhc,δRyc,drag,qh,vx0,vy0,vz0,rad,a9,ic,stride)
        Ryh=_facep_sources(m0yp,mhyp,δLyp,drag,qh,vx0,vy0,vz0,rad,a9,iyp,stride)
        Fyl = riemann(Lyl,Ryl,2,γ,ch,smallr,pfl,llf_dmin,llf_pmin,rsol)
        Fyh = riemann(Lyh,Ryh,2,γ,ch,smallr,pfl,llf_dmin,llf_pmin,rsol)
        # z faces
        (m0zm,mhzm,_,_,_,_,_,δRzm) = _mh_edges(u,Nx,Ny,Nz,i,j,k-1,cx,cy,cz,rec,γ,dtdx,smallr,pfl)
        (m0zp,mhzp,_,_,_,_,δLzp,_) = _mh_edges(u,Nx,Ny,Nz,i,j,k+1,cx,cy,cz,rec,γ,dtdx,smallr,pfl)
        izm=_periodic_corr_index(Nx,Ny,Nz,i,j,k-1); izp=_periodic_corr_index(Nx,Ny,Nz,i,j,k+1)
        Lzl=_facep_sources(m0zm,mhzm,δRzm,drag,qh,vx0,vy0,vz0,rad,a9,izm,stride)
        Rzl=_facep_sources(m0c,mhc,δLzc,drag,qh,vx0,vy0,vz0,rad,a9,ic,stride)
        Lzh=_facep_sources(m0c,mhc,δRzc,drag,qh,vx0,vy0,vz0,rad,a9,ic,stride)
        Rzh=_facep_sources(m0zp,mhzp,δLzp,drag,qh,vx0,vy0,vz0,rad,a9,izp,stride)
        Fzl = riemann(Lzl,Rzl,3,γ,ch,smallr,pfl,llf_dmin,llf_pmin,rsol)
        Fzh = riemann(Lzh,Rzh,3,γ,ch,smallr,pfl,llf_dmin,llf_pmin,rsol)
        U0 = (a1[c],a2[c],a3[c],a4[c],a5[c],a6[c],a7[c],a8[c],a9[c])
        drho=dtdx*((Fxl[1]-Fxh[1])+(Fyl[1]-Fyh[1])+(Fzl[1]-Fzh[1]))
        rho_new=_cube_density_update!(a9,4*stride+c,U0[1],drho,track_density)
        rh = (rho_new,
              U0[2]+dtdx*((Fxl[2]-Fxh[2])+(Fyl[2]-Fyh[2])+(Fzl[2]-Fzh[2])),
              U0[3]+dtdx*((Fxl[3]-Fxh[3])+(Fyl[3]-Fyh[3])+(Fzl[3]-Fzh[3])),
              U0[4]+dtdx*((Fxl[4]-Fxh[4])+(Fyl[4]-Fyh[4])+(Fzl[4]-Fzh[4])),
              U0[5]+dtdx*((Fxl[5]-Fxh[5])+(Fyl[5]-Fyh[5])+(Fzl[5]-Fzh[5])),
              U0[6]+dtdx*((Fxl[6]-Fxh[6])+(Fyl[6]-Fyh[6])+(Fzl[6]-Fzh[6])),
              U0[7]+dtdx*((Fxl[7]-Fxh[7])+(Fyl[7]-Fyh[7])+(Fzl[7]-Fzh[7])),
              U0[8]+dtdx*((Fxl[8]-Fxh[8])+(Fyl[8]-Fyh[8])+(Fzl[8]-Fzh[8])),
              U0[9]+dtdx*((Fxl[9]-Fxh[9])+(Fyl[9]-Fyh[9])+(Fzl[9]-Fzh[9])))
        r = DRAG ? drag_correct_conserved(U0,rh,qfull,vx0,vy0,vz0,smallr,T(1e-30)) : rh
        o1[c]=r[1];o2[c]=r[2];o3[c]=r[3];o4[c]=r[4];o5[c]=r[5];o6[c]=r[6];o7[c]=r[7];o8[c]=r[8]
        o9[c]=r[9]*decay              # parabolic GLM ψ damping
    end
end

"""
    step_ref!(s::MHDState, dt; ch, decay)

Advance one MUSCL-Hancock GLM-MHD step with the portable reference kernel, writing
into the scratch set and swapping it in. `ch` is the GLM cleaning speed; `decay` the
ψ damping factor for this step.
"""
function step_ref!(s::MHDState{T}, dt::Real; ch::Real, decay::Real) where {T}
    Nx,Ny,Nz = s.dims; dtdx = T(dt)/s.dx; cx,cy,cz = bc_codes(s); rec = Val(recon_code_of(s)); rsol = Val(riemann_code_of(s))
    step_ref_kernel!(s.be, 256)(s.scratch..., s.U..., Nx,Ny,Nz, cx,cy,cz, rec, rsol, dtdx, s.γ, T(ch), T(decay),
                                s.smallr, s.pfl, s.llf_dmin, s.llf_pmin, Val(false),
                                zero(T),zero(T),zero(T),zero(T),zero(T),Val(false),Val(false);
                                ndrange = ncells(s))
    KA.synchronize(s.be)
    s.U, s.scratch = s.scratch, s.U
    return s
end

function step_ref_drag!(s::MHDState{T},dt::Real; ch::Real,decay::Real,
                        drag_impulse::Real,target_velocity=(0,0,0)) where {T}
    Nx,Ny,Nz=s.dims; dtdx=T(dt)/s.dx; cx,cy,cz=bc_codes(s)
    rec=Val(recon_code_of(s)); rsol=Val(riemann_code_of(s)); vx0,vy0,vz0=target_velocity
    q=T(drag_impulse)
    step_ref_kernel!(s.be,256)(s.scratch...,s.U...,Nx,Ny,Nz,cx,cy,cz,rec,rsol,dtdx,s.γ,T(ch),T(decay),
        s.smallr,s.pfl,s.llf_dmin,s.llf_pmin,Val(true),T(0.5)*q,q,T(vx0),T(vy0),T(vz0),
        Val(false),Val(false);
        ndrange=ncells(s))
    KA.synchronize(s.be)
    s.U,s.scratch=s.scratch,s.U
    s
end


function step_ref_radiation!(s::MHDState{T},dt::Real;ch::Real,decay::Real,
        drag_impulse::Real,correction,track_density::Bool=true) where {T}
    Nx,Ny,Nz=s.dims; dtdx=T(dt)/s.dx; cx,cy,cz=bc_codes(s)
    rec=Val(recon_code_of(s)); rsol=Val(riemann_code_of(s)); q=T(drag_impulse)
    step_ref_kernel!(s.be,256)(s.scratch...,s.U[1],s.U[2],s.U[3],s.U[4],s.U[5],
        s.U[6],s.U[7],s.U[8],correction,Nx,Ny,Nz,cx,cy,cz,rec,rsol,
        dtdx,s.γ,T(ch),T(decay),s.smallr,s.pfl,s.llf_dmin,s.llf_pmin,
        Val(true),T(0.5)*q,q,zero(T),zero(T),zero(T),Val(true),Val(track_density);
        ndrange=ncells(s))
    KA.synchronize(s.be)
    s.U,s.scratch=s.scratch,s.U
    s
end
