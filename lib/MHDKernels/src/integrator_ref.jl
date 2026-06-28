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

export step_ref!

# A cell's Hancock-predicted primitive `mh` plus its per-axis face offsets (δL,δR),
# with per-axis BC synthesis at the domain edges and the chosen reconstruction `rec`.
@inline function _mh_edges(u::NTuple{9,A}, Nx::Int, Ny::Int, Nz::Int, i::Int, j::Int, k::Int,
                           cx::Int, cy::Int, cz::Int, rec::Val, γ::T, dtdx::T, smallr::T, pfl::T) where {T,A}
    m0 = _loadP_bc(u,Nx,Ny,Nz,i,j,k,cx,cy,cz,γ,smallr,pfl)
    δLx,δRx = recon_offsets(_loadP_bc(u,Nx,Ny,Nz,i-1,j,k,cx,cy,cz,γ,smallr,pfl), m0, _loadP_bc(u,Nx,Ny,Nz,i+1,j,k,cx,cy,cz,γ,smallr,pfl), rec)
    δLy,δRy = recon_offsets(_loadP_bc(u,Nx,Ny,Nz,i,j-1,k,cx,cy,cz,γ,smallr,pfl), m0, _loadP_bc(u,Nx,Ny,Nz,i,j+1,k,cx,cy,cz,γ,smallr,pfl), rec)
    δLz,δRz = recon_offsets(_loadP_bc(u,Nx,Ny,Nz,i,j,k-1,cx,cy,cz,γ,smallr,pfl), m0, _loadP_bc(u,Nx,Ny,Nz,i,j,k+1,cx,cy,cz,γ,smallr,pfl), rec)
    uh = hancock_edges(m0,δLx,δRx,δLy,δRy,δLz,δRz,dtdx,γ); mh = cons2prim(uh,γ,smallr,pfl)
    (mh, δLx,δRx, δLy,δRy, δLz,δRz)
end
@inline _facep(mh::NTuple{9,T}, δ::NTuple{9,T}) where {T} = ntuple(i->mh[i]+δ[i], 9)

@kernel function step_ref_kernel!(o1,o2,o3,o4,o5,o6,o7,o8,o9,
        @Const(a1),@Const(a2),@Const(a3),@Const(a4),@Const(a5),@Const(a6),@Const(a7),@Const(a8),@Const(a9),
        Nx::Int, Ny::Int, Nz::Int, cx::Int, cy::Int, cz::Int, rec::Val, dtdx::T, γ::T, ch::T, decay::T,
        smallr::T, pfl::T, llf_dmin::T, llf_pmin::T, use_hlld::Bool) where {T}
    c = @index(Global, Linear)
    @inbounds begin
        u = (a1,a2,a3,a4,a5,a6,a7,a8,a9)
        i=(c-1)%Nx+1; j=((c-1)÷Nx)%Ny+1; k=(c-1)÷(Nx*Ny)+1
        (mhc,δLxc,δRxc,δLyc,δRyc,δLzc,δRzc) = _mh_edges(u,Nx,Ny,Nz,i,j,k,cx,cy,cz,rec,γ,dtdx,smallr,pfl)
        # x faces (L = +x edge of left cell, R = -x edge of right cell)
        (mhxm,_,δRxm,_,_,_,_) = _mh_edges(u,Nx,Ny,Nz,i-1,j,k,cx,cy,cz,rec,γ,dtdx,smallr,pfl)
        (mhxp,δLxp,_,_,_,_,_) = _mh_edges(u,Nx,Ny,Nz,i+1,j,k,cx,cy,cz,rec,γ,dtdx,smallr,pfl)
        Fxl = riemann(_facep(mhxm,δRxm), _facep(mhc,δLxc), 1, γ,ch,smallr,pfl,llf_dmin,llf_pmin,use_hlld)
        Fxh = riemann(_facep(mhc,δRxc),  _facep(mhxp,δLxp),1, γ,ch,smallr,pfl,llf_dmin,llf_pmin,use_hlld)
        # y faces
        (mhym,_,_,_,δRym,_,_) = _mh_edges(u,Nx,Ny,Nz,i,j-1,k,cx,cy,cz,rec,γ,dtdx,smallr,pfl)
        (mhyp,_,_,δLyp,_,_,_) = _mh_edges(u,Nx,Ny,Nz,i,j+1,k,cx,cy,cz,rec,γ,dtdx,smallr,pfl)
        Fyl = riemann(_facep(mhym,δRym), _facep(mhc,δLyc), 2, γ,ch,smallr,pfl,llf_dmin,llf_pmin,use_hlld)
        Fyh = riemann(_facep(mhc,δRyc),  _facep(mhyp,δLyp),2, γ,ch,smallr,pfl,llf_dmin,llf_pmin,use_hlld)
        # z faces
        (mhzm,_,_,_,_,_,δRzm) = _mh_edges(u,Nx,Ny,Nz,i,j,k-1,cx,cy,cz,rec,γ,dtdx,smallr,pfl)
        (mhzp,_,_,_,_,δLzp,_) = _mh_edges(u,Nx,Ny,Nz,i,j,k+1,cx,cy,cz,rec,γ,dtdx,smallr,pfl)
        Fzl = riemann(_facep(mhzm,δRzm), _facep(mhc,δLzc), 3, γ,ch,smallr,pfl,llf_dmin,llf_pmin,use_hlld)
        Fzh = riemann(_facep(mhc,δRzc),  _facep(mhzp,δLzp),3, γ,ch,smallr,pfl,llf_dmin,llf_pmin,use_hlld)
        U0 = (a1[c],a2[c],a3[c],a4[c],a5[c],a6[c],a7[c],a8[c],a9[c])
        r = ntuple(v -> U0[v] + dtdx*((Fxl[v]-Fxh[v])+(Fyl[v]-Fyh[v])+(Fzl[v]-Fzh[v])), 9)
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
    Nx,Ny,Nz = s.dims; dtdx = T(dt)/s.dx; cx,cy,cz = bc_codes(s); rec = Val(recon_code_of(s))
    step_ref_kernel!(s.be, 256)(s.scratch..., s.U..., Nx,Ny,Nz, cx,cy,cz, rec, dtdx, s.γ, T(ch), T(decay),
                                s.smallr, s.pfl, s.llf_dmin, s.llf_pmin, s.use_hlld; ndrange = ncells(s))
    KA.synchronize(s.be)
    s.U, s.scratch = s.scratch, s.U
    return s
end
