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
export step_cube!

const CTB = 4                 # owned cells/dim/block
const CTT = CTB + 4           # 8: prim tile incl. 2-cell halo
const CNCP = CTT*CTT*CTT      # 512
const CNFX = (CTB+1)*CTB*CTB  # 80 faces per direction
const CUBE_GS = 192           # threads per group

@inline _ctile(pi,pj,pk) = pi + CTT*(pj + CTT*pk)
@inline _cfx(fi,fj,fk) = fi + (CTB+1)*(fj + CTB*fk)
@inline _cfy(fi,fj,fk) = fi + CTB*(fj + (CTB+1)*fk)
@inline _cfz(fi,fj,fk) = fi + CTB*(fj + CTB*fk)
# SoA variable-major shared (lin contiguous ⇒ coalesced / bank-conflict-free).
@inline _sg9(S,NC,lin) = @inbounds ntuple(v -> S[(v-1)*NC+lin+1], 9)
@inline function _sp9!(S,NC,lin,q)
    @inbounds for v in 1:9; S[(v-1)*NC+lin+1] = q[v]; end
end

@kernel function step_cube_kernel!(o1,o2,o3,o4,o5,o6,o7,o8,o9,
        @Const(a1),@Const(a2),@Const(a3),@Const(a4),@Const(a5),@Const(a6),@Const(a7),@Const(a8),@Const(a9),
        N::Int, nb::Int, cx::Int, cy::Int, cz::Int, rec::Val, per::Val, dtdx::T, γ::T, ch::T, decay::T,
        smallr::T, pfl::T, llf_dmin::T, llf_pmin::T, use_hlld::Bool) where {T}
    SP = @localmem T (9*CNCP)
    LX = @localmem T (9*CNFX); RX = @localmem T (9*CNFX)
    LY = @localmem T (9*CNFX); RY = @localmem T (9*CNFX)
    LZ = @localmem T (9*CNFX); RZ = @localmem T (9*CNFX)
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
            m0=_sg9(SP,CNCP,_ctile(pi,pj,pk))
            δLx,δRx=recon_offsets(_sg9(SP,CNCP,_ctile(pi-1,pj,pk)),m0,_sg9(SP,CNCP,_ctile(pi+1,pj,pk)),rec)
            δLy,δRy=recon_offsets(_sg9(SP,CNCP,_ctile(pi,pj-1,pk)),m0,_sg9(SP,CNCP,_ctile(pi,pj+1,pk)),rec)
            δLz,δRz=recon_offsets(_sg9(SP,CNCP,_ctile(pi,pj,pk-1)),m0,_sg9(SP,CNCP,_ctile(pi,pj,pk+1)),rec)
            uh=hancock_edges(m0,δLx,δRx,δLy,δRy,δLz,δRz,dtdx,γ); mh=cons2prim(uh,γ,smallr,pfl)
            inxt=(pj>=2&&pj<=CTB+1)&&(pk>=2&&pk<=CTB+1)
            inyt=(pi>=2&&pi<=CTB+1)&&(pk>=2&&pk<=CTB+1)
            inzt=(pi>=2&&pi<=CTB+1)&&(pj>=2&&pj<=CTB+1)
            if inxt
                if pi<=CTB+1; _sp9!(LX,CNFX,_cfx(pi-1,pj-2,pk-2),_facep(mh,δRx)); end
                if pi>=2;     _sp9!(RX,CNFX,_cfx(pi-2,pj-2,pk-2),_facep(mh,δLx)); end
            end
            if inyt
                if pj<=CTB+1; _sp9!(LY,CNFX,_cfy(pi-2,pj-1,pk-2),_facep(mh,δRy)); end
                if pj>=2;     _sp9!(RY,CNFX,_cfy(pi-2,pj-2,pk-2),_facep(mh,δLy)); end
            end
            if inzt
                if pk<=CTB+1; _sp9!(LZ,CNFX,_cfz(pi-2,pj-2,pk-1),_facep(mh,δRz)); end
                if pk>=2;     _sp9!(RZ,CNFX,_cfz(pi-2,pj-2,pk-2),_facep(mh,δLz)); end
            end
            t+=nth
        end
        @synchronize
        # Stage 3: GLM Riemann once per face, flux written back into L*
        nfx=(CTB+1)*CTB*CTB; t=tid
        while t<=3*nfx
            if t<=nfx
                l=t-1; fi=l%(CTB+1); fj=(l÷(CTB+1))%CTB; fk=l÷((CTB+1)*CTB); ln=_cfx(fi,fj,fk)
                F=riemann(_sg9(LX,CNFX,ln),_sg9(RX,CNFX,ln),1,γ,ch,smallr,pfl,llf_dmin,llf_pmin,use_hlld)
                _sp9!(LX,CNFX,ln,F)
            elseif t<=2*nfx
                l=t-1-nfx; fi=l%CTB; fj=(l÷CTB)%(CTB+1); fk=l÷(CTB*(CTB+1)); ln=_cfy(fi,fj,fk)
                F=riemann(_sg9(LY,CNFX,ln),_sg9(RY,CNFX,ln),2,γ,ch,smallr,pfl,llf_dmin,llf_pmin,use_hlld)
                _sp9!(LY,CNFX,ln,F)
            else
                l=t-1-2*nfx; fi=l%CTB; fj=(l÷CTB)%CTB; fk=l÷(CTB*CTB); ln=_cfz(fi,fj,fk)
                F=riemann(_sg9(LZ,CNFX,ln),_sg9(RZ,CNFX,ln),3,γ,ch,smallr,pfl,llf_dmin,llf_pmin,use_hlld)
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
            Fxl=_sg9(LX,CNFX,_cfx(a,b,c));   Fxh=_sg9(LX,CNFX,_cfx(a+1,b,c))
            Fyl=_sg9(LY,CNFX,_cfy(a,b,c));   Fyh=_sg9(LY,CNFX,_cfy(a,b+1,c))
            Fzl=_sg9(LZ,CNFX,_cfz(a,b,c));   Fzh=_sg9(LZ,CNFX,_cfz(a,b,c+1))
            U0=(a1[idx],a2[idx],a3[idx],a4[idx],a5[idx],a6[idx],a7[idx],a8[idx],a9[idx])
            r=ntuple(v->U0[v]+dtdx*((Fxl[v]-Fxh[v])+(Fyl[v]-Fyh[v])+(Fzl[v]-Fzh[v])), 9)
            o1[idx]=r[1];o2[idx]=r[2];o3[idx]=r[3];o4[idx]=r[4];o5[idx]=r[5];o6[idx]=r[6];o7[idx]=r[7];o8[idx]=r[8]
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
    rec = Val(recon_code_of(s)); per = Val(all_periodic(s))
    step_cube_kernel!(be, CUBE_GS)(s.scratch..., s.U..., N, nb, cx,cy,cz, rec, per, dtdx, s.γ, T(ch), T(decay),
            s.smallr, s.pfl, s.llf_dmin, s.llf_pmin, s.use_hlld; ndrange = nb*nb*nb*CUBE_GS)
    KA.synchronize(be)
end
