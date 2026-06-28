module MHDKernelsCUDAExt

using MHDKernels
using CUDA

# Reuse the EXACT device functions + tile geometry from the core, so the raw kernel is
# correct-by-construction and cross-checks against the reference integrator to round-off.
import MHDKernels: CTB, CTT, CNCP, CNFX, _ctile, _cfx, _cfy, _cfz, _sg9, _sp9!,
                   _loadP_sel, recon_offsets, hancock_edges, _facep, riemann, cons2prim,
                   bc_codes, recon_code_of, all_periodic, MHDState, _cube_launch!

function __init__()
    if CUDA.functional()
        MHDKernels.register_backend!(:cuda, CUDABackend())
    end
end

MHDKernels.device_zeros(::CUDABackend, ::Type{T}, dims::Dims) where {T} = CUDA.zeros(T, dims)

# ── raw-CUDA cube: same 4-stage shared-memory algorithm as the KA cube, but with
# native CuStaticSharedArray + a 3-D @cuda grid + default register allocation, which
# recovers the NVIDIA peak the KA abstraction leaves on the table. f32-only. ────────
const RAW_GS = 192

function raw_cube_kernel!(o1,o2,o3,o4,o5,o6,o7,o8,o9,
        a1,a2,a3,a4,a5,a6,a7,a8,a9, N::Int, cx::Int, cy::Int, cz::Int, rec::Val, per::Val,
        dtdx::Float32, γ::Float32, ch::Float32, decay::Float32,
        smallr::Float32, pfl::Float32, llf_dmin::Float32, llf_pmin::Float32, use_hlld::Bool)
    SP = CuStaticSharedArray(Float32, 9*CNCP)
    LX = CuStaticSharedArray(Float32, 9*CNFX); RX = CuStaticSharedArray(Float32, 9*CNFX)
    LY = CuStaticSharedArray(Float32, 9*CNFX); RY = CuStaticSharedArray(Float32, 9*CNFX)
    LZ = CuStaticSharedArray(Float32, 9*CNFX); RZ = CuStaticSharedArray(Float32, 9*CNFX)
    @fastmath @inbounds begin
        u = (a1,a2,a3,a4,a5,a6,a7,a8,a9)
        tid = threadIdx().x; nth = blockDim().x
        ox = (blockIdx().x-1)*CTB; oy = (blockIdx().y-1)*CTB; oz = (blockIdx().z-1)*CTB
        # Stage 1: cons2prim -> 8^3 tile (BC-synthesised halo)
        t = tid
        while t <= CNCP
            l=t-1; pi=l%CTT; pj=(l÷CTT)%CTT; pk=l÷(CTT*CTT)
            lx=ox+pi-1; ly=oy+pj-1; lz=oz+pk-1
            _sp9!(SP,CNCP,_ctile(pi,pj,pk), _loadP_sel(per,u,N,N,N,lx,ly,lz,cx,cy,cz,γ,smallr,pfl)); t+=nth
        end
        sync_threads()
        # Stage 2: trace -> per-face interface states
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
        sync_threads()
        # Stage 3: GLM Riemann once per face
        nfx=(CTB+1)*CTB*CTB; t=tid
        while t<=3*nfx
            if t<=nfx
                l=t-1; fi=l%(CTB+1); fj=(l÷(CTB+1))%CTB; fk=l÷((CTB+1)*CTB); ln=_cfx(fi,fj,fk)
                _sp9!(LX,CNFX,ln, riemann(_sg9(LX,CNFX,ln),_sg9(RX,CNFX,ln),1,γ,ch,smallr,pfl,llf_dmin,llf_pmin,use_hlld))
            elseif t<=2*nfx
                l=t-1-nfx; fi=l%CTB; fj=(l÷CTB)%(CTB+1); fk=l÷(CTB*(CTB+1)); ln=_cfy(fi,fj,fk)
                _sp9!(LY,CNFX,ln, riemann(_sg9(LY,CNFX,ln),_sg9(RY,CNFX,ln),2,γ,ch,smallr,pfl,llf_dmin,llf_pmin,use_hlld))
            else
                l=t-1-2*nfx; fi=l%CTB; fj=(l÷CTB)%CTB; fk=l÷(CTB*CTB); ln=_cfz(fi,fj,fk)
                _sp9!(LZ,CNFX,ln, riemann(_sg9(LZ,CNFX,ln),_sg9(RZ,CNFX,ln),3,γ,ch,smallr,pfl,llf_dmin,llf_pmin,use_hlld))
            end
            t+=nth
        end
        sync_threads()
        # Stage 4: conservative update of the 4^3 owned cells (+ψ damping)
        t=tid
        while t<=CTB*CTB*CTB
            l=t-1; a=l%CTB; b=(l÷CTB)%CTB; c=l÷(CTB*CTB)
            gi=mod(ox+a,N)+1; gj=mod(oy+b,N)+1; gk=mod(oz+c,N)+1; idx=((gk-1)*N+(gj-1))*N+gi
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
    return nothing
end

# raw-CUDA launch for :auto/:raw on a CUDA-f32 state (overrides the core KA default).
function MHDKernels._cube_launch!(::Union{Val{:auto},Val{:raw}}, be::CUDABackend,
                                  s::MHDState{Float32}, dt::Real, ch::Real, decay::Real)
    N = s.dims[1]; nb = N ÷ CTB; dtdx = Float32(dt/s.dx)
    cx,cy,cz = bc_codes(s); rec = recon_code_of(s); per = all_periodic(s)
    @cuda threads=RAW_GS blocks=(nb,nb,nb) raw_cube_kernel!(
        s.scratch..., s.U..., N, cx,cy,cz, Val(rec), Val(per), dtdx, s.γ, Float32(ch), Float32(decay),
        s.smallr, s.pfl, s.llf_dmin, s.llf_pmin, s.use_hlld)
    CUDA.synchronize()
end

end # module
