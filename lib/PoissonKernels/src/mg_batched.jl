# Batched multigrid — solve NB SAME-SIZE subgrids in ONE kernel launch per stage
# (a 4th "subgrid" batch dimension), to kill the per-subgrid GPU launch overhead
# that makes full-GPU AMR gravity launch-bound (hundreds of tiny V-cycles → one).
# Each batched kernel is the per-grid kernel (relax/defect/restrict/prolong/comp_accel)
# with a trailing batch index `b`; the math is identical, so a batched solve matches
# the per-subgrid solve. All grids in a batch share dims ⇒ shared h-factors/schedule.

using KernelAbstractions: @kernel, @index, @Const

@kernel function _relax_b!(sol, @Const(rhs), redblack::Int, h3, coef3)
    gi, gj, gk, b = @index(Global, NTuple)
    i = gi + 1; j = gj + 1; k = gk + 1
    @inbounds if (i + j + k) % 2 == redblack
        sol[i,j,k,b] = coef3 * (sol[i-1,j,k,b] + sol[i+1,j,k,b] + sol[i,j-1,k,b] +
                                sol[i,j+1,k,b] + sol[i,j,k-1,b] + sol[i,j,k+1,b] - h3 * rhs[i,j,k,b])
    end
end
function mg_relax_batched!(sol, rhs)
    be = KA.get_backend(sol); T = eltype(sol); d1,d2,d3,nb = size(sol)
    h1 = one(T)/T(d1-1); h2 = h1/T(d2-1); h3 = h2/T(d3-1); coef3 = one(T)/T(6)
    (d1<3||d2<3||d3<3) && return sol
    _relax_b!(be)(sol, rhs, 0, h3, coef3; ndrange=(d1-2,d2-2,d3-2,nb))
    _relax_b!(be)(sol, rhs, 1, h3, coef3; ndrange=(d1-2,d2-2,d3-2,nb))
    return sol
end

@kernel function _defect_b!(defect, @Const(sol), @Const(rhs), h3)
    gi, gj, gk, b = @index(Global, NTuple); i=gi+1;j=gj+1;k=gk+1; T=eltype(defect)
    @inbounds defect[i,j,k,b] = h3 * (sol[i-1,j,k,b]+sol[i+1,j,k,b]+sol[i,j-1,k,b]+
        sol[i,j+1,k,b]+sol[i,j,k-1,b]+sol[i,j,k+1,b] - T(6)*sol[i,j,k,b]) + rhs[i,j,k,b]
end
function mg_calc_defect_batched!(defect, sol, rhs)
    be = KA.get_backend(sol); T = eltype(sol); d1,d2,d3,nb = size(sol)
    h1=-T(d1-1); h2=h1*T(d2-1); h3=h2*T(d3-1); fill!(defect, zero(T))
    (d1>2&&d2>2&&d3>2) && _defect_b!(be)(defect, sol, rhs, h3; ndrange=(d1-2,d2-2,d3-2,nb))
    return defect
end

@kernel function _restrict_b!(dest, @Const(src), fact1, fact2, fact3, coef3,
                              sd1::Int, sd2::Int, sd3::Int, dd1::Int, dd2::Int, dd3::Int)
    i, j, k, b = @index(Global, NTuple); T = eltype(dest)
    @inbounds begin
        if k == 1 || k == dd3
            ksrc = (k==1) ? 1 : sd3
            i1 = min(max(unsafe_trunc(Int, T(i-1)*fact1+T(0.5))+1,1),sd1)
            j1 = min(max(unsafe_trunc(Int, T(j-1)*fact2+T(0.5))+1,1),sd2)
            dest[i,j,k,b] = src[i1,j1,ksrc,b]
        elseif j == 1 || j == dd2
            jsrc = (j==1) ? 1 : sd2
            i1 = min(max(unsafe_trunc(Int, T(i-1)*fact1+T(0.5))+1,1),sd1)
            k1 = unsafe_trunc(Int, T(k-1)*fact3+T(0.5))+1
            dest[i,j,k,b] = src[i1,jsrc,k1,b]
        elseif i == 1 || i == dd1
            isrc = (i==1) ? 1 : sd1
            j1 = unsafe_trunc(Int, T(j-1)*fact2+T(0.5))+1
            k1 = unsafe_trunc(Int, T(k-1)*fact3+T(0.5))+1
            dest[i,j,k,b] = src[isrc,j1,k1,b]
        else
            x=T(i-1)*fact1+T(0.5); i1=unsafe_trunc(Int,x)+1
            y=T(j-1)*fact2+T(0.5); j1=unsafe_trunc(Int,y)+1
            z=T(k-1)*fact3+T(0.5); k1=unsafe_trunc(Int,z)+1
            dxm=T(0.5)*(T(i1)-x)^2; dxp=T(0.5)*(one(T)+x-T(i1))^2; dx0=one(T)-dxp-dxm
            dym=T(0.5)*(T(j1)-y)^2; dyp=T(0.5)*(one(T)+y-T(j1))^2; dy0=one(T)-dyp-dym
            dzm=T(0.5)*(T(k1)-z)^2; dzp=T(0.5)*(one(T)+z-T(k1))^2; dz0=one(T)-dzp-dzm
            v = src[i1-1,j1-1,k1-1,b]*dxm*dym*dzm + src[i1,j1-1,k1-1,b]*dx0*dym*dzm + src[i1+1,j1-1,k1-1,b]*dxp*dym*dzm +
                src[i1-1,j1,k1-1,b]*dxm*dy0*dzm + src[i1,j1,k1-1,b]*dx0*dy0*dzm + src[i1+1,j1,k1-1,b]*dxp*dy0*dzm +
                src[i1-1,j1+1,k1-1,b]*dxm*dyp*dzm + src[i1,j1+1,k1-1,b]*dx0*dyp*dzm + src[i1+1,j1+1,k1-1,b]*dxp*dyp*dzm +
                src[i1-1,j1-1,k1,b]*dxm*dym*dz0 + src[i1,j1-1,k1,b]*dx0*dym*dz0 + src[i1+1,j1-1,k1,b]*dxp*dym*dz0 +
                src[i1-1,j1,k1,b]*dxm*dy0*dz0 + src[i1,j1,k1,b]*dx0*dy0*dz0 + src[i1+1,j1,k1,b]*dxp*dy0*dz0 +
                src[i1-1,j1+1,k1,b]*dxm*dyp*dz0 + src[i1,j1+1,k1,b]*dx0*dyp*dz0 + src[i1+1,j1+1,k1,b]*dxp*dyp*dz0 +
                src[i1-1,j1-1,k1+1,b]*dxm*dym*dzp + src[i1,j1-1,k1+1,b]*dx0*dym*dzp + src[i1+1,j1-1,k1+1,b]*dxp*dym*dzp +
                src[i1-1,j1,k1+1,b]*dxm*dy0*dzp + src[i1,j1,k1+1,b]*dx0*dy0*dzp + src[i1+1,j1,k1+1,b]*dxp*dy0*dzp +
                src[i1-1,j1+1,k1+1,b]*dxm*dyp*dzp + src[i1,j1+1,k1+1,b]*dx0*dyp*dzp + src[i1+1,j1+1,k1+1,b]*dxp*dyp*dzp
            dest[i,j,k,b] = coef3 * v
        end
    end
end
function mg_restrict_batched!(dest, src)
    be = KA.get_backend(dest); T = eltype(dest)
    sd1,sd2,sd3,nb = size(src); dd1,dd2,dd3,_ = size(dest)
    f1=T(sd1-1)/T(dd1-1); f2=T(sd2-1)/T(dd2-1); f3=T(sd3-1)/T(dd3-1); coef3=one(T)/T(8)
    _restrict_b!(be)(dest, src, f1,f2,f3,coef3, sd1,sd2,sd3, dd1,dd2,dd3; ndrange=(dd1,dd2,dd3,nb))
    return dest
end

@kernel function _prolong_b!(dest, @Const(src), fact1, fact2, fact3, half, edge1, edge2, edge3)
    i, j, k, b = @index(Global, NTuple); T = eltype(dest)
    @inbounds begin
        x=min(max(T(i-1)*fact1+T(0.5),half),edge1); i1=unsafe_trunc(Int,x+T(0.5)); dx=T(i1)+T(0.5)-x
        y=min(max(T(j-1)*fact2+T(0.5),half),edge2); j1=unsafe_trunc(Int,y+T(0.5)); dy=T(j1)+T(0.5)-y
        z=min(max(T(k-1)*fact3+T(0.5),half),edge3); k1=unsafe_trunc(Int,z+T(0.5)); dz=T(k1)+T(0.5)-z
        dest[i,j,k,b] = src[i1,j1,k1,b]*dx*dy*dz + src[i1+1,j1,k1,b]*(one(T)-dx)*dy*dz +
            src[i1,j1+1,k1,b]*dx*(one(T)-dy)*dz + src[i1+1,j1+1,k1,b]*(one(T)-dx)*(one(T)-dy)*dz +
            src[i1,j1,k1+1,b]*dx*dy*(one(T)-dz) + src[i1+1,j1,k1+1,b]*(one(T)-dx)*dy*(one(T)-dz) +
            src[i1,j1+1,k1+1,b]*dx*(one(T)-dy)*(one(T)-dz) + src[i1+1,j1+1,k1+1,b]*(one(T)-dx)*(one(T)-dy)*(one(T)-dz)
    end
end
function mg_prolong_batched!(dest, src)
    be = KA.get_backend(dest); T = eltype(dest)
    sd1,sd2,sd3,nb = size(src); dd1,dd2,dd3,_ = size(dest)
    f1=T(sd1-1)/T(dd1-1); f2=T(sd2-1)/T(dd2-1); f3=T(sd3-1)/T(dd3-1); half=T(0.5001)
    _prolong_b!(be)(dest, src, f1,f2,f3, half, T(sd1)-half,T(sd2)-half,T(sd3)-half; ndrange=(dd1,dd2,dd3,nb))
    return dest
end

@kernel function _compaccel_b!(d1f, d2f, d3f, @Const(src), f1, f2, f3, iflag::Int, s1::Int, s2::Int, s3::Int)
    i, j, k, b = @index(Global, NTuple)
    @inbounds begin
        d1f[i,j,k,b] = f1 * (src[i+s1+iflag,j+s2,k+s3,b] - src[i+s1-1,j+s2,k+s3,b])
        d2f[i,j,k,b] = f2 * (src[i+s1,j+s2+iflag,k+s3,b] - src[i+s1,j+s2-1,k+s3,b])
        d3f[i,j,k,b] = f3 * (src[i+s1,j+s2,k+s3+iflag,b] - src[i+s1,j+s2,k+s3-1,b])
    end
end
function comp_accel_batched!(d1, d2, d3, src; iflag::Integer, start, del)
    be = KA.get_backend(d1); T = eltype(d1); dd1,dd2,dd3,nb = size(d1)
    f1=-one(T)/(T(iflag+1)*T(del[1])); f2=-one(T)/(T(iflag+1)*T(del[2])); f3=-one(T)/(T(iflag+1)*T(del[3]))
    _compaccel_b!(be)(d1,d2,d3,src, f1,f2,f3, Int(iflag), Int(start[1]),Int(start[2]),Int(start[3]); ndrange=(dd1,dd2,dd3,nb))
    return d1,d2,d3
end

# save/re-impose the per-subgrid Dirichlet boundary ring of a batched array (faces of dims 1..3)
_save_faces_b(A) = (A[1,:,:,:], A[end,:,:,:], A[:,1,:,:], A[:,end,:,:], A[:,:,1,:], A[:,:,end,:])
function _impose_faces_b!(A, f)
    A[1,:,:,:].=f[1]; A[end,:,:,:].=f[2]; A[:,1,:,:].=f[3]; A[:,end,:,:].=f[4]; A[:,:,1,:].=f[5]; A[:,:,end,:].=f[6]; A
end

"""
    vcycle_batched!(sol, rhs; cycle=:W, ncyc=30, pre=2, post=3, dirichlet=true)

Fixed-count batched multigrid for `sol[d,d,d,NB]` (NB same-size subgrids), in place.
Same operators as `vcycle_solve!`; processes all NB grids per kernel launch. Uses a
fixed cycle count (no per-batch residual reduction). `dirichlet=true` re-imposes each
subgrid's initial boundary each cycle (parent-interpolated φ). Cubic dims assumed.
"""
function vcycle_batched!(sol::AbstractArray{T,4}, rhs::AbstractArray{T,4};
                         cycle::Symbol=:W, ncyc::Integer=30, pre::Integer=2, post::Integer=3,
                         dirichlet::Bool=true) where {T}
    be = KA.get_backend(sol); d0 = size(sol)[1:3]; nb = size(sol,4)
    dims = mg_dims_schedule(d0); nlev = length(dims); mu = cycle===:W ? 2 : 1
    faces = dirichlet ? _save_faces_b(sol) : nothing
    Sol=Vector{Any}(undef,nlev); RHS=Vector{Any}(undef,nlev); Def=Vector{Any}(undef,nlev)
    Sol[1]=sol; RHS[1]=rhs; Def[1]=KA.zeros(be,T,d0...,nb)
    for L in 2:nlev
        Sol[L]=KA.zeros(be,T,dims[L]...,nb); RHS[L]=KA.zeros(be,T,dims[L]...,nb); Def[L]=KA.zeros(be,T,dims[L]...,nb)
    end
    function mucycle(L)
        if L == nlev
            for _ in 1:(3pre); mg_relax_batched!(Sol[L],RHS[L]); end; return
        end
        for _ in 1:pre; mg_relax_batched!(Sol[L],RHS[L]); end
        mg_calc_defect_batched!(Def[L],Sol[L],RHS[L])
        mg_restrict_batched!(RHS[L+1],Def[L]); fill!(Sol[L+1],zero(T))
        for _ in 1:mu; mucycle(L+1); end
        mg_prolong_batched!(Def[L],Sol[L+1]); Sol[L].+=Def[L]
        for _ in 1:post; mg_relax_batched!(Sol[L],RHS[L]); end
    end
    for _ in 1:ncyc
        mucycle(1)
        faces===nothing || _impose_faces_b!(Sol[1],faces)
    end
    KA.synchronize(be)
    return sol
end
