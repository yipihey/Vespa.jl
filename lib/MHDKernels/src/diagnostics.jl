# ── Diagnostics: ∇·B monitor + conserved totals ──────────────────────────────
export max_divb, conserved_totals

# |∇·B| via BC-aware central differences (the GLM cleaning is judged by this). The
# normal-B flip from `_bcidx` is what makes a reflecting wall read ∂Bn correctly.
@kernel function _divb_kernel!(d, @Const(bx),@Const(by),@Const(bz), Nx::Int,Ny::Int,Nz::Int,
        cx::Int,cy::Int,cz::Int, inv2dx::T) where {T}
    c = @index(Global, Linear)
    @inbounds begin
        i=(c-1)%Nx+1; j=((c-1)÷Nx)%Ny+1; k=(c-1)÷(Nx*Ny)+1
        (ip,fxp)=_bcidx(i+1,Nx,cx); (im,fxm)=_bcidx(i-1,Nx,cx)
        (jp,fyp)=_bcidx(j+1,Ny,cy); (jm,fym)=_bcidx(j-1,Ny,cy)
        (kp,fzp)=_bcidx(k+1,Nz,cz); (km,fzm)=_bcidx(k-1,Nz,cz)
        L(ii,jj,kk)=((kk-1)*Ny+(jj-1))*Nx+ii
        db = (bx[L(ip,j,k)]*fxp - bx[L(im,j,k)]*fxm
            + by[L(i,jp,k)]*fyp - by[L(i,jm,k)]*fym
            + bz[L(i,j,kp)]*fzp - bz[L(i,j,km)]*fzm)*inv2dx
        d[c] = abs(db)
    end
end

"`max_divb(s)` — max |∇·B| over the grid (BC-aware central differences)."
function max_divb(s::MHDState{T}) where {T}
    Nx,Ny,Nz = s.dims; cx,cy,cz = bc_codes(s)
    d = device_zeros(s.be, T, (ncells(s),))
    _divb_kernel!(s.be, 256)(d, s.U[6],s.U[7],s.U[8], Nx,Ny,Nz, cx,cy,cz, one(T)/(2*s.dx); ndrange = ncells(s))
    KA.synchronize(s.be)
    return maximum(d)
end

"""
    conserved_totals(s) -> NamedTuple

Volume-integrated conserved quantities (mass, momentum, total energy, ∫B). ψ is
excluded — it is damped, not conserved.
"""
function conserved_totals(s::MHDState{T}) where {T}
    dV = s.dx^3
    h = fields_to_host(s)
    (mass = sum(Float64, h[1])*Float64(dV),
     momx = sum(Float64, h[2])*Float64(dV),
     momy = sum(Float64, h[3])*Float64(dV),
     momz = sum(Float64, h[4])*Float64(dV),
     energy = sum(Float64, h[5])*Float64(dV),
     Bx = sum(Float64, h[6])*Float64(dV),
     By = sum(Float64, h[7])*Float64(dV),
     Bz = sum(Float64, h[8])*Float64(dV))
end
