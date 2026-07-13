# ── Initial conditions ────────────────────────────────────────────────────────
# Cell centres at x=(i-½)dx etc.; domain [0,L]^3 with L=Nx·dx. All write the
# conserved fields of an `MHDState` via prim2cons. Smooth waves are the convergence
# gate; Orszag-Tang (periodic) is the divB-cleaning demo; Brio-Wu needs the
# (not-yet-wired) outflow BC to be physical at the ends.
export init_alfven_wave!, init_orszag_tang!, init_brio_wu!, alfven_By_exact, init_turb_field!
export init_field_loop!

@kernel function _turb_field_kernel!(o1,o2,o3,o4,o5,o6,o7,o8,o9, Nx::Int,Ny::Int,Nz::Int, dx::T, γ::T) where {T}
    c = @index(Global, Linear)
    @inbounds begin
        i=(c-1)%Nx+1; j=((c-1)÷Nx)%Ny+1; k=(c-1)÷(Nx*Ny)+1
        Lx=Nx*dx; Ly=Ny*dx; Lz=Nz*dx
        x=T(2)*T(π)*(i-T(0.5))*dx/Lx; y=T(2)*T(π)*(j-T(0.5))*dx/Ly; z=T(2)*T(π)*(k-T(0.5))*dx/Lz
        # smooth, all-3-axes-varying state (exercises every sweep direction)
        ρ=one(T); p=one(T)
        vx=T(0.2)*sin(z); vy=T(0.2)*sin(x); vz=T(0.2)*sin(y)
        Bx=T(0.3)+T(0.1)*sin(T(2)*y); By=T(0.1)*sin(T(2)*z); Bz=T(0.1)*sin(T(2)*x)
        U=prim2cons((ρ,vx,vy,vz,p,Bx,By,Bz,zero(T)), γ)
        o1[c]=U[1];o2[c]=U[2];o3[c]=U[3];o4[c]=U[4];o5[c]=U[5];o6[c]=U[6];o7[c]=U[7];o8[c]=U[8];o9[c]=U[9]
    end
end

"Smooth 3-D field varying in all directions (for ref↔cube cross-check + throughput)."
function init_turb_field!(s::MHDState{T}) where {T}
    Nx,Ny,Nz = s.dims
    _turb_field_kernel!(s.be,256)(s.U..., Nx,Ny,Nz, s.dx, s.γ; ndrange=ncells(s))
    KA.synchronize(s.be); return s
end

@kernel function _alfven_kernel!(o1,o2,o3,o4,o5,o6,o7,o8,o9, Nx::Int,Ny::Int,Nz::Int,
        dx::T, amp::T, B0::T, p0::T, γ::T) where {T}
    c = @index(Global, Linear)
    @inbounds begin
        i=(c-1)%Nx+1
        x = (i-T(0.5))*dx; L = Nx*dx; k = T(2)*T(π)/L; phase = k*x
        By = amp*sin(phase); Bz = amp*cos(phase)
        # right-going circularly-polarised Alfvén wave (vA = B0/√ρ = 1 with ρ=1): v⊥ = -B⊥
        U = prim2cons((one(T), zero(T), -By, -Bz, p0, B0, By, Bz, zero(T)), γ)
        o1[c]=U[1];o2[c]=U[2];o3[c]=U[3];o4[c]=U[4];o5[c]=U[5];o6[c]=U[6];o7[c]=U[7];o8[c]=U[8];o9[c]=U[9]
    end
end

"""
    init_alfven_wave!(s; amp=1e-3, B0=1, p0=0.1)

Right-going circularly-polarised Alfvén wave along x (vA=1). The exact solution is
the initial profile translated by +t in x; use [`alfven_By_exact`](@ref) for the L1
gate.
"""
function init_alfven_wave!(s::MHDState{T}; amp::Real=1e-3, B0::Real=1, p0::Real=0.1) where {T}
    Nx,Ny,Nz = s.dims
    _alfven_kernel!(s.be,256)(s.U..., Nx,Ny,Nz, s.dx, T(amp), T(B0), T(p0), s.γ; ndrange=ncells(s))
    KA.synchronize(s.be); return s
end

"Exact By(x,t) of the right-going CP Alfvén wave (vA=1), for the convergence L1."
alfven_By_exact(x::Real, t::Real, L::Real; amp::Real=1e-3) =
    amp*sin(2π/L*(mod(x - t, L)))

@kernel function _orszag_tang_kernel!(o1,o2,o3,o4,o5,o6,o7,o8,o9, Nx::Int,Ny::Int,Nz::Int,
        dx::T, γ::T) where {T}
    c = @index(Global, Linear)
    @inbounds begin
        i=(c-1)%Nx+1; j=((c-1)÷Nx)%Ny+1
        x=(i-T(0.5))*dx; y=(j-T(0.5))*dx; L=Nx*dx; tp=T(2)*T(π)/L
        ρ=γ*γ; p=γ; B0=one(T)/sqrt(T(4)*T(π))
        vx=-sin(tp*y); vy=sin(tp*x)
        Bx=-B0*sin(tp*y); By=B0*sin(T(2)*tp*x)
        U=prim2cons((ρ, vx, vy, zero(T), p, Bx, By, zero(T), zero(T)), γ)
        o1[c]=U[1];o2[c]=U[2];o3[c]=U[3];o4[c]=U[4];o5[c]=U[5];o6[c]=U[6];o7[c]=U[7];o8[c]=U[8];o9[c]=U[9]
    end
end

"Orszag-Tang vortex (2D, periodic, γ=5/3). Set the state's γ to 5/3."
function init_orszag_tang!(s::MHDState{T}) where {T}
    Nx,Ny,Nz = s.dims
    _orszag_tang_kernel!(s.be,256)(s.U..., Nx,Ny,Nz, s.dx, s.γ; ndrange=ncells(s))
    KA.synchronize(s.be); return s
end

@kernel function _brio_wu_kernel!(o1,o2,o3,o4,o5,o6,o7,o8,o9, Nx::Int,Ny::Int,Nz::Int, γ::T) where {T}
    c = @index(Global, Linear)
    @inbounds begin
        i=(c-1)%Nx+1; left = i <= Nx÷2
        ρ = left ? one(T) : T(0.125); p = left ? one(T) : T(0.1)
        By = left ? one(T) : -one(T); Bx = T(0.75)
        U=prim2cons((ρ, zero(T),zero(T),zero(T), p, Bx, By, zero(T), zero(T)), γ)
        o1[c]=U[1];o2[c]=U[2];o3[c]=U[3];o4[c]=U[4];o5[c]=U[5];o6[c]=U[6];o7[c]=U[7];o8[c]=U[8];o9[c]=U[9]
    end
end

"Brio-Wu MHD shock tube along x (γ=2). NOTE: needs outflow BC (not yet wired) to be physical."
function init_brio_wu!(s::MHDState{T}) where {T}
    Nx,Ny,Nz = s.dims
    _brio_wu_kernel!(s.be,256)(s.U..., Nx,Ny,Nz, s.γ; ndrange=ncells(s))
    KA.synchronize(s.be); return s
end

@kernel function _field_loop_A_k!(A, Nx::Int, Ny::Int, Nz::Int, dx, radius, amplitude)
    c = @index(Global, Linear)
    @inbounds begin
        T = eltype(A)
        i = (c-1) % Nx + 1
        j = ((c-1) ÷ Nx) % Ny + 1
        x = (T(i) - T(0.5)) * dx - T(0.5) * T(Nx) * dx
        y = (T(j) - T(0.5)) * dx - T(0.5) * T(Ny) * dx
        r = sqrt(x*x + y*y)
        A[c] = amplitude * max(radius - r, zero(T))
    end
end

@kernel function _field_loop_cons_k!(o1,o2,o3,o4,o5,o6,o7,o8,o9, @Const(A),
                                     Nx::Int, Ny::Int, Nz::Int, dx, gamma,
                                     rho0, p0, vx0, vy0, vz0)
    c = @index(Global, Linear)
    @inbounds begin
        T = eltype(o1)
        i = (c-1) % Nx + 1
        j = ((c-1) ÷ Nx) % Ny + 1
        k = (c-1) ÷ (Nx*Ny) + 1
        ip = mod(i, Nx) + 1; im = mod(i-2, Nx) + 1
        jp = mod(j, Ny) + 1; jm = mod(j-2, Ny) + 1
        L(ii,jj,kk) = ((kk-1)*Ny + (jj-1))*Nx + ii
        inv2dx = inv(T(2)*dx)
        bx = (A[L(i,jp,k)] - A[L(i,jm,k)]) * inv2dx
        by = -(A[L(ip,j,k)] - A[L(im,j,k)]) * inv2dx
        U = prim2cons((rho0,vx0,vy0,vz0,p0,bx,by,zero(T),zero(T)), gamma)
        o1[c]=U[1];o2[c]=U[2];o3[c]=U[3];o4[c]=U[4];o5[c]=U[5]
        o6[c]=U[6];o7[c]=U[7];o8[c]=U[8];o9[c]=U[9]
    end
end

"""
    init_field_loop!(s; radius=0.3, amplitude=1e-3, rho0=1, p0=1,
                     velocity=(1, 1, 0))

Initialize the periodic Gardiner-Stone magnetic-field-loop advection problem.
The in-plane field is formed as a discrete centered curl of `A_z`, making its
initial centered divergence zero to roundoff.
"""
function init_field_loop!(s::MHDState{T}; radius::Real=0.3, amplitude::Real=1e-3,
                          rho0::Real=1, p0::Real=1,
                          velocity=(1, 1, 0)) where {T}
    Nx,Ny,Nz = s.dims
    vx0,vy0,vz0 = velocity
    _field_loop_A_k!(s.be,256)(s.scratch[1], Nx,Ny,Nz, s.dx, T(radius), T(amplitude);
                               ndrange=ncells(s))
    _field_loop_cons_k!(s.be,256)(s.U..., s.scratch[1], Nx,Ny,Nz, s.dx, s.γ,
                                  T(rho0), T(p0), T(vx0), T(vy0), T(vz0);
                                  ndrange=ncells(s))
    KA.synchronize(s.be)
    return s
end
