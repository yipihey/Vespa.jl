# ── Boundary conditions (ghost-free: synthesised at neighbour-load time) ──────
# Per-axis BC, encoded as Int codes for the GPU kernels: 0=periodic, 1=outflow
# (zero-gradient), 2=reflecting (mirror about the boundary face, normal v & B
# flipped). `_bcidx` maps a possibly out-of-range logical index to an in-range source
# index plus a flip sign (-1 only for the reflected normal component). `_loadU_bc`
# loads the (flip-transformed) conserved state; flips compose per axis, so corner
# ghosts are handled correctly. Owned/interior cells never trigger a flip.
const BC_PERIODIC = 0
const BC_OUTFLOW  = 1
const BC_REFLECT  = 2

bc_code(s::Symbol) = s === :periodic ? BC_PERIODIC :
                     s === :outflow  ? BC_OUTFLOW  :
                     s === :reflecting ? BC_REFLECT :
                     error("unknown BC :$s (have :periodic, :outflow, :reflecting)")

@inline function _bcidx(i::Int, N::Int, code::Int)
    if code == BC_PERIODIC
        return (mod(i-1, N) + 1, 1)
    elseif code == BC_OUTFLOW
        return (i < 1 ? 1 : (i > N ? N : i), 1)
    else                                   # reflecting: mirror about the face
        if i < 1
            return (1 - i, -1)             # i=0→1, i=-1→2
        elseif i > N
            return (2N + 1 - i, -1)        # i=N+1→N, i=N+2→N-1
        else
            return (i, 1)
        end
    end
end

# Conserved state at logical (i,j,k) under the per-axis BCs (flips normal v & B).
@inline function _loadU_bc(u::NTuple{9,A}, Nx::Int, Ny::Int, Nz::Int, i::Int, j::Int, k::Int,
                           cx::Int, cy::Int, cz::Int) where {A}
    (ii,fx)=_bcidx(i,Nx,cx); (jj,fy)=_bcidx(j,Ny,cy); (kk,fz)=_bcidx(k,Nz,cz)
    idx = ((kk-1)*Ny + (jj-1))*Nx + ii
    @inbounds (u[1][idx], u[2][idx]*fx, u[3][idx]*fy, u[4][idx]*fz, u[5][idx],
               u[6][idx]*fx, u[7][idx]*fy, u[8][idx]*fz, u[9][idx])
end

@inline _loadP_bc(u, Nx, Ny, Nz, i, j, k, cx, cy, cz, γ::T, smallr::T, pfl::T) where {T} =
    cons2prim(_loadU_bc(u, Nx, Ny, Nz, i, j, k, cx, cy, cz), γ, smallr, pfl)

# All-periodic fast path: pure `mod` index, NO flip multiplies and NO BC branches — the
# common case, and what closes the cube's gap to the bare-periodic prototype.
@inline function _loadP_periodic(u::NTuple{9,A}, Nx::Int, Ny::Int, Nz::Int, i::Int, j::Int, k::Int,
                                 γ::T, smallr::T, pfl::T) where {T,A}
    ii=mod(i-1,Nx)+1; jj=mod(j-1,Ny)+1; kk=mod(k-1,Nz)+1; idx=((kk-1)*Ny+(jj-1))*Nx+ii
    @inbounds cons2prim((u[1][idx],u[2][idx],u[3][idx],u[4][idx],u[5][idx],u[6][idx],u[7][idx],u[8][idx],u[9][idx]),γ,smallr,pfl)
end
# Compile-time selector: `Val{true}` ⇒ periodic fast path, `Val{false}` ⇒ general BCs.
@inline _loadP_sel(::Val{true},  u, Nx, Ny, Nz, i, j, k, cx, cy, cz, γ::T, smallr::T, pfl::T) where {T} =
    _loadP_periodic(u, Nx, Ny, Nz, i, j, k, γ, smallr, pfl)
@inline _loadP_sel(::Val{false}, u, Nx, Ny, Nz, i, j, k, cx, cy, cz, γ::T, smallr::T, pfl::T) where {T} =
    _loadP_bc(u, Nx, Ny, Nz, i, j, k, cx, cy, cz, γ, smallr, pfl)
