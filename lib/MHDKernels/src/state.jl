# ── MHDState: 9 SoA conserved fields + a ping-pong scratch set + run parameters ─
# Fields are flat column-major length-Nx·Ny·Nz device vectors; cell (i,j,k) lives at
#   idx = (k-1)*Nx*Ny + (j-1)*Nx + i   (1-based).
# Boundaries are PERIODIC for now (the smooth-wave convergence gate needs nothing
# else); `bc` is carried for the outflow/reflecting extension. The conserved order is
# U=(ρ,ρvx,ρvy,ρvz,E,Bx,By,Bz,ψ); ψ is the GLM cleaning potential (damped, not
# conserved). γ is the adiabatic index; the floors/Riemann-switch params feed
# `riemann`. `be` is the KA backend the fields live on.

export MHDState, allocate_state, nvars, ncells, linidx, fields_to_host

const NVAR = 9

mutable struct MHDState{T,A,B}
    U::NTuple{NVAR,A}        # conserved
    scratch::NTuple{NVAR,A}  # ping-pong target for the integrator
    dims::NTuple{3,Int}      # (Nx, Ny, Nz)
    dx::T
    γ::T
    smallr::T
    pfl::T
    llf_dmin::T
    llf_pmin::T
    use_hlld::Bool
    bcs::NTuple{3,Symbol}    # per-axis: :periodic / :outflow / :reflecting
    recon::Symbol            # :plm (MonCen) or :ppm (local CW84 parabola)
    be::B
end

nvars(::MHDState) = NVAR
ncells(s::MHDState) = prod(s.dims)
@inline linidx(Nx::Int, Ny::Int, i::Int, j::Int, k::Int) = ((k-1)*Ny + (j-1))*Nx + i

"`bc_codes(s)` → the per-axis BC integer codes (cx,cy,cz) for the kernels."
bc_codes(s::MHDState) = (bc_code(s.bcs[1]), bc_code(s.bcs[2]), bc_code(s.bcs[3]))

"`recon_code_of(s)` → the reconstruction integer code (0=PLM, 1=PPM)."
recon_code_of(s::MHDState) = recon_code(s.recon)

"`all_periodic(s)` → true when every axis is periodic (enables the cube's fast load path)."
all_periodic(s::MHDState) = all(==(:periodic), s.bcs)

"""
    allocate_state(be, T, dims; dx, gamma=5/3, smallr=1e-6, pfl=1e-7,
                   llf_dmin=1e-4, llf_pmin=0, use_hlld=true,
                   bc=:periodic, bcs=(bc,bc,bc))

Allocate an `MHDState` of element type `T` and shape `dims=(Nx,Ny,Nz)` on backend
`be` (zero-filled; fill it with a problem initialiser). `dx` is the uniform cell
width (domain assumed `[0,L]^3` with L = Nx·dx). `bc` sets all three axes; `bcs`
overrides per-axis (e.g. `bcs=(:outflow,:periodic,:periodic)` for a 1-D shock tube).
"""
function allocate_state(be, ::Type{T}, dims::NTuple{3,Int};
                        dx::Real, gamma::Real = 5//3,
                        smallr::Real = 1e-6, pfl::Real = 1e-7,
                        llf_dmin::Real = 1e-4, llf_pmin::Real = 0,
                        use_hlld::Bool = true, bc::Symbol = :periodic,
                        bcs::NTuple{3,Symbol} = (bc, bc, bc), recon::Symbol = :plm) where {T}
    nc = prod(dims)
    U  = ntuple(_ -> device_zeros(be, T, (nc,)), NVAR)
    sc = ntuple(_ -> device_zeros(be, T, (nc,)), NVAR)
    MHDState{T,eltype(U),typeof(be)}(U, sc, dims, T(dx), T(gamma), T(smallr), T(pfl),
                                     T(llf_dmin), T(llf_pmin), use_hlld, bcs, recon, be)
end

"`fields_to_host(s)` → an `NTuple{9,Array}` host copy of the conserved fields."
fields_to_host(s::MHDState) = ntuple(v -> to_host(s.U[v]), NVAR)
