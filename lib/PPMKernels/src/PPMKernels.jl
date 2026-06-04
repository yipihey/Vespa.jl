"""
    PPMKernels

A KernelAbstractions.jl port of Enzo's Piecewise-Parabolic-Method (PPM)
DirectEuler hydro solver — the `inteuler → twoshock → flux_twoshock → euler`
chain, with the dual-energy formalism, gravity, and colour advection — written
**once** and run on both the CPU (the parity oracle) and the Metal GPU.

Design contract:

  * **One source, two devices.** Every compute kernel is a precision-generic
    `@kernel` parameterised on the element type `T`. The CPU backend runs it in
    `Float64` (to certify the algorithm *is* Enzo's, bit-tight against the Fortran
    golden fixtures) and in `Float32`; the Metal backend runs it only in `Float32`
    (Apple GPUs have no `Float64`). f32 CPU↔Metal agreement is the parity gate.

  * **Backend by name.** `backend(:cpu)` always works; `backend(:metal)` resolves
    only after `using Metal` has loaded the package extension (Apple Silicon).
    Allocation/host-transfer go through [`device_zeros`](@ref) / [`to_device`](@ref)
    / [`to_host`](@ref), which the Metal extension specialises — the kernels and
    tests never name a concrete array type.
"""
module PPMKernels

using KernelAbstractions
const KA = KernelAbstractions

export backend, has_backend, device_zeros, to_device, to_host

# ── backend registry ─────────────────────────────────────────────────────────
# `:cpu` is always present; the Metal extension registers `:metal` on load.
const _BACKENDS = Dict{Symbol,Any}(:cpu => CPU())

"Register a KernelAbstractions backend under `name` (used by the Metal extension)."
register_backend!(name::Symbol, be) = (_BACKENDS[name] = be)

"True when backend `name` is available (`:metal` needs `using Metal` first)."
has_backend(name::Symbol) = haskey(_BACKENDS, name)

"""
    backend(name::Symbol = :cpu)

The KernelAbstractions backend registered under `name`. `:cpu` is always
available; `:metal` requires `using Metal` (Apple Silicon) to have loaded the
`PPMKernelsMetalExt` extension.
"""
function backend(name::Symbol = :cpu)
    return get(_BACKENDS, name) do
        error("PPM backend :$name is not available. " *
              (name === :metal ? "Run `using Metal` first (Apple Silicon only)." :
               "Known backends: $(collect(keys(_BACKENDS)))."))
    end
end

# ── device array helpers (specialised by the Metal extension) ────────────────
"""
    device_zeros(be, T, dims::Dims) -> AbstractArray{T}

A zero-filled array of element type `T` and shape `dims` living on backend `be`.
The CPU default is `zeros(T, dims)`; the Metal extension returns an `MtlArray`.
"""
device_zeros(::CPU, ::Type{T}, dims::Dims) where {T} = zeros(T, dims)

"""
    to_device(be, a, T = eltype(a)) -> AbstractArray{T}

Copy host array `a` onto backend `be`, converting to element type `T`. Used by
tests to stage fixture inputs (always `Float64` on disk) at the working precision.
"""
function to_device(be, a::AbstractArray, ::Type{T} = eltype(a)) where {T}
    d = device_zeros(be, T, size(a))
    copyto!(d, convert(Array{T}, a))
    return d
end

"`to_host(a)` — a plain host `Array` copy of a device array (no-op-ish on CPU)."
to_host(a::AbstractArray) = Array(a)

# ── compute kernels (added as each component is ported + certified) ──────────
include("eos.jl")              # pgas2d / pgas2d_dual            (Phase 2.1) ✓
include("calcdiss.jl")         # diffusion + flattening          (Phase 2.2) ✓
include("intvar.jl")           # per-variable PPM reconstruction (Phase 2.3a) ✓
include("inteuler.jl")         # PPM parabolic reconstruction     (Phase 2.3) ✓
include("twoshock.jl")         # two-shock Riemann solver         (Phase 2.4) ✓
include("flux_twoshock.jl")    # physical fluxes                  (Phase 2.5) ✓
include("euler.jl")            # conservative update              (Phase 2.6) ✓
include("sweep.jl")            # composed 1-D directional sweep   (Phase 3) ✓
include("ppm_grid.jl")         # 3-D Strang-split sweeps          (Phase 4) ✓

end # module
