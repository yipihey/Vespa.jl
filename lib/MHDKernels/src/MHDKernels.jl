"""
    MHDKernels

A KernelAbstractions.jl ideal-MHD solver with **Dedner mixed-GLM divergence
cleaning** — MUSCL-Hancock + PLM(MonCen) reconstruction + HLLD (LLF fallback),
written **once** and run on the CPU (the f64 convergence/conservation oracle) and
on CUDA / Metal GPUs (f32).

Design contract (mirrors `PPMKernels`):

  * **One source, two+ devices, f32-first.** Every compute kernel is a
    precision-generic `@kernel` parameterised on the element type `T`, but the
    DEFAULT precision is `Float32` on EVERY backend — the CPU runs f32 too, so
    CPU↔GPU is an apples-to-apples comparison. `Float64` is available (pass it to
    `allocate_state`) only when a physical reason needs the dynamic range. The
    convergence gate runs in f32 (finite-amplitude exact Alfvén wave keeps the
    discretisation error above the f32 round-off floor).
  * **Backend by name.** `backend(:cpu)` always works; `backend(:cuda)` /
    `backend(:metal)` resolve after `using CUDA` / `using Metal` loads the matching
    package extension. Allocation/transfer go through [`device_zeros`](@ref) /
    [`to_device`](@ref) / [`to_host`](@ref).
  * **Conserved order** `U = (ρ, ρvx, ρvy, ρvz, E, Bx, By, Bz, ψ)`;
    **primitive order** `W = (ρ, vx, vy, vz, p, Bx, By, Bz, ψ)`.

Two integrators, cross-checked: a portable per-cell **reference** kernel
(`step_ref!`, runs on every backend incl. CPU-f64) and a shared-memory **cube**
kernel (`step_cube!`, GPU throughput path). Both implement the identical scheme.
"""
module MHDKernels

using KernelAbstractions
const KA = KernelAbstractions

export backend, has_backend, device_zeros, to_device, to_host

# ── backend registry ─────────────────────────────────────────────────────────
const _BACKENDS = Dict{Symbol,Any}(:cpu => CPU())
register_backend!(name::Symbol, be) = (_BACKENDS[name] = be)
has_backend(name::Symbol) = haskey(_BACKENDS, name)

"""
    backend(name::Symbol = :cpu)

The KernelAbstractions backend registered under `name`. `:cpu` is always
available; `:cuda`/`:metal` require `using CUDA`/`using Metal` first.
"""
function backend(name::Symbol = :cpu)
    return get(_BACKENDS, name) do
        error("MHD backend :$name is not available. " *
              (name === :cuda ? "Run `using CUDA` first." :
               name === :metal ? "Run `using Metal` first (Apple Silicon only)." :
               "Known backends: $(collect(keys(_BACKENDS)))."))
    end
end

# ── device array helpers (specialised by the CUDA/Metal extensions) ──────────
device_zeros(::CPU, ::Type{T}, dims::Dims) where {T} = zeros(T, dims)

function to_device(be, a::AbstractArray, ::Type{T} = eltype(a)) where {T}
    d = device_zeros(be, T, size(a))
    copyto!(d, convert(Array{T}, a))
    return d
end

"`to_host(a)` — a plain host `Array` copy of a device array; synchronizes first."
function to_host(a::AbstractArray)
    KA.synchronize(KA.get_backend(a))
    return Array(a)
end

include("glm_physics.jl")     # precision-generic @fastmath GLM-MHD device functions
include("boundary.jl")        # per-axis BCs (periodic/outflow/reflecting), ghost-free
include("state.jl")           # MHDState (9 SoA fields) + cons/prim host helpers
include("integrator_ref.jl")  # portable per-cell reference integrator (CPU + GPU)
include("integrator_cube.jl") # fused shared-memory cube integrator (GPU throughput)
include("timestep.jl")        # CFL Δt reduction
include("diagnostics.jl")     # max|∇·B|, conserved totals
include("problems.jl")        # initial conditions (smooth waves, Brio-Wu, Orszag-Tang)
include("driver.jl")          # step! / evolve! (ch + ψ-damping schedule)

end # module
