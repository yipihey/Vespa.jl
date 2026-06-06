"""
    PoissonKernels

A KernelAbstractions.jl port of Enzo's **multigrid Poisson solver** — the V-cycle
behind `grid::SolveForPotential` (`MultigridSolver.C` + the `mg_relax` /
`mg_calc_defect` / `mg_restrict` / `mg_prolong` / `comp_accel` Fortran kernels) —
written **once** and run on both the CPU (the parity oracle) and the Metal GPU.

Design contract (mirrors `PPMKernels`):

  * **One source, two devices.** Every compute kernel is a precision-generic
    `@kernel` parameterised on the element type `T = eltype(output)`. The CPU
    backend runs it in `Float64` (to certify the port *is* Enzo's, bit-tight
    against the live Fortran multigrid kernels) and in `Float32`; the Metal
    backend runs it only in `Float32`. f32 CPU↔Metal agreement is the parity gate.

  * **Column-major, 3-D.** Fields are 3-D `AbstractArray{T,3}` with `(dim1, dim2,
    dim3)` storage — identical to Enzo's Fortran column-major layout — so the
    stencils read `A[i,j,k]` directly and the bridge oracles take the array
    pointer with no transpose. Indices are 1-based, matching Fortran.

  * **Backend by name.** `backend(:cpu)` always works; `backend(:metal)` resolves
    only after `using Metal` has loaded the package extension (Apple Silicon).
    Allocation/host-transfer go through [`device_zeros`](@ref) / [`to_device`](@ref)
    / [`to_host`](@ref), which the Metal extension specialises — the kernels and
    tests never name a concrete array type.

The numerics target Enzo's **default build** (2nd-order 7-point Laplacian; the
`#else` branch of the order-switched Fortran kernels) on the **`b8`** library,
where the multigrid kernels run in `double` — so Layer A is a genuine f64-vs-f64
bit-tight comparison.
"""
module PoissonKernels

using KernelAbstractions
const KA = KernelAbstractions

export backend, has_backend, device_zeros, to_device, to_host
export mg_relax!, mg_calc_defect!, mg_restrict!, mg_prolong!, comp_accel!
export mg_dims_schedule, vcycle_solve!, fft_poisson_root!, fft_poisson_root_gpu!
export vcycle_batched!, comp_accel_batched!, mg_relax_batched!

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
`PoissonKernelsMetalExt` extension.
"""
function backend(name::Symbol = :cpu)
    return get(_BACKENDS, name) do
        error("Poisson backend :$name is not available. " *
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

# Host reads are THE synchronization boundary: `to_host` synchronizes the backend
# queue once before copying device memory to the host.
"`to_host(a)` — a plain host `Array` copy of a device array; synchronizes first."
function to_host(a::AbstractArray)
    KA.synchronize(KA.get_backend(a))
    return Array(a)
end

# ── compute kernels (one per Enzo Fortran multigrid routine, 2nd-order) ───────
include("relax.jl")        # mg_relax      — red/black Gauss-Seidel  (7-point)
include("defect.jl")       # mg_calc_defect — residual + L2 norm
include("restrict.jl")     # mg_restrict   — fine→coarse (quadratic)
include("prolong.jl")      # mg_prolong    — coarse→fine (trilinear)
include("comp_accel.jl")   # comp_accel    — g = -∇φ finite-difference gradient
include("vcycle.jl")       # MultigridSolver V-cycle host driver
include("fft_poisson.jl")  # root-grid FFT Poisson solve (CPU-host FFTW + Green's -1/k²)
include("gpu_fft.jl")      # GPU-resident radix-2 FFT + fft_poisson_root_gpu! (device root solve)
include("mg_batched.jl")   # batched multigrid: NB same-size subgrids per kernel launch

end # module
