"""
    ChemistryKernels

A table-free, KernelAbstractions.jl implementation of the primordial +
deuterium chemistry/cooling network of **Abel, Anninos, Zhang & Norman (1997,
New Astronomy 2, 181)** and **Anninos, Zhang, Abel & Norman (1997, New Astronomy
2, 209)** — the original Enzo primordial chemistry (the same physics later
packaged as the `grackle` library).  Reduced model: advect HII, H2I, HDI;
H⁻/H₂⁺/D⁺ in algebraic equilibrium; helium in ionisation equilibrium (or advected
He⁺); nₑ from charge conservation; primordial only, no metals, no UV background
(unless photo-ionisation rates are supplied).

Also implements density-dependent Lyα-mixing recombination (`solve_chem_mixing!`)
for early-Universe / PMF science, where the Peebles C-factor escape rate uses a
host-supplied smoothed neutral density instead of the cell-local value.

Design contract (mirrors `PPMKernels`/`PoissonKernels`):

  * **One source, two devices.** Every compute kernel is a precision-generic
    `@kernel` parameterised on `T = eltype(output)`. The CPU backend runs f64 and
    f32; Metal runs f32-only. f32 CPU↔Metal agreement is the parity gate. Rate and
    cooling coefficients are checked against the published Abel/Anninos et al.
    (1997) analytic fits (and, for recombination, against HyRec-2).

  * **No tables.** Every rate/cooling coefficient is evaluated DIRECTLY from its
    analytic fit (Abel/Anninos et al. 1997 and the references therein), not
    interpolated from a precomputed log-T grid.

  * **f32-safe representation.** State is carried as physical abundances relative
    to the hydrogen number density (`x_i = n_i/n_H`, dimensionless and O(1) for
    the dominant species) with reaction frequencies `k·n_H` (s⁻¹), keeping
    products in f32 range without coupling to any host code-unit system.

  * **AD-friendly.** The math core (rates, cooling, network update) is written as
    pure, allocation-free functions of `(state, T)`; mutation is confined to the
    outer driver, so Phase-3 differentiability (Enzyme) is a later add-on.

  * **Backend by name.** `backend(:cpu)` always works; `backend(:metal)` resolves
    after `using Metal`. Allocation/host-transfer go through `device_zeros` /
    `to_device` / `to_host`, specialised by the Metal extension.
"""
module ChemistryKernels

using KernelAbstractions
const KA = KernelAbstractions

export backend, has_backend, device_zeros, to_device, to_host

# ── backend registry ─────────────────────────────────────────────────────────
const _BACKENDS = Dict{Symbol,Any}(:cpu => CPU())

"Register a KernelAbstractions backend under `name` (used by the Metal extension)."
register_backend!(name::Symbol, be) = (_BACKENDS[name] = be)

"True when backend `name` is available (`:metal` needs `using Metal` first)."
has_backend(name::Symbol) = haskey(_BACKENDS, name)

"""
    backend(name::Symbol = :cpu)

The KernelAbstractions backend registered under `name`. `:cpu` is always
available; `:metal` requires `using Metal` (Apple Silicon) to have loaded the
`ChemistryKernelsMetalExt` extension.
"""
function backend(name::Symbol = :cpu)
    return get(_BACKENDS, name) do
        error("Chemistry backend :$name is not available. " *
              (name === :metal ? "Run `using Metal` first (Apple Silicon only)." :
               "Known backends: $(collect(keys(_BACKENDS)))."))
    end
end

# ── device array helpers (specialised by the Metal extension) ────────────────
"A zero-filled array of element type `T` and shape `dims` on backend `be`."
device_zeros(::CPU, ::Type{T}, dims::Dims) where {T} = zeros(T, dims)

"Copy host array `a` onto backend `be`, converting to element type `T`."
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

# ── component sources (added as each Wave lands) ─────────────────────────────
include("constants.jl")
include("kernelgen.jl")
include("uv_background.jl")
include("representation.jl")
# Wave 1 — table-free rate + cooling formula kernels.
include("rates_atomic.jl")
include("rates_h2.jl")
include("rates_deuterium.jl")
include("rates_cmb.jl")
include("cooling_atomic.jl")
include("cooling_h2.jl")
include("cooling_hd.jl")
include("cooling_compton.jl")
# Wave 2 — local composed: temperature (mmw/H2-γ) + algebraic equilibrium species.
include("temperature.jl")
include("equilibrium.jl")
# Wave 3 — assemblers: cooling rate + one backward-Euler network sweep.
include("edot.jl")
include("network_step.jl")
# Wave 4 — driver: Peebles recombination, the sub-cycle, and the host boundary.
include("recombination.jl")
include("subcycle.jl")
include("solve.jl")
# Wave 5 — Lyα-mixing recombination for early-Universe / PMF science.
include("tables.jl")
include("recombination_clumping.jl")

end # module ChemistryKernels
