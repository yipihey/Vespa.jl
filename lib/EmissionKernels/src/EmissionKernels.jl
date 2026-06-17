"""
    EmissionKernels

Per-channel / per-line radiative emissivity for the v2026 primordial+metal network —
the FOUNDATION layer beneath `ChemistryKernels`. One source of truth for two uses:

  • **Cooling**: `cooling_rate_total` sums every radiative channel (H/He collisional
    excitation & ionisation, recombination, bremsstrahlung, H₂, HD, Compton, metal
    fine-structure) into the total volumetric cooling the chemistry network needs.
    `ChemistryKernels.cooling_edot = -cooling_rate_total` (bit-identical to the legacy).
  • **Synthetic emission**: per-channel (`emiss_*`) and per-line (`lya_emissivity`,
    `metal_line_emissivities`) volumetric emissivities for mock spectra / line maps.

Each coefficient is a pure, `@inline`, precision-generic (`R = typeof(T)`),
allocation-free scalar function — runs on CPU (f64/f32) and Metal/CUDA (weak-dep
extensions). Physics: Abel/Anninos et al. (1997) primordial cooling; Glover & Jappsen
(2007) metal fine-structure; Galli-Palla (2008) H₂.
"""
module EmissionKernels

using KernelAbstractions
const KA = KernelAbstractions

export backend, has_backend, device_zeros, to_device, to_host
# radiative coefficients (re-exported by ChemistryKernels)
export ceHI, ceHeI, ceHeII, ciHI, ciHeI, ciHeII, ciHeIS,
       reHII, reHeII1, reHeII2, reHeIII, brem,
       GAHI, GAH2, GAHe, GAHp, GAel, H2LTE, HDlte, HDlow,
       comp1_cmb, comp2_cmb,
       MetalAbundances, metal_abund, metal_cooling_rate

# ── backend registry (own copy; each KA package owns its own — see PPMKernels etc.) ──
const _BACKENDS = Dict{Symbol,Any}(:cpu => CPU())

"Register a KernelAbstractions backend under `name` (used by the Metal/CUDA extensions)."
register_backend!(name::Symbol, be) = (_BACKENDS[name] = be)

"True when backend `name` is available (`:metal`/`:cuda` need the GPU package loaded)."
has_backend(name::Symbol) = haskey(_BACKENDS, name)

"""
    backend(name::Symbol = :cpu)

The KernelAbstractions backend registered under `name`. `:cpu` always available;
`:metal`/`:cuda` require `using Metal`/`using CUDA` (loads the package extension).
"""
function backend(name::Symbol = :cpu)
    return get(_BACKENDS, name) do
        error("Emission backend :$name is not available. " *
              (name === :metal ? "Run `using Metal` first (Apple Silicon only)." :
               name === :cuda  ? "Run `using CUDA` first (NVIDIA only)." :
               "Known backends: $(collect(keys(_BACKENDS)))."))
    end
end

device_zeros(::CPU, ::Type{T}, dims::Dims) where {T} = zeros(T, dims)

function to_device(be, a::AbstractArray, ::Type{T} = eltype(a)) where {T}
    d = device_zeros(be, T, size(a))
    copyto!(d, convert(Array{T}, a))
    return d
end

function to_host(a::AbstractArray)
    KA.synchronize(KA.get_backend(a))
    return Array(a)
end

# ── radiative-channel physics (each Wave-1-style: pure @inline coefficient kernels) ──
include("constants.jl")
include("kernelgen.jl")          # @scalarkernel (re-pointed at EmissionKernels.backend)
include("cooling_atomic.jl")     # ceHI…brem (+ _LOG_DHUGE, _k1/_k3/_k5_inline)
include("cooling_h2.jl")         # GAHI…H2LTE
include("cooling_hd.jl")         # HDlte, HDlow
include("cooling_compton.jl")    # comp1_cmb, comp2_cmb, COMPA
include("cooling_metal.jl")      # MetalAbundances, metal_*; the _cool_*/_fion_* internals
include("emission.jl")           # per-channel/per-line API + cooling_rate_total

end # module EmissionKernels
