"""
    BlockAMRMetalExt

Package extension lighting up the Metal (Apple GPU) backend for `BlockAMR`:
registers `:metal` and specialises `device_zeros` onto `MtlArray`. Loaded
automatically when `Metal` is present in the environment.

Metal is Float32-native but Apple GPUs support Float16 storage/arithmetic, so
BlockAMR's f16 fields + u16 species map over unchanged; positions/times stay
Float64 on the host (never in a kernel) exactly as on CUDA.

Not enabled by default here: the tiled single-pass CTU kernel (`_ctu_step_k!`).
It requests ~42 KB of `@localmem` (threadgroup memory) at NS=2, which exceeds
the 32 KB threadgroup-memory limit of current Apple GPUs. Until the target
device is confirmed to allow the launch, `ctu_level!` runs the per-cell CTU
oracle on Metal (correct on every backend, ~2× slower). To flip it on after
validating `device.maxThreadgroupMemoryLength ≥ 43_008` on the M-series GPU:

    using BlockAMR, Metal
    BlockAMR.enable_tiling!(:metal)          # then ctu_level! auto-tiles

or force per-call for a single test: `ctu_level!(hier, l, λ; tiled=true)`.
"""
module BlockAMRMetalExt

using BlockAMR
using Metal

function __init__()
    # Only register a usable GPU if the system actually has one (CI on Apple
    # hardware without a functional Metal device should degrade gracefully).
    if Metal.functional()
        BlockAMR.register_backend!(:metal, Metal.MetalBackend())
        # NOTE: tiling intentionally NOT enabled — see the module docstring.
        # After validating threadgroup memory on the target GPU, uncomment:
        # BlockAMR.enable_tiling!(:metal)
    end
end

# `Metal.zeros(T, dims)` allocates a zero-filled MtlArray on the default device.
BlockAMR.device_zeros(::Metal.MetalBackend, ::Type{T}, dims::Dims) where {T} =
    Metal.zeros(T, dims)

end # module
