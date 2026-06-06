# Shared verification harness for the PoissonKernels multigrid tests — the same
# three-layer "ladder" used by PPMKernels, expressed once and reused.
#
#   Layer A  port faithfulness : KA kernel on CPU in Float64  vs Fortran oracle (f64)
#   Layer B  CPU↔GPU parity     : same kernel, CPU f32        vs Metal f32
#   Layer C  f32 accuracy floor : Metal f32                   vs Fortran oracle (f64)
#
# The Fortran oracle is the LIVE Enzo multigrid kernel called through EnzoLib's
# grid dylib (mg_*_ref); the b8 library runs them in double, so Layer A is a
# genuine f64-vs-f64 bit-tight comparison.

using EnzoFixtures: Tolerance, compare, CompareResult
using Test

const RTOL_A = Tolerance(rtol = 1e-12, atol = 1e-14)   # f64 vs Fortran (bit-tight)
const RTOL_B = Tolerance(rtol = 1e-6,  atol = 1e-7)    # f32 CPU vs f32 Metal
const RTOL_C = Tolerance(rtol = 1e-5,  atol = 1e-6)    # f32 Metal vs f64 Fortran

"True once `using Metal` has registered a functional `:metal` backend."
metal_ready() = PoissonKernels.has_backend(:metal)

# `compare` wants flat sequences; flatten any array shape to a Vector for diffing.
_flat(a) = vec(collect(a))

"""
    @check label got expected tol

Assert `got ≈ expected` under tolerance `tol`, logging the worst deviation on
failure (maxabs / maxrel / worst-index). `label` is a human string.
"""
macro check(label, got, expected, tol)
    quote
        local r = compare(_flat($(esc(got))), _flat($(esc(expected))), $(esc(tol)))
        @test Bool(r)
        Bool(r) || @info "MISMATCH" field = $(esc(label)) maxabs = r.maxabs maxrel = r.maxrel worst = r.worst
        r
    end
end

"Layer A — CPU-Float64 kernel output `got` vs the Float64 Fortran reference `ref`."
layerA!(label, got, ref) =
    @check(string(label, " [A:cpu-f64 vs Fortran]"), got, ref, RTOL_A)

"Layer C — Metal-Float32 vs the Float64 Fortran reference (looser RTOL_C). Skips with no GPU."
function layerC!(label, run, ref)
    metal_ready() || return nothing
    @check(string(label, " [C:metal-f32 vs Fortran]"), run(:metal, Float32), ref, RTOL_C)
end

"Layer B — CPU-f32 ≡ Metal-f32 for the same kernel `run(name, T)`. Skips with no GPU."
function layerB!(label, run)
    metal_ready() || return nothing
    cpu = _flat(run(:cpu,   Float32))
    gpu = _flat(run(:metal, Float32))
    @check(string(label, " [B:cpu≡metal f32]"), gpu, cpu, RTOL_B)
end

# ── deterministic test fields (no RNG: reproducible across runs/processes) ─────
"""
    poisson_field(dims; amp=1.0, phase=0.0) -> Array{Float64,3}

A smooth, deterministic field on a `dims` grid: a product of sines plus a
higher-frequency cosine ripple. Used as a stand-in potential / RHS / source.
"""
function poisson_field(dims::NTuple{3,Int}; amp::Float64 = 1.0, phase::Float64 = 0.0)
    d1, d2, d3 = dims
    a = Array{Float64,3}(undef, dims)
    @inbounds for k in 1:d3, j in 1:d2, i in 1:d1
        x = (i - 1) / max(d1 - 1, 1)
        y = (j - 1) / max(d2 - 1, 1)
        z = (k - 1) / max(d3 - 1, 1)
        a[i, j, k] = amp * (sinpi(2x) * sinpi(2y) * sinpi(2z) +
                            0.1 * cospi(4x + phase) * sinpi(2y) +
                            0.05 * sinpi(2z) * cospi(2y))
    end
    return a
end
