# Shared verification harness for the PPM component tests — the "ladder" from the
# plan, expressed once and reused by every component/sweep test.
#
#   Layer A  port faithfulness : KA kernel on CPU in Float64  vs Fortran fixture (f64)
#   Layer B  CPU↔GPU parity     : same kernel, CPU f32        vs Metal f32
#   Layer C  f32 accuracy floor : Metal f32                   vs Fortran fixture (f64)
#
# All three diff through EnzoFixtures' language-neutral Tolerance/compare policy,
# the same gate the legacy `test_parity_ppm.jl` uses.

using EnzoFixtures: Tolerance, compare, CompareResult
using Test

# Tolerances (see plan §"Verification ladder").
const RTOL_A = Tolerance(rtol = 1e-12, atol = 1e-14)   # f64 vs Fortran (bit-tight)
const RTOL_B = Tolerance(rtol = 1e-6,  atol = 1e-7)    # f32 CPU vs f32 Metal
const RTOL_C = Tolerance(rtol = 1e-5,  atol = 1e-6)    # f32 Metal vs f64 Fortran

"True once `using Metal` has registered a functional `:metal` backend."
metal_ready() = PPMKernels.has_backend(:metal)

# `compare` wants flat sequences; flatten any array shape to a Vector for diffing.
_flat(a) = vec(collect(a))

"""
    @check label got expected tol

Assert `got ≈ expected` under tolerance `tol`, logging the worst deviation on
failure (maxabs / maxrel / worst-index — the `CompareResult` fields). `label` is
a human string naming the field under test.
"""
macro check(label, got, expected, tol)
    quote
        local r = compare(_flat($(esc(got))), _flat($(esc(expected))), $(esc(tol)))
        @test Bool(r)
        Bool(r) || @info "MISMATCH" field = $(esc(label)) maxabs = r.maxabs maxrel = r.maxrel worst = r.worst
        r
    end
end

"""
    layerA!(label, got, ref)

Layer A — port faithfulness. Assert the CPU-`Float64` kernel output `got` matches
the `Float64` Fortran reference `ref` under `RTOL_A` (bit-tight up to FP
reassociation). Both are arrays (any shape; flattened for the diff).
"""
layerA!(label, got, ref) =
    @check(string(label, " [A:cpu-f64 vs Fortran]"), got, ref, RTOL_A)

"""
    layerC!(label, run, ref)

Layer C — f32 accuracy floor. When Metal is present, run `run(:metal, Float32)`
and assert it matches the `Float64` Fortran reference `ref` under the looser
`RTOL_C`. Skips cleanly (returns `nothing`) with no GPU.
"""
function layerC!(label, run, ref)
    metal_ready() || return nothing
    @check(string(label, " [C:metal-f32 vs Fortran]"), run(:metal, Float32), ref, RTOL_C)
end

"""
    layerB!(label, run)

Run `run(:cpu, Float32)` and, when Metal is present, `run(:metal, Float32)`, and
assert the two agree under `RTOL_B`. `run(name, T)` must return a host `Vector`
(or array) of the kernel output at backend `name` in precision `T`. Skips cleanly
(returns `nothing`) when no Metal device is available so CPU-only CI stays green.
"""
function layerB!(label, run)
    metal_ready() || return nothing
    cpu  = _flat(run(:cpu,   Float32))
    gpu  = _flat(run(:metal, Float32))
    @check(string(label, " [B:cpu≡metal f32]"), gpu, cpu, RTOL_B)
end
