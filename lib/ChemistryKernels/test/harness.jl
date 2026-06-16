# Verification "ladder" for ChemistryKernels — self-contained (the reference is
# grackle's analytic functions via ChemOracle, not a fixture file).
#
#   Layer A  port faithfulness  : KA kernel on CPU f64  vs grackle fn (units=1, f64)
#   Layer B  CPU↔Metal parity   : same kernel, CPU f32  vs Metal f32
#   Layer C  f32 accuracy floor : Metal f32             vs grackle fn (f64)
#   Layer D  CPU↔CUDA parity    : same kernel, CPU f32  vs CUDA f32
#   Layer E  f32 accuracy floor : CUDA f32              vs grackle fn (f64)
#
# Layers B/C skip when Metal is absent; layers D/E skip when CUDA is absent.

using Test

# f64 vs grackle: "bit-tight up to FP reassociation". The exp(degree-8 logT
# polynomial) fits reassociate at ~1e-12 (Julia `^` vs C pow/Horner) and a few
# (k11) hit ~2e-11 from C's fused-multiply-add — so the floor is 5e-11, still 8+
# orders below any transcription typo (a wrong digit is O(1e-2+)).
const RTOL_A = (rtol = 5e-11, atol = 1e-30)
# f32 device-parity / accuracy: the cancellation-heavy log-T polynomials evaluate
# to ~1e-4 relative across CPU-libm vs Metal-GPU intrinsics — that IS the honest
# f32 floor for this kind of fit (not a bug). The Wave-5 one-zone f32-vs-f64
# trajectory is the real f32-adequacy gate.
const RTOL_B = (rtol = 1e-4, atol = 1e-30)    # f32 CPU vs f32 Metal
const RTOL_C = (rtol = 1e-4, atol = 1e-30)    # f32 Metal vs f64 grackle
# The Abel-1997 ionization/recombination polynomials cancel catastrophically in
# f32 ONLY at T > ~1e8 K — far outside the primordial-chemistry regime (the gas
# never gets there). The f32 layers are checked over the operating range; Layer A
# (f64) still spans the full grid.
const F32_TMAX = 1.0e8

metal_ready() = ChemistryKernels.has_backend(:metal)
cuda_ready()  = ChemistryKernels.has_backend(:cuda)
_flat(a) = vec(collect(a))

"Worst relative/absolute deviation between two flat sequences (matched scale)."
function _agree(got, ref, tol)
    g = _flat(got); r = _flat(ref)
    @assert length(g) == length(r) "length mismatch $(length(g)) vs $(length(r))"
    maxrel = 0.0; maxabs = 0.0; worst = 0
    @inbounds for i in eachindex(g)
        a = abs(g[i] - r[i]); maxabs = max(maxabs, a)
        denom = max(abs(r[i]), tol.atol)
        rel = a / denom
        if rel > maxrel; maxrel = rel; worst = i; end
    end
    ok = maxrel <= tol.rtol || maxabs <= tol.atol
    return (ok = ok, maxrel = maxrel, maxabs = maxabs, worst = worst)
end

macro check(label, got, expected, tol)
    quote
        local r = _agree($(esc(got)), $(esc(expected)), $(esc(tol)))
        @test r.ok
        r.ok || @info "MISMATCH" field=$(esc(label)) maxrel=r.maxrel maxabs=r.maxabs worst=r.worst
        r
    end
end

layerA!(label, got, ref) = @check(string(label, " [A:cpu-f64 vs grackle]"), got, ref, RTOL_A)

function layerC!(label, run, ref)
    metal_ready() || return nothing
    @check(string(label, " [C:metal-f32 vs grackle]"), run(:metal, Float32), ref, RTOL_C)
end

function layerB!(label, run)
    metal_ready() || return nothing
    cpu = run(:cpu, Float32); gpu = run(:metal, Float32)
    @check(string(label, " [B:cpu≡metal f32]"), gpu, cpu, RTOL_B)
end

# CUDA parity layers (D: cpu≡cuda f32, E: cuda-f32 vs grackle) — skip when CUDA absent.
function layerE!(label, run, ref)
    cuda_ready() || return nothing
    @check(string(label, " [E:cuda-f32 vs grackle]"), run(:cuda, Float32), ref, RTOL_C)
end

function layerD!(label, run)
    cuda_ready() || return nothing
    cpu = run(:cpu, Float32); gpu = run(:cuda, Float32)
    @check(string(label, " [D:cpu≡cuda f32]"), gpu, cpu, RTOL_B)
end

"""
    check_scalar_kernel(label, jl_fn, oracle_vals, Ts; f32rtol = RTOL_B.rtol)

Run the A/B/C ladder for a scalar formula. Layer A (f64 vs grackle, 5e-11) is the
hard CORRECTNESS gate — it catches any transcription typo (O(1e-2)) regardless of
precision. Layers B/C are the f32 device-parity / accuracy floor, restricted to
the operating range T ≤ F32_TMAX. Pass a larger `f32rtol` (~3e-3) for the
cancellation-heavy degree-8 log-T fits (k1/k3/k5/k11/k14, the ci* that embed
them, HDlte): summing O(1000)-magnitude terms in f32 leaves an intrinsic ~1e-3
floor on CPU-libm-vs-Metal-GPU agreement — NOT a bug (grackle's own f32 build
tabulates these same fits in f32). Real f32 *adequacy* is the Wave-5 one-zone
f32-vs-f64 trajectory.
"""
function check_scalar_kernel(label, jl_fn, oracle_vals, Ts; f32rtol = RTOL_B.rtol)
    layerA!(label, jl_fn(:cpu, Float64, Ts), oracle_vals)
    (metal_ready() || cuda_ready()) || return nothing
    # f32 layers: operating range (T ≤ F32_TMAX) AND where the value is
    # non-negligible (≥ 1e-6 of its grid peak). Relative error in the deep tail of
    # a rate (value 1e-17 vs its 1e-9 peak) is f32 noise with zero physical impact.
    refv = _flat(oracle_vals); Tv = collect(Ts)
    peak = maximum(abs, refv)
    tol  = (rtol = f32rtol, atol = 1e-30)
    mask = findall(i -> Tv[i] <= F32_TMAX && abs(refv[i]) >= 1e-6 * peak, eachindex(Tv))
    cpuf = _flat(jl_fn(:cpu, Float32, Ts))[mask]
    ref  = refv[mask]
    if metal_ready()
        gpuf = _flat(jl_fn(:metal, Float32, Ts))[mask]
        @check(string(label, " [B:cpu≡metal f32, T≤$(F32_TMAX)]"), gpuf, cpuf, tol)
        @check(string(label, " [C:metal-f32 vs grackle, T≤$(F32_TMAX)]"), gpuf, ref, tol)
    end
    if cuda_ready()
        cudf = _flat(jl_fn(:cuda, Float32, Ts))[mask]
        @check(string(label, " [D:cpu≡cuda f32, T≤$(F32_TMAX)]"), cudf, cpuf, tol)
        @check(string(label, " [E:cuda-f32 vs grackle, T≤$(F32_TMAX)]"), cudf, ref, tol)
    end
end

# f32 device-parity floor per fit (Layer A still pins correctness at 5e-11):
#   ~3e-3 for the degree-8 exp(poly) fits; ~5e-2 for the Savin-2004 k11 (a
#   degree-7 logT poly with ~1e5 internal cancellation — f32 simply cannot do
#   better, and k11 is a minor channel; Wave-5 confirms it doesn't move the
#   trajectory). k14's tail is handled by the magnitude floor above.
const F32_SOFT  = Set(["k1","k3","k5","k14","ciHI","ciHeI","ciHeII","HDlte"])
const F32_SOFT2 = Set(["k11"])
f32rtol_for(name) = name in F32_SOFT2 ? 5e-2 : (name in F32_SOFT ? 3e-3 : RTOL_B.rtol)
