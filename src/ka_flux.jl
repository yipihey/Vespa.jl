# Batched-per-face flux path (ADR P1 re-platform onto the native KA/Julia stack).
#
# The native per-face `accumulate_flux!` computes the HLLC flux one face at a time
# on the host (`_flux_face!`). For GPU execution we keep the conservation machinery
# *byte-identical* — the `for_each_face` enumeration, the +F/−F net-flux-out
# scatter, and the `_reflux_capture!` / `_bflux_capture*` coarse↔fine capture — and
# move ONLY the per-face reconstruction+Riemann onto a swappable *flux backend*.
#
# Two passes over the (deterministic) face enumeration, so no face/handle list is
# stored (the opaque-handle contract stays intact):
#   1. GATHER — walk `for_each_face`; reconstruct each (sub)face's L/R primitive
#      state (`_face_value` / `_boundary_ghost`, reused verbatim) into flat columns.
#   2. COMPUTE — one `compute_face_fluxes!(backend, …)` call fills the flux column.
#      `HostBatchedFlux` loops the same `riemann_flux` the per-face path uses (⇒
#      bit-identical); a KA backend injected from a lib runs the same algebra on a
#      device. Conservation telescopes from single-flux-per-face REGARDLESS of the
#      backend — the structural guarantee Enzo's C++ CorrectForRefinedFluxes lacked.
#   3. SCATTER — walk `for_each_face` again (same order); add each flux to the
#      net-flux-out accumulator and run the unchanged reflux/bflux capture.
#
# Opt in by setting `sim.flux` to an `AbstractFluxBackend` (default `nothing` keeps
# the native per-face path). The two paths produce identical results with
# `HostBatchedFlux`, so every existing tolerance and the AMR round-off conservation
# hold unchanged — this is the decisive proof that KA hydro + Vespa's FluxRegister
# is exact, which the Enzo-driven reflux could not reach.

"Swappable flux backend: computes the conserved face flux for a batch of faces."
abstract type AbstractFluxBackend end

"""
    HostBatchedFlux()

Reference flux backend: loops the model's `riemann_flux` on the host, so the
batched path is bit-identical to the native per-face `_flux_face!`. Proves the
batched gather/scatter/reflux machinery conserves before any device kernel is
involved, and is the parity oracle for a KA (GPU) backend.
"""
struct HostBatchedFlux <: AbstractFluxBackend end

"""
    compute_face_fluxes!(backend, model, F, WL, WR, axisv)

Fill `F[f]` with the conserved flux across face `f` from the gathered left/right
primitive states `WL[f]`,`WR[f]` (each an `NTuple{5}`) and face normal `axisv[f]`.
`F`,`WL`,`WR` are `Vector{NTuple{5,T}}`; `axisv` is `Vector{Int}`. A device backend
overrides this to run the same algebra on the GPU (packing the columns as needed).
"""
function compute_face_fluxes!(::HostBatchedFlux, model, F, WL, WR, axisv)
    @inbounds for f in eachindex(F)
        F[f] = riemann_flux(model, WL[f], WR[f], axisv[f])
    end
    return F
end

"""
    KAFlux(backend)

KernelAbstractions flux backend: runs the per-face Riemann solve as a batched KA
kernel on `backend` (a `KernelAbstractions.Backend` — `CPU()`, `CUDABackend()`,
`MetalBackend()`, …). The kernel calls the SAME `hllc_flux` the host path uses, so
the device result matches `HostBatchedFlux` to floating-point round-off and
conservation stays exact by construction (single flux per face). The
`compute_face_fluxes!(::KAFlux, …)` method lives in the `VespaKAFluxExt` package
extension — load `KernelAbstractions` (and a device package, e.g. `CUDA`) to enable
it; using `KAFlux` without `KernelAbstractions` loaded is a `MethodError`.
"""
struct KAFlux{B} <: AbstractFluxBackend
    backend::B
end

# ── gather: reconstructed L/R primitive face states (mirror of `_flux_face!`) ────
# interior↔interior: +axis normal points i→j.
@inline _face_states(sim::Simulation, left::Interior, right::Interior, axis::Int) =
    (_face_value(sim, left.cell, axis, :hi), _face_value(sim, right.cell, axis, :lo))

# hi-side domain boundary: interior i on the left, ghost on the right.
@inline function _face_states(sim::Simulation, left::Interior, right::DomainBoundary, axis::Int)
    WL = _face_value(sim, left.cell, axis, :hi)
    return (WL, _boundary_ghost(sim, WL, right.bc, axis, :hi, left.cell))
end

# lo-side domain boundary: ghost on the left, interior j on the right.
@inline function _face_states(sim::Simulation, left::DomainBoundary, right::Interior, axis::Int)
    WR = _face_value(sim, right.cell, axis, :lo)
    return (_boundary_ghost(sim, WR, left.bc, axis, :lo, right.cell), WR)
end

# ── scatter: identical net-flux-out accumulation + capture as `_flux_face!` ──────
@inline function _scatter_face!(sim::Simulation, left::Interior, right::Interior,
                                axis::Int, area::Real, F; reflux = nothing, bflux = nothing)
    i, j = left.cell, right.cell
    av = sim.accv
    aT = Base.eltype(F)(area)
    _acc_add!(av, i, F, aT)               # flux leaves i (+F·area)
    _acc_add!(av, j, F, -aT)              # flux enters j (−F·area)
    if reflux !== nothing
        for reg in reflux
            _reflux_capture!(sim, reg, i, j, F, area)
        end
    end
    bflux === nothing || _bflux_capture_interior!(bflux, axis, i, F, area)
    return nothing
end

@inline function _scatter_face!(sim::Simulation, left::Interior, right::DomainBoundary,
                                axis::Int, area::Real, F; reflux = nothing, bflux = nothing)
    i = left.cell
    av = sim.accv
    aT = Base.eltype(F)(area)
    _acc_add!(av, i, F, aT)            # outward normal +axis
    bflux === nothing || _bflux_capture!(bflux, axis, :hi, i, F, area)
    return nothing
end

@inline function _scatter_face!(sim::Simulation, left::DomainBoundary, right::Interior,
                                axis::Int, area::Real, F; reflux = nothing, bflux = nothing)
    j = right.cell
    av = sim.accv
    aT = Base.eltype(F)(area)
    bflux === nothing || _bflux_capture!(bflux, axis, :lo, j, F, area)
    _acc_add!(av, j, F, -aT)           # outward normal −axis
    return nothing
end

# Batched flux divergence: gather all face L/R states, compute the flux batch on
# `sim.flux`, scatter back through the unchanged accumulation+capture. Equivalent
# to `accumulate_flux!`'s per-face path; selected when `sim.flux !== nothing`.
function _accumulate_flux_batched!(sim::Simulation, fluxbackend; reflux = nothing, bflux = nothing)
    b = sim.backend
    av = sim.accv
    T = _Tf(sim)
    z = ntuple(_ -> zero(T), nvars_val(sim.model))
    for_each_cell(b) do cell
        set_U!(av, cell, z)
    end

    # PASS 1 — gather reconstructed L/R primitive states (fresh per call; Phase-1
    # correctness over speed — a persistent device buffer is the perf follow-up).
    WL = NTuple{5,T}[]
    WR = NTuple{5,T}[]
    axisv = Int[]
    for_each_face(b; bcs = sim.bcs) do leftref, rightref, axis, area
        wl, wr = _face_states(sim, leftref, rightref, axis)
        push!(WL, wl)
        push!(WR, wr)
        push!(axisv, axis)
    end

    # PASS 2 — compute the whole face batch on the (CPU or device) flux backend.
    F = Vector{NTuple{5,T}}(undef, length(WL))
    compute_face_fluxes!(fluxbackend, sim.model, F, WL, WR, axisv)

    # PASS 3 — scatter in the SAME face order (for_each_face is deterministic, and
    # `sim.sv`/topology are untouched between the gather and scatter walks, so the
    # face counter realigns exactly). `Ref` keeps the captured counter unboxed.
    fi = Ref(0)
    for_each_face(b; bcs = sim.bcs) do leftref, rightref, axis, area
        fi[] += 1
        _scatter_face!(sim, leftref, rightref, axis, area, F[fi[]]; reflux = reflux, bflux = bflux)
    end
    return nothing
end
