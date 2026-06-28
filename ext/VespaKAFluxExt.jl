# KernelAbstractions flux backend (the GPU/CPU device path for `KAFlux`).
#
# `_accumulate_flux_batched!` (core) gathers every (sub)face's reconstructed L/R
# primitive state on the host, calls `compute_face_fluxes!` once, then scatters the
# fluxes back through the unchanged conservation+reflux machinery. This extension
# provides the `KAFlux` method: pack the host columns into device matrices, run the
# HLLC over all faces as a single KA kernel, copy the fluxes back. The kernel calls
# the SAME `Vespa.hllc_flux` (a pure, allocation-free `NTuple{5}` function ⇒
# GPU-compilable) the host path uses, so the device result matches `HostBatchedFlux`
# to round-off — and conservation is exact regardless (single flux per face).
#
# Device-agnostic: only KernelAbstractions is needed here; the concrete array type
# (CuArray/MtlArray/Array) follows from the `KernelAbstractions.Backend` the user
# wraps in `KAFlux(backend)`.

module VespaKAFluxExt

using Vespa
using KernelAbstractions
const KA = KernelAbstractions

# One work-item per face: HLLC flux from the gathered L/R primitive columns.
@kernel function _ka_hllc_kernel!(F, @Const(WL), @Const(WR), @Const(axisv), gamma)
    f = @index(Global)
    @inbounds begin
        wl = (WL[1, f], WL[2, f], WL[3, f], WL[4, f], WL[5, f])
        wr = (WR[1, f], WR[2, f], WR[3, f], WR[4, f], WR[5, f])
        Ff = Vespa.hllc_flux(wl, wr, gamma, Int(axisv[f]))
        F[1, f] = Ff[1]
        F[2, f] = Ff[2]
        F[3, f] = Ff[3]
        F[4, f] = Ff[4]
        F[5, f] = Ff[5]
    end
end

function Vespa.compute_face_fluxes!(kf::Vespa.KAFlux, model, F, WL, WR, axisv)
    be = kf.backend
    nf = length(F)
    nf == 0 && return F
    T = eltype(eltype(F))                      # NTuple{5,T} → T
    γ = Vespa.adiabatic_index(model)

    # Host columns: 5×nf contiguous matrices over the packed NTuple{5,T} storage
    # (materialized so `copyto!` to a device array has a plain Array source).
    hWL = Array(reshape(reinterpret(T, WL), 5, nf))
    hWR = Array(reshape(reinterpret(T, WR), 5, nf))
    hax = convert(Vector{Int32}, axisv)

    # Upload, launch one kernel over all faces, download.
    dWL = KA.allocate(be, T, (5, nf));    copyto!(dWL, hWL)
    dWR = KA.allocate(be, T, (5, nf));    copyto!(dWR, hWR)
    dax = KA.allocate(be, Int32, (nf,));  copyto!(dax, hax)
    dF  = KA.allocate(be, T, (5, nf))

    _ka_hllc_kernel!(be)(dF, dWL, dWR, dax, T(γ); ndrange = nf)
    KA.synchronize(be)

    hF = Array(dF)                             # device → host (Array() works per-backend)
    Fv = reshape(reinterpret(T, F), 5, nf)     # writable view over F's storage
    copyto!(Fv, hF)
    return F
end

end # module
