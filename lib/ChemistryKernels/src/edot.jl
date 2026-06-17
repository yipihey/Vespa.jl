# edot.jl — net radiative cooling/heating rate of the v2026 reduced network.
#
# The per-channel physics (H/He collisional excitation & ionisation, recombination,
# bremsstrahlung, H2, HD, CMB-Compton, metal fine-structure) now lives in the
# foundation package EmissionKernels.  `cooling_edot` is the network's view of it: the
# negative of `EmissionKernels.cooling_rate_total` (the summed radiative cooling, He
# omitted as in the reduced model).  Bit-identical to the legacy assembler by
# construction — same expression, same term order. Pure & allocation-free.

export cooling_edot

"""
    cooling_edot(nHI, nHII, nHeI, nde, nH2, nHD, T, z; ih2optical=false, nH=nothing,
                 metals=nothing)

Net volumetric energy rate ė [erg cm⁻³ s⁻¹] (cooling ⇒ negative) for the reduced
network at gas temperature `T` [K] and redshift `z`. Number densities are physical
[cm⁻³]; `nH2`/`nHD` are H2 and HD *molecule* densities; `metals` an optional
`MetalAbundances` (n(X)/n_H per cell) and `nH` the H-nucleus density. Delegates to
`EmissionKernels.cooling_rate_total`. Pure.
"""
@inline function cooling_edot(nHI, nHII, nHeI, nde, nH2, nHD, T, z;
                              ih2optical::Bool = false, nH = nothing, metals = nothing)
    return -cooling_rate_total(nHI, nHII, nHeI, nde, nH2, nHD, T, z;
                               ih2optical = ih2optical, nH = nH, metals = metals)
end
