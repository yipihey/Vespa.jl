# solve.jl — host boundary + device launcher.  Converts a host's code-unit fields
# to physical CGS, sub-cycles every cell on any backend, and writes the evolved
# fields back — the drop-in for grackle_reduced_step! (lib/MultiCode/deps/
# grackle_reduced.c).  Unit convention = grackle's comoving_coordinates=0 (what
# grackle_reduced.c uses): ρ_cgs = field·density_units (NO a³), e_cgs =
# e·(length/time)², t_s = dt·time_units, z = 1/a_value − 1 (sets the CMB).

export solve_chem!

# Per-cell: convert code units → CGS, evolve, convert back. Keeps fields in place.
@kernel function _evolve_k!(e, HII, H2I, HDI, @Const(rho),
                            du, vu2, tu, dt, z, hubble, Om, OL, fh, deut)
    i = @index(Global)
    @inbounds begin
        T   = eltype(e)
        hd_in = deut ? HDI[i]*du : zero(T)
        en, hii, h2, hd = evolve_cell(rho[i]*du, e[i]*vu2, HII[i]*du, H2I[i]*du,
                                      hd_in, dt*tu, z; hubble=hubble, Om=Om, OL=OL,
                                      fh=fh, deuterium=deut)
        e[i]   = en  / vu2
        HII[i] = hii / du
        H2I[i] = h2  / du
        deut && (HDI[i] = hd / du)
    end
end

"""
    solve_chem!(rho, e_int, HII, H2I, [HDI]; a_value, dt, density_units,
                length_units, time_units, hubble=71, Om=0.27, OL=0.73, fh=0.76,
                deuterium=false, backend=:cpu, precision=Float64)

Evolve the v2026 reduced primordial+D chemistry/cooling over `dt` (code time
units) for every cell, updating `e_int`, `HII`, `H2I` (and `HDI` if `deuterium`)
in place.  `rho` is read-only; `HII`/`H2I`/`HDI` are MASS densities ρ·x in the
host code units defined by `density_units`/`length_units`/`time_units`.  Mirrors
`GrackleChem.grackle_reduced_step!`; the engine is a KA kernel (`backend=:cpu` or
`:metal`) at `precision` (Float64/Float32).
"""
function solve_chem!(rho::AbstractVector, e_int::AbstractVector,
                     HII::AbstractVector, H2I::AbstractVector,
                     HDI::Union{Nothing,AbstractVector} = nothing;
                     a_value::Real, dt::Real, density_units::Real,
                     length_units::Real, time_units::Real,
                     hubble::Real = 71.0, Om::Real = 0.27, OL::Real = 0.73,
                     fh::Real = 0.76, deuterium::Bool = false,
                     backend::Symbol = :cpu, precision::Type = Float64)
    n  = length(rho)
    @assert length(e_int) == n && length(HII) == n && length(H2I) == n
    deut = deuterium && HDI !== nothing
    deut && @assert length(HDI) == n

    P   = precision
    be  = ChemistryKernels.backend(backend)
    du  = P(density_units)
    vu2 = P((length_units / time_units)^2)
    tu  = P(time_units)
    z   = P(1.0 / a_value - 1.0)

    d_rho = to_device(be, collect(rho),   P)
    d_e   = to_device(be, collect(e_int), P)
    d_HII = to_device(be, collect(HII),   P)
    d_H2I = to_device(be, collect(H2I),   P)
    d_HDI = deut ? to_device(be, collect(HDI), P) : device_zeros(be, P, (n,))

    _evolve_k!(be)(d_e, d_HII, d_H2I, d_HDI, d_rho, du, vu2, tu,
                   P(dt), z, P(hubble), P(Om), P(OL), P(fh), deut; ndrange = n)

    e_int .= to_host(d_e)
    HII   .= to_host(d_HII)
    H2I   .= to_host(d_H2I)
    deut && (HDI .= to_host(d_HDI))
    return nothing
end
