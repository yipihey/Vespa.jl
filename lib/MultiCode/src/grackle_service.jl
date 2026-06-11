# ── reduced primordial-chemistry service (v2026) ──────────────────────────────
#
# A code-neutral Grackle chemistry+cooling service: any MultiCode host (RAMSES,
# Arepo, …) evolves early-universe primordial chemistry while carrying only TWO
# advected species, HII and H2I.  Helium is forced neutral, electrons = protons,
# HI is reconstructed and H-/H2+ are equilibrium -- all inside our Grackle fork
# (flags neutral_helium + equilibrium_h2_intermediates + cmb_dissociation +
# cmb_recombination).  Backed by lib/MultiCode/deps/libgrackle_reduced.dylib
# (built against ~/grackle_install; see deps/build_grackle_reduced.sh).
#
# Per-cell arrays are Float64 and density-weighted: HII, H2I are MASS densities
# rho*x in the code_units passed to `grackle_reduced_init!`.  This is exactly the
# convention RAMSES uses for its ion passive scalars (uold = rho*x).

module GrackleChem

const LIBGR = abspath(joinpath(@__DIR__, "..", "deps", "libgrackle_reduced.dylib"))

"Is the reduced-chemistry service library built and loadable?"
available() = isfile(LIBGR)

"""
    grackle_reduced_init!(; hubble, Om, OL, a_value, fh, density_units,
                            length_units, time_units, data_file)

Initialize the reduced network once for a host's code units.  `hubble` is H0 in
km/s/Mpc; `*_units` convert the host's code units to CGS (field*density_units →
g/cm³); `fh` = hydrogen mass fraction; `data_file` = Grackle Cloudy table path.
"""
function grackle_reduced_init!(; hubble::Real=71.0, Om::Real=0.27, OL::Real=0.73,
        a_value::Real, fh::Real=0.76, density_units::Real, length_units::Real,
        time_units::Real, data_file::AbstractString)
    available() || error("libgrackle_reduced not built ($LIBGR); run deps/build_grackle_reduced.sh")
    rc = ccall((:grackle_reduced_init, LIBGR), Cint,
        (Cdouble,Cdouble,Cdouble,Cdouble,Cdouble,Cdouble,Cdouble,Cdouble,Cstring),
        hubble, Om, OL, a_value, fh, density_units, length_units, time_units, data_file)
    rc == 1 || error("grackle_reduced_init failed (rc=$rc)")
    return nothing
end

"""
    grackle_reduced_step!(rho, e_int, HII, H2I; a_value, dt)

Evolve `n` cells for `dt` (code time units) at expansion factor `a_value`.
`e_int`, `HII`, `H2I` are updated in place; `rho` is read-only.  The 7
reconstructed species live in transient scratch inside the service.
"""
function grackle_reduced_step!(rho::Vector{Float64}, e_int::Vector{Float64},
        HII::Vector{Float64}, H2I::Vector{Float64}; a_value::Real, dt::Real)
    n = length(rho)
    @assert length(e_int)==n && length(HII)==n && length(H2I)==n
    rc = ccall((:grackle_reduced_step, LIBGR), Cint,
        (Clong,Cdouble,Cdouble,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),
        n, a_value, dt, rho, e_int, HII, H2I)
    rc == 1 || error("grackle_reduced_step failed (rc=$rc)")
    return nothing
end

"Per-cell gas temperature [K] (diagnostic; same reduced reconstruction)."
function grackle_reduced_temperature(rho::Vector{Float64}, e_int::Vector{Float64},
        HII::Vector{Float64}, H2I::Vector{Float64}; a_value::Real)
    n = length(rho); T = zeros(Float64, n)
    rc = ccall((:grackle_reduced_temperature, LIBGR), Cint,
        (Clong,Cdouble,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),
        n, a_value, rho, e_int, HII, H2I, T)
    rc == 1 || error("grackle_reduced_temperature failed (rc=$rc)")
    return T
end

end # module GrackleChem
