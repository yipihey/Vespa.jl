# ccall bindings to the grackle verification oracle (oracle/libchem_oracle.dylib).
# These return grackle's OWN analytic rate/cooling values (units=1.0, table-free),
# the gold reference every ported Julia formula is diffed against.

module ChemOracle

const LIB = abspath(joinpath(@__DIR__, "..", "oracle", "libchem_oracle.dylib"))

available() = isfile(LIB)

"Configure grackle's reduced-network flags (CaseB + the rate-family toggles)."
set_flags!(; caseB=1, colexc=1, colion=1, reccool=1, brems=1) =
    ccall((:chem_set_flags, LIB), Cvoid, (Cint,Cint,Cint,Cint,Cint),
          caseB, colexc, colion, reccool, brems)

"grackle reaction-rate kN at temperature T [K], in CGS (units=1)."
rate(name::AbstractString, T::Real) =
    ccall((:chem_rate, LIB), Cdouble, (Cstring, Cdouble), name, Float64(T))

"grackle cooling/heating coefficient at temperature T [K], in CGS (units=1)."
cool(name::AbstractString, T::Real) =
    ccall((:chem_cool, LIB), Cdouble, (Cstring, Cdouble), name, Float64(T))

# ── reduced-network temperature oracle (grackle calculate_temperature) ────────
# Self-contained: table-free init (metal_cooling/UVbackground off ⇒ no data file).
"Initialize the reduced-network temperature oracle (returns 1 on success)."
temperature_init!(; hubble=71.0, Om=0.27, OL=0.73, a_value, fh=0.76,
        density_units=1.0, length_units=1.0, time_units=1.0,
        data_file="", deuterium=false) =
    ccall((:chem_temperature_init, LIB), Cint,
          (Cdouble,Cdouble,Cdouble,Cdouble,Cdouble,Cdouble,Cdouble,Cdouble,Cstring,Cint),
          hubble, Om, OL, a_value, fh, density_units, length_units, time_units,
          data_file, deuterium ? 1 : 0)

"grackle reduced-network gas temperature [K] over arrays of (rho,e_int,HII,H2I)."
function temperature(rho::Vector{Float64}, eint::Vector{Float64},
                     HII::Vector{Float64}, H2I::Vector{Float64}; a_value::Real)
    n = length(rho); T = zeros(Float64, n)
    rc = ccall((:chem_temperature, LIB), Cint,
        (Clong,Cdouble,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),
        n, a_value, rho, eint, HII, H2I, T)
    rc == 1 || error("chem_temperature failed (rc=$rc)")
    return T
end

"grackle reduced-network cooling time [code time units] (init sets time_units)."
function cooling_time(rho::Vector{Float64}, eint::Vector{Float64},
                      HII::Vector{Float64}, H2I::Vector{Float64}; a_value::Real)
    n = length(rho); tc = zeros(Float64, n)
    rc = ccall((:chem_cooling_time, LIB), Cint,
        (Clong,Cdouble,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),
        n, a_value, rho, eint, HII, H2I, tc)
    rc == 1 || error("chem_cooling_time failed (rc=$rc)")
    return tc
end

# A log-spaced temperature grid spanning the rate-table range, with the known
# branch boundaries spliced in (k1 T_ev=0.8 -> T≈9284; k2 CaseA T=5500; the
# 1 K / 1e9 K table edges; k11/k12 T_ev=0.3 -> T≈3481; k15 T_ev=0.1 -> T≈1160).
function tgrid(; n::Int=200, lo::Float64=1.0, hi::Float64=1.0e9)
    base = exp.(range(log(lo), log(hi), length=n))
    bnd  = [0.8*11605.0, 5500.0, 0.3*11605.0, 0.1*11605.0, 0.04*11605.0,
            30.0, 617.0, 3.0e3, 1.0e4, 2.0e3, 2.0e2, 2.0e5, 3.2e4,
            1.0+1e-6, 1.0e9-1.0]
    sort!(unique(vcat(base, bnd)))
end

end # module ChemOracle
