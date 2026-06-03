# Shared logic for running an Enzo test problem through BOTH Enzo's own
# EvolveHierarchy and the new Julia EvolveLevel (full replication, with the
# problem's physics enabled), and comparing all fields. Used by the suite test
# and the discovery script.

const RUN_DIR = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "run"))

# Read a scalar Enzo parameter from a .enzo file (first match), else `default`.
function enzo_param(pf::AbstractString, key::AbstractString; default = 0.0)
    rx = Regex("^\\s*" * key * "\\s*=\\s*([-+0-9.eE]+)")
    for ln in eachline(pf)
        m = match(rx, ln)
        m === nothing || return parse(Float64, m.captures[1])
    end
    return default
end

"Infer the physics flags the Julia EvolveLevel needs from the .enzo parameters."
function problem_flags(pf::AbstractString)
    g    = enzo_param(pf, "SelfGravity") != 0
    cool = enzo_param(pf, "RadiativeCooling") != 0 || enzo_param(pf, "MultiSpecies") != 0
    rad  = enzo_param(pf, "RadiativeTransfer") != 0
    star = enzo_param(pf, "StarParticleCreation") != 0
    hm   = Int(enzo_param(pf, "HydroMethod"))
    return (gravity = g, cooling = cool, radiation = rad,
            star_formation = star, star_sources = star, hydromethod = hm)
end

# Per-field L∞ error normalized by the field's magnitude (robust for near-zero
# fields); returns the max over all fields shared by the two runs.
function _max_field_error(dj::Dict{Int,Vector{Float64}}, de::Dict{Int,Vector{Float64}})
    common = intersect(keys(dj), keys(de))
    isempty(common) && return (err = Inf, worst = -1, nfields = 0)
    maxerr = 0.0; worst = -1
    for ft in common
        a = dj[ft]; b = de[ft]
        length(a) == length(b) || return (err = Inf, worst = ft, nfields = length(common))
        scale = maximum(abs, b) + 1e-30
        e = maximum(abs.(a .- b)) / scale
        e > maxerr && (maxerr = e; worst = ft)
    end
    return (err = maxerr, worst = worst, nfields = length(common))
end

"""
    compare_problem(enzo_file) -> NamedTuple

Run the problem through Enzo's EvolveHierarchy and the Julia EvolveLevel (full
replication, physics from `problem_flags`), and return the max normalized field
error between them on the root grid. `err ≈ 0` for single-grid (bit-for-bit),
`~few×1e-5` for AMR (cross-level ordering).
"""
function compare_problem(pf::AbstractString)
    fl = problem_flags(pf)
    de = EnzoLib.evolve_problem_fields(pf)
    dj = EnzoLib.run_amr_fields(pf; gravity = fl.gravity, cooling = fl.cooling,
                                radiation = fl.radiation, star_sources = fl.star_sources,
                                star_formation = fl.star_formation)
    r = _max_field_error(dj, de)
    return (err = r.err, worst = r.worst, nfields = r.nfields, flags = fl)
end
