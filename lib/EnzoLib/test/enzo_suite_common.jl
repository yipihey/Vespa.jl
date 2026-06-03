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
    cosmo = enzo_param(pf, "ComovingCoordinates") != 0
    hm   = Int(enzo_param(pf, "HydroMethod"))
    mhdct = hm == 6 || enzo_param(pf, "UseMHDCT") != 0    # constrained-transport MHD
    return (gravity = g, cooling = cool, radiation = rad, star_formation = star,
            star_sources = star, cosmology = cosmo, mhdct = mhdct, hydromethod = hm)
end

# Per-field L∞ error, normalized by the field magnitude with an ABSOLUTE floor
# tied to the problem's characteristic scale (the largest field magnitude). This
# is the crucial bit for MHD/multi-D: fields that are DYNAMICALLY NEGLIGIBLE in
# the reference (e.g. Bz/Vz in a planar problem — zero up to the scheme's symmetry
# error) have an ill-defined relative error; dividing a small absolute deviation
# by the field's own ~0 magnitude turns it into a spurious O(1e5) "error". The
# floor must exceed not just machine roundoff (~1e-17) but the deviations a
# correct-but-not-bit-identical scheme legitimately produces in such fields (an
# AMR-CT regrid breaks planar symmetry at ~1e-3·scale). atol = 1e-2·(global scale)
# treats any field below 1% of the problem's peak as insignificant (its deviation
# is then measured against that 1% floor) while leaving active fields' relative
# error essentially unchanged. Returns the max over all fields shared by both runs.
function _max_field_error(dj::Dict{Int,Vector{Float64}}, de::Dict{Int,Vector{Float64}})
    common = intersect(keys(dj), keys(de))
    isempty(common) && return (err = Inf, worst = -1, nfields = 0)
    gscale = 0.0
    for ft in common; gscale = max(gscale, maximum(abs, de[ft])); end
    atol = 1e-2 * gscale + 1e-300
    maxerr = 0.0; worst = -1
    for ft in common
        a = dj[ft]; b = de[ft]
        length(a) == length(b) || return (err = Inf, worst = ft, nfields = length(common))
        scale = maximum(abs, b) + atol      # absolute floor kills dynamically-negligible-field noise
        e = maximum(abs.(a .- b)) / scale
        e > maxerr && (maxerr = e; worst = ft)
    end
    return (err = maxerr, worst = worst, nfields = length(common))
end

# Order-independent particle-position agreement: for each Julia particle, the
# distance to the nearest Enzo particle (matching is robust to cross-grid gather
# ordering), max over particles, normalized by the point-cloud extent. ≈0 means
# the two runs put the particles in the same places. Both pj/pe are Np×rank.
function _max_particle_error(pj::AbstractMatrix, pe::AbstractMatrix)
    (size(pj, 1) == 0 || size(pe, 1) == 0) && return (err = NaN, nparticles = 0)
    size(pj, 2) == size(pe, 2) || return (err = Inf, nparticles = size(pj, 1))
    # normalize by the largest per-axis span of the reference cloud (domain scale)
    scale = 0.0
    for d in 1:size(pe, 2)
        col = @view pe[:, d]
        scale = max(scale, maximum(col) - minimum(col))
    end
    scale += 1e-30
    worst = 0.0
    @inbounds for i in 1:size(pj, 1)
        best = Inf
        for j in 1:size(pe, 1)
            s = 0.0
            for d in 1:size(pj, 2)
                δ = pj[i, d] - pe[j, d]; s += δ * δ
            end
            s < best && (best = s)
        end
        worst = max(worst, sqrt(best))
    end
    return (err = worst / scale, nparticles = size(pj, 1))
end

# The serial (use-mpi-no) build's root-grid FFT gravity solver requires
# UnigridTranspose=0 (the default 2 is MPI-only); patch a temp copy of the .enzo
# for self-gravitating problems so BOTH runs use identical parameters.
function paramfile_for(pf::AbstractString)
    problem_flags(pf).gravity || return pf
    tmp = tempname() * ".enzo"
    cp(pf, tmp)
    open(tmp, "a") do io; println(io, "\nUnigridTranspose = 0"); end
    return tmp
end

"""
    compare_problem(enzo_file) -> NamedTuple

Run the problem through Enzo's EvolveHierarchy and the Julia EvolveLevel (full
replication, physics from `problem_flags`), and return the max normalized field
error between them on the root grid. `err ≈ 0` for single-grid (bit-for-bit),
`~few×1e-5` for AMR (cross-level ordering).
"""
function compare_problem(pf0::AbstractString)
    fl = problem_flags(pf0)
    pf = paramfile_for(pf0)
    se = EnzoLib.evolve_problem_state(pf)
    sj = EnzoLib.run_amr_state(pf; gravity = fl.gravity, cooling = fl.cooling,
                               radiation = fl.radiation, star_sources = fl.star_sources,
                               star_formation = fl.star_formation, cosmology = fl.cosmology,
                               mhdct = fl.mhdct)
    r = _max_field_error(sj.fields, se.fields)
    p = _max_particle_error(sj.particles, se.particles)
    return (err = r.err, worst = r.worst, nfields = r.nfields,
            perr = p.err, nparticles = p.nparticles, flags = fl)
end
