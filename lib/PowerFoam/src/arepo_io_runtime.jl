"""
    ArepoRuntimeFeatureSet

Compact capability summary derived from an AREPO parameter/config parse and the
currently loaded PowerFoam runtime. This is diagnostic glue for gates and
reports, not the production scheduler.
"""
struct ArepoRuntimeFeatureSet
    dimensionality::Int
    config_hdf5::Bool
    package_hdf5::Bool
    parameter_io::Bool
    snapshot_io::Bool
    hydro::Bool
    gravity::Bool
    cosmology::Bool
    hierarchical_timesteps::Bool
    local_ppm::Bool
    riemann::Symbol
end

"""
    arepo_cosmology_runtime(params)

Return a typed summary of the AREPO cosmology fields carried through
normalization.
"""
function arepo_cosmology_runtime(params::ArepoParameterSet)
    return params.normalized.cosmology
end

function arepo_cosmology_runtime(normalized::NamedTuple)
    return _runtime_cosmology_group(normalized)
end

"""
    arepo_runtime_features(params)

Return a typed feature summary for an `ArepoParameterSet`, normalized parameter
view, or `ArepoConfigFlags`.
"""
function arepo_runtime_features(params::ArepoParameterSet)
    return arepo_runtime_features(params.normalized, params.config_flags)
end

function arepo_runtime_features(normalized::NamedTuple,
                                config_flags::ArepoConfigFlags = ArepoConfigFlags())
    features = _runtime_feature_group(normalized, config_flags)
    cosmology = _runtime_cosmology_group(normalized)
    gravity = _runtime_namedtuple_group(normalized, :gravity)
    dimensionality = _runtime_feature_bool(features, :twodims) ? 2 : 3
    config_hdf5 = _runtime_feature_bool(features, :have_hdf5)
    package_hdf5 = arepo_snapshot_hdf5_available()
    gravity_on = (:SELFGRAVITY in config_flags.enabled) ||
                 any(value !== nothing for value in values(pairs(gravity)))
    cosmology_on = cosmology.enabled
    riemann = _runtime_feature_bool(features, :riemann_hll) ? :hll : :default
    return ArepoRuntimeFeatureSet(
        dimensionality,
        config_hdf5,
        package_hdf5,
        true,
        package_hdf5,
        true,
        gravity_on,
        cosmology_on,
        _runtime_feature_bool(features, :force_equal_timesteps) ||
            _runtime_feature_bool(features, :tree_based_timesteps),
        _runtime_feature_bool(features, :local_ppm),
        riemann,
    )
end

function arepo_runtime_features(config_flags::ArepoConfigFlags)
    normalized = (features = (
                      twodims = :TWODIMS in config_flags.enabled,
                      have_hdf5 = :HAVE_HDF5 in config_flags.enabled,
                      force_equal_timesteps = :FORCE_EQUAL_TIMESTEPS in config_flags.enabled,
                      tree_based_timesteps = :TREE_BASED_TIMESTEPS in config_flags.enabled,
                      local_ppm = :LOCAL_PPM in config_flags.enabled,
                      riemann_hll = :RIEMANN_HLL in config_flags.enabled,
                  ),
                  domain = (box_size = nothing, periodic_boundaries_on = false),
                  cosmology = ArepoCosmologyRuntime(false, false, false, nothing, nothing,
                                                   nothing, nothing, nothing),
                  gravity = (;))
    return arepo_runtime_features(normalized, config_flags)
end

function _runtime_feature_group(normalized::NamedTuple,
                                config_flags::ArepoConfigFlags)
    if :features in keys(normalized)
        return getfield(normalized, :features)
    end
    return (
        twodims = :TWODIMS in config_flags.enabled,
        have_hdf5 = :HAVE_HDF5 in config_flags.enabled,
        force_equal_timesteps = :FORCE_EQUAL_TIMESTEPS in config_flags.enabled,
        tree_based_timesteps = :TREE_BASED_TIMESTEPS in config_flags.enabled,
        local_ppm = :LOCAL_PPM in config_flags.enabled,
        riemann_hll = :RIEMANN_HLL in config_flags.enabled,
    )
end

function _runtime_namedtuple_group(normalized::NamedTuple, name::Symbol)
    return name in keys(normalized) ? getfield(normalized, name) : (;)
end

function _runtime_cosmology_group(normalized::NamedTuple)
    if :cosmology in keys(normalized)
        return getfield(normalized, :cosmology)
    end
    domain = _runtime_namedtuple_group(normalized, :domain)
    comoving_integration_on = _runtime_group_value(domain, :comoving_integration_on, false)
    return ArepoCosmologyRuntime(
        comoving_integration_on === true,
        comoving_integration_on,
        _runtime_group_value(domain, :periodic_boundaries_on, nothing),
        _runtime_group_value(domain, :box_size, nothing),
        _runtime_group_value(domain, :omega0, nothing),
        _runtime_group_value(domain, :omega_baryon, nothing),
        _runtime_group_value(domain, :omega_lambda, nothing),
        _runtime_group_value(domain, :hubble_param, nothing),
    )
end

function _runtime_feature_bool(features::NamedTuple, name::Symbol)
    return name in keys(features) ? getfield(features, name) === true : false
end

function _runtime_group_value(group::NamedTuple, name::Symbol, default)
    return name in keys(group) ? getfield(group, name) : default
end
