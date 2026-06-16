"""
    ArepoConfigFlags

Dependency-free decoded view of AREPO `Config.sh` feature toggles.
"""
struct ArepoConfigFlags
    enabled::Set{Symbol}
    values::Dict{Symbol,String}
end

ArepoConfigFlags() = ArepoConfigFlags(Set{Symbol}(), Dict{Symbol,String}())

"""
    ArepoParameterSet

Small include-only record pairing raw AREPO parameter text with a normalized
runtime view and decoded config flags.
"""
struct ArepoParameterSet
    raw::NamedTuple
    normalized::NamedTuple
    config_flags::ArepoConfigFlags
end

"""
    ArepoParameterValidation

Validation result for normalized AREPO runtime parameters.
"""
struct ArepoParameterValidation
    valid::Bool
    errors::Vector{String}
    warnings::Vector{String}
end

"""
    read_arepo_param_file(path)

Read `param.txt`-style text and forward to `parse_arepo_param_text`.
"""
read_arepo_param_file(path::AbstractString) = parse_arepo_param_text(read(path, String))

"""
    read_arepo_config_flags(path)

Read `Config.sh`-style text and forward to `parse_arepo_config_text`.
"""
read_arepo_config_flags(path::AbstractString) = parse_arepo_config_text(read(path, String))

"""
    parse_arepo_param_text(text)

Parse simple AREPO `param.txt` content into a raw `NamedTuple` of string values.
Accepted line forms are `Key value` and `Key = value`, with `#`, `%`, and `//`
comments stripped.
"""
function parse_arepo_param_text(text::AbstractString)
    entries = Pair{Symbol,String}[]
    seen = Set{Symbol}()
    for (lineno, raw_line) in enumerate(split(text, '\n'; keepempty = false))
        line = _strip_arepo_comment(raw_line)
        isempty(line) && continue
        key, value = _split_arepo_assignment(line, lineno, "parameter")
        key in seen && error("parse_arepo_param_text: duplicate key $(String(key)) on line $(lineno)")
        push!(seen, key)
        push!(entries, key => value)
    end
    return _named_tuple(entries)
end

"""
    parse_arepo_config_text(text)

Parse simple `Config.sh` feature lines into `ArepoConfigFlags`. Bare tokens are
treated as enabled flags. `NAME=value` entries preserve the raw assigned string
and are considered enabled when `value` is truthy (`1`, `true`, `yes`, `on`).
"""
function parse_arepo_config_text(text::AbstractString)
    enabled = Set{Symbol}()
    values = Dict{Symbol,String}()
    for (lineno, raw_line) in enumerate(split(text, '\n'; keepempty = false))
        line = _strip_arepo_comment(raw_line)
        isempty(line) && continue
        key, value = _split_arepo_config_line(line, lineno)
        if value === nothing
            push!(enabled, key)
            continue
        end
        values[key] = value
        _is_truthy_flag(value) && push!(enabled, key)
    end
    return ArepoConfigFlags(enabled, values)
end

"""
    normalize_arepo_parameters(raw, config_flags = ArepoConfigFlags())

Normalize raw AREPO parameter values into typed runtime groups. Unknown keys are
preserved under `normalized.extras` as strings.
"""
function normalize_arepo_parameters(raw, config_flags::ArepoConfigFlags = ArepoConfigFlags())
    raw_nt = _coerce_raw_parameters(raw)
    raw_dict = Dict{Symbol,String}(name => String(getfield(raw_nt, name)) for name in keys(raw_nt))

    io = (
        init_cond_file = _get_string(raw_dict, :InitCondFile),
        ic_format = _get_int(raw_dict, :ICFormat),
        output_dir = _get_string(raw_dict, :OutputDir),
        snapshot_file_base = _get_string(raw_dict, :SnapshotFileBase),
        snap_format = _get_int(raw_dict, :SnapFormat),
        num_files_per_snapshot = _get_int(raw_dict, :NumFilesPerSnapshot),
        num_files_written_in_parallel = _get_int(raw_dict, :NumFilesWrittenInParallel),
        output_list_on = _get_bool(raw_dict, :OutputListOn),
        output_list_filename = _get_string(raw_dict, :OutputListFilename),
    )

    time = (
        time_begin = _get_float(raw_dict, :TimeBegin),
        time_max = _get_float(raw_dict, :TimeMax),
        time_bet_snapshot = _get_float(raw_dict, :TimeBetSnapshot),
        time_of_first_snapshot = _get_float(raw_dict, :TimeOfFirstSnapshot),
        time_bet_statistics = _get_float(raw_dict, :TimeBetStatistics),
        cpu_time_bet_restart_file = _get_float(raw_dict, :CpuTimeBetRestartFile),
        time_limit_cpu = _get_float(raw_dict, :TimeLimitCPU),
    )

    domain = (
        box_size = _get_float(raw_dict, :BoxSize),
        periodic_boundaries_on = _get_bool(raw_dict, :PeriodicBoundariesOn),
        comoving_integration_on = _get_bool(raw_dict, :ComovingIntegrationOn),
        omega0 = _get_float(raw_dict, :Omega0),
        omega_baryon = _get_float(raw_dict, :OmegaBaryon),
        omega_lambda = _get_float(raw_dict, :OmegaLambda),
        hubble_param = _get_float(raw_dict, :HubbleParam),
    )

    hydro = (
        courant_fac = _get_float(raw_dict, :CourantFac),
        max_size_timestep = _get_float(raw_dict, :MaxSizeTimestep),
        min_size_timestep = _get_float(raw_dict, :MinSizeTimestep),
        type_of_timestep_criterion = _get_string(raw_dict, :TypeOfTimestepCriterion),
        limit_u_below_this_density = _get_float(raw_dict, :LimitUBelowThisDensity),
        limit_u_below_certain_density_to_this_value =
            _get_float(raw_dict, :LimitUBelowCertainDensityToThisValue),
        init_gas_temp = _get_float(raw_dict, :InitGasTemp),
        min_gas_temp = _get_float(raw_dict, :MinGasTemp),
        min_egy_spec = _get_float(raw_dict, :MinEgySpec),
        minimum_density_on_start_up = _get_float(raw_dict, :MinimumDensityOnStartUp),
    )

    mesh = (
        des_num_ngb = _get_int(raw_dict, :DesNumNgb),
        max_num_ngb_deviation = _get_int(raw_dict, :MaxNumNgbDeviation),
        multiple_domains = _get_int(raw_dict, :MultipleDomains),
        top_node_factor = _get_float(raw_dict, :TopNodeFactor),
        active_part_frac_for_new_domain_decomp =
            _get_float(raw_dict, :ActivePartFracForNewDomainDecomp),
        cell_shaping_speed = _get_float(raw_dict, :CellShapingSpeed),
        cell_max_angle_factor = _get_float(raw_dict, :CellMaxAngleFactor),
    )

    gravity = (
        err_tol_int_accuracy = _get_float(raw_dict, :ErrTolIntAccuracy),
        err_tol_theta = _get_float(raw_dict, :ErrTolTheta),
        err_tol_force_acc = _get_float(raw_dict, :ErrTolForceAcc),
        gas_soft_factor = _get_float(raw_dict, :GasSoftFactor),
        gravity_constant_internal = _get_float(raw_dict, :GravityConstantInternal),
        softening_comoving = _collect_indexed_values(raw_dict, :SofteningComovingType, Float64),
        softening_max_phys = _collect_indexed_values(raw_dict, :SofteningMaxPhysType, Float64),
        softening_type_of_part = _collect_indexed_values(raw_dict, :SofteningTypeOfPartType, Int),
    )

    features = (
        twodims = :TWODIMS in config_flags.enabled,
        double_precision = (:DOUBLEPRECISION in config_flags.enabled) ||
                           (_flag_value_is_truthy(config_flags, :DOUBLEPRECISION)),
        input_in_double_precision = :INPUT_IN_DOUBLEPRECISION in config_flags.enabled,
        output_in_double_precision = :OUTPUT_IN_DOUBLEPRECISION in config_flags.enabled,
        have_hdf5 = :HAVE_HDF5 in config_flags.enabled,
        output_center_of_mass = :OUTPUT_CENTER_OF_MASS in config_flags.enabled,
        output_volume = :OUTPUT_VOLUME in config_flags.enabled,
        output_pressure = :OUTPUT_PRESSURE in config_flags.enabled,
        output_vertex_velocity = :OUTPUT_VERTEX_VELOCITY in config_flags.enabled,
        regularize_mesh_cm_drift = :REGULARIZE_MESH_CM_DRIFT in config_flags.enabled,
        regularize_mesh_cm_drift_use_soundspeed =
            :REGULARIZE_MESH_CM_DRIFT_USE_SOUNDSPEED in config_flags.enabled,
        regularize_mesh_face_angle = :REGULARIZE_MESH_FACE_ANGLE in config_flags.enabled,
        force_equal_timesteps = :FORCE_EQUAL_TIMESTEPS in config_flags.enabled,
        tree_based_timesteps = :TREE_BASED_TIMESTEPS in config_flags.enabled,
        local_ppm = :LOCAL_PPM in config_flags.enabled,
        riemann_hll = :RIEMANN_HLL in config_flags.enabled,
        artificial_bulk_viscosity = :ARTIFICIAL_BULK_VISCOSITY in config_flags.enabled,
        shock_following_mesh = :SHOCK_FOLLOWING_MESH in config_flags.enabled,
        shock_following_mesh_gain = _get_config_float(config_flags, :SHOCK_FOLLOWING_MESH_GAIN),
        shock_following_mesh_width = _get_config_float(config_flags, :SHOCK_FOLLOWING_MESH_WIDTH),
        shock_following_mesh_rmin = _get_config_float(config_flags, :SHOCK_FOLLOWING_MESH_RMIN),
    )

    recognized = Set{Symbol}((
        :InitCondFile, :ICFormat, :OutputDir, :SnapshotFileBase, :SnapFormat,
        :NumFilesPerSnapshot, :NumFilesWrittenInParallel, :OutputListOn,
        :OutputListFilename, :TimeBegin, :TimeMax, :TimeBetSnapshot,
        :TimeOfFirstSnapshot, :TimeBetStatistics, :CpuTimeBetRestartFile,
        :TimeLimitCPU, :BoxSize, :PeriodicBoundariesOn, :ComovingIntegrationOn,
        :Omega0, :OmegaBaryon, :OmegaLambda, :HubbleParam, :CourantFac,
        :MaxSizeTimestep, :MinSizeTimestep, :TypeOfTimestepCriterion,
        :LimitUBelowThisDensity, :LimitUBelowCertainDensityToThisValue,
        :InitGasTemp, :MinGasTemp, :MinEgySpec, :MinimumDensityOnStartUp,
        :DesNumNgb, :MaxNumNgbDeviation, :MultipleDomains, :TopNodeFactor,
        :ActivePartFracForNewDomainDecomp, :CellShapingSpeed, :CellMaxAngleFactor,
        :ErrTolIntAccuracy, :ErrTolTheta, :ErrTolForceAcc, :GasSoftFactor,
        :GravityConstantInternal,
    ))
    for prefix in (:SofteningComovingType, :SofteningMaxPhysType, :SofteningTypeOfPartType)
        for idx in 0:5
            push!(recognized, Symbol(string(prefix), idx))
        end
    end

    extra_entries = Pair{Symbol,String}[]
    for name in keys(raw_nt)
        name in recognized && continue
        push!(extra_entries, name => raw_dict[name])
    end

    normalized = (
        io = io,
        time = time,
        domain = domain,
        hydro = hydro,
        mesh = mesh,
        gravity = gravity,
        features = features,
        extras = _named_tuple(extra_entries),
    )
    return ArepoParameterSet(raw_nt, normalized, config_flags)
end

"""
    validate_arepo_parameters(params)

Validate normalized AREPO parameters and return `ArepoParameterValidation`.
"""
function validate_arepo_parameters(params::ArepoParameterSet)
    errors = String[]
    warnings = String[]

    io = params.normalized.io
    time = params.normalized.time
    domain = params.normalized.domain
    hydro = params.normalized.hydro
    mesh = params.normalized.mesh
    gravity = params.normalized.gravity

    _require(io.init_cond_file !== nothing, errors, "InitCondFile is required")
    _require(io.output_dir !== nothing, errors, "OutputDir is required")
    _require(io.snapshot_file_base !== nothing, errors, "SnapshotFileBase is required")
    _require(time.time_begin !== nothing, errors, "TimeBegin is required")
    _require(time.time_max !== nothing, errors, "TimeMax is required")
    _require(domain.box_size !== nothing, errors, "BoxSize is required")
    _require(hydro.courant_fac !== nothing, errors, "CourantFac is required")

    if time.time_begin !== nothing && time.time_max !== nothing
        _require(time.time_max >= time.time_begin, errors,
                 "TimeMax must be >= TimeBegin")
    end

    if io.num_files_per_snapshot !== nothing
        _require(io.num_files_per_snapshot > 0, errors,
                 "NumFilesPerSnapshot must be positive")
    end

    if io.num_files_written_in_parallel !== nothing
        _require(io.num_files_written_in_parallel > 0, errors,
                 "NumFilesWrittenInParallel must be positive")
    end

    if domain.box_size !== nothing
        _require(domain.box_size > 0, errors, "BoxSize must be positive")
    end

    if hydro.courant_fac !== nothing
        _require(hydro.courant_fac > 0, errors, "CourantFac must be positive")
    end

    if hydro.min_size_timestep !== nothing && hydro.max_size_timestep !== nothing
        _require(hydro.min_size_timestep <= hydro.max_size_timestep, errors,
                 "MinSizeTimestep must be <= MaxSizeTimestep")
    end

    if io.output_list_on === true
        _require(io.output_list_filename !== nothing, errors,
                 "OutputListFilename is required when OutputListOn=1")
    elseif io.output_list_on === false && io.output_list_filename !== nothing
        push!(warnings, "OutputListFilename is set while OutputListOn is disabled")
    end

    if domain.comoving_integration_on === true
        _require(domain.omega0 !== nothing, errors,
                 "Omega0 is required when ComovingIntegrationOn=1")
        _require(domain.omega_lambda !== nothing, errors,
                 "OmegaLambda is required when ComovingIntegrationOn=1")
        _require(domain.hubble_param !== nothing, errors,
                 "HubbleParam is required when ComovingIntegrationOn=1")
    end

    if mesh.des_num_ngb !== nothing
        _require(mesh.des_num_ngb > 0, errors, "DesNumNgb must be positive")
    end

    if mesh.max_num_ngb_deviation !== nothing
        _require(mesh.max_num_ngb_deviation >= 0, errors,
                 "MaxNumNgbDeviation must be nonnegative")
    end

    if gravity.softening_type_of_part !== nothing
        any(x -> x === nothing, gravity.softening_type_of_part) &&
            push!(warnings, "SofteningTypeOfPartType0..5 is only partially specified")
    end

    if params.normalized.features.have_hdf5
        push!(warnings, "Config.sh requests HAVE_HDF5, but this slice is parser-only")
    end

    return ArepoParameterValidation(isempty(errors), errors, warnings)
end

function _strip_arepo_comment(line::AbstractString)
    line = strip(String(line))
    isempty(line) && return ""
    for marker in ("//", "#", "%")
        idx = findfirst(marker, line)
        idx === nothing && continue
        start = first(idx)
        start == firstindex(line) && return ""
        line = strip(line[firstindex(line):prevind(line, start)])
    end
    return line
end

function _split_arepo_assignment(line::AbstractString, lineno::Integer, label::AbstractString)
    if occursin('=', line)
        parts = split(line, '='; limit = 2)
        key = strip(parts[1])
        value = strip(parts[2])
    else
        parts = split(line; limit = 2)
        length(parts) == 2 ||
            error("parse_arepo_$(label)_text: expected `Key value` on line $(lineno)")
        key, value = strip.(parts)
    end
    isempty(key) && error("parse_arepo_$(label)_text: empty key on line $(lineno)")
    isempty(value) && error("parse_arepo_$(label)_text: empty value for $(key) on line $(lineno)")
    return Symbol(key), value
end

function _split_arepo_config_line(line::AbstractString, lineno::Integer)
    if occursin('=', line)
        key, value = _split_arepo_assignment(line, lineno, "config")
        return key, value
    end
    token = strip(line)
    isempty(token) && error("parse_arepo_config_text: empty config token on line $(lineno)")
    return Symbol(token), nothing
end

function _named_tuple(entries::Vector{Pair{Symbol,T}}) where {T}
    return (; entries...)
end

_named_tuple(entries::Vector{Pair{Symbol,String}}) = (; entries...)

function _coerce_raw_parameters(raw::NamedTuple)
    converted = Pair{Symbol,String}[]
    for name in keys(raw)
        value = getfield(raw, name)
        push!(converted, name => string(value))
    end
    return _named_tuple(converted)
end

function _coerce_raw_parameters(raw::AbstractDict)
    entries = Pair{Symbol,String}[]
    for key in sort!(collect(keys(raw)); by = x -> string(x))
        push!(entries, Symbol(string(key)) => string(raw[key]))
    end
    return _named_tuple(entries)
end

_coerce_raw_parameters(raw) = error("normalize_arepo_parameters: unsupported raw parameter container $(typeof(raw))")

_get_string(raw::Dict{Symbol,String}, key::Symbol) = get(raw, key, nothing)

function _get_float(raw::Dict{Symbol,String}, key::Symbol)
    value = get(raw, key, nothing)
    value === nothing && return nothing
    try
        return parse(Float64, value)
    catch err
        error("normalize_arepo_parameters: $(String(key)) must parse as Float64; got `$(value)` ($(err))")
    end
end

function _get_int(raw::Dict{Symbol,String}, key::Symbol)
    value = get(raw, key, nothing)
    value === nothing && return nothing
    try
        return parse(Int, value)
    catch err
        error("normalize_arepo_parameters: $(String(key)) must parse as Int; got `$(value)` ($(err))")
    end
end

function _get_bool(raw::Dict{Symbol,String}, key::Symbol)
    value = get(raw, key, nothing)
    value === nothing && return nothing
    lowered = lowercase(strip(value))
    lowered in ("1", "true", "yes", "on") && return true
    lowered in ("0", "false", "no", "off") && return false
    error("normalize_arepo_parameters: $(String(key)) must be boolean-like; got `$(value)`")
end

function _collect_indexed_values(raw::Dict{Symbol,String}, prefix::Symbol, ::Type{T}) where {T}
    found = false
    values = Vector{Union{Nothing,T}}(undef, 6)
    for idx in 0:5
        key = Symbol(string(prefix), idx)
        if haskey(raw, key)
            found = true
            values[idx + 1] = T === Int ? _get_int(raw, key) : _get_float(raw, key)
        else
            values[idx + 1] = nothing
        end
    end
    return found ? Tuple(values) : nothing
end

function _get_config_float(flags::ArepoConfigFlags, key::Symbol)
    value = get(flags.values, key, nothing)
    value === nothing && return nothing
    try
        return parse(Float64, value)
    catch err
        error("normalize_arepo_parameters: config flag $(String(key)) must parse as Float64; got `$(value)` ($(err))")
    end
end

_flag_value_is_truthy(flags::ArepoConfigFlags, key::Symbol) =
    haskey(flags.values, key) && _is_truthy_flag(flags.values[key])

function _is_truthy_flag(value::AbstractString)
    lowercase(strip(value)) in ("1", "true", "yes", "on")
end

function _require(condition::Bool, errors::Vector{String}, message::AbstractString)
    condition || push!(errors, String(message))
    return nothing
end
