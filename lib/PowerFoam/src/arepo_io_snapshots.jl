"""
    ArepoSnapshotLocator

Resolved file-surface description for an AREPO-style snapshot root.
"""
struct ArepoSnapshotLocator
    root::String
    snapshot_index::Int
    layout::Symbol
    resolved_paths::Vector{String}
end

"""
    ArepoSnapshotHeader

Minimal runtime header for gas snapshot compatibility.
"""
struct ArepoSnapshotHeader
    time::Float64
    box_size::Union{Nothing,Float64}
    num_files::Int
    fields_present::Set{Symbol}
end

"""
    ArepoGasSnapshotBlock

Typed gas payload used by the first runtime snapshot surface.
"""
mutable struct ArepoGasSnapshotBlock{TExtras}
    density::Vector{Float64}
    masses::Vector{Float64}
    internal_energy::Vector{Float64}
    velocities::Matrix{Float64}
    volume::Union{Nothing,Vector{Float64}}
    pressure::Union{Nothing,Vector{Float64}}
    center::Union{Nothing,Matrix{Float64}}
    particle_ids::Union{Nothing,Vector{Int}}
    extras::TExtras
end

"""
    ArepoSnapshotData

Typed snapshot runtime record for the first PowerFoam/Arepo.jl IO slice.
"""
struct ArepoSnapshotData{TG<:ArepoGasSnapshotBlock}
    locator::ArepoSnapshotLocator
    header::ArepoSnapshotHeader
    gas::TG
    derived::NamedTuple
end

"""
    ArepoHydroRuntimePayload

Bounded snapshot-to-runtime adapter payload for the current PowerFoam hydro
surface.  The payload preserves the resolved gas arrays, the 2-D and 3-D
conserved projections, and the provenance flags needed to reason about how the
runtime inputs were derived.
"""
struct ArepoHydroRuntimePayload{P}
    locator::ArepoSnapshotLocator
    header::ArepoSnapshotHeader
    source_derived::NamedTuple
    dimensionality::Int
    gamma::Float64
    density::Vector{Float64}
    masses::Vector{Float64}
    volume::Vector{Float64}
    internal_energy::Vector{Float64}
    pressure::Vector{Float64}
    velocities::Matrix{Float64}
    center::Matrix{Float64}
    primitive::P
    conserved_2d::EulerState2D
    conserved_3d::EulerState3D
    conserved::Union{EulerState2D,EulerState3D}
    mass_from_volume::Vector{Float64}
    volume_consistent::Bool
end

"""
    ArepoSnapshotValidation

Validation result for the in-memory snapshot schema.
"""
struct ArepoSnapshotValidation
    valid::Bool
    errors::Vector{String}
    warnings::Vector{String}
end

"""
    ArepoSnapshotIOResult

Result of a file-surface read/write attempt or preflight.
"""
struct ArepoSnapshotIOResult
    ok::Bool
    status::Symbol
    path::String
    backend::Symbol
    messages::Vector{String}
end

@static if Base.find_package("HDF5") !== nothing
    import HDF5
end

arepo_snapshot_hdf5_available() = isdefined(@__MODULE__, :HDF5)

"""
    arepo_snapshot_hdf5_preflight(; project_file = Base.active_project())

Report whether the active project can resolve `HDF5.jl` for snapshot IO and, if
not, name the next dependency action to take.
"""
function arepo_snapshot_hdf5_preflight(; project_file::Union{Nothing,AbstractString} = Base.active_project())
    project_path = project_file === nothing ? nothing : normpath(String(project_file))
    manifest_path = _active_manifest_path(project_path)
    package_path = Base.find_package("HDF5")
    project_has_hdf5 = _project_declares_hdf5(project_path)
    manifest_has_hdf5 = _manifest_declares_hdf5(manifest_path)

    action, command, detail = if package_path !== nothing
        (:ready,
         nothing,
         "HDF5.jl resolves in the active project")
    elseif !project_has_hdf5
        (:add_hdf5,
         "julia --project=lib/PowerFoam -e 'using Pkg; Pkg.add(\"HDF5\")'",
         "lib/PowerFoam/Project.toml does not declare HDF5")
    elseif !manifest_has_hdf5
        (:instantiate_hdf5,
         "julia --project=lib/PowerFoam -e 'using Pkg; Pkg.instantiate()'",
         "lib/PowerFoam/Manifest.toml does not contain an HDF5 resolution")
    else
        (:resolve_hdf5,
         "julia --project=lib/PowerFoam -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'",
         "the project declares HDF5, but Base.find_package(\"HDF5\") still returns nothing")
    end

    return (
        available = package_path !== nothing,
        package_path = package_path,
        project_file = project_path,
        manifest_file = manifest_path,
        project_declares_hdf5 = project_has_hdf5,
        manifest_declares_hdf5 = manifest_has_hdf5,
        action = action,
        command = command,
        detail = detail,
    )
end

snapshot_available_fields(header::ArepoSnapshotHeader) = copy(header.fields_present)

"""
    locate_arepo_snapshot(root, snapshot_index; snapshot_file_base = "snap", extension = ".hdf5", must_exist = false)

Resolve the minimum direct and split AREPO snapshot layouts.
"""
function locate_arepo_snapshot(root::AbstractString, snapshot_index::Integer;
                               snapshot_file_base::AbstractString = "snap",
                               extension::AbstractString = ".hdf5",
                               must_exist::Bool = false)
    idx = Int(snapshot_index)
    idx >= 0 || error("locate_arepo_snapshot: snapshot_index must be >= 0")
    ext = startswith(extension, ".") ? String(extension) : "." * String(extension)
    snap = lpad(string(idx), 3, '0')
    root_path = normpath(String(root))
    direct = isfile(root_path) ? root_path :
             joinpath(root_path, string(snapshot_file_base, "_", snap, ext))
    split = isfile(root_path) ? root_path :
            joinpath(root_path, string(snapshot_file_base, "dir_", snap),
                     string(snapshot_file_base, "_", snap, ".0", ext))

    direct_exists = isfile(direct)
    split_exists = isfile(split)
    layout = if direct_exists && split_exists
        :ambiguous
    elseif direct_exists
        :direct
    elseif split_exists
        :split
    else
        must_exist ? :missing : :planned
    end

    if must_exist && layout == :missing
        error("locate_arepo_snapshot: no snapshot found for index $(idx) under `$(root_path)`")
    end

    return ArepoSnapshotLocator(root_path, idx, layout, [direct, split])
end

"""
    arepo_snapshot_read_preflight(root, snapshot_index; snapshot_file_base = "snap", extension = ".hdf5")

Resolve the read-side snapshot surface without opening HDF5.  This reports the
planned direct or split layout, or a blocker when both layouts exist.
"""
function arepo_snapshot_read_preflight(root::AbstractString, snapshot_index::Integer;
                                       snapshot_file_base::AbstractString = "snap",
                                       extension::AbstractString = ".hdf5")
    locator = locate_arepo_snapshot(root, snapshot_index;
                                    snapshot_file_base = snapshot_file_base,
                                    extension = extension,
                                    must_exist = false)
    messages = String[]

    if locator.layout == :ambiguous
        push!(messages, "snapshot index $(locator.snapshot_index) resolves to both direct and split layouts")
        push!(messages, "direct=$(locator.resolved_paths[1])")
        push!(messages, "split=$(locator.resolved_paths[2])")
        return ArepoSnapshotIOResult(false, :ambiguous_layout, locator.root, :preflight, messages)
    elseif locator.layout == :planned
        push!(messages, "no snapshot files exist yet")
        push!(messages, "direct=$(locator.resolved_paths[1])")
        push!(messages, "split=$(locator.resolved_paths[2])")
        return ArepoSnapshotIOResult(true, :planned, locator.resolved_paths[1], :preflight, messages)
    else
        read_path = _locator_path(locator)
        push!(messages, "resolved layout=$(locator.layout) read_path=$(read_path)")
        push!(messages, "direct=$(locator.resolved_paths[1])")
        push!(messages, "split=$(locator.resolved_paths[2])")
        return ArepoSnapshotIOResult(true, :ready, read_path, :preflight, messages)
    end
end

"""
    derive_arepo_snapshot_volume!(gas)

Fill `gas.volume` from `gas.masses ./ gas.density` when it is missing.
"""
function derive_arepo_snapshot_volume!(gas::ArepoGasSnapshotBlock)
    gas.volume !== nothing && return gas
    density = Float64.(collect(gas.density))
    masses = Float64.(collect(gas.masses))
    any(!isfinite, density) && return gas
    any(x -> x <= 0.0, density) && return gas
    gas.volume = masses ./ density
    return gas
end

"""
    derive_arepo_snapshot_pressure!(gas; gamma = 5/3)

Fill `gas.pressure` from the ideal-gas fallback used by existing proxy analyzers.
"""
function derive_arepo_snapshot_pressure!(gas::ArepoGasSnapshotBlock; gamma::Real = 5 / 3)
    gas.pressure !== nothing && return gas
    density = Float64.(collect(gas.density))
    internal_energy = Float64.(collect(gas.internal_energy))
    gas.pressure = (float(gamma) - 1.0) .* density .* internal_energy
    return gas
end

"""
    resolve_arepo_snapshot_centers!(gas)

Resolve `gas.center` from extras when a compatible coordinates field is present.
"""
function resolve_arepo_snapshot_centers!(gas::ArepoGasSnapshotBlock)
    gas.center !== nothing && return gas
    extras = gas.extras
    for key in (:CenterOfMass, :Coordinates, :center, :coordinates)
        if extras isa NamedTuple && hasproperty(extras, key)
            gas.center = _coerce_matrix_float(getproperty(extras, key), "center")
            return gas
        elseif extras isa AbstractDict && haskey(extras, key)
            gas.center = _coerce_matrix_float(extras[key], "center")
            return gas
        end
    end
    return gas
end

"""
    validate_arepo_snapshot(snapshot)

Validate the first-slice snapshot schema and report errors plus soft blockers.
"""
function validate_arepo_snapshot(snapshot::ArepoSnapshotData)
    errors = String[]
    warnings = String[]

    gas = snapshot.gas
    ncells = length(gas.density)
    _require(ncells > 0, errors, "gas density must contain at least one cell")
    _require(length(gas.masses) == ncells, errors, "gas masses length must match density length")
    _require(length(gas.internal_energy) == ncells,
             errors, "gas internal energy length must match density length")

    vel = gas.velocities
    _require(ndims(vel) == 2, errors, "gas velocities must be a matrix")
    if ndims(vel) == 2
        _require(size(vel, 1) == ncells, errors,
                 "gas velocities row count must match density length")
        _require(1 <= size(vel, 2) <= 3, errors,
                 "gas velocities must have 1 to 3 columns")
    end

    if gas.volume === nothing
        push!(warnings, "gas volume missing; mass/density fallback is available")
    else
        _require(length(gas.volume) == ncells,
                 errors, "gas volume length must match density length")
    end

    if gas.pressure === nothing
        push!(warnings, "gas pressure missing; EOS fallback is available")
    else
        _require(length(gas.pressure) == ncells,
                 errors, "gas pressure length must match density length")
    end

    if gas.center === nothing
        push!(warnings, "gas center missing; coordinates fallback may be available")
    else
        _require(ndims(gas.center) == 2, errors, "gas center must be a matrix")
        if ndims(gas.center) == 2
            _require(size(gas.center, 1) == ncells,
                     errors, "gas center row count must match density length")
            _require(size(gas.center, 2) == size(vel, 2),
                     errors, "gas center column count must match velocity dimensionality")
        end
    end

    if gas.particle_ids !== nothing
        _require(length(gas.particle_ids) == ncells,
                 errors, "gas particle_ids length must match density length")
    end

    any(x -> !isfinite(x) || x <= 0.0, gas.density) &&
        push!(errors, "gas density entries must be finite and positive")
    any(x -> !isfinite(x) || x <= 0.0, gas.masses) &&
        push!(errors, "gas mass entries must be finite and positive")
    any(x -> !isfinite(x), gas.internal_energy) &&
        push!(errors, "gas internal energy entries must be finite")
    any(x -> !isfinite(x), gas.velocities) &&
        push!(errors, "gas velocity entries must be finite")

    _require(isfinite(snapshot.header.time), errors, "header time must be finite")
    _require(snapshot.header.num_files >= 1, errors, "header num_files must be >= 1")
    if snapshot.header.box_size !== nothing
        _require(isfinite(snapshot.header.box_size) && snapshot.header.box_size > 0.0,
                 errors, "header box_size must be finite and positive when present")
    end

    for field in (:density, :masses, :internal_energy, :velocities)
        field in snapshot.header.fields_present ||
            push!(warnings, "header fields_present is missing required marker $(field)")
    end

    return ArepoSnapshotValidation(isempty(errors), errors, warnings)
end

function _snapshot_hydro_component(values::AbstractMatrix, column::Integer,
                                   ::Type{T}) where {T<:AbstractFloat}
    n = size(values, 1)
    if size(values, 2) >= column
        return T.(vec(@view values[:, column]))
    end
    return zeros(T, n)
end

function _snapshot_hydro_velocity_components(values::AbstractMatrix,
                                             ::Type{T}) where {T<:AbstractFloat}
    return (_snapshot_hydro_component(values, 1, T),
            _snapshot_hydro_component(values, 2, T),
            _snapshot_hydro_component(values, 3, T))
end

function _snapshot_hydro_conserved_2d(rho::AbstractVector, vx::AbstractVector,
                                      vy::AbstractVector, pressure::AbstractVector,
                                      gamma::Real, ::Type{T}) where {T<:AbstractFloat}
    r = T.(collect(rho))
    ux = T.(collect(vx))
    uy = T.(collect(vy))
    p = T.(collect(pressure))
    return EulerState2D(copy(r), r .* ux, r .* uy,
                        p ./ T(gamma - 1) .+ T(0.5) .* r .* (ux .* ux .+ uy .* uy))
end

function _snapshot_hydro_conserved_3d(rho::AbstractVector, vx::AbstractVector,
                                      vy::AbstractVector, vz::AbstractVector,
                                      pressure::AbstractVector, gamma::Real,
                                      ::Type{T}) where {T<:AbstractFloat}
    r = T.(collect(rho))
    ux = T.(collect(vx))
    uy = T.(collect(vy))
    uz = T.(collect(vz))
    p = T.(collect(pressure))
    return EulerState3D(copy(r), r .* ux, r .* uy, r .* uz,
                        p ./ T(gamma - 1) .+
                        T(0.5) .* r .* (ux .* ux .+ uy .* uy .+ uz .* uz))
end

function _arepo_snapshot_hydro_payload(snapshot::ArepoSnapshotData;
                                       dimensionality::Integer,
                                       gamma::Real,
                                       T::Type{<:AbstractFloat})
    dim = Int(dimensionality)
    dim in (2, 3) || error("arepo_snapshot_hydro_payload: dimensionality must be 2 or 3")

    gas = snapshot.gas
    if gas.volume === nothing
        derive_arepo_snapshot_volume!(gas)
    end
    if gas.pressure === nothing
        derive_arepo_snapshot_pressure!(gas; gamma = gamma)
    end
    if gas.center === nothing
        resolve_arepo_snapshot_centers!(gas)
    end
    gas.center !== nothing ||
        error("arepo_snapshot_hydro_payload: gas center must be resolved before runtime conversion")

    density = T.(collect(gas.density))
    masses = T.(collect(gas.masses))
    volume = T.(collect(gas.volume))
    internal_energy = T.(collect(gas.internal_energy))
    pressure = T.(collect(gas.pressure))
    velocities = T.(Matrix(gas.velocities))
    center = T.(Matrix(gas.center))

    size(velocities, 2) >= dim ||
        error("arepo_snapshot_hydro_payload: gas velocities need at least $(dim) columns for dimensionality=$(dim)")
    size(center, 2) >= dim ||
        error("arepo_snapshot_hydro_payload: gas centers need at least $(dim) columns for dimensionality=$(dim)")

    vx, vy, vz = _snapshot_hydro_velocity_components(velocities, T)
    conserved_2d = _snapshot_hydro_conserved_2d(density, vx, vy, pressure, gamma, T)
    conserved_3d = _snapshot_hydro_conserved_3d(density, vx, vy, vz, pressure, gamma, T)
    primitive = (; rho = copy(density), vx = copy(vx), vy = copy(vy),
                 vz = copy(vz), pressure = copy(pressure))
    mass_from_volume = density .* volume
    denom = max.(abs.(masses), eps(T))
    volume_consistent = isfinite(maximum(abs.(mass_from_volume .- masses) ./ denom))
    volume_consistent || error("arepo_snapshot_hydro_payload: mass and volume are inconsistent")
    maximum(abs.(mass_from_volume .- masses) ./ denom) <= T(1e-10) ||
        error("arepo_snapshot_hydro_payload: mass and volume consistency exceeds tolerance")

    conserved = dim == 2 ? conserved_2d : conserved_3d
    return ArepoHydroRuntimePayload(snapshot.locator, snapshot.header,
                                    snapshot.derived, dim, float(gamma),
                                    density, masses, volume, internal_energy,
                                    pressure, velocities, center, primitive,
                                    conserved_2d, conserved_3d, conserved,
                                    mass_from_volume, true)
end

"""
    arepo_snapshot_hydro_payload(snapshot; dimensionality=3, gamma=5/3, T=Float64)

Convert an `ArepoSnapshotData` gas block into a bounded runtime payload that
preserves the resolved hydro fields plus both 2-D and 3-D conserved states.
The `conserved` field selects the requested dimensionality.
"""
function arepo_snapshot_hydro_payload(snapshot::ArepoSnapshotData;
                                      dimensionality::Integer = 3,
                                      gamma::Real = 5 / 3,
                                      T::Type{<:AbstractFloat} = Float64)
    return _arepo_snapshot_hydro_payload(snapshot;
                                         dimensionality = dimensionality,
                                         gamma = gamma,
                                         T = T)
end

"""
    arepo_snapshot_hydro_state(snapshot; dimensionality=3, gamma=5/3, T=Float64)

Return the selected conserved hydro state from the bounded snapshot payload.
"""
function arepo_snapshot_hydro_state(snapshot::ArepoSnapshotData;
                                    dimensionality::Integer = 3,
                                    gamma::Real = 5 / 3,
                                    T::Type{<:AbstractFloat} = Float64)
    return arepo_snapshot_hydro_payload(snapshot;
                                        dimensionality = dimensionality,
                                        gamma = gamma,
                                        T = T).conserved
end

arepo_snapshot_hydro_state_2d(snapshot::ArepoSnapshotData; gamma::Real = 5 / 3,
                              T::Type{<:AbstractFloat} = Float64) =
    arepo_snapshot_hydro_payload(snapshot; dimensionality = 2, gamma = gamma, T = T).conserved_2d

arepo_snapshot_hydro_state_3d(snapshot::ArepoSnapshotData; gamma::Real = 5 / 3,
                              T::Type{<:AbstractFloat} = Float64) =
    arepo_snapshot_hydro_payload(snapshot; dimensionality = 3, gamma = gamma, T = T).conserved_3d

"""
    read_arepo_snapshot(source::NamedTuple; root = "memory", snapshot_index = 0, layout = :memory, gamma = 5/3)

Construct a typed snapshot payload from a minimal in-memory schema.
"""
function read_arepo_snapshot(source::NamedTuple; root::AbstractString = "memory",
                             snapshot_index::Integer = 0, layout::Symbol = :memory,
                             gamma::Real = 5 / 3)
    header_src = hasproperty(source, :header) ? getproperty(source, :header) : NamedTuple()
    gas_src = hasproperty(source, :gas) ? getproperty(source, :gas) : source

    gas = ArepoGasSnapshotBlock(
        _coerce_vector_float(_required_field(gas_src, :density), "density"),
        _coerce_vector_float(_required_field(gas_src, :masses), "masses"),
        _coerce_vector_float(_required_field(gas_src, :internal_energy), "internal_energy"),
        _coerce_matrix_float(_required_field(gas_src, :velocities), "velocities"),
        _optional_vector_float(_optional_field(gas_src, :volume), "volume"),
        _optional_vector_float(_optional_field(gas_src, :pressure), "pressure"),
        _optional_matrix_float(_optional_field(gas_src, :center), "center"),
        _optional_vector_int(_optional_field(gas_src, :particle_ids), "particle_ids"),
        _coerce_extras(gas_src, (:density, :masses, :internal_energy, :velocities,
                                 :volume, :pressure, :center, :particle_ids)),
    )

    volume_derived = gas.volume === nothing
    pressure_derived = gas.pressure === nothing
    center_derived = gas.center === nothing
    derive_arepo_snapshot_volume!(gas)
    derive_arepo_snapshot_pressure!(gas; gamma = gamma)
    resolve_arepo_snapshot_centers!(gas)

    fields_present = Set{Symbol}((:density, :masses, :internal_energy, :velocities))
    gas.volume !== nothing && push!(fields_present, :volume)
    gas.pressure !== nothing && push!(fields_present, :pressure)
    gas.center !== nothing && push!(fields_present, :center)
    gas.particle_ids !== nothing && push!(fields_present, :particle_ids)

    for name in propertynames(header_src)
        name == :fields_present || push!(fields_present, name)
    end
    if hasproperty(header_src, :fields_present)
        union!(fields_present, Set(Symbol.(collect(getproperty(header_src, :fields_present)))))
    end

    header = ArepoSnapshotHeader(
        _coerce_float(get(header_src, :time, 0.0), "header.time"),
        _coerce_optional_float(get(header_src, :box_size, nothing), "header.box_size"),
        Int(get(header_src, :num_files, 1)),
        fields_present,
    )

    locator = ArepoSnapshotLocator(String(root), Int(snapshot_index), layout, String[])
    snapshot = ArepoSnapshotData(
        locator,
        header,
        gas,
        (volume_derived = volume_derived,
         pressure_derived = pressure_derived,
         center_derived = center_derived,
         backend = :memory),
    )
    validation = validate_arepo_snapshot(snapshot)
    validation.valid || error("read_arepo_snapshot: invalid in-memory payload: $(join(validation.errors, "; "))")
    return snapshot
end

"""
    read_arepo_snapshot(root, snapshot_index; snapshot_file_base = "snap", gamma = 5/3)

Read a minimal AREPO-style HDF5 snapshot when `HDF5.jl` is available.
Otherwise this function reports that only preflight is available.
"""
function read_arepo_snapshot(root::AbstractString, snapshot_index::Integer;
                             snapshot_file_base::AbstractString = "snap",
                             gamma::Real = 5 / 3)
    locator = locate_arepo_snapshot(root, snapshot_index;
                                    snapshot_file_base = snapshot_file_base,
                                    must_exist = true)
    arepo_snapshot_hdf5_available() ||
        error("read_arepo_snapshot: HDF5.jl is not available in the active project; only in-memory snapshot reads and file-surface preflight are supported")
    locator.layout == :ambiguous &&
        error("read_arepo_snapshot: snapshot index $(snapshot_index) resolves to both direct and split layouts")
    return _read_arepo_snapshot_hdf5(locator; gamma = gamma)
end

"""
    write_arepo_snapshot(path, snapshot; create_parent = false)

Write a minimal AREPO-style HDF5 snapshot when `HDF5.jl` is available. Without
that backend this function still performs file-surface preflight and returns a
machine-readable blocker result.
"""
function write_arepo_snapshot(path::AbstractString, snapshot::ArepoSnapshotData;
                              create_parent::Bool = false)
    validation = validate_arepo_snapshot(snapshot)
    fullpath = normpath(String(path))
    parent = dirname(fullpath)
    messages = String[]
    append!(messages, validation.errors)
    append!(messages, validation.warnings)

    if !isdir(parent)
        if create_parent
            mkpath(parent)
        else
            push!(messages, "parent directory does not exist: $(parent)")
            return ArepoSnapshotIOResult(false, :missing_parent, fullpath, :preflight, messages)
        end
    end

    if !validation.valid
        return ArepoSnapshotIOResult(false, :invalid_snapshot, fullpath, :preflight, messages)
    end

    if !endswith(lowercase(fullpath), ".hdf5")
        push!(messages, "target path does not end in .hdf5")
    end

    if !arepo_snapshot_hdf5_available()
        push!(messages, "HDF5.jl is not available in the active project; snapshot write stopped after preflight")
        return ArepoSnapshotIOResult(true, :preflight_only, fullpath, :preflight, messages)
    end

    _write_arepo_snapshot_hdf5(fullpath, snapshot)
    push!(messages, "wrote minimal Header and PartType0 groups")
    return ArepoSnapshotIOResult(true, :wrote_hdf5, fullpath, :hdf5, messages)
end

read_arepo_snapshot_header(reader, locator::ArepoSnapshotLocator) =
    _read_arepo_snapshot_header_hdf5(reader, locator)

read_arepo_gas_snapshot_block(reader, locator::ArepoSnapshotLocator) =
    _read_arepo_gas_snapshot_block_hdf5(reader, locator)

function _require(condition::Bool, errors::Vector{String}, msg::String)
    condition || push!(errors, msg)
    return nothing
end

function _required_field(src, name::Symbol)
    value = _optional_field(src, name)
    value === nothing && error("read_arepo_snapshot: missing required field $(name)")
    return value
end

function _optional_field(src, name::Symbol)
    if src isa NamedTuple
        return hasproperty(src, name) ? getproperty(src, name) : nothing
    elseif src isa AbstractDict
        return haskey(src, name) ? src[name] : nothing
    else
        return hasproperty(src, name) ? getproperty(src, name) : nothing
    end
end

function _coerce_extras(src, excluded::Tuple)
    names = Symbol[]
    values = Any[]
    if src isa NamedTuple
        for name in propertynames(src)
            name in excluded && continue
            push!(names, name)
            push!(values, getproperty(src, name))
        end
    elseif src isa AbstractDict
        for (name, value) in pairs(src)
            key = Symbol(name)
            key in excluded && continue
            push!(names, key)
            push!(values, value)
        end
    else
        for name in propertynames(src)
            name in excluded && continue
            push!(names, name)
            push!(values, getproperty(src, name))
        end
    end
    return NamedTuple{Tuple(names)}(Tuple(values))
end

_coerce_float(x, label::AbstractString) = x isa Real ? Float64(x) :
    something(tryparse(Float64, string(x)),
              error("$(label) must parse as Float64; got `$(x)`"))

function _coerce_optional_float(x, label::AbstractString)
    x === nothing && return nothing
    return _coerce_float(x, label)
end

function _coerce_vector_float(x, label::AbstractString)
    x isa AbstractVector || error("$(label) must be a vector")
    return Float64.(collect(x))
end

function _optional_vector_float(x, label::AbstractString)
    x === nothing && return nothing
    return _coerce_vector_float(x, label)
end

function _coerce_vector_int(x, label::AbstractString)
    x isa AbstractVector || error("$(label) must be a vector")
    return Int.(collect(x))
end

function _optional_vector_int(x, label::AbstractString)
    x === nothing && return nothing
    return _coerce_vector_int(x, label)
end

function _coerce_matrix_float(x, label::AbstractString)
    x isa AbstractMatrix || error("$(label) must be a matrix")
    return Float64.(Matrix(x))
end

function _optional_matrix_float(x, label::AbstractString)
    x === nothing && return nothing
    return _coerce_matrix_float(x, label)
end

function _hdf5_dataset_optional(group, key::AbstractString)
    haskey(group, key) || return nothing
    return read(group[key])
end

function _locator_path(locator::ArepoSnapshotLocator)
    locator.layout == :split && return locator.resolved_paths[2]
    return locator.resolved_paths[1]
end

if arepo_snapshot_hdf5_available()
    function _read_arepo_snapshot_hdf5(locator::ArepoSnapshotLocator; gamma::Real = 5 / 3)
        path = _locator_path(locator)
        HDF5.h5open(path, "r") do io
            header = read_arepo_snapshot_header(io, locator)
            gas = read_arepo_gas_snapshot_block(io, locator)
            volume_derived = gas.volume === nothing
            pressure_derived = gas.pressure === nothing
            center_derived = gas.center === nothing
            derive_arepo_snapshot_volume!(gas)
            derive_arepo_snapshot_pressure!(gas; gamma = gamma)
            resolve_arepo_snapshot_centers!(gas)
            snapshot = ArepoSnapshotData(
                ArepoSnapshotLocator(locator.root, locator.snapshot_index, locator.layout, [path]),
                header,
                gas,
                (volume_derived = volume_derived,
                 pressure_derived = pressure_derived,
                 center_derived = center_derived,
                 backend = :hdf5),
            )
            validation = validate_arepo_snapshot(snapshot)
            validation.valid || error("read_arepo_snapshot: invalid HDF5 snapshot: $(join(validation.errors, "; "))")
            return snapshot
        end
    end

    function _read_arepo_snapshot_header_hdf5(reader, locator::ArepoSnapshotLocator)
        attrs = HDF5.attributes(reader["Header"])
        time = Float64(read(attrs["Time"]))
        box_size = haskey(attrs, "BoxSize") ? Float64(read(attrs["BoxSize"])) : nothing
        num_files = haskey(attrs, "NumFilesPerSnapshot") ?
                    Int(read(attrs["NumFilesPerSnapshot"])) : 1
        fields_present = Set{Symbol}()
        if haskey(reader, "PartType0")
            for key in keys(reader["PartType0"])
                push!(fields_present, _dataset_field_symbol(String(key)))
            end
        end
        return ArepoSnapshotHeader(time, box_size, num_files, fields_present)
    end

    function _read_arepo_gas_snapshot_block_hdf5(reader, locator::ArepoSnapshotLocator)
        gas = reader["PartType0"]
        return ArepoGasSnapshotBlock(
            Float64.(vec(read(gas["Density"]))),
            Float64.(vec(read(gas["Masses"]))),
            Float64.(vec(read(gas["InternalEnergy"]))),
            Float64.(Matrix(read(gas["Velocities"]))),
            _optional_vector_float(_hdf5_dataset_optional(gas, "Volume"), "Volume"),
            _optional_vector_float(_hdf5_dataset_optional(gas, "Pressure"), "Pressure"),
            _optional_matrix_float(_hdf5_dataset_optional(gas, "CenterOfMass"), "CenterOfMass"),
            _optional_vector_int(_hdf5_dataset_optional(gas, "ParticleIDs"), "ParticleIDs"),
            (Coordinates = _hdf5_dataset_optional(gas, "Coordinates"),),
        )
    end

    function _write_arepo_snapshot_hdf5(path::AbstractString, snapshot::ArepoSnapshotData)
        HDF5.h5open(path, "w") do io
            header = HDF5.create_group(io, "Header")
            attrs = HDF5.attributes(header)
            attrs["Time"] = snapshot.header.time
            snapshot.header.box_size !== nothing && (attrs["BoxSize"] = snapshot.header.box_size)
            attrs["NumFilesPerSnapshot"] = Int32(snapshot.header.num_files)

            gas = HDF5.create_group(io, "PartType0")
            gas["Density"] = Float64.(snapshot.gas.density)
            gas["Masses"] = Float64.(snapshot.gas.masses)
            gas["InternalEnergy"] = Float64.(snapshot.gas.internal_energy)
            gas["Velocities"] = Float64.(snapshot.gas.velocities)
            snapshot.gas.volume !== nothing && (gas["Volume"] = Float64.(snapshot.gas.volume))
            snapshot.gas.pressure !== nothing && (gas["Pressure"] = Float64.(snapshot.gas.pressure))
            snapshot.gas.center !== nothing && (gas["CenterOfMass"] = Float64.(snapshot.gas.center))
            snapshot.gas.particle_ids !== nothing && (gas["ParticleIDs"] = Int.(snapshot.gas.particle_ids))
            hasproperty(snapshot.gas.extras, :Coordinates) &&
                (gas["Coordinates"] = Float64.(getproperty(snapshot.gas.extras, :Coordinates)))
        end
        return path
    end
else
    _read_arepo_snapshot_hdf5(locator::ArepoSnapshotLocator; gamma::Real = 5 / 3) =
        error("read_arepo_snapshot: HDF5.jl backend is unavailable")
    _read_arepo_snapshot_header_hdf5(reader, locator::ArepoSnapshotLocator) =
        error("read_arepo_snapshot_header: HDF5.jl backend is unavailable")
    _read_arepo_gas_snapshot_block_hdf5(reader, locator::ArepoSnapshotLocator) =
        error("read_arepo_gas_snapshot_block: HDF5.jl backend is unavailable")
    _write_arepo_snapshot_hdf5(path::AbstractString, snapshot::ArepoSnapshotData) =
        error("write_arepo_snapshot: HDF5.jl backend is unavailable")
end

function _dataset_field_symbol(name::AbstractString)
    mapping = Dict(
        "Density" => :density,
        "Masses" => :masses,
        "InternalEnergy" => :internal_energy,
        "Velocities" => :velocities,
        "Volume" => :volume,
        "Pressure" => :pressure,
        "CenterOfMass" => :center,
        "Coordinates" => :coordinates,
        "ParticleIDs" => :particle_ids,
    )
    return get(mapping, String(name), Symbol(lowercase(String(name))))
end

function _active_manifest_path(project_path::Union{Nothing,String})
    project_path === nothing && return nothing
    project_dir = dirname(project_path)
    for name in ("Manifest.toml", "JuliaManifest.toml")
        candidate = joinpath(project_dir, name)
        isfile(candidate) && return candidate
    end
    return nothing
end

function _project_declares_hdf5(project_path::Union{Nothing,String})
    project_path === nothing && return false
    isfile(project_path) || return false
    text = read(project_path, String)
    return occursin(r"(?m)^\s*HDF5\s*=", text)
end

function _manifest_declares_hdf5(manifest_path::Union{Nothing,String})
    manifest_path === nothing && return false
    isfile(manifest_path) || return false
    return occursin("[[deps.HDF5]]", read(manifest_path, String))
end
