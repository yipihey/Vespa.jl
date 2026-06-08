const LOCAL_PPM_HYDROMETHOD = 10

function _real_parameter(paramfile::AbstractString, name::AbstractString, default::Real)
    rx = Regex("^\\s*" * name * "\\s*=\\s*([-+0-9.eEdD]+)")
    for ln in eachline(paramfile)
        m = match(rx, ln)
        m === nothing || return parse(Float64, replace(m.captures[1], 'd' => 'e', 'D' => 'E'))
    end
    return Float64(default)
end

function _integer_vector_parameter(paramfile::AbstractString, name::AbstractString)
    rx = Regex("^\\s*" * name * "\\s*=\\s*(.*)")
    for ln in eachline(paramfile)
        m = match(rx, split(ln, '#'; limit = 2)[1])
        m === nothing || return parse.(Int, split(strip(m.captures[1])))
    end
    return Int[]
end

function _periodic_root(paramfile::AbstractString)
    left = _integer_vector_parameter(paramfile, "LeftFaceBoundaryCondition")
    right = _integer_vector_parameter(paramfile, "RightFaceBoundaryCondition")
    return length(left) >= 3 && length(right) >= 3 &&
           all(==(3), left[1:3]) && all(==(3), right[1:3])
end

@inline function _local_ppm_flux_slot(field_type::Integer)
    field_type == 0 && return 1 # Density
    field_type == 4 && return 2 # Velocity1 -> x momentum
    field_type == 5 && return 3 # Velocity2 -> y momentum
    field_type == 6 && return 4 # Velocity3 -> z momentum
    field_type == 1 && return 5 # TotalEnergy
    field_type == 2 && return 6 # GasEnergy
    return 0
end

function _local_ppm_flux_plane(frec, slot, dims, ng, dim, m, st, en, g0, dtdx)
    nx, ny, _ = dims
    plane_dims = ntuple(d -> en[d] - st[d] + 1, 3)
    out = Vector{Float64}(undef, prod(plane_dims))
    flux = slot == 0 ? nothing : frec[dim + 1][slot]
    @inbounds for lin in 0:length(out)-1
        rem = lin
        idx = ntuple(3) do d
            offset = rem % plane_dims[d]
            rem ÷= plane_dims[d]
            global_i = st[d] + offset
            d == dim + 1 ? ng + m : ng + (global_i - g0[d] + 1)
        end
        cell = idx[1] + nx * (idx[2] - 1) + nx * ny * (idx[3] - 1)
        out[lin + 1] = flux === nothing ? 0.0 : flux[cell] * dtdx
    end
    return out
end

function _write_local_ppm_fluxes!(h, level, gi, grid, dims, ng, frec, dt, widths)
    g0 = problem_grid_global_start(h, grid)
    field_types = problem_field_types(h, grid)
    nsub = problem_num_subgrids(h, level, gi)
    active = ntuple(d -> dims[d] - 2ng, 3)
    function set_plane(sub, dim, side, m, st, en)
        for (field, field_type) in enumerate(field_types)
            plane = _local_ppm_flux_plane(
                frec, _local_ppm_flux_slot(field_type), dims, ng, dim, m,
                st, en, g0, dt / widths[dim + 1],
            )
            problem_set_subgrid_flux(h, level, gi, sub, field - 1, dim, side, plane)
        end
    end
    for sub in 0:nsub-2, dim in 0:2, side in 0:1
        st, en = problem_subgrid_flux_extent(h, level, gi, sub, dim, side)
        set_plane(sub, dim, side, (st[dim + 1] - g0[dim + 1]) + side + 1, st, en)
    end
    own = nsub - 1
    for dim in 0:2, side in 0:1
        st, en = problem_subgrid_flux_extent(h, level, gi, own, dim, side)
        set_plane(own, dim, side, side == 0 ? 1 : active[dim + 1] + 1, st, en)
    end
    return nothing
end

"""
    local_ppm_hydro(; gamma=1.4, nghost=1, periodic_root=false)

Build the conservative EnzoLib hydro hook for `HydroMethod = 10`. The hook uses
the tuned one-ghost local PPM reconstruction, characteristic tracing, and the
two-shock Riemann solver. Enzo continues to own ghost filling, AMR subcycling,
projection, and coarse-fine refluxing.
"""
function local_ppm_hydro(; gamma::Real = 1.4, nghost::Integer = 1,
                         periodic_root::Bool = false)
    nghost >= 1 || error("HydroMethod=10 requires at least one ghost zone")
    function hydro!(h, level, dt)
        count = session_num_grids_on_level(h, level)
        rank = session_my_rank(h)
        for gi in 0:count-1
            grid = problem_grid_index_on_level(h, level, gi)
            problem_grid_processor(h, grid) == rank || continue
            problem_grid_rank(h, grid) == 3 ||
                error("HydroMethod=10 currently supports 3-D grids only")
            dims = Tuple(problem_grid_dims(h, grid))
            active = ntuple(d -> dims[d] - 2nghost, 3)
            all(>(0), active) ||
                error("HydroMethod=10: grid $grid dimensions $dims are incompatible with nghost=$nghost")
            left, right = problem_grid_edge(h, grid)
            widths = ntuple(d -> (right[d] - left[d]) / active[d], 3)
            all(isapprox(widths[d], widths[1]; rtol = 32eps(Float64)) for d in 2:3) ||
                error("HydroMethod=10 currently requires cubic cells; got widths $widths")

            iD = field_index(h, 0; grid = grid)
            iTE = field_index(h, 1; grid = grid)
            iV1 = field_index(h, 4; grid = grid)
            iV2 = field_index(h, 5; grid = grid)
            iV3 = field_index(h, 6; grid = grid)
            gas_pos = findfirst(==(2), problem_field_types(h, grid))
            iGE = gas_pos === nothing ? nothing : gas_pos - 1

            D = problem_get_field(h, iD, grid)
            v1 = problem_get_field(h, iV1, grid)
            v2 = problem_get_field(h, iV2, grid)
            v3 = problem_get_field(h, iV3, grid)
            S1 = D .* v1
            S2 = D .* v2
            S3 = D .* v3
            Tau = D .* problem_get_field(h, iTE, grid)
            Ge = iGE === nothing ? nothing : D .* problem_get_field(h, iGE, grid)
            frec = ntuple(_ -> ntuple(_ -> zeros(Float64, length(D)), 6), 3)

            PPMKernels.muscl_hancock_step_3d!(
                D, S1, S2, S3, Tau, dims, Int(nghost);
                dt = dt, gamma = gamma, dx = widths[1], ge = Ge, fluxrec = frec,
                recon = :ppm_local, predictor = :trace, riemann = :twoshock,
                face_periodic = periodic_root && level == 0,
            )

            problem_set_field(h, iD, D; grid = grid)
            problem_set_field(h, iV1, S1 ./ D; grid = grid)
            problem_set_field(h, iV2, S2 ./ D; grid = grid)
            problem_set_field(h, iV3, S3 ./ D; grid = grid)
            problem_set_field(h, iTE, Tau ./ D; grid = grid)
            iGE === nothing || problem_set_field(h, iGE, Ge ./ D; grid = grid)
            _write_local_ppm_fluxes!(h, level, gi, grid, dims, Int(nghost), frec, dt, widths)
        end
        return nothing
    end
    return hydro!
end

function local_ppm_engine(paramfile::AbstractString; gravity::Bool = false,
                          cooling::Bool = false, radiation::Bool = false,
                          star_sources::Bool = false, star_formation::Bool = false,
                          cosmology::Bool = false, mhdct::Bool = false)
    mhdct && error("HydroMethod=10 is a hydrodynamics solver and does not support CT-MHD")
    hook = local_ppm_hydro(
        gamma = _real_parameter(paramfile, "Gamma", 5 / 3),
        nghost = _integer_parameter(paramfile, "NumberOfGhostZones", 3),
        periodic_root = _periodic_root(paramfile),
    )
    cfg = engine_from_flags(
        hydro = :julia, gravity = gravity, cooling = cooling, radiation = radiation,
        star_sources = star_sources, star_formation = star_formation,
        cosmology = cosmology, mhdct = false,
        hooks = Dict{Symbol,Function}(:hydro => hook),
    )
    return EngineConfig(
        hydro = cfg.hydro, gravity = cfg.gravity, cooling = cfg.cooling,
        comoving_expansion = cfg.comoving_expansion, mhd_ct = cfg.mhd_ct,
        radiation = cfg.radiation, star_formation = cfg.star_formation,
        star_sources = cfg.star_sources, hooks = cfg.hooks, reflux = true,
    )
end
