using Statistics

const PF_NOH2D_DEFAULT_GAMMA = 5 / 3
const PF_SOUNDWAVE2D_DEFAULT_GAMMA = 5 / 3
const PF_GRESHO2D_DEFAULT_GAMMA = 5 / 3

function pf_noh2d_run_tag(n_side::Integer, t_final::Real, riemann::Symbol)
    return replace("N$(n_side)_t$(Float64(t_final))_$(lowercase(String(riemann)))", "." => "p")
end

function pf_noh2d_uniform_points(n_side::Integer; domain_radius::Real = 3.0)
    n_side > 0 || error("n_side must be positive")
    radius = float(domain_radius)
    dx = 2radius / n_side
    points = Matrix{Float64}(undef, n_side * n_side, 2)
    q = 1
    for j in 1:n_side, i in 1:n_side
        points[q, 1] = -radius + (i - 0.5) * dx
        points[q, 2] = -radius + (j - 0.5) * dx
        q += 1
    end
    return points
end

function pf_noh2d_mesh(n_side::Integer; domain_radius::Real = 3.0)
    radius = float(domain_radius)
    points = pf_noh2d_uniform_points(n_side; domain_radius = radius)
    mesh = power_diagram(PowerSites2D(points; domain = ((-radius, radius), (-radius, radius))))
    geom = arepo_mesh_arrays(mesh; T = Float64)
    return (; points, mesh, geom)
end

function pf_noh2d_initial_primitives(mesh::PolygonMesh2D;
                                     rho0::Real = 1.0,
                                     p0::Real = 1e-4,
                                     vrad::Real = 1.0)
    center = cell_centroids(mesh)
    n = size(center, 1)
    rho = fill(float(rho0), n)
    pressure = fill(float(p0), n)
    vx = zeros(Float64, n)
    vy = zeros(Float64, n)
    eps_r = sqrt(eps(Float64))
    @inbounds for i in 1:n
        x = center[i, 1]
        y = center[i, 2]
        r = hypot(x, y)
        if r > eps_r
            vx[i] = -float(vrad) * x / r
            vy[i] = -float(vrad) * y / r
        end
    end
    return (; rho, vx, vy, pressure, center)
end

function pf_noh2d_initial_state(mesh::PolygonMesh2D; gamma::Real = PF_NOH2D_DEFAULT_GAMMA,
                                rho0::Real = 1.0, p0::Real = 1e-4, vrad::Real = 1.0)
    prim = pf_noh2d_initial_primitives(mesh; rho0, p0, vrad)
    state = euler_state_2d(mesh; rho = prim.rho, vx = prim.vx, vy = prim.vy,
                           pressure = prim.pressure, gamma, T = Float64)
    return (; state, prim)
end

function pf_noh2d_stable_dt(state::EulerState2D, geom::ArepoMeshArrays2D;
                            cfl::Real = 0.18, gamma::Real = PF_NOH2D_DEFAULT_GAMMA)
    dx = sqrt(minimum(Array(geom.volume)))
    return float(cfl) * dx / max_signal_speed_2d(state; gamma)
end

function pf_noh2d_radial_bins(mesh::PolygonMesh2D, state::EulerState2D;
                              gamma::Real = PF_NOH2D_DEFAULT_GAMMA,
                              nbins::Integer = 24,
                              analysis_radius::Union{Nothing,Real} = nothing,
                              rho0::Real = 1.0)
    nbins > 0 || error("nbins must be positive")
    prim = conserved_to_primitive_2d(state; gamma)
    center = cell_centroids(mesh)
    volume = cell_areas(mesh)
    radius_limit = analysis_radius === nothing ? minimum((
        abs(mesh.domain[1][1]), abs(mesh.domain[1][2]),
        abs(mesh.domain[2][1]), abs(mesh.domain[2][2]),
    )) : float(analysis_radius)
    edges = collect(range(0.0, radius_limit; length = nbins + 1))
    counts = zeros(Int, nbins)
    volume_sum = zeros(Float64, nbins)
    mass_sum = zeros(Float64, nbins)
    rho_sum = zeros(Float64, nbins)
    pressure_sum = zeros(Float64, nbins)
    vrad_sum = zeros(Float64, nbins)
    @inbounds for i in eachindex(volume)
        x = center[i, 1]
        y = center[i, 2]
        r = hypot(x, y)
        r > radius_limit && continue
        frac = radius_limit == 0 ? 0.0 : clamp(r / radius_limit, 0.0, 1.0 - eps(Float64))
        b = clamp(floor(Int, frac * nbins) + 1, 1, nbins)
        counts[b] += 1
        vol = volume[i]
        rho = prim.rho[i]
        pressure = prim.pressure[i]
        vrad = r > 0 ? (prim.vx[i] * x + prim.vy[i] * y) / r : 0.0
        volume_sum[b] += vol
        mass_sum[b] += rho * vol
        rho_sum[b] += rho * vol
        pressure_sum[b] += pressure * vol
        vrad_sum[b] += vrad * vol
    end
    rows = NamedTuple[]
    shock_proxy = 0.0
    threshold = 2.0 * float(rho0)
    for b in 1:nbins
        vol = volume_sum[b]
        rho_mean = vol > 0 ? rho_sum[b] / vol : NaN
        pressure_mean = vol > 0 ? pressure_sum[b] / vol : NaN
        vrad_mean = vol > 0 ? vrad_sum[b] / vol : NaN
        r_inner = edges[b]
        r_outer = edges[b + 1]
        r_mid = 0.5 * (r_inner + r_outer)
        if isfinite(rho_mean) && rho_mean >= threshold
            shock_proxy = r_outer
        end
        push!(rows, (; bin = b, r_inner, r_outer, r_mid, count = counts[b],
                     volume = vol, mass = mass_sum[b], rho_mean,
                     pressure_mean, vrad_mean))
    end
    return (; rows, shock_radius_proxy = shock_proxy, analysis_radius = radius_limit)
end

function pf_noh2d_metric_row(mesh::PolygonMesh2D, geom::ArepoMeshArrays2D,
                             state::EulerState2D;
                             label::AbstractString,
                             step::Integer,
                             time::Real,
                             gamma::Real = PF_NOH2D_DEFAULT_GAMMA,
                             nbins::Integer = 24,
                             analysis_radius::Union{Nothing,Real} = nothing,
                             rho0::Real = 1.0,
                             initial_mass::Union{Nothing,Real} = nothing,
                             initial_energy::Union{Nothing,Real} = nothing)
    prim = conserved_to_primitive_2d(state; gamma)
    total = total_conserved_2d(state, geom)
    bins = pf_noh2d_radial_bins(mesh, state; gamma, nbins, analysis_radius, rho0)
    mass0 = initial_mass === nothing ? total.mass : float(initial_mass)
    energy0 = initial_energy === nothing ? total.energy : float(initial_energy)
    return (; label = String(label), step, t = float(time),
            mass = total.mass, energy = total.energy, mx = total.mx, my = total.my,
            mass_rel_drift = abs(total.mass - mass0) / max(abs(mass0), eps(Float64)),
            energy_rel_drift = abs(total.energy - energy0) / max(abs(energy0), eps(Float64)),
            rho_min = minimum(prim.rho), rho_max = maximum(prim.rho),
            p_min = minimum(prim.pressure), p_max = maximum(prim.pressure),
            shock_radius_proxy = bins.shock_radius_proxy,
            analytic_shock_radius = float(time) / 3,
            shock_radius_error = bins.shock_radius_proxy - float(time) / 3,
            analysis_radius = bins.analysis_radius)
end

function pf_noh2d_run(; n_side::Integer = 24,
                      t_final::Real = 0.2,
                      nbins::Integer = 24,
                      cfl::Real = 0.18,
                      gamma::Real = PF_NOH2D_DEFAULT_GAMMA,
                      rho0::Real = 1.0,
                      p0::Real = 1e-4,
                      vrad::Real = 1.0,
                      domain_radius::Real = 3.0,
                      riemann::Symbol = :hll,
                      max_steps::Integer = 10_000)
    built = pf_noh2d_mesh(n_side; domain_radius)
    init = pf_noh2d_initial_state(built.mesh; gamma, rho0, p0, vrad)
    state = init.state
    mesh = built.mesh
    geom = built.geom
    total0 = total_conserved_2d(state, geom)
    history = NamedTuple[
        pf_noh2d_metric_row(mesh, geom, state;
                            label = "powerfoam_noh2d", step = 0, time = 0.0,
                            gamma, nbins, analysis_radius = domain_radius,
                            rho0, initial_mass = total0.mass,
                            initial_energy = total0.energy)
    ]
    logs = NamedTuple[]
    time = 0.0
    step = 0
    while time < float(t_final) - 1e-14
        step += 1
        step > max_steps && error("pf_noh2d_run exceeded max_steps=$max_steps")
        dt = min(pf_noh2d_stable_dt(state, geom; cfl, gamma), float(t_final) - time)
        finite_volume_step_2d!(state, geom; dt, gamma, riemann)
        time += dt
        metric = pf_noh2d_metric_row(mesh, geom, state;
                                     label = "powerfoam_noh2d", step, time,
                                     gamma, nbins, analysis_radius = domain_radius,
                                     rho0, initial_mass = total0.mass,
                                     initial_energy = total0.energy)
        push!(history, metric)
        push!(logs, (; step, dt, t = time,
                     mass_rel_drift = metric.mass_rel_drift,
                     energy_rel_drift = metric.energy_rel_drift,
                     rho_min = metric.rho_min, rho_max = metric.rho_max,
                     p_min = metric.p_min, p_max = metric.p_max,
                     shock_radius_proxy = metric.shock_radius_proxy))
        isfinite(metric.rho_min) && isfinite(metric.p_min) || error("encountered non-finite primitive state")
        metric.rho_min > 0 || error("encountered non-positive density")
        metric.p_min > 0 || error("encountered non-positive pressure")
    end
    final_metric = history[end]
    bins = pf_noh2d_radial_bins(mesh, state; gamma, nbins,
                                analysis_radius = domain_radius, rho0)
    numerics_ok = final_metric.mass_rel_drift <= 1e-8 &&
                  final_metric.energy_rel_drift <= 1e-8 &&
                  final_metric.rho_max >= 1.5 * float(rho0) &&
                  final_metric.shock_radius_proxy > 0
    status = numerics_ok ? "calibration-PENDING" : "run-FAIL"
    return (; mesh, geom, state, history, logs, radial_bins = bins.rows,
            final_metric, initial_totals = total0, status, numerics_ok,
            domain_radius = float(domain_radius), gamma = float(gamma),
            rho0 = float(rho0), p0 = float(p0), vrad = float(vrad),
            n_side, t_final = float(t_final), riemann)
end

@inline pf_periodic_cell_id_2d(i, j, nx) = i + nx * (j - 1)
@inline pf_periodic_wrap1(i, n) = i > n ? 1 : i

function pf_cell_face_csr_periodic_2d(ncells::Integer, c1, c2, ::Type{I}) where {I<:Integer}
    counts = zeros(Int, ncells)
    for f in eachindex(c1)
        counts[Int(c1[f])] += 1
        counts[Int(c2[f])] += 1
    end
    offsets = Vector{I}(undef, ncells + 1)
    offsets[1] = one(I)
    for i in 1:ncells
        offsets[i + 1] = offsets[i] + I(counts[i])
    end
    faces = Vector{I}(undef, Int(offsets[end] - one(I)))
    signs = Vector{I}(undef, length(faces))
    cursor = Int.(offsets[1:end-1])
    for f in eachindex(c1)
        i = Int(c1[f])
        p = cursor[i]
        faces[p] = I(f)
        signs[p] = -one(I)
        cursor[i] += 1
        j = Int(c2[f])
        p = cursor[j]
        faces[p] = I(f)
        signs[p] = one(I)
        cursor[j] += 1
    end
    return offsets, faces, signs
end

function pf_cartesian_periodic_mesh_arrays_2d(nx::Integer, ny::Integer;
                                              xlim::Tuple{<:Real,<:Real} = (0.0, 1.0),
                                              ylim::Tuple{<:Real,<:Real} = (0.0, 1.0),
                                              T::Type = Float64,
                                              index_type::Type{<:Integer} = Int32)
    nx > 0 || error("nx must be positive")
    ny > 0 || error("ny must be positive")
    xlo = float(xlim[1])
    xhi = float(xlim[2])
    ylo = float(ylim[1])
    yhi = float(ylim[2])
    dx = (xhi - xlo) / nx
    dy = (yhi - ylo) / ny
    nc = nx * ny
    nf = 2 * nc
    c1 = Vector{index_type}(undef, nf)
    c2 = Vector{index_type}(undef, nf)
    normal_x = zeros(T, nf)
    normal_y = zeros(T, nf)
    area = Vector{T}(undef, nf)
    centers = Matrix{Float64}(undef, nc, 2)
    q = 1
    for j in 1:ny, i in 1:nx
        centers[q, 1] = xlo + (i - 0.5) * dx
        centers[q, 2] = ylo + (j - 0.5) * dy
        q += 1
    end
    f = 1
    for j in 1:ny, i in 1:nx
        id = pf_periodic_cell_id_2d(i, j, nx)
        c1[f] = index_type(id)
        c2[f] = index_type(pf_periodic_cell_id_2d(pf_periodic_wrap1(i + 1, nx), j, nx))
        normal_x[f] = one(T)
        area[f] = T(dy)
        f += 1
        c1[f] = index_type(id)
        c2[f] = index_type(pf_periodic_cell_id_2d(i, pf_periodic_wrap1(j + 1, ny), nx))
        normal_y[f] = one(T)
        area[f] = T(dx)
        f += 1
    end
    offsets, faces, signs = pf_cell_face_csr_periodic_2d(nc, c1, c2, index_type)
    volume = fill(T(dx * dy), nc)
    geom = ArepoMeshArrays2D(c1, c2, offsets, faces, signs, volume, area,
                             normal_x, normal_y, zeros(T, nf), zeros(T, nf))
    return (; geom, centers, dx = float(dx), dy = float(dy),
            xlim = (xlo, xhi), ylim = (ylo, yhi), nx, ny)
end

function pf_soundwave2d_run_tag(nx::Integer, ny::Integer, t_final::Real, riemann::Symbol)
    raw = "Nx$(nx)_Ny$(ny)_t$(Float64(t_final))_$(lowercase(String(riemann)))"
    return replace(raw, "." => "p")
end

function pf_soundwave2d_mesh(nx::Integer, ny::Integer;
                             xlim::Tuple{<:Real,<:Real} = (0.0, 1.0),
                             ylim::Tuple{<:Real,<:Real} = (0.0, 1.0))
    return pf_cartesian_periodic_mesh_arrays_2d(nx, ny; xlim, ylim, T = Float64)
end

function pf_soundwave2d_phase(centers::AbstractMatrix, time::Real;
                              xlim::Tuple{<:Real,<:Real} = (0.0, 1.0),
                              rho0::Real = 1.0,
                              p0::Real = 1.0,
                              gamma::Real = PF_SOUNDWAVE2D_DEFAULT_GAMMA,
                              amplitude::Real = 1e-3,
                              mode::Integer = 1)
    size(centers, 2) == 2 || error("centers must be n x 2")
    mode > 0 || error("mode must be positive")
    xlo = float(xlim[1])
    xhi = float(xlim[2])
    lx = xhi - xlo
    cs = sqrt(float(gamma) * float(p0) / float(rho0))
    k = 2pi * mode / lx
    x = @view centers[:, 1]
    return @. k * (x - xlo) - cs * k * float(time)
end

function pf_soundwave2d_exact_primitives(centers::AbstractMatrix, time::Real;
                                         xlim::Tuple{<:Real,<:Real} = (0.0, 1.0),
                                         rho0::Real = 1.0,
                                         p0::Real = 1.0,
                                         gamma::Real = PF_SOUNDWAVE2D_DEFAULT_GAMMA,
                                         amplitude::Real = 1e-3,
                                         mode::Integer = 1)
    phase = pf_soundwave2d_phase(centers, time; xlim, rho0, p0, gamma, amplitude, mode)
    a = float(amplitude)
    rho_base = float(rho0)
    p_base = float(p0)
    cs = sqrt(float(gamma) * p_base / rho_base)
    drho = @. a * rho_base * sin(phase)
    rho = @. rho_base + drho
    vx = @. a * cs * sin(phase)
    vy = zeros(Float64, size(centers, 1))
    pressure = @. p_base + cs^2 * drho
    return (; rho, vx, vy, pressure, cs)
end

function pf_soundwave2d_initial_state(nx::Integer, ny::Integer;
                                      gamma::Real = PF_SOUNDWAVE2D_DEFAULT_GAMMA,
                                      rho0::Real = 1.0,
                                      p0::Real = 1.0,
                                      amplitude::Real = 1e-3,
                                      mode::Integer = 1,
                                      xlim::Tuple{<:Real,<:Real} = (0.0, 1.0),
                                      ylim::Tuple{<:Real,<:Real} = (0.0, 1.0))
    built = pf_soundwave2d_mesh(nx, ny; xlim, ylim)
    prim = pf_soundwave2d_exact_primitives(built.centers, 0.0;
                                           xlim, rho0, p0, gamma, amplitude, mode)
    D = copy(prim.rho)
    Mx = prim.rho .* prim.vx
    My = prim.rho .* prim.vy
    E = prim.pressure ./ (float(gamma) - 1) .+
        0.5 .* prim.rho .* (prim.vx .* prim.vx .+ prim.vy .* prim.vy)
    state = EulerState2D(D, Mx, My, E)
    return (; state, centers = built.centers, geom = built.geom, dx = built.dx, dy = built.dy,
            xlim = built.xlim, ylim = built.ylim, nx, ny, cs = prim.cs)
end

function pf_soundwave2d_stable_dt(state::EulerState2D, geom::ArepoMeshArrays2D;
                                  cfl::Real = 0.25,
                                  gamma::Real = PF_SOUNDWAVE2D_DEFAULT_GAMMA)
    dx = sqrt(minimum(Array(geom.volume)))
    return float(cfl) * dx / max_signal_speed_2d(state; gamma)
end

function pf_soundwave2d_fourier_mode(centers::AbstractMatrix, values::AbstractVector,
                                     volume::AbstractVector;
                                     xlim::Tuple{<:Real,<:Real} = (0.0, 1.0),
                                     mode::Integer = 1)
    mode > 0 || error("mode must be positive")
    xlo = float(xlim[1])
    xhi = float(xlim[2])
    lx = xhi - xlo
    k = 2pi * mode / lx
    total_volume = sum(volume)
    coeff = zero(ComplexF64)
    @inbounds for i in eachindex(values, volume)
        phase = -k * (centers[i, 1] - xlo)
        coeff += volume[i] * values[i] * cis(phase)
    end
    return (2 / total_volume) * coeff
end

function pf_soundwave2d_metric_row(state::EulerState2D, geom::ArepoMeshArrays2D,
                                   centers::AbstractMatrix;
                                   label::AbstractString,
                                   step::Integer,
                                   time::Real,
                                   gamma::Real = PF_SOUNDWAVE2D_DEFAULT_GAMMA,
                                   rho0::Real = 1.0,
                                   p0::Real = 1.0,
                                   amplitude::Real = 1e-3,
                                   mode::Integer = 1,
                                   xlim::Tuple{<:Real,<:Real} = (0.0, 1.0),
                                   initial_mass::Union{Nothing,Real} = nothing,
                                   initial_energy::Union{Nothing,Real} = nothing)
    prim = conserved_to_primitive_2d(state; gamma)
    total = total_conserved_2d(state, geom)
    exact = pf_soundwave2d_exact_primitives(centers, time; xlim, rho0, p0, gamma, amplitude, mode)
    volume = Array(geom.volume)
    rho_err = prim.rho .- exact.rho
    vx_err = prim.vx .- exact.vx
    pressure_err = prim.pressure .- exact.pressure
    mass0 = initial_mass === nothing ? total.mass : float(initial_mass)
    energy0 = initial_energy === nothing ? total.energy : float(initial_energy)
    rho_mode = pf_soundwave2d_fourier_mode(centers, prim.rho .- float(rho0), volume; xlim, mode)
    rho_ref_mode = pf_soundwave2d_fourier_mode(centers, exact.rho .- float(rho0), volume; xlim, mode)
    mode_scale = max(float(rho0) * abs(float(amplitude)), eps(Float64))
    phase_error = angle(rho_mode / rho_ref_mode)
    return (; label = String(label), step, t = float(time),
            mass = total.mass, energy = total.energy, mx = total.mx, my = total.my,
            mass_rel_drift = abs(total.mass - mass0) / max(abs(mass0), eps(Float64)),
            energy_rel_drift = abs(total.energy - energy0) / max(abs(energy0), eps(Float64)),
            rho_min = minimum(prim.rho), rho_max = maximum(prim.rho),
            p_min = minimum(prim.pressure), p_max = maximum(prim.pressure),
            rho_l1 = sum(abs.(rho_err) .* volume) / sum(volume) / float(rho0),
            rho_l2 = sqrt(sum((rho_err .^ 2) .* volume) / sum(volume)) / float(rho0),
            vx_l1 = sum(abs.(vx_err) .* volume) / sum(volume) / max(exact.cs * float(amplitude), eps(Float64)),
            vx_l2 = sqrt(sum((vx_err .^ 2) .* volume) / sum(volume)) / max(exact.cs * float(amplitude), eps(Float64)),
            pressure_l1 = sum(abs.(pressure_err) .* volume) / sum(volume) / float(p0),
            pressure_l2 = sqrt(sum((pressure_err .^ 2) .* volume) / sum(volume)) / float(p0),
            rho_mode_amp = abs(rho_mode) / mode_scale,
            rho_exact_mode_amp = abs(rho_ref_mode) / mode_scale,
            rho_mode_amp_ratio = abs(rho_mode) / max(abs(rho_ref_mode), eps(Float64)),
            rho_mode_phase_error = phase_error)
end

function pf_soundwave2d_profile_rows(state::EulerState2D, centers::AbstractMatrix;
                                     gamma::Real = PF_SOUNDWAVE2D_DEFAULT_GAMMA,
                                     rho0::Real = 1.0,
                                     p0::Real = 1.0,
                                     amplitude::Real = 1e-3,
                                     mode::Integer = 1,
                                     time::Real = 0.0,
                                     xlim::Tuple{<:Real,<:Real} = (0.0, 1.0))
    prim = conserved_to_primitive_2d(state; gamma)
    exact = pf_soundwave2d_exact_primitives(centers, time; xlim, rho0, p0, gamma, amplitude, mode)
    order = sortperm(view(centers, :, 1))
    rows = NamedTuple[]
    for idx in order
        push!(rows, (; x = centers[idx, 1], y = centers[idx, 2],
                     rho = prim.rho[idx], rho_exact = exact.rho[idx],
                     vx = prim.vx[idx], vx_exact = exact.vx[idx],
                     pressure = prim.pressure[idx], pressure_exact = exact.pressure[idx]))
    end
    return rows
end

function pf_soundwave2d_run(; nx::Integer = 32,
                            ny::Integer = 8,
                            t_final::Real = 0.05,
                            cfl::Real = 0.25,
                            gamma::Real = PF_SOUNDWAVE2D_DEFAULT_GAMMA,
                            rho0::Real = 1.0,
                            p0::Real = 1.0,
                            amplitude::Real = 1e-3,
                            mode::Integer = 1,
                            riemann::Symbol = :hll,
                            max_steps::Integer = 10_000,
                            xlim::Tuple{<:Real,<:Real} = (0.0, 1.0),
                            ylim::Tuple{<:Real,<:Real} = (0.0, 1.0))
    built = pf_soundwave2d_initial_state(nx, ny; gamma, rho0, p0, amplitude, mode, xlim, ylim)
    state = built.state
    geom = built.geom
    centers = built.centers
    total0 = total_conserved_2d(state, geom)
    history = NamedTuple[
        pf_soundwave2d_metric_row(state, geom, centers;
                                  label = "powerfoam_soundwave2d", step = 0, time = 0.0,
                                  gamma, rho0, p0, amplitude, mode, xlim,
                                  initial_mass = total0.mass,
                                  initial_energy = total0.energy)
    ]
    logs = NamedTuple[]
    time = 0.0
    step = 0
    while time < float(t_final) - 1e-14
        step += 1
        step > max_steps && error("pf_soundwave2d_run exceeded max_steps=$max_steps")
        dt = min(pf_soundwave2d_stable_dt(state, geom; cfl, gamma), float(t_final) - time)
        finite_volume_step_2d!(state, geom; dt, gamma, riemann)
        time += dt
        metric = pf_soundwave2d_metric_row(state, geom, centers;
                                           label = "powerfoam_soundwave2d", step, time,
                                           gamma, rho0, p0, amplitude, mode, xlim,
                                           initial_mass = total0.mass,
                                           initial_energy = total0.energy)
        push!(history, metric)
        push!(logs, (; step, dt, t = time,
                     mass_rel_drift = metric.mass_rel_drift,
                     energy_rel_drift = metric.energy_rel_drift,
                     rho_l2 = metric.rho_l2,
                     vx_l2 = metric.vx_l2,
                     pressure_l2 = metric.pressure_l2,
                     rho_mode_amp_ratio = metric.rho_mode_amp_ratio,
                     rho_mode_phase_error = metric.rho_mode_phase_error))
        isfinite(metric.rho_min) && isfinite(metric.p_min) || error("encountered non-finite primitive state")
        metric.rho_min > 0 || error("encountered non-positive density")
        metric.p_min > 0 || error("encountered non-positive pressure")
    end
    final_metric = history[end]
    profile_rows = pf_soundwave2d_profile_rows(state, centers;
                                               gamma, rho0, p0, amplitude, mode,
                                               time = final_metric.t, xlim)
    numerics_ok = final_metric.mass_rel_drift <= 1e-10 &&
                  final_metric.energy_rel_drift <= 5e-4 &&
                  final_metric.rho_l2 <= 0.2 &&
                  final_metric.vx_l2 <= 0.2 &&
                  abs(final_metric.rho_mode_amp_ratio) > 0.1 &&
                  isfinite(final_metric.rho_mode_phase_error)
    status = numerics_ok ? "calibration-PENDING" : "run-FAIL"
    return (; geom, centers, state, history, logs, profile_rows,
            final_metric, initial_totals = total0, status, numerics_ok,
            gamma = float(gamma), rho0 = float(rho0), p0 = float(p0),
            amplitude = float(amplitude), mode, nx, ny, t_final = float(t_final),
            riemann, xlim = (float(xlim[1]), float(xlim[2])),
            ylim = (float(ylim[1]), float(ylim[2])))
end

function pf_gresho2d_run_tag(nx::Integer, ny::Integer, t_final::Real, riemann::Symbol)
    raw = "Nx$(nx)_Ny$(ny)_t$(Float64(t_final))_$(lowercase(String(riemann)))"
    return replace(raw, "." => "p")
end

function pf_gresho2d_mesh(nx::Integer, ny::Integer;
                          xlim::Tuple{<:Real,<:Real} = (0.0, 1.0),
                          ylim::Tuple{<:Real,<:Real} = (0.0, 1.0))
    return pf_cartesian_periodic_mesh_arrays_2d(nx, ny; xlim, ylim, T = Float64)
end

function pf_gresho2d_profile_at_radius(r::Real)
    radius = float(r)
    if radius < 0.2
        return 5.0 * radius, 5.0 + 12.5 * radius^2
    elseif radius < 0.4
        return 2.0 - 5.0 * radius,
               9.0 + 12.5 * radius^2 - 20.0 * radius + 4.0 * log(5.0 * radius)
    else
        return 0.0, 3.0 + 4.0 * log(2.0)
    end
end

function pf_gresho2d_exact_primitives(centers::AbstractMatrix, time::Real;
                                      xlim::Tuple{<:Real,<:Real} = (0.0, 1.0),
                                      ylim::Tuple{<:Real,<:Real} = (0.0, 1.0),
                                      rho0::Real = 1.0,
                                      gamma::Real = PF_GRESHO2D_DEFAULT_GAMMA,
                                      center::Tuple{<:Real,<:Real} = (0.5, 0.5))
    size(centers, 2) == 2 || error("centers must be n x 2")
    _ = time
    _ = xlim
    _ = ylim
    n = size(centers, 1)
    rho = fill(float(rho0), n)
    vx = zeros(Float64, n)
    vy = zeros(Float64, n)
    pressure = zeros(Float64, n)
    cx = float(center[1])
    cy = float(center[2])
    eps_r = sqrt(eps(Float64))
    @inbounds for i in 1:n
        rx = centers[i, 1] - cx
        ry = centers[i, 2] - cy
        radius = hypot(rx, ry)
        vt, p = pf_gresho2d_profile_at_radius(radius)
        pressure[i] = p
        if radius > eps_r
            vx[i] = -vt * ry / radius
            vy[i] = vt * rx / radius
        end
    end
    return (; rho, vx, vy, pressure, gamma = float(gamma))
end

function pf_gresho2d_initial_state(nx::Integer, ny::Integer;
                                   gamma::Real = PF_GRESHO2D_DEFAULT_GAMMA,
                                   rho0::Real = 1.0,
                                   center::Tuple{<:Real,<:Real} = (0.5, 0.5),
                                   xlim::Tuple{<:Real,<:Real} = (0.0, 1.0),
                                   ylim::Tuple{<:Real,<:Real} = (0.0, 1.0))
    built = pf_gresho2d_mesh(nx, ny; xlim, ylim)
    prim = pf_gresho2d_exact_primitives(built.centers, 0.0;
                                        xlim, ylim, rho0, gamma, center)
    rho = Float64.(prim.rho)
    vx = Float64.(prim.vx)
    vy = Float64.(prim.vy)
    pressure = Float64.(prim.pressure)
    state = EulerState2D(copy(rho), rho .* vx, rho .* vy,
                         pressure ./ (gamma - 1) .+
                         0.5 .* rho .* (vx .* vx .+ vy .* vy))
    return (; state, centers = built.centers, geom = built.geom, dx = built.dx,
            dy = built.dy, xlim = built.xlim, ylim = built.ylim, nx, ny,
            center = (float(center[1]), float(center[2])))
end

function pf_gresho2d_stable_dt(state::EulerState2D, geom::ArepoMeshArrays2D;
                               cfl::Real = 0.18,
                               gamma::Real = PF_GRESHO2D_DEFAULT_GAMMA)
    dx = sqrt(minimum(Array(geom.volume)))
    return float(cfl) * dx / max_signal_speed_2d(state; gamma)
end

function pf_gresho2d_tangential_velocity(prim, centers::AbstractMatrix;
                                         center::Tuple{<:Real,<:Real} = (0.5, 0.5))
    size(centers, 2) == 2 || error("centers must be n x 2")
    vt = similar(prim.rho)
    cx = float(center[1])
    cy = float(center[2])
    eps_r = sqrt(eps(Float64))
    @inbounds for i in eachindex(vt)
        rx = centers[i, 1] - cx
        ry = centers[i, 2] - cy
        radius = hypot(rx, ry)
        vt[i] = radius > eps_r ? (-prim.vx[i] * ry + prim.vy[i] * rx) / radius : 0.0
    end
    return vt
end

function pf_gresho2d_profile_rows(state::EulerState2D, geom::ArepoMeshArrays2D,
                                  centers::AbstractMatrix;
                                  gamma::Real = PF_GRESHO2D_DEFAULT_GAMMA,
                                  nbins::Integer = 24,
                                  xlim::Tuple{<:Real,<:Real} = (0.0, 1.0),
                                  ylim::Tuple{<:Real,<:Real} = (0.0, 1.0),
                                  center::Tuple{<:Real,<:Real} = (0.5, 0.5),
                                  analysis_radius::Union{Nothing,Real} = nothing,
                                  rho0::Real = 1.0)
    nbins > 0 || error("nbins must be positive")
    prim = conserved_to_primitive_2d(state; gamma)
    exact = pf_gresho2d_exact_primitives(centers, 0.0; xlim, ylim, rho0, gamma, center)
    volume = Array(geom.volume)
    vt = pf_gresho2d_tangential_velocity(prim, centers; center)
    vt_exact = pf_gresho2d_tangential_velocity(exact, centers; center)
    radius_limit = analysis_radius === nothing ? minimum((
        float(center[1]) - float(xlim[1]),
        float(xlim[2]) - float(center[1]),
        float(center[2]) - float(ylim[1]),
        float(ylim[2]) - float(center[2]),
    )) : float(analysis_radius)
    edges = collect(range(0.0, radius_limit; length = nbins + 1))
    counts = zeros(Int, nbins)
    volume_sum = zeros(Float64, nbins)
    mass_sum = zeros(Float64, nbins)
    rho_sum = zeros(Float64, nbins)
    pressure_sum = zeros(Float64, nbins)
    vt_sum = zeros(Float64, nbins)
    vt_exact_sum = zeros(Float64, nbins)
    rows = NamedTuple[]
    @inbounds for i in eachindex(volume)
        rx = centers[i, 1] - float(center[1])
        ry = centers[i, 2] - float(center[2])
        radius = hypot(rx, ry)
        radius > radius_limit && continue
        frac = radius_limit == 0 ? 0.0 : clamp(radius / radius_limit, 0.0, 1.0 - eps(Float64))
        b = clamp(floor(Int, frac * nbins) + 1, 1, nbins)
        counts[b] += 1
        vol = volume[i]
        volume_sum[b] += vol
        mass_sum[b] += prim.rho[i] * vol
        rho_sum[b] += prim.rho[i] * vol
        pressure_sum[b] += prim.pressure[i] * vol
        vt_sum[b] += vt[i] * vol
        vt_exact_sum[b] += vt_exact[i] * vol
    end
    peak_index = argmax(abs.(vt))
    peak_radius = hypot(centers[peak_index, 1] - float(center[1]),
                        centers[peak_index, 2] - float(center[2]))
    vt_peak_ratio = abs(vt[peak_index])
    for b in 1:nbins
        vol = volume_sum[b]
        rho_mean = vol > 0 ? rho_sum[b] / vol : NaN
        pressure_mean = vol > 0 ? pressure_sum[b] / vol : NaN
        vt_mean = vol > 0 ? vt_sum[b] / vol : NaN
        vt_exact_mean = vol > 0 ? vt_exact_sum[b] / vol : NaN
        r_inner = edges[b]
        r_outer = edges[b + 1]
        r_mid = 0.5 * (r_inner + r_outer)
        _, pressure_exact = pf_gresho2d_profile_at_radius(r_mid)
        push!(rows, (; bin = b, r_inner, r_outer, r_mid, count = counts[b],
                     volume = vol, mass = mass_sum[b], rho_mean, vt_mean,
                     vt_exact_mean, vt_abs_error = abs(vt_mean - vt_exact_mean),
                     pressure_mean, pressure_exact_mean = pressure_exact))
    end
    return (; rows, vt_peak_ratio, vt_peak_radius = peak_radius,
            analysis_radius = radius_limit)
end

function pf_gresho2d_metric_row(state::EulerState2D, geom::ArepoMeshArrays2D,
                                centers::AbstractMatrix;
                                label::AbstractString,
                                step::Integer,
                                time::Real,
                                gamma::Real = PF_GRESHO2D_DEFAULT_GAMMA,
                                nbins::Integer = 24,
                                xlim::Tuple{<:Real,<:Real} = (0.0, 1.0),
                                ylim::Tuple{<:Real,<:Real} = (0.0, 1.0),
                                center::Tuple{<:Real,<:Real} = (0.5, 0.5),
                                rho0::Real = 1.0,
                                initial_mass::Union{Nothing,Real} = nothing,
                                initial_energy::Union{Nothing,Real} = nothing)
    prim = conserved_to_primitive_2d(state; gamma)
    total = total_conserved_2d(state, geom)
    profile = pf_gresho2d_profile_rows(state, geom, centers;
                                       gamma, nbins, xlim, ylim, center, rho0)
    exact = pf_gresho2d_exact_primitives(centers, time; xlim, ylim, rho0, gamma, center)
    volume = Array(geom.volume)
    vt = pf_gresho2d_tangential_velocity(prim, centers; center)
    vt_exact = pf_gresho2d_tangential_velocity(exact, centers; center)
    vt_err = vt .- vt_exact
    mass0 = initial_mass === nothing ? total.mass : float(initial_mass)
    energy0 = initial_energy === nothing ? total.energy : float(initial_energy)
    vt_l1 = sum(abs.(vt_err) .* volume) / sum(volume)
    vt_l2 = sqrt(sum((vt_err .^ 2) .* volume) / sum(volume))
    return (; label = String(label), step, t = float(time),
            mass = total.mass, energy = total.energy, mx = total.mx, my = total.my,
            mass_rel_drift = abs(total.mass - mass0) / max(abs(mass0), eps(Float64)),
            energy_rel_drift = abs(total.energy - energy0) / max(abs(energy0), eps(Float64)),
            rho_min = minimum(prim.rho), rho_max = maximum(prim.rho),
            p_min = minimum(prim.pressure), p_max = maximum(prim.pressure),
            vt_l1, vt_l2,
            vt_peak_ratio = profile.vt_peak_ratio,
            vt_peak_radius = profile.vt_peak_radius,
            vt_peak_radius_error = profile.vt_peak_radius - 0.2,
            analysis_radius = profile.analysis_radius)
end

function pf_gresho2d_run(; nx::Integer = 32,
                         ny::Integer = 32,
                         t_final::Real = 0.02,
                         cfl::Real = 0.18,
                         gamma::Real = PF_GRESHO2D_DEFAULT_GAMMA,
                         rho0::Real = 1.0,
                         center::Tuple{<:Real,<:Real} = (0.5, 0.5),
                         riemann::Symbol = :hll,
                         max_steps::Integer = 10_000,
                         nbins::Integer = 24,
                         xlim::Tuple{<:Real,<:Real} = (0.0, 1.0),
                         ylim::Tuple{<:Real,<:Real} = (0.0, 1.0))
    built = pf_gresho2d_initial_state(nx, ny; gamma, rho0, center, xlim, ylim)
    state = built.state
    geom = built.geom
    centers = built.centers
    total0 = total_conserved_2d(state, geom)
    history = NamedTuple[
        pf_gresho2d_metric_row(state, geom, centers;
                               label = "powerfoam_gresho2d", step = 0, time = 0.0,
                               gamma, nbins, xlim, ylim, center, rho0,
                               initial_mass = total0.mass,
                               initial_energy = total0.energy)
    ]
    logs = NamedTuple[]
    time = 0.0
    step = 0
    while time < float(t_final) - 1e-14
        step += 1
        step > max_steps && error("pf_gresho2d_run exceeded max_steps=$max_steps")
        dt = min(pf_gresho2d_stable_dt(state, geom; cfl, gamma), float(t_final) - time)
        finite_volume_step_2d!(state, geom; dt, gamma, riemann)
        time += dt
        metric = pf_gresho2d_metric_row(state, geom, centers;
                                        label = "powerfoam_gresho2d", step, time,
                                        gamma, nbins, xlim, ylim, center, rho0,
                                        initial_mass = total0.mass,
                                        initial_energy = total0.energy)
        push!(history, metric)
        push!(logs, (; step, dt, t = time,
                     mass_rel_drift = metric.mass_rel_drift,
                     energy_rel_drift = metric.energy_rel_drift,
                     rho_min = metric.rho_min, rho_max = metric.rho_max,
                     p_min = metric.p_min, p_max = metric.p_max,
                     vt_l2 = metric.vt_l2, vt_peak_ratio = metric.vt_peak_ratio))
        isfinite(metric.rho_min) && isfinite(metric.p_min) || error("encountered non-finite primitive state")
        metric.rho_min > 0 || error("encountered non-positive density")
        metric.p_min > 0 || error("encountered non-positive pressure")
    end
    final_metric = history[end]
    profile_rows = pf_gresho2d_profile_rows(state, geom, centers;
                                            gamma, nbins, xlim, ylim, center,
                                            rho0).rows
    numerics_ok = final_metric.mass_rel_drift <= 1e-10 &&
                  final_metric.energy_rel_drift <= 1e-4 &&
                  final_metric.rho_min > 0 &&
                  final_metric.p_min > 0 &&
                  isfinite(final_metric.vt_l1) &&
                  isfinite(final_metric.vt_l2) &&
                  isfinite(final_metric.vt_peak_ratio)
    status = numerics_ok ? "calibration-PENDING" : "run-FAIL"
    return (; geom, centers, state, history, logs, profile_rows,
            final_metric, initial_totals = total0, status, numerics_ok,
            gamma = float(gamma), rho0 = float(rho0),
            center = (float(center[1]), float(center[2])),
            nx, ny, t_final = float(t_final), riemann, nbins,
            xlim = (float(xlim[1]), float(xlim[2])),
            ylim = (float(ylim[1]), float(ylim[2])))
end
