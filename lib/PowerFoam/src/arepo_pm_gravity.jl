using Statistics

function poissonkernels_monorepo_paths()
    lib_root = normpath(joinpath(@__DIR__, "..", ".."))
    pkg_path = joinpath(lib_root, "PoissonKernels")
    env_path = joinpath(pkg_path, "test")
    return (
        pkg_path = pkg_path,
        env_path = env_path,
    )
end

function with_temporary_load_path(f::Function, paths)
    old_load_path = copy(LOAD_PATH)
    try
        for path in reverse(paths)
            path in LOAD_PATH || pushfirst!(LOAD_PATH, path)
        end
        return f()
    finally
        empty!(LOAD_PATH)
        append!(LOAD_PATH, old_load_path)
    end
end

function probe_poissonkernels_monorepo()
    paths = poissonkernels_monorepo_paths()
    scoped_paths = filter(isdir, (paths.env_path, paths.pkg_path))
    return with_temporary_load_path(scoped_paths) do
        pkg_entry = Base.find_package("PoissonKernels")
        fftw_entry = Base.find_package("FFTW")
        ka_entry = Base.find_package("KernelAbstractions")
        if pkg_entry === nothing
            return (;
                pm_module = nothing,
                error = ErrorException("PoissonKernels was not resolvable after overlaying the monorepo test env"),
                pkg_entry = pkg_entry,
                fftw_entry = fftw_entry,
                ka_entry = ka_entry,
                scoped_paths = collect(scoped_paths),
            )
        end
        try
            Core.eval(@__MODULE__, :(using PoissonKernels))
            pm_module = Base.invokelatest(() -> getfield(@__MODULE__, :PoissonKernels))
            return (;
                pm_module = pm_module,
                error = nothing,
                pkg_entry = pkg_entry,
                fftw_entry = fftw_entry,
                ka_entry = ka_entry,
                scoped_paths = collect(scoped_paths),
            )
        catch err
            return (;
                pm_module = nothing,
                error = err,
                pkg_entry = pkg_entry,
                fftw_entry = fftw_entry,
                ka_entry = ka_entry,
                scoped_paths = collect(scoped_paths),
            )
        end
    end
end

function arepo_pm_gravity_fixture(; Npm::Int = 16)
    return (
        Npm = Npm,
        boxsize = 1.0,
        ng = 3,
        x = [3.5, 10.5, 6.5, 12.5] ./ Npm,
        y = [5.5, 5.5, 11.5, 9.5] ./ Npm,
        z = [7.5, 7.5, 4.5, 13.5] ./ Npm,
        m = ones(Float64, 4),
        vx = zeros(Float64, 4),
        vy = zeros(Float64, 4),
        vz = zeros(Float64, 4),
    )
end

function periodic_image_sum_accel(x, y, z, m; boxsize::Real = 1.0,
                                  nimg::Int = 1, G::Real = 1.0,
                                  softening::Real = 0.0)
    n = length(x)
    ax = zeros(Float64, n)
    ay = zeros(Float64, n)
    az = zeros(Float64, n)
    eps2 = float(softening)^2
    @inbounds for i in 1:n
        xi = x[i]
        yi = y[i]
        zi = z[i]
        for j in 1:n
            mj = m[j]
            for nx in -nimg:nimg, ny in -nimg:nimg, nz in -nimg:nimg
                if i == j && nx == 0 && ny == 0 && nz == 0
                    continue
                end
                dx = (x[j] - xi) + nx * boxsize
                dy = (y[j] - yi) + ny * boxsize
                dz = (z[j] - zi) + nz * boxsize
                r2 = dx * dx + dy * dy + dz * dz + eps2
                invr = inv(sqrt(r2))
                invr3 = invr * invr * invr
                coeff = float(G) * mj * invr3
                ax[i] += coeff * dx
                ay[i] += coeff * dy
                az[i] += coeff * dz
            end
        end
    end
    return ax, ay, az
end

function periodic_background_subtracted_image_oracle(x, y, z, m;
                                                     boxsize::Real = 1.0,
                                                     nimg::Int = 2,
                                                     previous_nimg::Union{Nothing,Int} = nothing,
                                                     G::Real = 1.0,
                                                     softening::Real = 0.0)
    nimg >= 0 || error("periodic_background_subtracted_image_oracle: nimg must be nonnegative")
    if previous_nimg !== nothing
        0 <= previous_nimg < nimg ||
            error("periodic_background_subtracted_image_oracle: previous_nimg must be smaller than nimg")
    end
    ax, ay, az = periodic_image_sum_accel(x, y, z, m;
                                          boxsize = boxsize, nimg = nimg,
                                          G = G, softening = softening)
    mtot = sum(m)
    mtot > 0 || error("periodic_background_subtracted_image_oracle: total mass must be positive")
    residual = momentum_residual(m, ax, ay, az)
    ax .-= residual.x / mtot
    ay .-= residual.y / mtot
    az .-= residual.z / mtot

    if previous_nimg === nothing
        shell_max_component_change = nothing
        shell_rms_component_change = nothing
    else
        previous = periodic_background_subtracted_image_oracle(
            x, y, z, m; boxsize = boxsize, nimg = previous_nimg,
            previous_nimg = nothing, G = G, softening = softening)
        diffs = vcat(abs.(ax .- previous.ax),
                     abs.(ay .- previous.ay),
                     abs.(az .- previous.az))
        shell_max_component_change = maximum(diffs)
        shell_rms_component_change = sqrt(sum(abs2, diffs) / length(diffs))
    end

    neutralized_residual = momentum_residual(m, ax, ay, az)
    return (
        ax = ax,
        ay = ay,
        az = az,
        raw_net_force = residual,
        neutralized_net_force = neutralized_residual,
        nimg = nimg,
        previous_nimg = previous_nimg,
        shell_max_component_change = shell_max_component_change,
        shell_rms_component_change = shell_rms_component_change,
        max_abs_accel = max_abs_accel(ax, ay, az),
    )
end

function momentum_residual(m, ax, ay, az)
    return (
        x = sum(m .* ax),
        y = sum(m .* ay),
        z = sum(m .* az),
    )
end

function max_abs_accel(ax, ay, az)
    return maximum(sqrt.(ax .* ax .+ ay .* ay .+ az .* az))
end

struct ArepoPMGravityWorkspace
    Npm::Int
    boxsize::Float64
    ng::Int
    greens::Symbol
    cellsize::Float64
    leftedge::NTuple{3,Float64}
    rho_flat::Vector{Float64}
    rhs::Array{Float64,3}
    phi::Array{Float64,3}
    phi_pad::Array{Float64,3}
    gx_active::Array{Float64,3}
    gy_active::Array{Float64,3}
    gz_active::Array{Float64,3}
    gx::Array{Float64,3}
    gy::Array{Float64,3}
    gz::Array{Float64,3}
    ax::Vector{Float64}
    ay::Vector{Float64}
    az::Vector{Float64}
end

struct ArepoPMGravityResult
    workspace::ArepoPMGravityWorkspace
    rho::AbstractArray{Float64,3}
    rhs::AbstractArray{Float64,3}
    phi::AbstractArray{Float64,3}
    gx::AbstractArray{Float64,3}
    gy::AbstractArray{Float64,3}
    gz::AbstractArray{Float64,3}
    ax::Vector{Float64}
    ay::Vector{Float64}
    az::Vector{Float64}
    mass_sum::Float64
    rhs_mean::Float64
    rhs_sum::Float64
    phi_mean::Float64
    net_force::NamedTuple{(:x, :y, :z),NTuple{3,Float64}}
    max_abs_accel::Float64
end

function arepo_pm_gravity_workspace(; fixture = arepo_pm_gravity_fixture(),
                                    particle_count::Integer = length(fixture.x),
                                    greens::Symbol = :spectral)
    Npm = fixture.Npm
    boxsize = fixture.boxsize
    ng = fixture.ng
    cellsize = boxsize / Npm
    leftedge = ntuple(_ -> -ng * cellsize, 3)
    M = Npm + 2ng
    return ArepoPMGravityWorkspace(
        Npm,
        boxsize,
        ng,
        greens,
        cellsize,
        leftedge,
        zeros(Float64, Npm^3),
        zeros(Float64, Npm, Npm, Npm),
        zeros(Float64, Npm, Npm, Npm),
        zeros(Float64, M, M, M),
        zeros(Float64, Npm, Npm, Npm),
        zeros(Float64, Npm, Npm, Npm),
        zeros(Float64, Npm, Npm, Npm),
        zeros(Float64, M, M, M),
        zeros(Float64, M, M, M),
        zeros(Float64, M, M, M),
        zeros(Float64, Int(particle_count)),
        zeros(Float64, Int(particle_count)),
        zeros(Float64, Int(particle_count)),
    )
end

function arepo_pm_gravity!(workspace::ArepoPMGravityWorkspace, pkmod,
                           x, y, z, m, vx, vy, vz)
    pkmod === nothing &&
        error("arepo_pm_gravity!: PoissonKernels module is required for the reusable PM chain")

    cellsize = workspace.cellsize
    leftedge = workspace.leftedge
    ng = workspace.ng
    Npm = workspace.Npm
    rho_flat = workspace.rho_flat
    rhs = workspace.rhs
    phi = workspace.phi
    phi_pad = workspace.phi_pad
    gx_active = workspace.gx_active
    gy_active = workspace.gy_active
    gz_active = workspace.gz_active
    gx = workspace.gx
    gy = workspace.gy
    gz = workspace.gz
    ax = workspace.ax
    ay = workspace.ay
    az = workspace.az

    Base.invokelatest(pkmod.cic_deposit!, rho_flat, x, y, z, vx, vy, vz, m;
                      N = Npm, disp = 0.0, shift = -0.5)
    rho = reshape(rho_flat, Npm, Npm, Npm)
    copyto!(rhs, rho)
    rhs .-= mean(rho)
    Base.invokelatest(pkmod.fft_poisson_root!, phi, rhs; G = 1.0, a = 1.0,
                      boxsize = workspace.boxsize, greens = workspace.greens)

    M = Npm + 2ng
    fill!(phi_pad, 0.0)
    interior = ng + 1:ng + Npm
    phi_pad[interior, interior, interior] .= phi
    Base.invokelatest(pkmod.fill_periodic_ghosts!, phi_pad; ng = ng)

    Base.invokelatest(pkmod.comp_accel!, gx_active, gy_active, gz_active, phi_pad;
                      iflag = 1, start = (ng, ng, ng),
                      del = (cellsize, cellsize, cellsize))
    fill!(gx, 0.0)
    fill!(gy, 0.0)
    fill!(gz, 0.0)
    gx[interior, interior, interior] .= gx_active
    gy[interior, interior, interior] .= gy_active
    gz[interior, interior, interior] .= gz_active
    Base.invokelatest(pkmod.fill_periodic_ghosts!, gx; ng = ng)
    Base.invokelatest(pkmod.fill_periodic_ghosts!, gy; ng = ng)
    Base.invokelatest(pkmod.fill_periodic_ghosts!, gz; ng = ng)

    Base.invokelatest(pkmod.interp_accel_to_particles!, ax, ay, az,
                      x, y, z, vx, vy, vz,
                      gx, gy, gz;
                      dcoef = 0.0,
                      cellsize = cellsize,
                      leftedge = leftedge)

    return ArepoPMGravityResult(
        workspace,
        rho,
        rhs,
        phi,
        gx,
        gy,
        gz,
        ax,
        ay,
        az,
        sum(rho),
        mean(rhs),
        sum(rhs),
        mean(phi),
        momentum_residual(m, ax, ay, az),
        max_abs_accel(ax, ay, az),
    )
end

function arepo_pm_gravity(pkmod; fixture = arepo_pm_gravity_fixture(),
                          greens::Symbol = :spectral)
    workspace = arepo_pm_gravity_workspace(; fixture = fixture, greens = greens)
    return arepo_pm_gravity!(workspace, pkmod,
                             fixture.x, fixture.y, fixture.z, fixture.m,
                             fixture.vx, fixture.vy, fixture.vz)
end

function arepo_pm_gravity_result_rows(result::ArepoPMGravityResult;
                                      reference_mass::Real,
                                      direct_oracle,
                                      direct_nimg::Int = 2)
    residual = result.net_force
    comp_diffs = vcat(abs.(result.ax .- direct_oracle.ax),
                      abs.(result.ay .- direct_oracle.ay),
                      abs.(result.az .- direct_oracle.az))
    rows = NamedTuple[]
    push!(rows, _numeric_row("pm", "mass_sum", result.mass_sum,
                             reference = reference_mass,
                             delta = result.mass_sum - reference_mass))
    push!(rows, _numeric_row("pm", "rhs_mean", result.rhs_mean))
    push!(rows, _numeric_row("pm", "rhs_sum", result.rhs_sum))
    push!(rows, _numeric_row("pm", "phi_mean", result.phi_mean))
    push!(rows, _numeric_row("pm", "net_force_x", residual.x))
    push!(rows, _numeric_row("pm", "net_force_y", residual.y))
    push!(rows, _numeric_row("pm", "net_force_z", residual.z))
    push!(rows, _numeric_row("pm", "max_abs_accel", result.max_abs_accel))
    push!(rows, _numeric_row("pm_vs_direct_oracle",
                             "nimg$(direct_nimg)_max_component_diff",
                             maximum(comp_diffs),
                             note = "PM zero-mode convention compared to finite neutralized image oracle"))
    push!(rows, _numeric_row("pm_vs_direct_oracle",
                             "nimg$(direct_nimg)_rms_component_diff",
                             sqrt(sum(abs2, comp_diffs) / length(comp_diffs)),
                             note = "PM zero-mode convention compared to finite neutralized image oracle"))
    return rows
end

function periodic_cell_center_residual(coords, Npm::Int)
    scaled = coords .* Npm
    return maximum(abs.(scaled .- (round.(scaled .- 0.5) .+ 0.5)))
end

function _numeric_row(category::String, label::String, value;
                      status::String = "ok", reference = nothing,
                      delta = nothing, note::String = "")
    return (
        category = category,
        label = label,
        status = status,
        value = float(value),
        reference = reference === nothing ? nothing : float(reference),
        delta = delta === nothing ? nothing : float(delta),
        note = note,
    )
end

function _status_row(category::String, label::String, status::String; note::String = "")
    return (
        category = category,
        label = label,
        status = status,
        value = nothing,
        reference = nothing,
        delta = nothing,
        note = note,
    )
end

function _poissonkernels_blocker_rows(pk_probe)
    pk_probe === nothing && return NamedTuple[]
    rows = NamedTuple[]
    for (idx, path) in enumerate(get(pk_probe, :scoped_paths, String[]))
        push!(rows, _status_row("blocker", "poissonkernels_load_path_$(idx)", "info";
                                note = path))
    end
    push!(rows, _status_row("blocker", "poissonkernels_package_entry",
                            pk_probe.pkg_entry === nothing ? "blocker" : "info";
                            note = something(pk_probe.pkg_entry, "missing")))
    push!(rows, _status_row("blocker", "fftw_entry",
                            pk_probe.fftw_entry === nothing ? "blocker" : "info";
                            note = something(pk_probe.fftw_entry, "missing")))
    push!(rows, _status_row("blocker", "kernelabstractions_entry",
                            pk_probe.ka_entry === nothing ? "blocker" : "info";
                            note = something(pk_probe.ka_entry, "missing")))
    if pk_probe.error !== nothing
        push!(rows, _status_row("blocker", "poissonkernels_load_error", "blocker";
                                note = sprint(showerror, pk_probe.error)))
    end
    return rows
end

function run_arepo_pm_gravity_preflight(pkmod = nothing;
                                        fixture = arepo_pm_gravity_fixture(),
                                        greens::Symbol = :spectral,
                                        nimg_values = 0:2,
                                        pk_probe = nothing)
    rows = NamedTuple[]
    x = fixture.x
    y = fixture.y
    z = fixture.z
    m = fixture.m
    vx = fixture.vx
    vy = fixture.vy
    vz = fixture.vz
    Npm = fixture.Npm
    boxsize = fixture.boxsize
    ng = fixture.ng
    cellsize = boxsize / Npm
    leftedge = ntuple(_ -> -ng * cellsize, 3)

    push!(rows, _numeric_row("fixture", "particle_count", length(x)))
    push!(rows, _numeric_row("fixture", "npm", Npm))
    push!(rows, _numeric_row("fixture", "ghost_depth", ng))
    push!(rows, _numeric_row("fixture", "cell_center_residual_x",
                             periodic_cell_center_residual(x, Npm)))
    push!(rows, _numeric_row("fixture", "cell_center_residual_y",
                             periodic_cell_center_residual(y, Npm)))
    push!(rows, _numeric_row("fixture", "cell_center_residual_z",
                             periodic_cell_center_residual(z, Npm)))

    direct_diags = NamedTuple[]
    for nimg in nimg_values
        ax, ay, az = periodic_image_sum_accel(x, y, z, m; boxsize = boxsize, nimg = nimg)
        residual = momentum_residual(m, ax, ay, az)
        push!(rows, _numeric_row("direct_diag", "nimg$(nimg)_max_abs_accel",
                                 max_abs_accel(ax, ay, az)))
        push!(rows, _numeric_row("direct_diag", "nimg$(nimg)_net_force_x", residual.x))
        push!(rows, _numeric_row("direct_diag", "nimg$(nimg)_net_force_y", residual.y))
        push!(rows, _numeric_row("direct_diag", "nimg$(nimg)_net_force_z", residual.z))
        push!(direct_diags, (nimg = nimg, ax = ax, ay = ay, az = az))
    end
    nimg_list = collect(nimg_values)
    oracle_nimg = maximum(nimg_list)
    oracle_previous = oracle_nimg > minimum(nimg_list) ? oracle_nimg - 1 : nothing
    direct_oracle = periodic_background_subtracted_image_oracle(
        x, y, z, m; boxsize = boxsize, nimg = oracle_nimg,
        previous_nimg = oracle_previous)
    push!(rows, _status_row("direct_oracle",
                            "periodic_background_subtracted_image_sum",
                            "ok";
                            note = "finite symmetric image oracle with zero net-force projection"))
    push!(rows, _numeric_row("direct_oracle", "nimg", direct_oracle.nimg))
    push!(rows, _numeric_row("direct_oracle", "max_abs_accel",
                             direct_oracle.max_abs_accel))
    push!(rows, _numeric_row("direct_oracle", "net_force_x",
                             direct_oracle.neutralized_net_force.x))
    push!(rows, _numeric_row("direct_oracle", "net_force_y",
                             direct_oracle.neutralized_net_force.y))
    push!(rows, _numeric_row("direct_oracle", "net_force_z",
                             direct_oracle.neutralized_net_force.z))
    if direct_oracle.shell_max_component_change !== nothing
        push!(rows, _numeric_row("direct_oracle",
                                 "shell_max_component_change",
                                 direct_oracle.shell_max_component_change;
                                 note = "difference from one smaller symmetric image stencil"))
        push!(rows, _numeric_row("direct_oracle",
                                 "shell_rms_component_change",
                                 direct_oracle.shell_rms_component_change;
                                 note = "difference from one smaller symmetric image stencil"))
    end

    pm = nothing
    if pkmod === nothing
        push!(rows, _status_row("blocker", "pm_fft_chain", "blocker";
                                note = "PoissonKernels not available on LOAD_PATH for this run"))
        append!(rows, _poissonkernels_blocker_rows(pk_probe))
    else
        pm = arepo_pm_gravity(pkmod; fixture = fixture, greens = greens)
        append!(rows, arepo_pm_gravity_result_rows(pm;
                                                   reference_mass = sum(m),
                                                   direct_oracle = direct_oracle,
                                                   direct_nimg = direct_oracle.nimg))
        one_x = [3.5 / Npm]
        one_y = [5.5 / Npm]
        one_z = [7.5 / Npm]
        one_m = [1.0]
        one_v = [0.0]
        self_workspace = arepo_pm_gravity_workspace(; fixture = fixture,
                                                    particle_count = 1,
                                                    greens = greens)
        self_pm = arepo_pm_gravity!(self_workspace, pkmod,
                                     one_x, one_y, one_z, one_m,
                                     one_v, one_v, one_v)
        push!(rows, _numeric_row("pm_self_control", "one_particle_max_abs_accel",
                                 self_pm.max_abs_accel;
                                 note = "single cell-centered particle in periodic mean-free PM solve"))
    end

    return (
        fixture = fixture,
        rows = rows,
        direct_diags = direct_diags,
        pm = pm,
        direct_oracle = direct_oracle,
        greens = greens,
        leftedge = leftedge,
        cellsize = cellsize,
    )
end
