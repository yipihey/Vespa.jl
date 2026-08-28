export NonlocalRecombinationTable, load_nonlocal_recombination_table,
       apply_nonlocal_recombination!

"""Device-resident `R(k,z)-R(0,z)` free-electron response coefficients."""
struct NonlocalRecombinationTable{T,A}
    log_negative_density_rate::A
    log_negative_ionization_rate::A
    log_positive_velocity_divergence_rate::A
    log_positive_temperature_rate::A
    local_velocity_divergence_rate::A
    has_temperature::Bool
    zlog_min::T
    zlog_max::T
    inv_zlog_step::T
    logk_min::T
    logk_max::T
    inv_logk_step::T
    nz::Int32
    nk::Int32
end

function _uniform_log_grid(values, transform, label)
    length(values) >= 3 || error("nonlocal recombination $label grid needs >= 3 points")
    transformed = transform.(Float64.(values))
    all(isfinite, transformed) || error("nonlocal recombination $label grid is non-finite")
    all(diff(transformed) .> 0) || error("nonlocal recombination $label grid is not increasing")
    step = transformed[2] - transformed[1]
    maximum(abs.(diff(transformed) .- step)) <= 1e-8 * max(abs(step), 1.0) ||
        error("nonlocal recombination $label grid must be uniform after its log transform")
    transformed, step
end

function NonlocalRecombinationTable(be, redshifts, positive_k_mpc,
        density_rate::AbstractMatrix, ionization_rate::AbstractMatrix,
        velocity_divergence_rate::AbstractMatrix;
        temperature_rate::Union{Nothing,AbstractMatrix}=nothing,
        local_velocity_divergence_rate::Union{Nothing,AbstractVector}=nothing,
        precision::Type{T}=Float32) where {T}
    T <: AbstractFloat || error("nonlocal recombination precision must be floating point")
    zlog, zstep = _uniform_log_grid(redshifts, log1p, "redshift")
    logk, kstep = _uniform_log_grid(positive_k_mpc, log, "wavenumber")
    nz = length(redshifts); nk = length(positive_k_mpc)
    expected = (nz, nk)
    for (label, values) in (("density", density_rate),
                            ("ionization", ionization_rate),
                            ("velocity-divergence", velocity_divergence_rate))
        size(values) == expected || error(
            "nonlocal recombination $label table has size $(size(values)); expected $expected")
        all(isfinite, values) || error("nonlocal recombination $label table is non-finite")
    end
    # These signs are fixed by the analytic k -> infinity limit. Enforcing them
    # catches cancellation-contaminated low-k tables and the x_e/x_1s sign trap.
    maximum(density_rate) < 0 || error(
        "nonlocal density response must be strictly negative")
    maximum(ionization_rate) < 0 || error(
        "nonlocal ionization response must be strictly negative after k=0 subtraction")
    minimum(velocity_divergence_rate) > 0 || error(
        "nonlocal velocity-divergence response must be strictly positive")
    has_temperature = temperature_rate !== nothing
    temperature_values = has_temperature ? temperature_rate : fill(eps(Float64), expected)
    size(temperature_values) == expected || error(
        "nonlocal recombination temperature table has size $(size(temperature_values)); expected $expected")
    all(isfinite, temperature_values) || error(
        "nonlocal recombination temperature table is non-finite")
    minimum(temperature_values) > 0 || error(
        "nonlocal temperature response must be strictly positive")
    density_device = to_device(be,
        Matrix{T}(log.(-permutedims(density_rate))), T)
    ionization_device = to_device(be,
        Matrix{T}(log.(-permutedims(ionization_rate))), T)
    velocity_device = to_device(be,
        Matrix{T}(log.(permutedims(velocity_divergence_rate))), T)
    temperature_device = to_device(be,
        Matrix{T}(log.(permutedims(temperature_values))), T)
    local_velocity_values = local_velocity_divergence_rate === nothing ?
                            zeros(Float64, nz) :
                            Float64.(local_velocity_divergence_rate)
    length(local_velocity_values) == nz || error(
        "local velocity-divergence response has length $(length(local_velocity_values)); expected $nz")
    all(isfinite, local_velocity_values) || error(
        "local velocity-divergence response is non-finite")
    local_velocity_device = to_device(
        be, Matrix{T}(permutedims(local_velocity_values)), T)
    NonlocalRecombinationTable(
        density_device, ionization_device, velocity_device, temperature_device,
        local_velocity_device, has_temperature,
        T(first(zlog)), T(last(zlog)), T(inv(zstep)),
        T(first(logk)), T(last(logk)), T(inv(kstep)),
        Int32(nz), Int32(nk),
    )
end

"""Load the campaign TSV table and upload its three or four differential responses."""
function load_nonlocal_recombination_table(path::AbstractString, be;
                                           precision::Type{T}=Float32) where {T}
    isempty(strip(path)) && error("nonlocal recombination table path is empty")
    lines = readlines(path)
    isempty(lines) && error("nonlocal recombination table is empty: $path")
    header = split(strip(lines[1]), '\t')
    required = (
        "redshift", "k_mpc", "delta_density_rate_s",
        "delta_ionization_rate_s", "delta_velocity_divergence_rate_s",
    )
    columns = Dict(name => findfirst(==(name), header) for name in required)
    any(isnothing, values(columns)) && error(
        "nonlocal recombination table is missing required columns: $path")
    temperature_column = findfirst(==("delta_temperature_rate_s"), header)
    local_velocity_column = findfirst(==("local_velocity_divergence_rate_s"), header)
    rows = NTuple{7,Float64}[]
    for (line_offset, line) in enumerate(lines[2:end])
        line_number = line_offset + 1
        isempty(strip(line)) && continue
        fields = split(strip(line), '\t')
        length(fields) == length(header) || error(
            "nonlocal recombination table row $line_number has the wrong field count")
        base = ntuple(i -> parse(Float64, fields[Int(columns[required[i]])]), 5)
        temperature = temperature_column === nothing ? NaN :
                      parse(Float64, fields[Int(temperature_column)])
        local_velocity = local_velocity_column === nothing ? 0.0 :
                         parse(Float64, fields[Int(local_velocity_column)])
        push!(rows, (base..., temperature, local_velocity))
    end
    isempty(rows) && error("nonlocal recombination table has no data rows: $path")
    redshifts = sort!(unique([first(row) for row in rows]))
    positive_k = sort!(unique([row[2] for row in rows]))
    nz = length(redshifts); nk = length(positive_k)
    length(rows) == nz * nk || error("nonlocal recombination table is not rectangular")
    zindex = Dict(value => index for (index, value) in enumerate(redshifts))
    kindex = Dict(value => index for (index, value) in enumerate(positive_k))
    density = Matrix{Float64}(undef, nz, nk)
    ionization = similar(density)
    velocity = similar(density)
    temperature = similar(density)
    local_velocity = fill(NaN, nz)
    populated = falses(nz, nk)
    for row in rows
        iz = zindex[row[1]]; ik = kindex[row[2]]
        populated[iz, ik] && error("duplicate nonlocal recombination table row at z=$(row[1]) k=$(row[2])")
        density[iz, ik] = row[3]
        ionization[iz, ik] = row[4]
        velocity[iz, ik] = row[5]
        temperature[iz, ik] = row[6]
        if isnan(local_velocity[iz])
            local_velocity[iz] = row[7]
        elseif row[7] != local_velocity[iz]
            error("local velocity-divergence response varies with k at z=$(row[1])")
        end
        populated[iz, ik] = true
    end
    all(populated) || error("nonlocal recombination table has missing rows")
    NonlocalRecombinationTable(be, redshifts, positive_k,
        density, ionization, velocity;
        temperature_rate=temperature_column === nothing ? nothing : temperature,
        local_velocity_divergence_rate=local_velocity,
        precision=T)
end

@inline function _recombination_lin3(i::Int32, j::Int32, k::Int32, n::Int32)
    ii = mod(i, n); jj = mod(j, n); kk = mod(k, n)
    ii + n * (jj + n * kk) + Int32(1)
end

@kernel function _prepare_nonlocal_recombination_k!(delta_b, delta_xe,
        theta_over_aH, delta_temperature, @Const(rho), @Const(mx), @Const(my),
        @Const(mz), @Const(energy), @Const(bx), @Const(by), @Const(bz),
        @Const(HII), @Const(H2I), rho_mean, xe_mean, fh, gamma,
        velocity_unit2, temperature_reference, n::Int32, inv2dx,
        theta_conversion)
    c = @index(Global)
    @inbounds begin
        T = eltype(delta_b)
        q = Int32(c - 1)
        i = q % n; q = q ÷ n
        j = q % n; k = q ÷ n
        center_rho = max(rho[c], eps(T))
        delta_b[c] = (rho[c] - rho_mean) / max(rho_mean, eps(T))
        delta_xe[c] = HII[c] / max(fh * center_rho, eps(T)) - xe_mean
        onei = Int32(1)
        cxp = _recombination_lin3(i + onei, j, k, n)
        cxm = _recombination_lin3(i - onei, j, k, n)
        cyp = _recombination_lin3(i, j + onei, k, n)
        cym = _recombination_lin3(i, j - onei, k, n)
        czp = _recombination_lin3(i, j, k + onei, n)
        czm = _recombination_lin3(i, j, k - onei, n)
        divv = ((mx[cxp] / max(rho[cxp], eps(T)) - mx[cxm] / max(rho[cxm], eps(T))) +
                (my[cyp] / max(rho[cyp], eps(T)) - my[cym] / max(rho[cym], eps(T))) +
                (mz[czp] / max(rho[czp], eps(T)) - mz[czm] / max(rho[czm], eps(T)))) * inv2dx
        theta_over_aH[c] = divv * theta_conversion
        kinetic = T(0.5) * (mx[c] * mx[c] + my[c] * my[c] + mz[c] * mz[c]) /
                  center_rho
        magnetic = T(0.5) * (bx[c] * bx[c] + by[c] * by[c] + bz[c] * bz[c])
        specific_e = max((energy[c] - kinetic - magnetic) / center_rho, T(1e-30))
        hii_fraction = max(HII[c] / center_rho, zero(T))
        h2_fraction = max(H2I[c] / center_rho, zero(T))
        particles_per_mass_h = max(
            fh + hii_fraction - T(0.5) * h2_fraction + (one(T) - fh) / T(4),
            eps(T),
        )
        temperature = (gamma - one(T)) * specific_e * velocity_unit2 *
                      T(1.6735575e-24) /
                      (T(1.380649e-16) * particles_per_mass_h)
        delta_temperature[c] = temperature / max(temperature_reference, one(T)) - one(T)
    end
end

@inline function _recombination_table_interp(values, ik::Int32, iz::Int32,
                                              kfraction, zfraction)
    @inbounds begin
        lower = muladd(kfraction, values[ik + 1, iz] - values[ik, iz], values[ik, iz])
        upper = muladd(kfraction, values[ik + 1, iz + 1] - values[ik, iz + 1],
                       values[ik, iz + 1])
        muladd(zfraction, upper - lower, lower)
    end
end

@kernel function _nonlocal_recombination_response_k!(correction_hat,
        @Const(delta_b_hat), @Const(delta_xe_hat), @Const(theta_hat),
        @Const(log_negative_density_rate),
        @Const(log_negative_ionization_rate),
        @Const(log_positive_velocity_rate),
        @Const(log_positive_temperature_rate), @Const(delta_temperature_hat),
        @Const(local_velocity_rate),
        has_temperature::Bool,
        n::Int32, nk::Int32, iz::Int32, zfraction, logk_min, logk_max,
        inv_logk_step, fundamental_k_mpc, dt_seconds)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        T = typeof(zfraction)
        ii = Int32(i - 1)
        jj0 = Int32(j - 1); kk0 = Int32(k - 1)
        jj = jj0 <= n ÷ Int32(2) ? jj0 : jj0 - n
        kk = kk0 <= n ÷ Int32(2) ? kk0 : kk0 - n
        mode2 = T(ii * ii + jj * jj + kk * kk)
        if iszero(mode2)
            correction_hat[i, j, k] = zero(eltype(correction_hat))
        else
            kmode = fundamental_k_mpc * sqrt(mode2)
            logk = log(kmode)
            kfraction = zero(T)
            ik = Int32(1)
            low_scale = one(T)
            if logk <= logk_min
                low_scale = (kmode / exp(logk_min))^2
            elseif logk >= logk_max
                ik = nk - Int32(1)
                kfraction = one(T)
            else
                coordinate = (logk - logk_min) * inv_logk_step
                ik = min(unsafe_trunc(Int32, coordinate) + Int32(1),
                         nk - Int32(1))
                kfraction = coordinate - T(ik - Int32(1))
            end
            rb = -low_scale * exp(_recombination_table_interp(
                log_negative_density_rate, ik, iz, kfraction, zfraction))
            rx = -low_scale * exp(_recombination_table_interp(
                log_negative_ionization_rate, ik, iz, kfraction, zfraction))
            local_rt = muladd(
                zfraction,
                local_velocity_rate[1, iz + 1] - local_velocity_rate[1, iz],
                local_velocity_rate[1, iz])
            rt = local_rt + low_scale * exp(_recombination_table_interp(
                log_positive_velocity_rate, ik, iz, kfraction, zfraction))
            rtemp = has_temperature ? low_scale * exp(_recombination_table_interp(
                log_positive_temperature_rate, ik, iz, kfraction, zfraction)) : zero(T)
            q = min(rx * dt_seconds, zero(T))
            em1 = if abs(q) < T(1e-3)
                q * (one(T) + q * (T(0.5) + q * (T(1) / T(6) + q / T(24))))
            else
                exp(q) - one(T)
            end
            phi = abs(rx) > eps(T) ? em1 / rx : dt_seconds
            source = rb * delta_b_hat[i, j, k] + rt * theta_hat[i, j, k] +
                     rtemp * delta_temperature_hat[i, j, k]
            correction_hat[i, j, k] = em1 * delta_xe_hat[i, j, k] + phi * source
        end
    end
end

@kernel function _apply_nonlocal_recombination_k!(HII, @Const(H2I), @Const(rho),
                                                   @Const(delta_xe), fh)
    c = @index(Global)
    @inbounds begin
        T = eltype(HII)
        cap = max(fh * rho[c] - max(H2I[c], zero(T)), zero(T))
        HII[c] = min(max(HII[c] + fh * rho[c] * delta_xe[c], zero(T)), cap)
    end
end

function _nonlocal_redshift_coordinate(table::NonlocalRecombinationTable,
                                       redshift::Real)
    zlog = log1p(float(redshift))
    zlog < table.zlog_min && return nothing
    zlog > table.zlog_max && return nothing
    coordinate = (zlog - table.zlog_min) * table.inv_zlog_step
    iz = min(floor(Int, coordinate) + 1, Int(table.nz) - 1)
    iz, coordinate - (iz - 1)
end

"""
    apply_nonlocal_recombination!(HII, H2I, s, workspace, table; ...)

Apply the exact frozen-coefficient Fourier update for the differential
`R(k,z)-R(0,z)` free-electron response. The existing local chemistry remains
authoritative. `workspace` is the radiation FFT workspace and is reused in a
stage where its real/hat batches are scratch; no full-grid allocation occurs.
"""
function apply_nonlocal_recombination!(HII, H2I, s::MHDState,
        workspace::RadiationWorkspace, table::NonlocalRecombinationTable;
        redshift::Real, xe_mean::Real, dt_seconds::Real,
        fundamental_k_mpc::Real, theta_over_aH_conversion::Real,
        hydrogen_mass_fraction::Real=0.76,
        velocity_unit2::Real=1,
        temperature_reference::Real=1)
    coordinate = _nonlocal_redshift_coordinate(table, redshift)
    coordinate === nothing && return false
    dt_seconds > 0 || return false
    n = Int32(s.dims[1]); cells = ncells(s); T = eltype(s.U[1])
    iz, zfraction = coordinate
    _prepare_nonlocal_recombination_k!(s.be)(
        view(workspace.real_batch, :, :, :, 1),
        view(workspace.real_batch, :, :, :, 2),
        view(workspace.real_batch, :, :, :, 3),
        view(workspace.real_batch, :, :, :, 4),
        s.U[1], s.U[2], s.U[3], s.U[4], s.U[5], s.U[6], s.U[7], s.U[8],
        HII, H2I, T(workspace.rho_mean), T(xe_mean), T(hydrogen_mass_fraction),
        T(s.γ), T(velocity_unit2), T(temperature_reference), n,
        T(0.5) / s.dx, T(theta_over_aH_conversion); ndrange=cells)
    KA.synchronize(s.be)
    _radiation_forward_batch!(workspace.hat_batch, workspace.real_batch)
    _nonlocal_recombination_response_k!(s.be)(
        workspace.density_hat, workspace.density_hat,
        view(workspace.hat_batch, :, :, :, 2),
        view(workspace.hat_batch, :, :, :, 3),
        table.log_negative_density_rate, table.log_negative_ionization_rate,
        table.log_positive_velocity_divergence_rate,
        table.log_positive_temperature_rate,
        view(workspace.hat_batch, :, :, :, 4),
        table.local_velocity_divergence_rate, table.has_temperature,
        n, table.nk, Int32(iz), T(zfraction),
        T(table.logk_min), T(table.logk_max), T(table.inv_logk_step),
        T(fundamental_k_mpc), T(dt_seconds); ndrange=size(workspace.density_hat))
    KA.synchronize(s.be)
    _radiation_inverse!(workspace.fft_real, workspace.density_hat)
    _apply_nonlocal_recombination_k!(s.be)(HII, H2I, s.U[1],
        workspace.fft_real, T(hydrogen_mass_fraction); ndrange=cells)
    KA.synchronize(s.be)
    true
end
