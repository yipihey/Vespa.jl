export NonlocalRecombinationTable, load_nonlocal_recombination_table,
       apply_nonlocal_recombination!

"""Device-resident `R(k,z)-R(0,z)` free-electron response coefficients."""
struct NonlocalRecombinationTable{T,A}
    log_negative_density_rate::A
    log_negative_ionization_rate::A
    log_positive_velocity_divergence_rate::A
    log_positive_temperature_rate::A
    local_velocity_divergence_rate::A
    local_response_rate::A
    has_temperature::Bool
    has_local_response::Bool
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
        local_response_rate::Union{Nothing,AbstractMatrix}=nothing,
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
    has_local_response = local_response_rate !== nothing
    local_response_values = has_local_response ? Matrix{Float64}(local_response_rate) :
                            zeros(Float64, nz, 4)
    size(local_response_values) == (nz, 4) || error(
        "local recombination response has size $(size(local_response_values)); expected $((nz, 4))")
    all(isfinite, local_response_values) || error(
        "local recombination response is non-finite")
    local_velocity_values = local_velocity_divergence_rate === nothing ?
                            copy(view(local_response_values, :, 3)) :
                            Float64.(local_velocity_divergence_rate)
    length(local_velocity_values) == nz || error(
        "local velocity-divergence response has length $(length(local_velocity_values)); expected $nz")
    all(isfinite, local_velocity_values) || error(
        "local velocity-divergence response is non-finite")
    local_velocity_device = to_device(
        be, Matrix{T}(permutedims(local_velocity_values)), T)
    local_response_device = to_device(
        be, Matrix{T}(permutedims(local_response_values)), T)
    NonlocalRecombinationTable(
        density_device, ionization_device, velocity_device, temperature_device,
        local_velocity_device, local_response_device,
        has_temperature, has_local_response,
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
    local_names = (
        "local_density_rate_s", "local_ionization_rate_s",
        "local_velocity_divergence_rate_s", "local_temperature_rate_s",
    )
    local_columns = map(name -> findfirst(==(name), header), local_names)
    local_count = count(!isnothing, local_columns)
    local_count in (0, 1, 4) || error(
        "nonlocal recombination table must provide all four local-response columns")
    local_count == 1 && local_columns[3] === nothing && error(
        "a partial local response table may provide only local_velocity_divergence_rate_s")
    has_local_response = all(!isnothing, local_columns)
    rows = NTuple{10,Float64}[]
    for (line_offset, line) in enumerate(lines[2:end])
        line_number = line_offset + 1
        isempty(strip(line)) && continue
        fields = split(strip(line), '\t')
        length(fields) == length(header) || error(
            "nonlocal recombination table row $line_number has the wrong field count")
        base = ntuple(i -> parse(Float64, fields[Int(columns[required[i]])]), 5)
        temperature = temperature_column === nothing ? NaN :
                      parse(Float64, fields[Int(temperature_column)])
        local_values = has_local_response ?
            ntuple(i -> parse(Float64, fields[Int(local_columns[i])]), 4) :
            (0.0, 0.0,
             local_columns[3] === nothing ? 0.0 :
                 parse(Float64, fields[Int(local_columns[3])]),
             0.0)
        push!(rows, (base..., temperature, local_values...))
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
    local_response = fill(NaN, nz, 4)
    populated = falses(nz, nk)
    for row in rows
        iz = zindex[row[1]]; ik = kindex[row[2]]
        populated[iz, ik] && error("duplicate nonlocal recombination table row at z=$(row[1]) k=$(row[2])")
        density[iz, ik] = row[3]
        ionization[iz, ik] = row[4]
        velocity[iz, ik] = row[5]
        temperature[iz, ik] = row[6]
        if isnan(local_response[iz, 1])
            local_response[iz, :] .= row[7:10]
        elseif any(local_response[iz, component] != row[6 + component]
                   for component in 1:4)
            error("local recombination response varies with k at z=$(row[1])")
        end
        populated[iz, ik] = true
    end
    all(populated) || error("nonlocal recombination table has missing rows")
    NonlocalRecombinationTable(be, redshifts, positive_k,
        density, ionization, velocity;
        temperature_rate=temperature_column === nothing ? nothing : temperature,
        local_velocity_divergence_rate=local_response[:, 3],
        local_response_rate=has_local_response ? local_response : nothing,
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
        theta_conversion, hydrogen_reference, hydrogen_complement_reference,
        ::Val{NH},
        ::Val{CH}) where {NH,CH}
    c = @index(Global)
    @inbounds begin
        T = eltype(delta_b)
        q = Int32(c - 1)
        i = q % n; q = q ÷ n
        j = q % n; k = q ÷ n
        center_rho = max(rho[c], eps(T))
        delta_b[c] = (rho[c] - rho_mean) / max(rho_mean, eps(T))
        hden = max(fh * center_rho, eps(T))
        h2_fraction_h = max(H2I[c] / hden, zero(T))
        if CH
            xhii = min(max(hydrogen_reference + HII[c], zero(T)),
                       max(one(T) - h2_fraction_h, zero(T)))
            # Keep the perturbation arithmetic centered. Forming xhii-xe_mean
            # would discard sub-ULP modes when both backgrounds are near one.
            delta_xe[c] = (hydrogen_reference - xe_mean) + HII[c]
        else
            complement = (HII[c] + H2I[c]) / hden
            xhii = NH ? max(one(T) - complement, zero(T)) :
                        max(HII[c] / hden, zero(T))
            delta_xe[c] = xhii - xe_mean
        end
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
        h2_fraction = max(H2I[c] / center_rho, zero(T))
        hii_fraction = CH ? fh * xhii :
                       NH ? max(fh - HII[c] / center_rho - h2_fraction, zero(T)) :
                            max(HII[c] / center_rho, zero(T))
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

@inline function _recombination_local_interp(values, component::Int32,
                                              iz::Int32, zfraction)
    @inbounds muladd(
        zfraction,
        values[component, iz + 1] - values[component, iz],
        values[component, iz],
    )
end

@inline function _recombination_expm1(q)
    T = typeof(q)
    abs(q) < T(1e-3) ?
        q * (one(T) + q * (T(0.5) + q * (T(1) / T(6) + q / T(24)))) :
        exp(q) - one(T)
end

@inline function _recombination_phi(rate, dt_seconds, em1)
    iszero(rate) ? dt_seconds : em1 / rate
end

@kernel function _nonlocal_recombination_response_k!(correction_hat,
        @Const(delta_b_hat), @Const(delta_xe_hat), @Const(theta_hat),
        @Const(log_negative_density_rate),
        @Const(log_negative_ionization_rate),
        @Const(log_positive_velocity_rate),
        @Const(log_positive_temperature_rate), @Const(delta_temperature_hat),
        @Const(local_velocity_rate), @Const(local_response_rate),
        has_temperature::Bool, has_local_response::Bool,
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
            # The output intentionally aliases one input spectrum to avoid a
            # fifth complex grid. Load all four source modes before the store.
            delta_b_value = delta_b_hat[i, j, k]
            delta_xe_value = delta_xe_hat[i, j, k]
            theta_value = theta_hat[i, j, k]
            temperature_value = delta_temperature_hat[i, j, k]
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
            delta_rt = low_scale * exp(_recombination_table_interp(
                log_positive_velocity_rate, ik, iz, kfraction, zfraction))
            rtemp = has_temperature ? low_scale * exp(_recombination_table_interp(
                log_positive_temperature_rate, ik, iz, kfraction, zfraction)) : zero(T)
            q = min(rx * dt_seconds, zero(T))
            em1 = _recombination_expm1(q)
            phi = _recombination_phi(rx, dt_seconds, em1)
            if has_local_response
                local_rb = _recombination_local_interp(
                    local_response_rate, Int32(1), iz, zfraction)
                local_rx = _recombination_local_interp(
                    local_response_rate, Int32(2), iz, zfraction)
                local_rt = _recombination_local_interp(
                    local_response_rate, Int32(3), iz, zfraction)
                local_rtemp = _recombination_local_interp(
                    local_response_rate, Int32(4), iz, zfraction)
                total_rx = local_rx + rx
                local_em1 = _recombination_expm1(min(local_rx * dt_seconds, zero(T)))
                total_em1 = _recombination_expm1(min(total_rx * dt_seconds, zero(T)))
                local_phi = _recombination_phi(
                    local_rx, dt_seconds, local_em1)
                total_phi = _recombination_phi(
                    total_rx, dt_seconds, total_em1)
                differential_decay = em1 + one(T)
                local_source = local_rb * delta_b_value +
                               local_rtemp * temperature_value
                total_source = (local_rb + rb) * delta_b_value +
                               (local_rt + delta_rt) * theta_value +
                               (local_rtemp + rtemp) * temperature_value
                correction_hat[i, j, k] =
                    em1 * delta_xe_value + total_phi * total_source -
                    differential_decay * local_phi * local_source
            else
                source = rb * delta_b_value +
                         (local_rt + delta_rt) * theta_value +
                         rtemp * temperature_value
                correction_hat[i, j, k] = em1 * delta_xe_value +
                                          phi * source
            end
        end
    end
end

@kernel function _apply_nonlocal_recombination_k!(HII, @Const(H2I), @Const(rho),
                                                   @Const(delta_xe), fh,
                                                   hydrogen_reference,
                                                   hydrogen_complement_reference,
                                                   ::Val{NH}, ::Val{CH}) where {NH,CH}
    c = @index(Global)
    @inbounds begin
        T = eltype(HII)
        if CH
            hden = max(fh * rho[c], eps(T))
            xh2 = max(H2I[c], zero(T)) / hden
            lower = -hydrogen_reference
            upper = max(hydrogen_complement_reference - xh2, lower)
            HII[c] = min(max(HII[c] + delta_xe[c], lower), upper)
        else
            cap = max(fh * rho[c] - max(H2I[c], zero(T)), zero(T))
            correction = fh * rho[c] * delta_xe[c]
            HII[c] = min(max(HII[c] + (NH ? -correction : correction), zero(T)), cap)
        end
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
        neutral_hydrogen_storage::Bool=false,
        centered_hydrogen_storage::Bool=false,
        hydrogen_reference::Real=xe_mean,
        hydrogen_complement_reference::Real=1 - hydrogen_reference,
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
        T(0.5) / s.dx, T(theta_over_aH_conversion),
        T(hydrogen_reference), T(hydrogen_complement_reference),
        Val(neutral_hydrogen_storage),
        Val(centered_hydrogen_storage); ndrange=cells)
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
        table.local_velocity_divergence_rate, table.local_response_rate,
        table.has_temperature, table.has_local_response,
        n, table.nk, Int32(iz), T(zfraction),
        T(table.logk_min), T(table.logk_max), T(table.inv_logk_step),
        T(fundamental_k_mpc), T(dt_seconds); ndrange=size(workspace.density_hat))
    KA.synchronize(s.be)
    _radiation_inverse!(workspace.fft_real, workspace.density_hat)
    _apply_nonlocal_recombination_k!(s.be)(HII, H2I, s.U[1],
        workspace.fft_real, T(hydrogen_mass_fraction),
        T(hydrogen_reference), T(hydrogen_complement_reference),
        Val(neutral_hydrogen_storage),
        Val(centered_hydrogen_storage); ndrange=cells)
    KA.synchronize(s.be)
    true
end
