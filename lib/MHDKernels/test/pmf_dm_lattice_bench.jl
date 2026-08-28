using MHDKernels
using PoissonKernels
using KernelAbstractions
using ChemistryKernels
using Printf
using Dates
using Serialization

const KA = KernelAbstractions
const BACKEND_NAME = Symbol(lowercase(get(ENV, "MHD_BACKEND", "metal")))
const T = let precision = lowercase(get(ENV, "MHD_PRECISION", "f32"))
    precision in ("f32", "float32") ? Float32 :
    precision in ("f64", "float64") ? Float64 :
    error("MHD_PRECISION must be f32 or f64")
end
BACKEND_NAME !== :cpu && T !== Float32 &&
    error("MHD_PRECISION=f64 is supported only by the CPU reference backend")
const _PMF_ARAD  = 7.5657e-15
const _PMF_SIGT  = 6.6524e-25
const _PMF_CL    = 2.99792458e10
# Keep pressure initialization, drag, photon closure, and chemistry on the same
# Fixsen/Grackle convention used by ChemistryKernels.comp2_cmb.
const _PMF_T_CMB = 2.725
const _PMF_KPC_CM = 3.0856775814913673e21
const _PMF_MPC_CM = 1000 * _PMF_KPC_CM
const _PMF_CHECKPOINT_MAGIC = "MHDKernels PMF checkpoint v1"

function _write_pmf_checkpoint(path::AbstractString, config, runtime, arrays)
    isempty(path) && return nothing
    parent = dirname(path)
    isempty(parent) || mkpath(parent)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        serialize(io, _PMF_CHECKPOINT_MAGIC)
        serialize(io, config)
        serialize(io, runtime)
        serialize(io, length(arrays))
        for (name, device_array) in arrays
            serialize(io, String(name))
            # Stream one field at a time so checkpointing does not retain a
            # second full simulation state in host memory.
            serialize(io, Array(device_array))
        end
        flush(io)
    end
    mv(tmp, path; force=true)
    return filesize(path)
end

function _restore_pmf_checkpoint!(path::AbstractString, expected_config, arrays;
                                  allowed_config_changes=(),
                                  allowed_config_validator=nothing)
    open(path, "r") do io
        deserialize(io) == _PMF_CHECKPOINT_MAGIC ||
            error("unsupported PMF checkpoint format: $path")
        stored_config = deserialize(io)
        runtime = deserialize(io)
        if !isequal(stored_config, expected_config)
            stored_keys = keys(stored_config)
            compatible_legacy = all(k -> haskey(expected_config, k) &&
                                         isequal(getproperty(stored_config, k),
                                                 getproperty(expected_config, k)),
                                    stored_keys)
            # The requested stopping redshift does not enter any update. Permit
            # extending or shortening a run while keeping every evolution
            # parameter in the checkpoint fingerprint strict.
            compatible_endpoint = all(k -> k === :zend ||
                                           (haskey(expected_config, k) &&
                                            isequal(getproperty(stored_config, k),
                                                    getproperty(expected_config, k))),
                                      stored_keys) &&
                                  all(k -> k === :zend || haskey(stored_config, k),
                                      keys(expected_config))
            allowed = Set(Symbol.(allowed_config_changes))
            structurally_compatible_override = !isempty(allowed) &&
                all(k -> k === :zend || k in allowed ||
                         (haskey(expected_config, k) &&
                          isequal(getproperty(stored_config, k),
                                  getproperty(expected_config, k))), stored_keys) &&
                all(k -> k === :zend || k in allowed || haskey(stored_config, k),
                    keys(expected_config)) &&
                all(k -> haskey(stored_config, k) && haskey(expected_config, k), allowed)
            changed = structurally_compatible_override ?
                filter(k -> !isequal(getproperty(stored_config, k),
                                     getproperty(expected_config, k)),
                       collect(allowed)) : Symbol[]
            compatible_override = structurally_compatible_override &&
                (allowed_config_validator === nothing ||
                 allowed_config_validator(stored_config, expected_config,
                                          runtime, changed))
            (compatible_legacy || compatible_endpoint || compatible_override) || error(
                "PMF checkpoint configuration mismatch\n" *
                "stored:   $(stored_config)\nexpected: $(expected_config)")
            if compatible_legacy
                @warn "restoring legacy PMF checkpoint without newer fingerprint fields" missing=filter(
                    k -> !(k in stored_keys), collect(keys(expected_config)))
            elseif compatible_endpoint
                @info "restoring PMF checkpoint with changed stopping redshift" stored_zend=stored_config.zend expected_zend=expected_config.zend
            else
                @warn "restoring PMF checkpoint with explicitly allowed configuration changes" changed
            end
        end
        narrays = deserialize(io)
        narrays == length(arrays) || error(
            "PMF checkpoint array-count mismatch: stored=$narrays expected=$(length(arrays))")
        for (expected_name, device_array) in arrays
            stored_name = deserialize(io)
            stored_name == String(expected_name) || error(
                "PMF checkpoint field mismatch: stored=$stored_name expected=$expected_name")
            host_array = deserialize(io)
            size(host_array) == size(device_array) || error(
                "PMF checkpoint size mismatch for $stored_name: " *
                "stored=$(size(host_array)) expected=$(size(device_array))")
            eltype(host_array) == eltype(device_array) || error(
                "PMF checkpoint type mismatch for $stored_name: " *
                "stored=$(eltype(host_array)) expected=$(eltype(device_array))")
            copyto!(device_array, host_array)
        end
        return runtime
    end
end

function _nonlocal_table_redshift_support(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("nonlocal response table is empty: $path")
    header = split(strip(lines[1]), '\t')
    column = findfirst(==("redshift"), header)
    column === nothing && error("nonlocal response table has no redshift column: $path")
    redshifts = Float64[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        fields = split(strip(line), '\t')
        push!(redshifts, parse(Float64, fields[column]))
    end
    isempty(redshifts) && error("nonlocal response table has no data: $path")
    extrema(redshifts)
end

function _checkpoint_nonlocal_table_change_is_safe(stored_config, expected_config,
                                                    runtime, changed)
    :chem_nonlocal_table in changed || return true
    checkpoint_redshift = a_to_z(runtime.a)
    stored_support = _nonlocal_table_redshift_support(stored_config.chem_nonlocal_table)
    expected_support = _nonlocal_table_redshift_support(expected_config.chem_nonlocal_table)
    outside(support) = checkpoint_redshift < support[1] || checkpoint_redshift > support[2]
    outside(stored_support) && outside(expected_support) || error(
        "cannot change the nonlocal response table at checkpoint z=$(checkpoint_redshift): " *
        "stored support=$(stored_support), expected support=$(expected_support)")
    @info "validated nonlocal-table checkpoint migration outside both supports" checkpoint_redshift stored_support expected_support
    true
end

z_to_a(z) = 1.0 / (1.0 + z)
a_to_z(a) = 1.0 / a - 1.0
dadtau(Om, OL, Ok, Or, a) = sqrt(a^3 * (Om + OL*a^3 + Ok*a + Or/a))
dtau_for_dlna(Om, OL, Ok, Or, a, dlna) = a * dlna / dadtau(Om, OL, Ok, Or, a)

if BACKEND_NAME === :metal
    using Metal
    Metal.functional() || error("Metal backend is not functional")
    MHDKernels.has_backend(:metal) || error("MHDKernelsMetalExt did not load")
    PoissonKernels.has_backend(:metal) || error("PoissonKernelsMetalExt did not load")
    ChemistryKernels.has_backend(:metal) || error("ChemistryKernelsMetalExt did not load")
elseif BACKEND_NAME === :cuda
    using CUDA
end

using KernelAbstractions: @Const, @index, @kernel

@kernel function _init_lattice_k!(px, py, pz, vx, vy, vz, N::Int)
    p = @index(Global)
    @inbounds begin
        T = eltype(px)
        q = p - 1
        i = q % N
        j = (q ÷ N) % N
        k = q ÷ (N * N)
        invN = one(T) / T(N)
        px[p] = (T(i) + T(0.5)) * invN
        py[p] = (T(j) + T(0.5)) * invN
        pz[p] = (T(k) + T(0.5)) * invN
        vx[p] = zero(T)
        vy[p] = zero(T)
        vz[p] = zero(T)
    end
end

@kernel function _particle_speed2_k!(work, @Const(vx), @Const(vy), @Const(vz))
    p = @index(Global)
    @inbounds begin
        x = Float32(vx[p])
        y = Float32(vy[p])
        z = Float32(vz[p])
        work[p] = x*x + y*y + z*z
    end
end

@kernel function _grid_accel2_k!(work, @Const(φ), N::Int)
    c = @index(Global)
    @inbounds begin
        q = c - 1
        i = q % N
        j = (q ÷ N) % N
        k = q ÷ (N * N)
        ip = (i + 1) % N
        im = (i + N - 1) % N
        jp = (j + 1) % N
        jm = (j + N - 1) % N
        kp = (k + 1) % N
        km = (k + N - 1) % N
        lin(ii, jj, kk) = ii + N * jj + N * N * kk + 1
        hc = Float32(0.5) * Float32(N)
        gx = -hc * (Float32(φ[lin(ip, j, k)]) - Float32(φ[lin(im, j, k)]))
        gy = -hc * (Float32(φ[lin(i, jp, k)]) - Float32(φ[lin(i, jm, k)]))
        gz = -hc * (Float32(φ[lin(i, j, kp)]) - Float32(φ[lin(i, j, km)]))
        work[c] = gx*gx + gy*gy + gz*gz
    end
end

function _particle_dt_limit(vmax::Float64, gmax::Float64, maxdisp::Float64)
    if gmax > 0
        return (sqrt(max(0.0, vmax*vmax + 2*gmax*maxdisp)) - vmax) / gmax
    elseif vmax > 0
        return maxdisp / vmax
    else
        return Inf
    end
end

@kernel function _push_kdk_global_phi_mix_k!(px, py, pz, vx, vy, vz,
                                             @Const(φ0), @Const(φ1),
                                             theta, dcoef, ts, driftcoef,
                                             wrap, dowrap::Int,
                                             invcell, c05, c1,
                                             n1::Int, n2::Int, n3::Int, hc)
    p = @index(Global)
    @inbounds begin
        xq = px[p] + dcoef * vx[p]
        yq = py[p] + dcoef * vy[p]
        zq = pz[p] + dcoef * vz[p]
        xpos = xq * invcell
        ypos = yq * invcell
        zpos = zq * invcell
        i1 = unsafe_trunc(Int, xpos + c05)
        j1 = unsafe_trunc(Int, ypos + c05)
        k1 = unsafe_trunc(Int, zpos + c05)
        dxr = oftype(xpos, i1) + c05 - xpos
        dyr = oftype(ypos, j1) + c05 - ypos
        dzr = oftype(zpos, k1) + c05 - zpos
        ex = c1 - dxr
        ey = c1 - dyr
        ez = c1 - dzr
        ax0, ay0, az0 = PoissonKernels._global_phi_force(φ0, i1, j1, k1,
                                                          dxr, dyr, dzr, ex, ey, ez,
                                                          n1, n2, n3, hc)
        ax1, ay1, az1 = PoissonKernels._global_phi_force(φ1, i1, j1, k1,
                                                          dxr, dyr, dzr, ex, ey, ez,
                                                          n1, n2, n3, hc)
        omθ = c1 - theta
        ax = omθ * ax0 + theta * ax1
        ay = omθ * ay0 + theta * ay1
        az = omθ * az0 + theta * az1

        Tv = eltype(vx)
        nvx = Tv(vx[p] + ax * ts)
        nvy = Tv(vy[p] + ay * ts)
        nvz = Tv(vz[p] + az * ts)

        x = px[p] + driftcoef * nvx
        y = py[p] + driftcoef * nvy
        z = pz[p] + driftcoef * nvz
        if dowrap == 1
            x = mod(x, wrap)
            y = mod(y, wrap)
            z = mod(z, wrap)
        end
        px[p] = x
        py[p] = y
        pz[p] = z
        vx[p] = Tv(nvx + ax * ts)
        vy[p] = Tv(nvy + ay * ts)
        vz[p] = Tv(nvz + az * ts)
    end
end

function _push_particles_fused_global_phi_mix!(px::AbstractVector{T}, py, pz, vx, vy, vz,
                                               φ0, φ1; theta::Real, dtau::Real, nc,
                                               wrap::Real=1) where {T}
    be = KA.get_backend(px)
    n1, n2, n3 = Int(nc[1]), Int(nc[2]), Int(nc[3])
    half = T(0.5) * T(dtau)
    dowrap = wrap > 0 ? 1 : 0
    _push_kdk_global_phi_mix_k!(be)(px, py, pz, vx, vy, vz, φ0, φ1,
                                    T(theta), half, half, T(dtau),
                                    T(wrap), dowrap, T(n1), T(0.5), T(1),
                                    n1, n2, n3, T(0.5) * T(n1);
                                    ndrange=length(px))
    return nothing
end

@kernel function _push_kdk_global_lattice_k!(dxp, dyp, dzp, vx, vy, vz, @Const(phi),
                                             dcoef, ts, driftcoef, dowrap::Int,
                                             c05, c1, N::Int, hc)
    p = @index(Global)
    @inbounds begin
        Tv = eltype(vx)
        q = p - 1
        i = q % N
        j = (q ÷ N) % N
        k = q ÷ (N * N)
        nT = Tv(N)
        # Reconstruct directly in cell coordinates. Adding dxp to an absolute
        # Float32 box position would round away the tiny high-z drift.
        xpos = mod(Tv(i) + c05 + nT * (Tv(dxp[p]) + dcoef * Tv(vx[p])), nT)
        ypos = mod(Tv(j) + c05 + nT * (Tv(dyp[p]) + dcoef * Tv(vy[p])), nT)
        zpos = mod(Tv(k) + c05 + nT * (Tv(dzp[p]) + dcoef * Tv(vz[p])), nT)
        i1 = unsafe_trunc(Int, xpos + c05)
        j1 = unsafe_trunc(Int, ypos + c05)
        k1 = unsafe_trunc(Int, zpos + c05)
        dxr = Tv(i1) + c05 - xpos
        dyr = Tv(j1) + c05 - ypos
        dzr = Tv(k1) + c05 - zpos
        ex = c1 - dxr; ey = c1 - dyr; ez = c1 - dzr
        ax, ay, az = PoissonKernels._global_phi_force(phi, i1, j1, k1,
                                                       dxr, dyr, dzr, ex, ey, ez,
                                                       N, N, N, hc)

        nvx = Tv(vx[p] + ax * ts)
        nvy = Tv(vy[p] + ay * ts)
        nvz = Tv(vz[p] + az * ts)
        x = Tv(dxp[p]) + driftcoef * nvx
        y = Tv(dyp[p]) + driftcoef * nvy
        z = Tv(dzp[p]) + driftcoef * nvz
        if dowrap == 1
            x -= floor(x + c05)
            y -= floor(y + c05)
            z -= floor(z + c05)
        end
        dxp[p] = x; dyp[p] = y; dzp[p] = z
        vx[p] = Tv(nvx + ax * ts)
        vy[p] = Tv(nvy + ay * ts)
        vz[p] = Tv(nvz + az * ts)
    end
end

function _push_particles_fused_global_lattice!(dxp::AbstractVector{T}, dyp, dzp,
                                               vx, vy, vz, phi;
                                               dtau::Real, N::Int,
                                               wrap::Bool=true) where {T}
    be = KA.get_backend(dxp)
    half = T(0.5) * T(dtau)
    _push_kdk_global_lattice_k!(be)(dxp, dyp, dzp, vx, vy, vz, phi,
                                     half, half, T(dtau), wrap ? 1 : 0,
                                     T(0.5), T(1), N, T(0.5) * T(N);
                                     ndrange=length(dxp))
    return nothing
end

function _deposit_particles!(rho, px, py, pz, vx, vy, vz;
                             N::Int, disp::Real, lattice_displacements::Bool)
    if lattice_displacements
        MHDKernels.deposit_lattice_displacements!(rho, px, py, pz, vx, vy, vz;
                                                   N=N, disp=disp, shift=-0.5)
    else
        PoissonKernels.cic_deposit!(rho, px, py, pz, vx, vy, vz, one(eltype(rho));
                                    N=N, disp=disp, shift=-0.5)
    end
    return rho
end

function _push_particles!(px, py, pz, vx, vy, vz, phi;
                          dtau::Real, N::Int, wrap::Bool,
                          lattice_displacements::Bool)
    if lattice_displacements
        _push_particles_fused_global_lattice!(px, py, pz, vx, vy, vz, phi;
                                               dtau=dtau, N=N, wrap=wrap)
    else
        PoissonKernels.push_particles_fused_global!(px, py, pz, vx, vy, vz, phi;
                                                     dtau=dtau, nc=(N,N,N),
                                                     wrap=wrap ? 1 : 0)
    end
    return nothing
end

@kernel function _assemble_total_delta_k!(ρtot, @Const(gasρ), @Const(dmρ),
                                          fb, fdm, N::Int)
    c = @index(Global)
    @inbounds ρtot[c] = fb * (gasρ[c] - one(eltype(ρtot))) +
                        fdm * (dmρ[c] - one(eltype(ρtot)))
end

@kernel function _assemble_total_from_gas_delta_k!(ρtot,@Const(gas_delta),
                                                   @Const(dmρ),fb,fdm,N::Int)
    c=@index(Global)
    @inbounds ρtot[c]=fb*gas_delta[c]+fdm*(dmρ[c]-one(eltype(ρtot)))
end

@kernel function _delta_from_density_k!(dst, @Const(src), N::Int)
    c = @index(Global)
    @inbounds dst[c] = src[c] - one(eltype(dst))
end

@kernel function _init_single_b_k!(rho, mx, my, mz, E, Bx, By, Bz, psi,
                                   N::Int, γ, p0, b0, mode::Int, kmode::Int)
    c = @index(Global)
    @inbounds begin
        T = eltype(rho)
        i = (c - 1) % N + 1
        x = (T(i) - T(0.5)) / T(N)
        phase = T(2π) * T(kmode) * x
        bx = zero(T)
        by = mode == 1 ? b0 * sin(phase) : b0 * sin(phase)
        bz = mode == 1 ? b0 * cos(phase) : zero(T)
        rho[c] = one(T)
        mx[c] = zero(T); my[c] = zero(T); mz[c] = zero(T)
        Bx[c] = bx; By[c] = by; Bz[c] = bz; psi[c] = zero(T)
        E[c] = p0 / (γ - one(T)) + T(0.5) * (bx * bx + by * by + bz * bz)
    end
end

@kernel function _init_alfven_linear_k!(rho, mx, my, mz, E, Bx, By, Bz, psi,
                                        N::Int, γ, p0, bguide, bpert, kmode::Int)
    c = @index(Global)
    @inbounds begin
        T = eltype(rho)
        i = (c - 1) % N + 1
        x = (T(i) - T(0.5)) / T(N)
        phase = T(2π) * T(kmode) * x
        bx = bguide
        by = bpert * sin(phase)
        bz = zero(T)
        rho[c] = one(T)
        mx[c] = zero(T); my[c] = zero(T); mz[c] = zero(T)
        Bx[c] = bx; By[c] = by; Bz[c] = bz; psi[c] = zero(T)
        E[c] = p0 / (γ - one(T)) + T(0.5) * (bx * bx + by * by + bz * bz)
    end
end

@kernel function _init_meanfield_linear_k!(rho, mx, my, mz, E, Bx, By, Bz, psi,
                                           N::Int, γ, p0, bguide, bpert, theta,
                                           polarization::Int, kmode::Int)
    c = @index(Global)
    @inbounds begin
        T = eltype(rho)
        i = (c - 1) % N + 1
        x = (T(i) - T(0.5)) / T(N)
        phase = T(2π) * T(kmode) * x
        bx = bguide * cos(theta)
        by = bguide * sin(theta) + (polarization == 1 ? bpert * cos(phase) : zero(T))
        bz = polarization == 2 ? bpert * cos(phase) : zero(T)
        rho[c] = one(T)
        mx[c] = zero(T); my[c] = zero(T); mz[c] = zero(T)
        Bx[c] = bx; By[c] = by; Bz[c] = bz; psi[c] = zero(T)
        E[c] = p0 / (γ - one(T)) + T(0.5) * (bx * bx + by * by + bz * bz)
    end
end

function _single_b_mode(env::AbstractString)
    s = Symbol(lowercase(env))
    s in (:forcefree, :circular) && return 1
    s in (:sine, :single, :single_sine, :nonforcefree,
          :magpressure, :compressive, :pressure, :density) && return 2
    error("MHD_PMF_INIT must be batchelor, forcefree, sine/magpressure, or alfven_linear; got '$env'")
end

@kernel function _init_reduced_chem_k!(HII, H2I, @Const(rho), xhii, xh2, fh, N::Int)
    c = @index(Global)
    @inbounds begin
        r = rho[c]
        HII[c] = xhii * fh * r
        H2I[c] = xh2 * fh * r
    end
end

@kernel function _init_reduced_helium_k!(HeII, @Const(rho), heii_mass_fraction, N::Int)
    c = @index(Global)
    @inbounds HeII[c] = heii_mass_fraction * rho[c]
end

@kernel function _mhd_to_eint_k!(eint, @Const(rho), @Const(mx), @Const(my), @Const(mz),
                                 @Const(E), @Const(Bx), @Const(By), @Const(Bz), small)
    c = @index(Global)
    @inbounds begin
        T = eltype(eint)
        r = max(rho[c], small)
        ek = T(0.5) * (mx[c]*mx[c] + my[c]*my[c] + mz[c]*mz[c]) / r
        eb = T(0.5) * (Bx[c]*Bx[c] + By[c]*By[c] + Bz[c]*Bz[c])
        eint[c] = max((E[c] - ek - eb) / r, T(1e-30))
    end
end

@kernel function _eint_to_mhd_k!(E, @Const(eint), @Const(rho), @Const(mx), @Const(my), @Const(mz),
                                 @Const(Bx), @Const(By), @Const(Bz), small)
    c = @index(Global)
    @inbounds begin
        T = eltype(E)
        r = max(rho[c], small)
        ek = T(0.5) * (mx[c]*mx[c] + my[c]*my[c] + mz[c]*mz[c]) / r
        eb = T(0.5) * (Bx[c]*Bx[c] + By[c]*By[c] + Bz[c]*Bz[c])
        E[c] = max(r * max(eint[c], T(1e-30)) + ek + eb, T(1e-30))
    end
end

@kernel function _chem_analytic_mixing_mhd_k!(rho, @Const(mx), @Const(my), @Const(mz),
                                              E, @Const(Bx), @Const(By), @Const(Bz),
                                              HII, H2I, @Const(nsm),
                                              nsm_scalar, du, vu2, tu, dt, z,
                                              f_alpha, Xe_mean, fudge, gauss,
                                              hubble, Om, OL, fh,
                                              hubble_expansion, adot_over_a,
                                              rate_tables, dtfrac, itcap, small,
                                              ::Val{NS}, ::Val{SN}) where {NS,SN}
    c = @index(Global)
    @inbounds begin
        T = eltype(E)
        r_code = max(rho[c], small)
        ek = T(0.5) * (mx[c]*mx[c] + my[c]*my[c] + mz[c]*mz[c]) / r_code
        eb = T(0.5) * (Bx[c]*Bx[c] + By[c]*By[c] + Bz[c]*Bz[c])
        e_code = max((E[c] - ek - eb) / r_code, T(1e-30))
        nsm_code = NS ? T(nsm_scalar) : nsm[c]
        nsm_h = nsm_code * du * (SN ? one(T) : T(fh)) / T(ChemistryKernels.MH)
        en, hii, h2, _ = ChemistryKernels.evolve_cell_analytic_mixing(
            r_code * du, e_code * vu2, HII[c] * du, H2I[c] * du,
            nsm_h, dt * tu, z;
            f_alpha=T(f_alpha), Xe_mean=T(Xe_mean),
            smoothed_is_neutral=Val(SN), fudge=T(fudge), gauss=T(gauss),
            hubble=T(hubble), Om=T(Om), OL=T(OL), fh=T(fh),
            hubble_expansion=hubble_expansion, adot_over_a=T(adot_over_a),
            rate_tables=rate_tables, cool_tables=nothing,
            dtfrac=T(dtfrac), itcap=itcap)
        cap = max(T(fh) * r_code, zero(T))
        h2_code = min(max(h2 / du, zero(T)), cap)
        hii_code = min(max(hii / du, zero(T)), max(cap - h2_code, zero(T)))
        H2I[c] = h2_code
        HII[c] = hii_code
        e_new = max(en / vu2, T(1e-30))
        E[c] = max(r_code * e_new + ek + eb, T(1e-30))
    end
end

@kernel function _chem_mixing_mhd_k!(rho, @Const(mx), @Const(my), @Const(mz),
                                     E, @Const(Bx), @Const(By), @Const(Bz),
                                     HII, H2I, @Const(nsm),
                                     nsm_scalar, du, vu2, tu, dt, z,
                                     f_alpha, Xe_mean, fudge, gauss,
                                     hubble, Om, OL, fh, hubble_expansion,
                                     rate_tables, dtfrac, itcap, small,
                                     ::Val{NS}, ::Val{SN}) where {NS,SN}
    c = @index(Global)
    @inbounds begin
        T = eltype(E)
        r_code = max(rho[c], small)
        ek = T(0.5) * (mx[c]*mx[c] + my[c]*my[c] + mz[c]*mz[c]) / r_code
        eb = T(0.5) * (Bx[c]*Bx[c] + By[c]*By[c] + Bz[c]*Bz[c])
        e_code = max((E[c] - ek - eb) / r_code, T(1e-30))
        nsm_code = NS ? T(nsm_scalar) : nsm[c]
        nsm_h = nsm_code * du * (SN ? one(T) : T(fh)) / T(ChemistryKernels.MH)
        en, hii, h2, _, _ = ChemistryKernels.evolve_cell_mixing(
            r_code * du, e_code * vu2, HII[c] * du, H2I[c] * du, zero(T),
            nsm_h, dt * tu, z;
            f_alpha=T(f_alpha), Xe_mean=T(Xe_mean),
            smoothed_is_neutral=Val(SN), fudge=T(fudge), gauss=T(gauss),
            hubble=T(hubble), Om=T(Om), OL=T(OL), fh=T(fh),
            deuterium=false, helium=false, hubble_expansion=hubble_expansion,
            rate_tables=rate_tables, dtfrac=T(dtfrac), itcap=itcap)
        cap = max(T(fh) * r_code, zero(T))
        h2_code = min(max(h2 / du, zero(T)), cap)
        hii_code = min(max(hii / du, zero(T)), max(cap - h2_code, zero(T)))
        H2I[c] = h2_code
        HII[c] = hii_code
        e_new = max(en / vu2, T(1e-30))
        E[c] = max(r_code * e_new + ek + eb, T(1e-30))
    end
end

@kernel function _chem_mixing_helium_mhd_k!(rho, @Const(mx), @Const(my), @Const(mz),
                                            E, @Const(Bx), @Const(By), @Const(Bz),
                                            HII, H2I, HeII, @Const(nsm),
                                            nsm_scalar, du, vu2, tu, dt, z,
                                            f_alpha, Xe_mean, fudge, gauss,
                                            hubble, Om, OL, fh, hubble_expansion,
                                            rate_tables, dtfrac, itcap, small,
                                            ::Val{NS}, ::Val{SN}) where {NS,SN}
    c = @index(Global)
    @inbounds begin
        T = eltype(E)
        r_code = max(rho[c], small)
        ek = T(0.5) * (mx[c]*mx[c] + my[c]*my[c] + mz[c]*mz[c]) / r_code
        eb = T(0.5) * (Bx[c]*Bx[c] + By[c]*By[c] + Bz[c]*Bz[c])
        e_code = max((E[c] - ek - eb) / r_code, T(1e-30))
        nsm_code = NS ? T(nsm_scalar) : nsm[c]
        nsm_h = nsm_code * du * (SN ? one(T) : T(fh)) / T(ChemistryKernels.MH)
        en, hii, h2, _, heii = ChemistryKernels.evolve_cell_mixing(
            r_code * du, e_code * vu2, HII[c] * du, H2I[c] * du, zero(T),
            nsm_h, dt * tu, z;
            f_alpha=T(f_alpha), Xe_mean=T(Xe_mean),
            smoothed_is_neutral=Val(SN), fudge=T(fudge), gauss=T(gauss),
            hubble=T(hubble), Om=T(Om), OL=T(OL), fh=T(fh),
            deuterium=false, helium=true, HeII_m=HeII[c] * du,
            hubble_expansion=hubble_expansion,
            rate_tables=rate_tables, dtfrac=T(dtfrac), itcap=itcap)
        hcap = max(T(fh) * r_code, zero(T))
        h2_code = min(max(h2 / du, zero(T)), hcap)
        hii_code = min(max(hii / du, zero(T)), max(hcap - h2_code, zero(T)))
        hecap = max((one(T) - T(fh)) * r_code, zero(T))
        H2I[c] = h2_code
        HII[c] = hii_code
        HeII[c] = min(max(heii / du, zero(T)), hecap)
        e_new = max(en / vu2, T(1e-30))
        E[c] = max(r_code * e_new + ek + eb, T(1e-30))
    end
end

function _apply_compton_drag_mhd!(be, s, f::Real, vx0::Real, vy0::Real, vz0::Real)
    MHDKernels.apply_linear_drag!(s, clamp(f, 0.0, 1.0);
                                  target_velocity=(vx0, vy0, vz0))
    return nothing
end

@inline function _lin3(i, j, k, N)
    ii = mod(i - 1, N) + 1
    jj = mod(j - 1, N) + 1
    kk = mod(k - 1, N) + 1
    return ((kk - 1) * N + (jj - 1)) * N + ii
end

@inline function _pressure_at(rho, mx, my, mz, E, Bx, By, Bz, c, gamma_gas, smallr, smallE)
    r = max(rho[c], smallr)
    ek = (mx[c] * mx[c] + my[c] * my[c] + mz[c] * mz[c]) / (2 * r)
    eb = (Bx[c] * Bx[c] + By[c] * By[c] + Bz[c] * Bz[c]) / 2
    eint = max(E[c] - ek - eb, smallE)
    return (gamma_gas - one(gamma_gas)) * eint
end

@kernel function _gas_force_balance_diag_k!(pax, pay, paz, hax, hay, haz,
        @Const(rho), @Const(mx), @Const(my), @Const(mz), @Const(E),
        @Const(Bx), @Const(By), @Const(Bz), N::Int, inv2dx,
        gamma_gas, smallr, smallE)
    c = @index(Global)
    @inbounds begin
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N * N) + 1
        cxp = _lin3(i + 1, j, k, N); cxm = _lin3(i - 1, j, k, N)
        cyp = _lin3(i, j + 1, k, N); cym = _lin3(i, j - 1, k, N)
        czp = _lin3(i, j, k + 1, N); czm = _lin3(i, j, k - 1, N)
        px = -(_pressure_at(rho, mx, my, mz, E, Bx, By, Bz, cxp,
                            gamma_gas, smallr, smallE) -
               _pressure_at(rho, mx, my, mz, E, Bx, By, Bz, cxm,
                            gamma_gas, smallr, smallE)) * inv2dx
        py = -(_pressure_at(rho, mx, my, mz, E, Bx, By, Bz, cyp,
                            gamma_gas, smallr, smallE) -
               _pressure_at(rho, mx, my, mz, E, Bx, By, Bz, cym,
                            gamma_gas, smallr, smallE)) * inv2dx
        pz = -(_pressure_at(rho, mx, my, mz, E, Bx, By, Bz, czp,
                            gamma_gas, smallr, smallE) -
               _pressure_at(rho, mx, my, mz, E, Bx, By, Bz, czm,
                            gamma_gas, smallr, smallE)) * inv2dx
        lx, ly, lz = _magnetic_force_at(Bx, By, Bz, i, j, k, N, inv2dx)
        irho = inv(max(rho[c], smallr))
        pax[c] = px * irho; pay[c] = py * irho; paz[c] = pz * irho
        hax[c] = (px + lx) * irho
        hay[c] = (py + ly) * irho
        haz[c] = (pz + lz) * irho
    end
end

@kernel function _gas_total_accel_diag_k!(tx, ty, tz,
        @Const(hx), @Const(hy), @Const(hz),
        @Const(sx), @Const(sy), @Const(sz), gamma_drag)
    c = @index(Global)
    @inbounds begin
        tx[c] = hx[c] + gamma_drag * sx[c]
        ty[c] = hy[c] + gamma_drag * sy[c]
        tz[c] = hz[c] + gamma_drag * sz[c]
    end
end

@kernel function _gas_velocity_diag_k!(vx, vy, vz,
        @Const(rho), @Const(mx), @Const(my), @Const(mz), smallr)
    c = @index(Global)
    @inbounds begin
        irho = inv(max(rho[c], smallr))
        vx[c] = mx[c] * irho
        vy[c] = my[c] * irho
        vz[c] = mz[c] * irho
    end
end

@kernel function _mass_flux_divergence_diag_k!(divm,
        @Const(mx), @Const(my), @Const(mz), N::Int, inv2dx)
    c = @index(Global)
    @inbounds begin
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N * N) + 1
        divm[c] = ((mx[_lin3(i + 1, j, k, N)] - mx[_lin3(i - 1, j, k, N)]) +
                   (my[_lin3(i, j + 1, k, N)] - my[_lin3(i, j - 1, k, N)]) +
                   (mz[_lin3(i, j, k + 1, N)] - mz[_lin3(i, j, k - 1, N)])) * inv2dx
    end
end

@inline function _terminal_v_at(rho, mx, my, mz, E, Bx, By, Bz, i, j, k, N,
                                inv2dx, gamma_code, gamma_gas, smallr, smallE, vcap)
    if gamma_code <= eps(gamma_code)
        return zero(gamma_code), zero(gamma_code), zero(gamma_code)
    end
    T = typeof(gamma_code)
    c = _lin3(i, j, k, N)
    cxp = _lin3(i + 1, j, k, N); cxm = _lin3(i - 1, j, k, N)
    cyp = _lin3(i, j + 1, k, N); cym = _lin3(i, j - 1, k, N)
    czp = _lin3(i, j, k + 1, N); czm = _lin3(i, j, k - 1, N)
    # Use conservative Maxwell-stress differences instead of pointwise J x B.
    # The continuum forms are equivalent for div(B)=0, but the stress form avoids
    # severe cell-centered aliasing of magnetic-pressure modes near marginal
    # resolution.
    bxp = Bx[cxp]; byp = By[cxp]; bzp = Bz[cxp]
    bxm = Bx[cxm]; bym = By[cxm]; bzm = Bz[cxm]
    b2p = bxp*bxp + byp*byp + bzp*bzp
    b2m = bxm*bxm + bym*bym + bzm*bzm
    txxp = bxp*bxp - T(0.5)*b2p
    txxm = bxm*bxm - T(0.5)*b2m
    tyxp = byp*bxp
    tyxm = bym*bxm
    tzxp = bzp*bxp
    tzxm = bzm*bxm

    bxp = Bx[cyp]; byp = By[cyp]; bzp = Bz[cyp]
    bxm = Bx[cym]; bym = By[cym]; bzm = Bz[cym]
    b2p = bxp*bxp + byp*byp + bzp*bzp
    b2m = bxm*bxm + bym*bym + bzm*bzm
    txyp = bxp*byp
    txym = bxm*bym
    tyyp = byp*byp - T(0.5)*b2p
    tyym = bym*bym - T(0.5)*b2m
    tzyp = bzp*byp
    tzym = bzm*bym

    bxp = Bx[czp]; byp = By[czp]; bzp = Bz[czp]
    bxm = Bx[czm]; bym = By[czm]; bzm = Bz[czm]
    b2p = bxp*bxp + byp*byp + bzp*bzp
    b2m = bxm*bxm + bym*bym + bzm*bzm
    txzp = bxp*bzp
    txzm = bxm*bzm
    tyzp = byp*bzp
    tyzm = bym*bzm
    tzzp = bzp*bzp - T(0.5)*b2p
    tzzm = bzm*bzm - T(0.5)*b2m

    fx = ((txxp - txxm) + (txyp - txym) + (txzp - txzm)) * inv2dx
    fy = ((tyxp - tyxm) + (tyyp - tyym) + (tyzp - tyzm)) * inv2dx
    fz = ((tzxp - tzxm) + (tzyp - tzym) + (tzzp - tzzm)) * inv2dx
    pxp = _pressure_at(rho, mx, my, mz, E, Bx, By, Bz, cxp, gamma_gas, smallr, smallE)
    pxm = _pressure_at(rho, mx, my, mz, E, Bx, By, Bz, cxm, gamma_gas, smallr, smallE)
    pyp = _pressure_at(rho, mx, my, mz, E, Bx, By, Bz, cyp, gamma_gas, smallr, smallE)
    pym = _pressure_at(rho, mx, my, mz, E, Bx, By, Bz, cym, gamma_gas, smallr, smallE)
    pzp = _pressure_at(rho, mx, my, mz, E, Bx, By, Bz, czp, gamma_gas, smallr, smallE)
    pzm = _pressure_at(rho, mx, my, mz, E, Bx, By, Bz, czm, gamma_gas, smallr, smallE)
    fx -= (pxp - pxm) * inv2dx
    fy -= (pyp - pym) * inv2dx
    fz -= (pzp - pzm) * inv2dx
    invrg = one(gamma_code) / (max(rho[c], smallr) * gamma_code)
    vx = fx * invrg
    vy = fy * invrg
    vz = fz * invrg
    v2 = vx * vx + vy * vy + vz * vz
    if v2 > vcap * vcap
        fac = vcap / sqrt(v2)
        vx *= fac; vy *= fac; vz *= fac
    end
    return vx, vy, vz
end

@inline function _vxb_at(rho, mx, my, mz, E, Bx, By, Bz, i, j, k, N,
                         inv2dx, gamma_code, gamma_gas, smallr, smallE, vcap)
    c = _lin3(i, j, k, N)
    vx, vy, vz = _terminal_v_at(rho, mx, my, mz, E, Bx, By, Bz, i, j, k, N,
                                inv2dx, gamma_code, gamma_gas, smallr, smallE, vcap)
    bx = Bx[c]; by = By[c]; bz = Bz[c]
    return vy * bz - vz * by, vz * bx - vx * bz, vx * by - vy * bx
end

@inline function _vxb_cached_at(vx, vy, vz, Bx, By, Bz, i, j, k, N)
    c = _lin3(i, j, k, N)
    vxc = vx[c]; vyc = vy[c]; vzc = vz[c]
    bx = Bx[c]; by = By[c]; bz = Bz[c]
    return vyc * bz - vzc * by, vzc * bx - vxc * bz, vxc * by - vyc * bx
end

@inline function _terminal_cached_v_at(tvx, tvy, tvz, i, j, k, N, vsmooth)
    c = _lin3(i, j, k, N)
    vx = tvx[c]; vy = tvy[c]; vz = tvz[c]
    if vsmooth <= zero(vsmooth)
        return vx, vy, vz
    end
    cxp = _lin3(i + 1, j, k, N); cxm = _lin3(i - 1, j, k, N)
    cyp = _lin3(i, j + 1, k, N); cym = _lin3(i, j - 1, k, N)
    czp = _lin3(i, j, k + 1, N); czm = _lin3(i, j, k - 1, N)
    a = min(max(vsmooth, zero(vsmooth)), one(vsmooth))
    cc = one(a) - a
    cn = a / typeof(a)(6)
    sx = tvx[cxp] + tvx[cxm] + tvx[cyp] + tvx[cym] + tvx[czp] + tvx[czm]
    sy = tvy[cxp] + tvy[cxm] + tvy[cyp] + tvy[cym] + tvy[czp] + tvy[czm]
    sz = tvz[cxp] + tvz[cxm] + tvz[cyp] + tvz[cym] + tvz[czp] + tvz[czm]
    return cc * vx + cn * sx, cc * vy + cn * sy, cc * vz + cn * sz
end

@inline function _vxb_cached_smooth_at(tvx, tvy, tvz, Bx, By, Bz, i, j, k, N, vsmooth)
    c = _lin3(i, j, k, N)
    vxc, vyc, vzc = _terminal_cached_v_at(tvx, tvy, tvz, i, j, k, N, vsmooth)
    bx = Bx[c]; by = By[c]; bz = Bz[c]
    return vyc * bz - vzc * by, vzc * bx - vxc * bz, vxc * by - vyc * bx
end

@inline function _magnetic_force_at(Bx, By, Bz, i, j, k, N, inv2dx)
    T = typeof(inv2dx)
    c = _lin3(i, j, k, N)
    cxp = _lin3(i + 1, j, k, N); cxm = _lin3(i - 1, j, k, N)
    cyp = _lin3(i, j + 1, k, N); cym = _lin3(i, j - 1, k, N)
    czp = _lin3(i, j, k + 1, N); czm = _lin3(i, j, k - 1, N)

    bxp = Bx[cxp]; byp = By[cxp]; bzp = Bz[cxp]
    bxm = Bx[cxm]; bym = By[cxm]; bzm = Bz[cxm]
    b2p = bxp*bxp + byp*byp + bzp*bzp
    b2m = bxm*bxm + bym*bym + bzm*bzm
    txxp = bxp*bxp - T(0.5)*b2p
    txxm = bxm*bxm - T(0.5)*b2m
    tyxp = byp*bxp
    tyxm = bym*bxm
    tzxp = bzp*bxp
    tzxm = bzm*bxm

    bxp = Bx[cyp]; byp = By[cyp]; bzp = Bz[cyp]
    bxm = Bx[cym]; bym = By[cym]; bzm = Bz[cym]
    b2p = bxp*bxp + byp*byp + bzp*bzp
    b2m = bxm*bxm + bym*bym + bzm*bzm
    txyp = bxp*byp
    txym = bxm*bym
    tyyp = byp*byp - T(0.5)*b2p
    tyym = bym*bym - T(0.5)*b2m
    tzyp = bzp*byp
    tzym = bzm*bym

    bxp = Bx[czp]; byp = By[czp]; bzp = Bz[czp]
    bxm = Bx[czm]; bym = By[czm]; bzm = Bz[czm]
    b2p = bxp*bxp + byp*byp + bzp*bzp
    b2m = bxm*bxm + bym*bym + bzm*bzm
    txzp = bxp*bzp
    txzm = bxm*bzm
    tyzp = byp*bzp
    tyzm = bym*bzm
    tzzp = bzp*bzp - T(0.5)*b2p
    tzzm = bzm*bzm - T(0.5)*b2m

    fx = ((txxp - txxm) + (txyp - txym) + (txzp - txzm)) * inv2dx
    fy = ((tyxp - tyxm) + (tyyp - tyym) + (tyzp - tyzm)) * inv2dx
    fz = ((tzxp - tzxm) + (tzyp - tzym) + (tzzp - tzzm)) * inv2dx
    return fx, fy, fz
end

@inline function _pressure_implicit_v_at(rho_new, fxm, fym, fzm,
                                         i, j, k, N, inv2dx, gamma_code,
                                         pressure_coeff, smallr, vcap)
    c = _lin3(i, j, k, N)
    cxp = _lin3(i + 1, j, k, N); cxm = _lin3(i - 1, j, k, N)
    cyp = _lin3(i, j + 1, k, N); cym = _lin3(i, j - 1, k, N)
    czp = _lin3(i, j, k + 1, N); czm = _lin3(i, j, k - 1, N)
    r = max(rho_new[c], smallr)
    gx = pressure_coeff * (rho_new[cxp] - rho_new[cxm]) * inv2dx
    gy = pressure_coeff * (rho_new[cyp] - rho_new[cym]) * inv2dx
    gz = pressure_coeff * (rho_new[czp] - rho_new[czm]) * inv2dx
    invrg = one(gamma_code) / (r * max(gamma_code, eps(gamma_code)))
    vx = (fxm[c] - gx) * invrg
    vy = (fym[c] - gy) * invrg
    vz = (fzm[c] - gz) * invrg
    v2 = vx * vx + vy * vy + vz * vz
    if v2 > vcap * vcap
        fac = vcap / sqrt(v2)
        vx *= fac; vy *= fac; vz *= fac
    end
    return vx, vy, vz
end

@inline function _magnetic_capped_v_at(rho, fxm, fym, fzm, c, gamma_code, smallr, vcap)
    r = max(rho[c], smallr)
    invrg = one(gamma_code) / (r * max(gamma_code, eps(gamma_code)))
    vx = fxm[c] * invrg
    vy = fym[c] * invrg
    vz = fzm[c] * invrg
    v2 = vx * vx + vy * vy + vz * vz
    if v2 > vcap * vcap
        fac = vcap / sqrt(v2)
        vx *= fac; vy *= fac; vz *= fac
    end
    return vx, vy, vz
end

@inline function _vxb_pressure_implicit_at(rho_new, fxm, fym, fzm, Bx, By, Bz,
                                           i, j, k, N, inv2dx, gamma_code,
                                           pressure_coeff, smallr, vcap)
    c = _lin3(i, j, k, N)
    vx, vy, vz = _pressure_implicit_v_at(rho_new, fxm, fym, fzm, i, j, k, N,
                                         inv2dx, gamma_code, pressure_coeff,
                                         smallr, vcap)
    bx = Bx[c]; by = By[c]; bz = Bz[c]
    return vy * bz - vz * by, vz * bx - vx * bz, vx * by - vy * bx
end

@kernel function _magnetic_force_field_k!(fx, fy, fz, @Const(Bx), @Const(By), @Const(Bz),
                                          N::Int, inv2dx)
    c = @index(Global)
    @inbounds begin
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N * N) + 1
        fxi, fyi, fzi = _magnetic_force_at(Bx, By, Bz, i, j, k, N, inv2dx)
        fx[c] = fxi
        fy[c] = fyi
        fz[c] = fzi
    end
end

@kernel function _pressure_implicit_source_k!(source, @Const(rho), @Const(fx), @Const(fy), @Const(fz),
                                              N::Int, inv2dx, gamma_code, smallr, vcap)
    c = @index(Global)
    @inbounds begin
        T = eltype(source)
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N * N) + 1
        cxp = _lin3(i + 1, j, k, N); cxm = _lin3(i - 1, j, k, N)
        cyp = _lin3(i, j + 1, k, N); cym = _lin3(i, j - 1, k, N)
        czp = _lin3(i, j, k + 1, N); czm = _lin3(i, j, k - 1, N)
        vx, vy, vz = _magnetic_capped_v_at(rho, fx, fy, fz, c, gamma_code, smallr, vcap)
        vxp, _, _ = _magnetic_capped_v_at(rho, fx, fy, fz, cxp, gamma_code, smallr, vcap)
        vxm, _, _ = _magnetic_capped_v_at(rho, fx, fy, fz, cxm, gamma_code, smallr, vcap)
        _, vyp, _ = _magnetic_capped_v_at(rho, fx, fy, fz, cyp, gamma_code, smallr, vcap)
        _, vym, _ = _magnetic_capped_v_at(rho, fx, fy, fz, cym, gamma_code, smallr, vcap)
        _, _, vzp = _magnetic_capped_v_at(rho, fx, fy, fz, czp, gamma_code, smallr, vcap)
        _, _, vzm = _magnetic_capped_v_at(rho, fx, fy, fz, czm, gamma_code, smallr, vcap)
        vxf_p = T(0.5) * (vx + vxp)
        vxf_m = T(0.5) * (vxm + vx)
        vyf_p = T(0.5) * (vy + vyp)
        vyf_m = T(0.5) * (vym + vy)
        vzf_p = T(0.5) * (vz + vzp)
        vzf_m = T(0.5) * (vzm + vz)
        fx_p = vxf_p * max(ifelse(vxf_p >= zero(T), rho[c], rho[cxp]), smallr)
        fx_m = vxf_m * max(ifelse(vxf_m >= zero(T), rho[cxm], rho[c]), smallr)
        fy_p = vyf_p * max(ifelse(vyf_p >= zero(T), rho[c], rho[cyp]), smallr)
        fy_m = vyf_m * max(ifelse(vyf_m >= zero(T), rho[cym], rho[c]), smallr)
        fz_p = vzf_p * max(ifelse(vzf_p >= zero(T), rho[c], rho[czp]), smallr)
        fz_m = vzf_m * max(ifelse(vzf_m >= zero(T), rho[czm], rho[c]), smallr)
        div_flux = ((fx_p - fx_m) + (fy_p - fy_m) + (fz_p - fz_m)) * (T(2) * inv2dx)
        source[c] = -div_flux
    end
end

@kernel function _terminal_velocity_field_k!(tvx, tvy, tvz,
                                             @Const(rho), @Const(mx), @Const(my), @Const(mz), @Const(E),
                                             @Const(Bx), @Const(By), @Const(Bz),
                                             N::Int, inv2dx, gamma_code, gamma_gas, smallr, smallE, vcap)
    c = @index(Global)
    @inbounds begin
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N * N) + 1
        vx, vy, vz = _terminal_v_at(rho, mx, my, mz, E, Bx, By, Bz,
                                    i, j, k, N, inv2dx, gamma_code, gamma_gas,
                                    smallr, smallE, vcap)
        tvx[c] = vx
        tvy[c] = vy
        tvz[c] = vz
    end
end

@kernel function _terminal_velocity_overlap_diag_k!(delta2, terminal2, full2,
                                                    @Const(rho), @Const(mx), @Const(my), @Const(mz),
                                                    @Const(tvx), @Const(tvy), @Const(tvz), smallr)
    c = @index(Global)
    @inbounds begin
        r = max(rho[c], smallr)
        vx = mx[c] / r
        vy = my[c] / r
        vz = mz[c] / r
        tx = tvx[c]
        ty = tvy[c]
        tz = tvz[c]
        dx = vx - tx
        dy = vy - ty
        dz = vz - tz
        delta2[c] = r * (dx * dx + dy * dy + dz * dz)
        terminal2[c] = r * (tx * tx + ty * ty + tz * tz)
        full2[c] = r * (vx * vx + vy * vy + vz * vz)
    end
end

@kernel function _terminal_velocity_update_cached_k!(orho, omx, omy, omz, oE, oBx, oBy, oBz, opsi,
                                                     @Const(rho), @Const(mx), @Const(my), @Const(mz), @Const(E),
                                                     @Const(Bx), @Const(By), @Const(Bz), @Const(psi),
                                                     @Const(tvx), @Const(tvy), @Const(tvz),
                                                     N::Int, dt, inv2dx, gamma_gas, smallr, smallE, vsmooth)
    c = @index(Global)
    @inbounds begin
        T = eltype(E)
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N * N) + 1

        r = max(rho[c], smallr)
        ek0 = T(0.5) * (mx[c] * mx[c] + my[c] * my[c] + mz[c] * mz[c]) / r
        eb0 = T(0.5) * (Bx[c] * Bx[c] + By[c] * By[c] + Bz[c] * Bz[c])
        eint = max(E[c] - ek0 - eb0, smallE)

        vx, vy, vz = _terminal_cached_v_at(tvx, tvy, tvz, i, j, k, N, vsmooth)
        cxp = _lin3(i + 1, j, k, N); cxm = _lin3(i - 1, j, k, N)
        cyp = _lin3(i, j + 1, k, N); cym = _lin3(i, j - 1, k, N)
        czp = _lin3(i, j, k + 1, N); czm = _lin3(i, j, k - 1, N)
        vxp_x, _, _ = _terminal_cached_v_at(tvx, tvy, tvz, i + 1, j, k, N, vsmooth)
        vxm_x, _, _ = _terminal_cached_v_at(tvx, tvy, tvz, i - 1, j, k, N, vsmooth)
        _, vyp_y, _ = _terminal_cached_v_at(tvx, tvy, tvz, i, j + 1, k, N, vsmooth)
        _, vym_y, _ = _terminal_cached_v_at(tvx, tvy, tvz, i, j - 1, k, N, vsmooth)
        _, _, vzp_z = _terminal_cached_v_at(tvx, tvy, tvz, i, j, k + 1, N, vsmooth)
        _, _, vzm_z = _terminal_cached_v_at(tvx, tvy, tvz, i, j, k - 1, N, vsmooth)
        vxf_p = T(0.5) * (vx + vxp_x)
        vxf_m = T(0.5) * (vxm_x + vx)
        vyf_p = T(0.5) * (vy + vyp_y)
        vyf_m = T(0.5) * (vym_y + vy)
        vzf_p = T(0.5) * (vz + vzp_z)
        vzf_m = T(0.5) * (vzm_z + vz)
        fx_p = vxf_p * max(ifelse(vxf_p >= zero(T), rho[c], rho[cxp]), smallr)
        fx_m = vxf_m * max(ifelse(vxf_m >= zero(T), rho[cxm], rho[c]), smallr)
        fy_p = vyf_p * max(ifelse(vyf_p >= zero(T), rho[c], rho[cyp]), smallr)
        fy_m = vyf_m * max(ifelse(vyf_m >= zero(T), rho[cym], rho[c]), smallr)
        fz_p = vzf_p * max(ifelse(vzf_p >= zero(T), rho[c], rho[czp]), smallr)
        fz_m = vzf_m * max(ifelse(vzf_m >= zero(T), rho[czm], rho[c]), smallr)
        div_rhov = ((fx_p - fx_m) + (fy_p - fy_m) + (fz_p - fz_m)) * (T(2) * inv2dx)
        nrho = max(rho[c] - dt * div_rhov, smallr)
        cyp_x, cyp_y, cyp_z = _vxb_cached_smooth_at(tvx, tvy, tvz, Bx, By, Bz, i, j + 1, k, N, vsmooth)
        cym_x, cym_y, cym_z = _vxb_cached_smooth_at(tvx, tvy, tvz, Bx, By, Bz, i, j - 1, k, N, vsmooth)
        czp_x, czp_y, czp_z = _vxb_cached_smooth_at(tvx, tvy, tvz, Bx, By, Bz, i, j, k + 1, N, vsmooth)
        czm_x, czm_y, czm_z = _vxb_cached_smooth_at(tvx, tvy, tvz, Bx, By, Bz, i, j, k - 1, N, vsmooth)
        cxp_x, cxp_y, cxp_z = _vxb_cached_smooth_at(tvx, tvy, tvz, Bx, By, Bz, i + 1, j, k, N, vsmooth)
        cxm_x, cxm_y, cxm_z = _vxb_cached_smooth_at(tvx, tvy, tvz, Bx, By, Bz, i - 1, j, k, N, vsmooth)

        nbx = Bx[c] + dt * ((cyp_z - cym_z) - (czp_y - czm_y)) * inv2dx
        nby = By[c] + dt * ((czp_x - czm_x) - (cxp_z - cxm_z)) * inv2dx
        nbz = Bz[c] + dt * ((cxp_y - cxm_y) - (cyp_x - cym_x)) * inv2dx
        nmx = nrho * vx
        nmy = nrho * vy
        nmz = nrho * vz
        ek1 = T(0.5) * (nmx * nmx + nmy * nmy + nmz * nmz) / nrho
        eb1 = T(0.5) * (nbx * nbx + nby * nby + nbz * nbz)
        orho[c] = nrho
        omx[c] = nmx; omy[c] = nmy; omz[c] = nmz
        oE[c] = max(eint * (nrho / r) + ek1 + eb1, smallE)
        oBx[c] = nbx; oBy[c] = nby; oBz[c] = nbz
        opsi[c] = psi[c]
    end
end

@kernel function _terminal_velocity_step_k!(orho, omx, omy, omz, oE, oBx, oBy, oBz, opsi,
                                            @Const(rho), @Const(mx), @Const(my), @Const(mz), @Const(E),
                                            @Const(Bx), @Const(By), @Const(Bz), @Const(psi),
                                            N::Int, dt, inv2dx, gamma_code, gamma_gas, smallr, smallE, vcap)
    c = @index(Global)
    @inbounds begin
        T = eltype(E)
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N * N) + 1

        r = max(rho[c], smallr)
        ek0 = T(0.5) * (mx[c] * mx[c] + my[c] * my[c] + mz[c] * mz[c]) / r
        eb0 = T(0.5) * (Bx[c] * Bx[c] + By[c] * By[c] + Bz[c] * Bz[c])
        eint = max(E[c] - ek0 - eb0, smallE)

        vx, vy, vz = _terminal_v_at(rho, mx, my, mz, E, Bx, By, Bz, i, j, k, N, inv2dx, gamma_code, gamma_gas, smallr, smallE, vcap)
        vxp_x, _, _ = _terminal_v_at(rho, mx, my, mz, E, Bx, By, Bz, i + 1, j, k, N, inv2dx, gamma_code, gamma_gas, smallr, smallE, vcap)
        vxm_x, _, _ = _terminal_v_at(rho, mx, my, mz, E, Bx, By, Bz, i - 1, j, k, N, inv2dx, gamma_code, gamma_gas, smallr, smallE, vcap)
        _, vyp_y, _ = _terminal_v_at(rho, mx, my, mz, E, Bx, By, Bz, i, j + 1, k, N, inv2dx, gamma_code, gamma_gas, smallr, smallE, vcap)
        _, vym_y, _ = _terminal_v_at(rho, mx, my, mz, E, Bx, By, Bz, i, j - 1, k, N, inv2dx, gamma_code, gamma_gas, smallr, smallE, vcap)
        _, _, vzp_z = _terminal_v_at(rho, mx, my, mz, E, Bx, By, Bz, i, j, k + 1, N, inv2dx, gamma_code, gamma_gas, smallr, smallE, vcap)
        _, _, vzm_z = _terminal_v_at(rho, mx, my, mz, E, Bx, By, Bz, i, j, k - 1, N, inv2dx, gamma_code, gamma_gas, smallr, smallE, vcap)
        cxp = _lin3(i + 1, j, k, N); cxm = _lin3(i - 1, j, k, N)
        cyp = _lin3(i, j + 1, k, N); cym = _lin3(i, j - 1, k, N)
        czp = _lin3(i, j, k + 1, N); czm = _lin3(i, j, k - 1, N)
        vxf_p = T(0.5) * (vx + vxp_x)
        vxf_m = T(0.5) * (vxm_x + vx)
        vyf_p = T(0.5) * (vy + vyp_y)
        vyf_m = T(0.5) * (vym_y + vy)
        vzf_p = T(0.5) * (vz + vzp_z)
        vzf_m = T(0.5) * (vzm_z + vz)
        fx_p = vxf_p * max(ifelse(vxf_p >= zero(T), rho[c], rho[cxp]), smallr)
        fx_m = vxf_m * max(ifelse(vxf_m >= zero(T), rho[cxm], rho[c]), smallr)
        fy_p = vyf_p * max(ifelse(vyf_p >= zero(T), rho[c], rho[cyp]), smallr)
        fy_m = vyf_m * max(ifelse(vyf_m >= zero(T), rho[cym], rho[c]), smallr)
        fz_p = vzf_p * max(ifelse(vzf_p >= zero(T), rho[c], rho[czp]), smallr)
        fz_m = vzf_m * max(ifelse(vzf_m >= zero(T), rho[czm], rho[c]), smallr)
        div_rhov = ((fx_p - fx_m) + (fy_p - fy_m) + (fz_p - fz_m)) * (T(2) * inv2dx)
        nrho = max(rho[c] - dt * div_rhov, smallr)
        cyp_x, cyp_y, cyp_z = _vxb_at(rho, mx, my, mz, E, Bx, By, Bz, i, j + 1, k, N, inv2dx, gamma_code, gamma_gas, smallr, smallE, vcap)
        cym_x, cym_y, cym_z = _vxb_at(rho, mx, my, mz, E, Bx, By, Bz, i, j - 1, k, N, inv2dx, gamma_code, gamma_gas, smallr, smallE, vcap)
        czp_x, czp_y, czp_z = _vxb_at(rho, mx, my, mz, E, Bx, By, Bz, i, j, k + 1, N, inv2dx, gamma_code, gamma_gas, smallr, smallE, vcap)
        czm_x, czm_y, czm_z = _vxb_at(rho, mx, my, mz, E, Bx, By, Bz, i, j, k - 1, N, inv2dx, gamma_code, gamma_gas, smallr, smallE, vcap)
        cxp_x, cxp_y, cxp_z = _vxb_at(rho, mx, my, mz, E, Bx, By, Bz, i + 1, j, k, N, inv2dx, gamma_code, gamma_gas, smallr, smallE, vcap)
        cxm_x, cxm_y, cxm_z = _vxb_at(rho, mx, my, mz, E, Bx, By, Bz, i - 1, j, k, N, inv2dx, gamma_code, gamma_gas, smallr, smallE, vcap)

        nbx = Bx[c] + dt * ((cyp_z - cym_z) - (czp_y - czm_y)) * inv2dx
        nby = By[c] + dt * ((czp_x - czm_x) - (cxp_z - cxm_z)) * inv2dx
        nbz = Bz[c] + dt * ((cxp_y - cxm_y) - (cyp_x - cym_x)) * inv2dx
        nmx = nrho * vx
        nmy = nrho * vy
        nmz = nrho * vz
        ek1 = T(0.5) * (nmx * nmx + nmy * nmy + nmz * nmz) / nrho
        eb1 = T(0.5) * (nbx * nbx + nby * nby + nbz * nbz)
        orho[c] = nrho
        omx[c] = nmx; omy[c] = nmy; omz[c] = nmz
        oE[c] = max(eint * (nrho / r) + ek1 + eb1, smallE)
        oBx[c] = nbx; oBy[c] = nby; oBz[c] = nbz
        opsi[c] = psi[c]
    end
end

@kernel function _pressure_implicit_finalize_k!(orho, omx, omy, omz, oE, oBx, oBy, oBz, opsi,
                                                @Const(rho), @Const(mx), @Const(my), @Const(mz), @Const(E),
                                                @Const(Bx), @Const(By), @Const(Bz), @Const(psi),
                                                @Const(rho_new), @Const(fxm), @Const(fym), @Const(fzm),
                                                N::Int, dt, inv2dx, gamma_code, pressure_coeff,
                                                gamma_gas, smallr, smallE, vcap)
    c = @index(Global)
    @inbounds begin
        T = eltype(E)
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N * N) + 1

        r0 = max(rho[c], smallr)
        ek0 = T(0.5) * (mx[c] * mx[c] + my[c] * my[c] + mz[c] * mz[c]) / r0
        eb0 = T(0.5) * (Bx[c] * Bx[c] + By[c] * By[c] + Bz[c] * Bz[c])
        eint = max(E[c] - ek0 - eb0, smallE)

        nrho = max(rho_new[c], smallr)
        vx, vy, vz = _pressure_implicit_v_at(rho_new, fxm, fym, fzm, i, j, k, N,
                                             inv2dx, gamma_code, pressure_coeff,
                                             smallr, vcap)
        cyp_x, cyp_y, cyp_z = _vxb_pressure_implicit_at(rho_new, fxm, fym, fzm, Bx, By, Bz,
                                                        i, j + 1, k, N, inv2dx, gamma_code,
                                                        pressure_coeff, smallr, vcap)
        cym_x, cym_y, cym_z = _vxb_pressure_implicit_at(rho_new, fxm, fym, fzm, Bx, By, Bz,
                                                        i, j - 1, k, N, inv2dx, gamma_code,
                                                        pressure_coeff, smallr, vcap)
        czp_x, czp_y, czp_z = _vxb_pressure_implicit_at(rho_new, fxm, fym, fzm, Bx, By, Bz,
                                                        i, j, k + 1, N, inv2dx, gamma_code,
                                                        pressure_coeff, smallr, vcap)
        czm_x, czm_y, czm_z = _vxb_pressure_implicit_at(rho_new, fxm, fym, fzm, Bx, By, Bz,
                                                        i, j, k - 1, N, inv2dx, gamma_code,
                                                        pressure_coeff, smallr, vcap)
        cxp_x, cxp_y, cxp_z = _vxb_pressure_implicit_at(rho_new, fxm, fym, fzm, Bx, By, Bz,
                                                        i + 1, j, k, N, inv2dx, gamma_code,
                                                        pressure_coeff, smallr, vcap)
        cxm_x, cxm_y, cxm_z = _vxb_pressure_implicit_at(rho_new, fxm, fym, fzm, Bx, By, Bz,
                                                        i - 1, j, k, N, inv2dx, gamma_code,
                                                        pressure_coeff, smallr, vcap)

        nbx = Bx[c] + dt * ((cyp_z - cym_z) - (czp_y - czm_y)) * inv2dx
        nby = By[c] + dt * ((czp_x - czm_x) - (cxp_z - cxm_z)) * inv2dx
        nbz = Bz[c] + dt * ((cxp_y - cxm_y) - (cyp_x - cym_x)) * inv2dx
        nmx = nrho * vx
        nmy = nrho * vy
        nmz = nrho * vz
        ek1 = T(0.5) * (nmx * nmx + nmy * nmy + nmz * nmz) / nrho
        eb1 = T(0.5) * (nbx * nbx + nby * nby + nbz * nbz)
        orho[c] = nrho
        omx[c] = nmx; omy[c] = nmy; omz[c] = nmz
        oE[c] = max(eint * (nrho / r0)^gamma_gas + ek1 + eb1, smallE)
        oBx[c] = nbx; oBy[c] = nby; oBz[c] = nbz
        opsi[c] = psi[c]
    end
end

@kernel function _stash_eint_and_force_k!(eint, mx, my, mz,
                                          @Const(rho), @Const(E),
                                          @Const(Bx), @Const(By), @Const(Bz),
                                          @Const(fx), @Const(fy), @Const(fz),
                                          smallr, smallE)
    c = @index(Global)
    @inbounds begin
        T = eltype(E)
        r = max(rho[c], smallr)
        ek = T(0.5) * (mx[c] * mx[c] + my[c] * my[c] + mz[c] * mz[c]) / r
        eb = T(0.5) * (Bx[c] * Bx[c] + By[c] * By[c] + Bz[c] * Bz[c])
        eint[c] = max(E[c] - ek - eb, smallE)
        mx[c] = fx[c]; my[c] = fy[c]; mz[c] = fz[c]
    end
end

@kernel function _pressure_implicit_hydro_finalize_k!(orho, omx, omy, omz, oE, oBx, oBy, oBz, opsi,
                                                      @Const(rho),
                                                      @Const(Bx), @Const(By), @Const(Bz), @Const(psi),
                                                      @Const(eint), @Const(rho_new),
                                                      @Const(fxm), @Const(fym), @Const(fzm),
                                                      N::Int, inv2dx, gamma_code, pressure_coeff,
                                                      gamma_gas, smallr, smallE, vcap)
    c = @index(Global)
    @inbounds begin
        T = eltype(eint)
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N * N) + 1
        r0 = max(rho[c], smallr)
        eb = T(0.5) * (Bx[c] * Bx[c] + By[c] * By[c] + Bz[c] * Bz[c])
        nr = max(rho_new[c], smallr)
        vx, vy, vz = _pressure_implicit_v_at(rho_new, fxm, fym, fzm, i, j, k, N,
                                             inv2dx, gamma_code, pressure_coeff,
                                             smallr, vcap)
        nmx = nr * vx
        nmy = nr * vy
        nmz = nr * vz
        ek = T(0.5) * (nmx * nmx + nmy * nmy + nmz * nmz) / nr
        orho[c] = nr
        omx[c] = nmx; omy[c] = nmy; omz[c] = nmz
        oE[c] = max(eint[c] * (nr / r0)^gamma_gas + ek + eb, smallE)
        oBx[c] = Bx[c]; oBy[c] = By[c]; oBz[c] = Bz[c]
        opsi[c] = psi[c]
    end
end

@inline function _minmod3(a, b, c)
    if a > zero(a) && b > zero(b) && c > zero(c)
        return min(a, min(b, c))
    elseif a < zero(a) && b < zero(b) && c < zero(c)
        return max(a, max(b, c))
    end
    return zero(a)
end

@inline function _plm_face(qll, ql, qr, qrr)
    T = typeof(ql)
    sl = _minmod3(T(1.5) * (ql - qll), T(0.5) * (qr - qll), T(1.5) * (qr - ql))
    sr = _minmod3(T(1.5) * (qr - ql), T(0.5) * (qrr - ql), T(1.5) * (qrr - qr))
    return ql + T(0.5) * sl, qr - T(0.5) * sr
end

@inline function _induction_flux_x_at(rho, fx, fy, fz, Bx, By, Bz,
                                      i, j, k, N, inv2dx, gamma_code,
                                      pressure_coeff, smallr, vcap)
    ll = _lin3(i - 1, j, k, N)
    l = _lin3(i, j, k, N)
    r = _lin3(i + 1, j, k, N)
    rr = _lin3(i + 2, j, k, N)
    vlx, vly, vlz = _pressure_implicit_v_at(rho, fx, fy, fz, i, j, k, N,
                                             inv2dx, gamma_code, pressure_coeff, smallr, vcap)
    vrx, vry, vrz = _pressure_implicit_v_at(rho, fx, fy, fz, i + 1, j, k, N,
                                             inv2dx, gamma_code, pressure_coeff, smallr, vcap)
    a = max(abs(vlx), abs(vrx))
    z = zero(a)
    bxl, bxr = _plm_face(Bx[ll], Bx[l], Bx[r], Bx[rr])
    byl, byr = _plm_face(By[ll], By[l], By[r], By[rr])
    bzl, bzr = _plm_face(Bz[ll], Bz[l], Bz[r], Bz[rr])
    fby_l = vlx * byl - vly * bxl
    fby_r = vrx * byr - vry * bxr
    fbz_l = vlx * bzl - vlz * bxl
    fbz_r = vrx * bzr - vrz * bxr
    half = one(a) / typeof(a)(2)
    return z, (fby_l + fby_r - a * (byr - byl)) * half,
           (fbz_l + fbz_r - a * (bzr - bzl)) * half
end

@inline function _induction_flux_y_at(rho, fx, fy, fz, Bx, By, Bz,
                                      i, j, k, N, inv2dx, gamma_code,
                                      pressure_coeff, smallr, vcap)
    ll = _lin3(i, j - 1, k, N)
    l = _lin3(i, j, k, N)
    r = _lin3(i, j + 1, k, N)
    rr = _lin3(i, j + 2, k, N)
    vlx, vly, vlz = _pressure_implicit_v_at(rho, fx, fy, fz, i, j, k, N,
                                             inv2dx, gamma_code, pressure_coeff, smallr, vcap)
    vrx, vry, vrz = _pressure_implicit_v_at(rho, fx, fy, fz, i, j + 1, k, N,
                                             inv2dx, gamma_code, pressure_coeff, smallr, vcap)
    a = max(abs(vly), abs(vry))
    z = zero(a)
    bxl, bxr = _plm_face(Bx[ll], Bx[l], Bx[r], Bx[rr])
    byl, byr = _plm_face(By[ll], By[l], By[r], By[rr])
    bzl, bzr = _plm_face(Bz[ll], Bz[l], Bz[r], Bz[rr])
    fbx_l = vly * bxl - vlx * byl
    fbx_r = vry * bxr - vrx * byr
    fbz_l = vly * bzl - vlz * byl
    fbz_r = vry * bzr - vrz * byr
    half = one(a) / typeof(a)(2)
    return (fbx_l + fbx_r - a * (bxr - bxl)) * half, z,
           (fbz_l + fbz_r - a * (bzr - bzl)) * half
end

@inline function _induction_flux_z_at(rho, fx, fy, fz, Bx, By, Bz,
                                      i, j, k, N, inv2dx, gamma_code,
                                      pressure_coeff, smallr, vcap)
    ll = _lin3(i, j, k - 1, N)
    l = _lin3(i, j, k, N)
    r = _lin3(i, j, k + 1, N)
    rr = _lin3(i, j, k + 2, N)
    vlx, vly, vlz = _pressure_implicit_v_at(rho, fx, fy, fz, i, j, k, N,
                                             inv2dx, gamma_code, pressure_coeff, smallr, vcap)
    vrx, vry, vrz = _pressure_implicit_v_at(rho, fx, fy, fz, i, j, k + 1, N,
                                             inv2dx, gamma_code, pressure_coeff, smallr, vcap)
    a = max(abs(vlz), abs(vrz))
    z = zero(a)
    bxl, bxr = _plm_face(Bx[ll], Bx[l], Bx[r], Bx[rr])
    byl, byr = _plm_face(By[ll], By[l], By[r], By[rr])
    bzl, bzr = _plm_face(Bz[ll], Bz[l], Bz[r], Bz[rr])
    fbx_l = vlz * bxl - vlx * bzl
    fbx_r = vrz * bxr - vrx * bzr
    fby_l = vlz * byl - vly * bzl
    fby_r = vrz * byr - vry * bzr
    half = one(a) / typeof(a)(2)
    return (fbx_l + fbx_r - a * (bxr - bxl)) * half,
           (fby_l + fby_r - a * (byr - byl)) * half, z
end

@inline function _induction_rhs_at(rho, fx, fy, fz, Bx, By, Bz,
                                   i, j, k, N, inv2dx, gamma_code,
                                   pressure_coeff, smallr, vcap)
    fxp = _induction_flux_x_at(rho, fx, fy, fz, Bx, By, Bz, i, j, k, N,
                                inv2dx, gamma_code, pressure_coeff, smallr, vcap)
    fxm = _induction_flux_x_at(rho, fx, fy, fz, Bx, By, Bz, i - 1, j, k, N,
                                inv2dx, gamma_code, pressure_coeff, smallr, vcap)
    fyp = _induction_flux_y_at(rho, fx, fy, fz, Bx, By, Bz, i, j, k, N,
                                inv2dx, gamma_code, pressure_coeff, smallr, vcap)
    fym = _induction_flux_y_at(rho, fx, fy, fz, Bx, By, Bz, i, j - 1, k, N,
                                inv2dx, gamma_code, pressure_coeff, smallr, vcap)
    fzp = _induction_flux_z_at(rho, fx, fy, fz, Bx, By, Bz, i, j, k, N,
                                inv2dx, gamma_code, pressure_coeff, smallr, vcap)
    fzm = _induction_flux_z_at(rho, fx, fy, fz, Bx, By, Bz, i, j, k - 1, N,
                                inv2dx, gamma_code, pressure_coeff, smallr, vcap)
    invdx = typeof(inv2dx)(2) * inv2dx
    return -((fxp[1] - fxm[1]) + (fyp[1] - fym[1]) + (fzp[1] - fzm[1])) * invdx,
           -((fxp[2] - fxm[2]) + (fyp[2] - fym[2]) + (fzp[2] - fzm[2])) * invdx,
           -((fxp[3] - fxm[3]) + (fyp[3] - fym[3]) + (fzp[3] - fzm[3])) * invdx
end

@kernel function _induction_predict_k!(pBx, pBy, pBz,
                                       @Const(rho), @Const(fx), @Const(fy), @Const(fz),
                                       @Const(Bx), @Const(By), @Const(Bz),
                                       N::Int, dt, inv2dx, gamma_code, pressure_coeff,
                                       smallr, vcap)
    c = @index(Global)
    @inbounds begin
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N * N) + 1
        dbx, dby, dbz = _induction_rhs_at(rho, fx, fy, fz, Bx, By, Bz, i, j, k, N,
                                          inv2dx, gamma_code, pressure_coeff, smallr, vcap)
        pBx[c] = Bx[c] + dt * dbx
        pBy[c] = By[c] + dt * dby
        pBz[c] = Bz[c] + dt * dbz
    end
end

@kernel function _induction_correct_k!(Bx, By, Bz, E,
                                       @Const(rho), @Const(fx), @Const(fy), @Const(fz),
                                       @Const(pBx), @Const(pBy), @Const(pBz),
                                       N::Int, dt, inv2dx, gamma_code, pressure_coeff,
                                       smallr, smallE, vcap)
    c = @index(Global)
    @inbounds begin
        T = eltype(E)
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N * N) + 1
        dbx, dby, dbz = _induction_rhs_at(rho, fx, fy, fz, pBx, pBy, pBz, i, j, k, N,
                                          inv2dx, gamma_code, pressure_coeff, smallr, vcap)
        obx = Bx[c]; oby = By[c]; obz = Bz[c]
        nbx = T(0.5) * (obx + pBx[c] + dt * dbx)
        nby = T(0.5) * (oby + pBy[c] + dt * dby)
        nbz = T(0.5) * (obz + pBz[c] + dt * dbz)
        eb0 = T(0.5) * (obx * obx + oby * oby + obz * obz)
        eb1 = T(0.5) * (nbx * nbx + nby * nby + nbz * nbz)
        Bx[c] = nbx; By[c] = nby; Bz[c] = nbz
        E[c] = max(E[c] + eb1 - eb0, smallE)
    end
end

@kernel function _divb_centered_k!(divb, @Const(Bx), @Const(By), @Const(Bz), N::Int, inv2dx)
    c = @index(Global)
    @inbounds begin
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N * N) + 1
        divb[c] = ((Bx[_lin3(i + 1, j, k, N)] - Bx[_lin3(i - 1, j, k, N)]) +
                   (By[_lin3(i, j + 1, k, N)] - By[_lin3(i, j - 1, k, N)]) +
                   (Bz[_lin3(i, j, k + 1, N)] - Bz[_lin3(i, j, k - 1, N)])) * inv2dx
    end
end

@kernel function _project_b_centered_k!(Bx, By, Bz, E, @Const(phi),
                                        N::Int, inv2dx, smallE)
    c = @index(Global)
    @inbounds begin
        T = eltype(E)
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N * N) + 1
        obx = Bx[c]; oby = By[c]; obz = Bz[c]
        nbx = obx - (phi[_lin3(i + 1, j, k, N)] - phi[_lin3(i - 1, j, k, N)]) * inv2dx
        nby = oby - (phi[_lin3(i, j + 1, k, N)] - phi[_lin3(i, j - 1, k, N)]) * inv2dx
        nbz = obz - (phi[_lin3(i, j, k + 1, N)] - phi[_lin3(i, j, k - 1, N)]) * inv2dx
        eb0 = T(0.5) * (obx * obx + oby * oby + obz * obz)
        eb1 = T(0.5) * (nbx * nbx + nby * nby + nbz * nbz)
        Bx[c] = nbx; By[c] = nby; Bz[c] = nbz
        E[c] = max(E[c] + eb1 - eb0, smallE)
    end
end

@kernel function _pressure_velocity_speed_k!(speed, @Const(rho), @Const(fx), @Const(fy), @Const(fz),
                                             N::Int, inv2dx, gamma_code, pressure_coeff,
                                             smallr, vcap)
    c = @index(Global)
    @inbounds begin
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N * N) + 1
        vx, vy, vz = _pressure_implicit_v_at(rho, fx, fy, fz, i, j, k, N,
                                             inv2dx, gamma_code, pressure_coeff,
                                             smallr, vcap)
        speed[c] = sqrt(vx * vx + vy * vy + vz * vz)
    end
end

@kernel function _terminal_nonlinear_limit_k!(rho, mx, my, mz, E, Bx, By, Bz,
                                              @Const(rho0), @Const(Bx0), @Const(By0), @Const(Bz0),
                                              smallr, smallE, rho_rel_limit, b_rel_limit)
    c = @index(Global)
    @inbounds begin
        T = eltype(E)
        r_before = max(rho[c], smallr)
        r_after = r_before
        if isfinite(rho_rel_limit)
            rref = max(rho0[c], smallr)
            rlo = rref / (one(T) + rho_rel_limit)
            rhi = rref * (one(T) + rho_rel_limit)
            r_after = min(max(r_before, max(rlo, smallr)), rhi)
        end

        bx_before = Bx[c]
        by_before = By[c]
        bz_before = Bz[c]
        bx_after = bx_before
        by_after = by_before
        bz_after = bz_before
        if isfinite(b_rel_limit)
            dbx = bx_before - Bx0[c]
            dby = by_before - By0[c]
            dbz = bz_before - Bz0[c]
            bmag = sqrt(Bx0[c] * Bx0[c] + By0[c] * By0[c] + Bz0[c] * Bz0[c])
            dbmag = sqrt(dbx * dbx + dby * dby + dbz * dbz)
            maxdb = b_rel_limit * max(bmag, smallE)
            if dbmag > maxdb
                fac = maxdb / max(dbmag, smallE)
                bx_after = Bx0[c] + fac * dbx
                by_after = By0[c] + fac * dby
                bz_after = Bz0[c] + fac * dbz
            end
        end

        if r_after != r_before || bx_after != bx_before || by_after != by_before || bz_after != bz_before
            mom_scale = r_after / max(r_before, smallr)
            ek_before = T(0.5) * (mx[c] * mx[c] + my[c] * my[c] + mz[c] * mz[c]) / max(r_before, smallr)
            eb_before = T(0.5) * (bx_before * bx_before + by_before * by_before + bz_before * bz_before)
            eint = max(E[c] - ek_before - eb_before, smallE)
            mx[c] *= mom_scale
            my[c] *= mom_scale
            mz[c] *= mom_scale
            ek_after = T(0.5) * (mx[c] * mx[c] + my[c] * my[c] + mz[c] * mz[c]) / max(r_after, smallr)
            eb_after = T(0.5) * (bx_after * bx_after + by_after * by_after + bz_after * bz_after)
            rho[c] = r_after
            Bx[c] = bx_after
            By[c] = by_after
            Bz[c] = bz_after
            E[c] = max(eint * (r_after / max(r_before, smallr)) + ek_after + eb_after, smallE)
        end
    end
end

function _terminal_velocity_step!(be, s, dt::Real, gamma_code::Real, vcap::Real)
    _terminal_velocity_step_k!(be)(s.scratch..., s.U...,
                                   Int(s.dims[1]), T(dt), T(1) / (T(2) * s.dx),
                                   T(gamma_code), T(s.γ), T(s.smallr), T(1e-30), T(vcap);
                                   ndrange=length(s.U[1]))
    KA.synchronize(be)
    s.U, s.scratch = s.scratch, s.U
    return nothing
end

function _terminal_pressure_exponential_solve!(dst, state, source, coeff_cells2, dt)
    PoissonKernels.rfft_exponential_source!(dst, state, source;
                                            coeff_cells2=coeff_cells2, dt=dt)
    return dst
end

function _project_mhd_b!(be, s)
    N = Int(s.dims[1])
    inv2dx = T(1) / (T(2) * s.dx)
    divb = s.scratch[1]
    phi = s.scratch[5]
    _divb_centered_k!(be)(divb, s.U[6], s.U[7], s.U[8], N, inv2dx;
                          ndrange=length(s.U[1]))
    PoissonKernels.fft_poisson_rfft!(reshape(phi, N, N, N),
                                     reshape(divb, N, N, N);
                                     boxsize=Float64(N) * Float64(s.dx),
                                     greens=:centered)
    _project_b_centered_k!(be)(s.U[6], s.U[7], s.U[8], s.U[5], phi,
                                N, inv2dx, T(1e-30); ndrange=length(s.U[1]))
    fill!(s.U[9], zero(T))
    KA.synchronize(be)
    return nothing
end

function _terminal_pressure_implicit_step!(be, s, dt::Real, gamma_code::Real,
                                           vcap::Real, pressure_coeff::Real,
                                           rho_rel_limit::Real, b_rel_limit::Real;
                                           induction::Symbol=:centered,
                                           induction_cfl::Real=0.4,
                                           induction_dissipation::Real=0.05,
                                           induction_dissipation_order::Int=4,
                                           project_b::Bool=true,
                                           max_induction_subcycles::Int=256)
    force_s = 0.0
    pressure_fft_s = 0.0
    finalize_s = 0.0
    induction_s = 0.0
    projection_s = 0.0
    if !(isfinite(gamma_code) && gamma_code > 0)
        finalize_s = @elapsed _terminal_velocity_step!(be, s, dt, gamma_code, vcap)
        return (nsub=1, force_s, pressure_fft_s, finalize_s, induction_s, projection_s)
    end
    if induction === :ct
        result = MHDKernels.terminal_pressure_ct_step!(_terminal_pressure_exponential_solve!, s, dt;
            gamma_drag=gamma_code, pressure_coeff=pressure_coeff,
            induction_cfl=induction_cfl,
            induction_dissipation=induction_dissipation,
            induction_dissipation_order=induction_dissipation_order,
            vcap=vcap, max_induction_subcycles=max_induction_subcycles)
        if project_b
            projection_s = @elapsed _project_mhd_b!(be, s)
            return merge(result, (; projection_s))
        end
        return result
    end
    N = Int(s.dims[1])
    inv2dx = T(1) / (T(2) * s.dx)
    fx = s.scratch[2]
    fy = s.scratch[3]
    fz = s.scratch[4]
    source = s.scratch[1]
    sol = s.scratch[5]
    force_s = @elapsed begin
        _magnetic_force_field_k!(be)(fx, fy, fz, s.U[6], s.U[7], s.U[8],
                                     N, inv2dx; ndrange=length(s.U[1]))
        _pressure_implicit_source_k!(be)(source, s.U[1], fx, fy, fz,
                                         N, inv2dx, T(gamma_code), T(s.smallr), T(vcap);
                                         ndrange=length(s.U[1]))
        KA.synchronize(be)
    end
    coeff_cells2 = Float64(dt) * max(Float64(pressure_coeff), 0.0) /
                   max(Float64(gamma_code), eps(Float64)) /
                   max(Float64(s.dx)^2, eps(Float64))
    pressure_fft_s = @elapsed begin
        PoissonKernels.rfft_exponential_source!(reshape(sol, N, N, N),
                                                reshape(s.U[1], N, N, N),
                                                reshape(source, N, N, N);
                                                coeff_cells2=coeff_cells2, dt=dt)
        copyto!(source, sol)
        KA.synchronize(be)
    end
    if induction in (:ssprk2, :ct)
        # Preserve thermal energy before reusing the old momentum arrays for
        # non-aliased Lorentz-force storage during the hydro finalize.
        finalize_s = @elapsed begin
            _stash_eint_and_force_k!(be)(sol, s.U[2], s.U[3], s.U[4],
                                         s.U[1], s.U[5], s.U[6], s.U[7], s.U[8],
                                         fx, fy, fz, T(s.smallr), T(1e-30);
                                         ndrange=length(s.U[1]))
            _pressure_implicit_hydro_finalize_k!(be)(s.scratch..., s.U[1],
                                                      s.U[6], s.U[7], s.U[8], s.U[9],
                                                      sol, source, s.U[2], s.U[3], s.U[4],
                                                      N, inv2dx, T(gamma_code), T(pressure_coeff),
                                                      T(s.γ), T(s.smallr), T(1e-30), T(vcap);
                                                      ndrange=length(s.U[1]))
            KA.synchronize(be)
            s.U, s.scratch = s.scratch, s.U
        end

        # The induction transport has its own CFL.  The pressure response stays
        # at macro cadence; only the nonlinear v x B coupling is subcycled.
        fx = s.scratch[2]; fy = s.scratch[3]; fz = s.scratch[4]
        speed = s.scratch[1]
        _magnetic_force_field_k!(be)(fx, fy, fz, s.U[6], s.U[7], s.U[8],
                                     N, inv2dx; ndrange=length(s.U[1]))
        _pressure_velocity_speed_k!(be)(speed, s.U[1], fx, fy, fz,
                                        N, inv2dx, T(gamma_code), T(pressure_coeff),
                                        T(s.smallr), T(vcap); ndrange=length(s.U[1]))
        KA.synchronize(be)
        vmax = Float64(maximum(speed))
        nind = max(1, ceil(Int, Float64(dt) * vmax /
                               max(Float64(induction_cfl) * Float64(s.dx), eps(Float64))))
        nind <= max_induction_subcycles ||
            error("terminal induction needs $nind subcycles, above MHD_TERMINAL_INDUCTION_MAX_SUBCYCLES=$max_induction_subcycles")
        dti = T(dt / nind)
        induction_s = @elapsed begin
            for _ in 1:nind
                _magnetic_force_field_k!(be)(fx, fy, fz, s.U[6], s.U[7], s.U[8],
                                             N, inv2dx; ndrange=length(s.U[1]))
                _induction_predict_k!(be)(s.scratch[6], s.scratch[7], s.scratch[8],
                                          s.U[1], fx, fy, fz, s.U[6], s.U[7], s.U[8],
                                          N, dti, inv2dx, T(gamma_code), T(pressure_coeff),
                                          T(s.smallr), T(vcap); ndrange=length(s.U[1]))
                _magnetic_force_field_k!(be)(fx, fy, fz,
                                             s.scratch[6], s.scratch[7], s.scratch[8],
                                             N, inv2dx; ndrange=length(s.U[1]))
                _induction_correct_k!(be)(s.U[6], s.U[7], s.U[8], s.U[5],
                                          s.U[1], fx, fy, fz,
                                          s.scratch[6], s.scratch[7], s.scratch[8],
                                          N, dti, inv2dx, T(gamma_code), T(pressure_coeff),
                                          T(s.smallr), T(1e-30), T(vcap);
                                          ndrange=length(s.U[1]))
            end
            KA.synchronize(be)
        end
        if project_b
            projection_s = @elapsed begin
                divb = s.scratch[1]
                phi = s.scratch[5]
                _divb_centered_k!(be)(divb, s.U[6], s.U[7], s.U[8], N, inv2dx;
                                      ndrange=length(s.U[1]))
                PoissonKernels.fft_poisson_rfft!(reshape(phi, N, N, N),
                                                 reshape(divb, N, N, N);
                                                 boxsize=Float64(N) * Float64(s.dx),
                                                 greens=:centered)
                _project_b_centered_k!(be)(s.U[6], s.U[7], s.U[8], s.U[5], phi,
                                            N, inv2dx, T(1e-30);
                                            ndrange=length(s.U[1]))
                KA.synchronize(be)
            end
        end
        return (nsub=nind, force_s, pressure_fft_s, finalize_s, induction_s, projection_s)
    elseif induction !== :centered
        error("unknown terminal induction mode $induction")
    end
    finalize_s = @elapsed begin
        _pressure_implicit_finalize_k!(be)(s.scratch..., s.U..., source, fx, fy, fz,
                                           N, T(dt), inv2dx, T(gamma_code), T(pressure_coeff),
                                           T(s.γ), T(s.smallr), T(1e-30), T(vcap);
                                           ndrange=length(s.U[1]))
        if isfinite(rho_rel_limit) || isfinite(b_rel_limit)
            _terminal_nonlinear_limit_k!(be)(s.scratch[1], s.scratch[2], s.scratch[3], s.scratch[4],
                                             s.scratch[5], s.scratch[6], s.scratch[7], s.scratch[8],
                                             s.U[1], s.U[6], s.U[7], s.U[8],
                                             T(s.smallr), T(1e-30), T(rho_rel_limit), T(b_rel_limit);
                                             ndrange=length(s.U[1]))
        end
        KA.synchronize(be)
        s.U, s.scratch = s.scratch, s.U
    end
    return (nsub=1, force_s, pressure_fft_s, finalize_s, induction_s, projection_s)
end

function _terminal_velocity_step_cached!(be, s, dt::Real, gamma_code::Real, vcap::Real,
                                         tvx, tvy, tvz, vsmooth::Real)
    _terminal_velocity_field_k!(be)(tvx, tvy, tvz, s.U[1], s.U[2], s.U[3], s.U[4],
                                    s.U[5], s.U[6], s.U[7], s.U[8],
                                    Int(s.dims[1]), T(1) / (T(2) * s.dx),
                                    T(gamma_code), T(s.γ), T(s.smallr), T(1e-30), T(vcap);
                                    ndrange=length(s.U[1]))
    KA.synchronize(be)
    _terminal_velocity_update_cached_k!(be)(s.scratch..., s.U..., tvx, tvy, tvz,
                                            Int(s.dims[1]), T(dt), T(1) / (T(2) * s.dx),
                                            T(s.γ), T(s.smallr), T(1e-30), T(vsmooth);
                                            ndrange=length(s.U[1]))
    KA.synchronize(be)
    s.U, s.scratch = s.scratch, s.U
    return nothing
end

@kernel function _terminal_handoff_momenta_cached_k!(mx, my, mz, E,
                                                     @Const(rho), @Const(Bx), @Const(By), @Const(Bz),
                                                     @Const(tvx), @Const(tvy), @Const(tvz),
                                                     smallr, smallE, alpha)
    c = @index(Global)
    @inbounds begin
        R = eltype(E)
        r = max(rho[c], smallr)
        ox = mx[c]
        oy = my[c]
        oz = mz[c]
        ek0 = R(0.5) * (ox * ox + oy * oy + oz * oz) / r
        eb = R(0.5) * (Bx[c] * Bx[c] + By[c] * By[c] + Bz[c] * Bz[c])
        eint = max(E[c] - ek0 - eb, smallE)
        a = min(max(alpha, zero(alpha)), one(alpha))
        nx = ox + a * (r * tvx[c] - ox)
        ny = oy + a * (r * tvy[c] - oy)
        nz = oz + a * (r * tvz[c] - oz)
        ek1 = R(0.5) * (nx * nx + ny * ny + nz * nz) / r
        mx[c] = nx
        my[c] = ny
        mz[c] = nz
        E[c] = max(eint + ek1 + eb, smallE)
    end
end

function _terminal_handoff_momenta!(be, s, gamma_code::Real, vcap::Real, alpha::Real;
                                    measure::Bool=false)
    tvx = s.scratch[2]
    tvy = s.scratch[3]
    tvz = s.scratch[4]
    _terminal_velocity_field_k!(be)(tvx, tvy, tvz, s.U[1], s.U[2], s.U[3], s.U[4],
                                    s.U[5], s.U[6], s.U[7], s.U[8],
                                    Int(s.dims[1]), T(1) / (T(2) * s.dx),
                                    T(gamma_code), T(s.γ), T(s.smallr), T(1e-30), T(vcap);
                                    ndrange=length(s.U[1]))
    KA.synchronize(be)
    overlap = (relative=NaN, delta_v_rms=NaN, terminal_v_rms=NaN, full_v_rms=NaN)
    if measure
        delta2 = s.scratch[1]
        terminal2 = s.scratch[5]
        full2 = s.scratch[9]
        _terminal_velocity_overlap_diag_k!(be)(delta2, terminal2, full2,
                                               s.U[1], s.U[2], s.U[3], s.U[4],
                                               tvx, tvy, tvz, T(s.smallr);
                                               ndrange=length(s.U[1]))
        KA.synchronize(be)
        n = Float64(length(s.U[1]))
        d2 = max(Float64(sum(delta2)), 0.0)
        t2 = max(Float64(sum(terminal2)), 0.0)
        f2 = max(Float64(sum(full2)), 0.0)
        scale2 = 0.5 * (t2 + f2)
        relative = scale2 > eps(Float64) ? sqrt(d2 / scale2) : 0.0
        overlap = (relative=relative,
                   delta_v_rms=sqrt(d2 / n),
                   terminal_v_rms=sqrt(t2 / n),
                   full_v_rms=sqrt(f2 / n))
    end
    _terminal_handoff_momenta_cached_k!(be)(s.U[2], s.U[3], s.U[4], s.U[5],
                                            s.U[1], s.U[6], s.U[7], s.U[8],
                                            tvx, tvy, tvz,
                                            T(s.smallr), T(1e-30), T(alpha);
                                            ndrange=length(s.U[1]))
    KA.synchronize(be)
    return overlap
end

function _hybrid_handoff_alpha(alpha_max::Real, remaining::Int, total::Int)
    total <= 1 && return Float64(alpha_max)
    progress = clamp(Float64(total - remaining) / Float64(total - 1), 0.0, 1.0)
    return Float64(alpha_max) * 0.5 * (1.0 + cos(pi * progress))
end

function _limit_source_dt(dt::Real, dt_explicit::Real, max_factor::Real)
    if isfinite(max_factor) && dt > max_factor * dt_explicit
        return (dt=Float64(max_factor * dt_explicit), limited=true)
    end
    return (dt=Float64(dt), limited=false)
end

function _limit_hybrid_aware_dt(dt::Real, dt_explicit::Real, max_factor::Real)
    limited_dt = min(Float64(dt), Float64(max_factor * dt_explicit))
    nsub = max(1, ceil(Int, limited_dt / max(Float64(dt_explicit), eps(Float64))))
    return (dt=limited_dt, nsub=nsub, limited=limited_dt < dt)
end

function _terminal_implicit_diffuse_B!(s, coeff_cells2::Real)
    coeff = Float64(coeff_cells2)
    if !(isfinite(coeff) && coeff > 0)
        return nothing
    end
    N = Int(s.dims[1])
    for v in 6:8
        dst = reshape(s.scratch[v], N, N, N)
        src = reshape(s.U[v], N, N, N)
        PoissonKernels.rfft_implicit_diffuse!(dst, src; coeff_cells2=coeff)
        copyto!(s.U[v], s.scratch[v])
    end
    KA.synchronize(s.be)
    return nothing
end

@kernel function _clamp_reduced_chem_k!(HII, H2I, @Const(rho), fh, N::Int)
    c = @index(Global)
    @inbounds begin
        T = eltype(HII)
        cap = max(fh * rho[c], zero(T))
        h2 = min(max(H2I[c], zero(T)), cap)
        hii = min(max(HII[c], zero(T)), max(cap - h2, zero(T)))
        H2I[c] = h2
        HII[c] = hii
    end
end

@kernel function _clamp_reduced_helium_k!(HeII, @Const(rho), fh, N::Int)
    c = @index(Global)
    @inbounds begin
        T = eltype(HeII)
        cap = max((one(T) - T(fh)) * rho[c], zero(T))
        HeII[c] = min(max(HeII[c], zero(T)), cap)
    end
end

@kernel function _neutral_h_mass_k!(dst, @Const(rho), @Const(HII), @Const(H2I), fh, N::Int)
    c = @index(Global)
    @inbounds begin
        T = eltype(dst)
        dst[c] = max(fh * rho[c] - HII[c] - H2I[c], zero(T))
    end
end

@kernel function _b2_diag_k!(dst, @Const(Bx), @Const(By), @Const(Bz))
    c = @index(Global, Linear)
    @inbounds begin
        bx = Float32(Bx[c])
        by = Float32(By[c])
        bz = Float32(Bz[c])
        dst[c] = bx*bx + by*by + bz*bz
    end
end

@kernel function _divb_abs_periodic_diag_k!(dst, @Const(Bx), @Const(By), @Const(Bz), N::Int, inv2dx)
    c = @index(Global, Linear)
    @inbounds begin
        q = c - 1
        i = q % N
        j = (q ÷ N) % N
        k = q ÷ (N * N)
        ip = (i + 1) % N
        im = (i + N - 1) % N
        jp = (j + 1) % N
        jm = (j + N - 1) % N
        kp = (k + 1) % N
        km = (k + N - 1) % N
        lin(ii, jj, kk) = ii + N * jj + N * N * kk + 1
        db = (Float32(Bx[lin(ip, j, k)]) - Float32(Bx[lin(im, j, k)]) +
              Float32(By[lin(i, jp, k)]) - Float32(By[lin(i, jm, k)]) +
              Float32(Bz[lin(i, j, kp)]) - Float32(Bz[lin(i, j, km)])) * Float32(inv2dx)
        dst[c] = abs(db)
    end
end

@kernel function _component2_diag_k!(dst, @Const(src))
    c = @index(Global, Linear)
    @inbounds begin
        x = Float32(src[c])
        dst[c] = x*x
    end
end

@kernel function _photon_slip_diag_k!(sx,sy,sz,@Const(gx),@Const(gy),@Const(gz),
        @Const(rho),@Const(mx),@Const(my),@Const(mz),smallr)
    c=@index(Global)
    @inbounds begin
        r=max(rho[c],smallr)
        sx[c]=gx[c]-mx[c]/r
        sy[c]=gy[c]-my[c]/r
        sz[c]=gz[c]-mz[c]/r
    end
end

@kernel function _delta2_diag_k!(dst, @Const(src), mean)
    c = @index(Global, Linear)
    @inbounds begin
        d = Float32(src[c]) - Float32(mean)
        dst[c] = d*d
    end
end

@kernel function _density_cos_mode_diag_k!(dst,@Const(rho),N::Int,kmode::Int,
                                           harmonic::Int,mean)
    c = @index(Global, Linear)
    @inbounds begin
        i = (c - 1) % N + 1
        x = (Float32(i) - 0.5f0) / Float32(N)
        w = cos(2f0 * Float32(pi) * Float32(harmonic * kmode) * x)
        dst[c] = (Float32(rho[c]) - Float32(mean)) * w
    end
end

@kernel function _scalar_fourier_mode_diag_k!(dst, @Const(src), N::Int, kmode::Int,
                                              quadrature::Int, mean)
    c = @index(Global, Linear)
    @inbounds begin
        T = eltype(dst)
        i = (c - 1) % N + 1
        x = (T(i) - T(0.5)) / T(N)
        phase = T(2) * T(pi) * T(kmode) * x
        w = quadrature == 1 ? cos(phase) : sin(phase)
        dst[c] = (T(src[c]) - T(mean)) * w
    end
end

@kernel function _velocity_fourier_mode_diag_k!(dst, @Const(momentum), @Const(rho),
                                                N::Int, kmode::Int,
                                                quadrature::Int, smallr)
    c = @index(Global, Linear)
    @inbounds begin
        T = eltype(dst)
        i = (c - 1) % N + 1
        x = (T(i) - T(0.5)) / T(N)
        phase = T(2) * T(pi) * T(kmode) * x
        w = quadrature == 1 ? cos(phase) : sin(phase)
        dst[c] = T(momentum[c]) / max(T(rho[c]), T(smallr)) * w
    end
end

@kernel function _slip_fourier_mode_diag_k!(dst, @Const(photon_velocity),
                                            @Const(momentum), @Const(rho),
                                            N::Int, kmode::Int,
                                            quadrature::Int, smallr)
    c = @index(Global, Linear)
    @inbounds begin
        T = eltype(dst)
        i = (c - 1) % N + 1
        x = (T(i) - T(0.5)) / T(N)
        phase = T(2) * T(pi) * T(kmode) * x
        w = quadrature == 1 ? cos(phase) : sin(phase)
        vb = T(momentum[c]) / max(T(rho[c]), T(smallr))
        dst[c] = (T(photon_velocity[c]) - vb) * w
    end
end

@kernel function _species_fourier_mode_diag_k!(dst, @Const(species), @Const(rho),
                                               N::Int, kmode::Int,
                                               quadrature::Int, fh, mean)
    c = @index(Global, Linear)
    @inbounds begin
        T = eltype(dst)
        i = (c - 1) % N + 1
        x = (T(i) - T(0.5)) / T(N)
        phase = T(2) * T(pi) * T(kmode) * x
        w = quadrature == 1 ? cos(phase) : sin(phase)
        fraction = T(species[c]) / max(T(fh) * T(rho[c]), eps(T))
        dst[c] = (fraction - T(mean)) * w
    end
end

@kernel function _species_fraction_diag_k!(dst, @Const(rho), @Const(sp), fh)
    c = @index(Global, Linear)
    @inbounds begin
        den = max(Float32(fh) * Float32(rho[c]), eps(Float32))
        dst[c] = max(Float32(sp[c]) / den, 0f0)
    end
end

@kernel function _total_electron_fraction_diag_k!(dst, @Const(rho), @Const(HII),
                                                  @Const(HeII), density_unit,
                                                  z, fh)
    c = @index(Global, Linear)
    @inbounds begin
        R = eltype(dst)
        r = max(R(rho[c]), eps(R))
        xhii = max(R(HII[c]) / max(R(fh) * r, eps(R)), zero(R))
        xheii = max(R(HeII[c]) / max(R(4) * R(fh) * r, eps(R)), zero(R))
        nH = max(R(density_unit) * r * R(fh) / R(ChemistryKernels.MH), R(1e-30))
        dst[c] = ChemistryKernels.total_electron_fraction(
            xhii, xheii, nH, R(_PMF_T_CMB) * (one(R) + R(z)); fh=R(fh))
    end
end

@kernel function _thermal_diag_k!(dst, @Const(rho), @Const(mx), @Const(my), @Const(mz),
                                  @Const(E), @Const(Bx), @Const(By), @Const(Bz),
                                  gamma, smallr, smallE, mode::Int)
    c = @index(Global, Linear)
    @inbounds begin
        r = max(Float32(rho[c]), Float32(smallr))
        ek = 0.5f0 * (Float32(mx[c]) * Float32(mx[c]) +
                      Float32(my[c]) * Float32(my[c]) +
                      Float32(mz[c]) * Float32(mz[c])) / r
        eb = 0.5f0 * (Float32(Bx[c]) * Float32(Bx[c]) +
                      Float32(By[c]) * Float32(By[c]) +
                      Float32(Bz[c]) * Float32(Bz[c]))
        eint = max(Float32(E[c]) - ek - eb, Float32(smallE))
        p = (Float32(gamma) - 1f0) * eint
        dst[c] = mode == 1 ? p : eint / r
    end
end

@kernel function _gas_temperature_diag_k!(dst, @Const(rho), @Const(mx), @Const(my),
        @Const(mz), @Const(E), @Const(Bx), @Const(By), @Const(Bz), @Const(HII),
        @Const(H2I), velocity_unit2, fh, gamma, smallr)
    c = @index(Global, Linear)
    @inbounds begin
        T = eltype(dst)
        r = max(T(rho[c]), T(smallr))
        kinetic = T(0.5) * (T(mx[c])^2 + T(my[c])^2 + T(mz[c])^2) / r
        magnetic = T(0.5) * (T(Bx[c])^2 + T(By[c])^2 + T(Bz[c])^2)
        specific_e = max((T(E[c]) - kinetic - magnetic) / r, T(1e-30))
        hii_fraction = max(T(HII[c]) / r, zero(T))
        h2_fraction = max(T(H2I[c]) / r, zero(T))
        particles_per_mass_h = max(
            T(fh) + hii_fraction - T(0.5) * h2_fraction +
            (one(T) - T(fh)) / T(4),
            eps(T),
        )
        dst[c] = (T(gamma) - one(T)) * specific_e * T(velocity_unit2) *
                 T(1.6735575e-24) /
                 (T(1.380649e-16) * particles_per_mass_h)
    end
end

@kernel function _gas_temperature_helium_diag_k!(dst, @Const(rho), @Const(mx),
        @Const(my), @Const(mz), @Const(E), @Const(Bx), @Const(By), @Const(Bz),
        @Const(HII), @Const(H2I), @Const(HeII), density_unit, velocity_unit2, z,
        fh, gamma, smallr)
    c = @index(Global, Linear)
    @inbounds begin
        R = eltype(dst)
        r = max(R(rho[c]), R(smallr))
        kinetic = R(0.5) * (R(mx[c])^2 + R(my[c])^2 + R(mz[c])^2) / r
        magnetic = R(0.5) * (R(Bx[c])^2 + R(By[c])^2 + R(Bz[c])^2)
        specific_e = max((R(E[c]) - kinetic - magnetic) / r, R(1e-30))
        du = R(density_unit)
        dst[c] = ChemistryKernels.temperature_from_reduced_helium(
            r * du, specific_e * R(velocity_unit2), R(HII[c]) * du,
            R(H2I[c]) * du, R(HeII[c]) * du, R(z);
            fh=R(fh), gamma=R(gamma))
    end
end

@kernel function _logrho_hist_k!(counts, @Const(rho), nbins::Int, loglo, invw)
    c = @index(Global, Linear)
    @inbounds begin
        lr = log10(max(Float32(rho[c]), 1.0f-30))
        b = unsafe_trunc(Int, floor((lr - Float32(loglo)) * Float32(invw))) + 1
        if 1 <= b <= nbins
            KA.@atomic counts[b] += Int32(1)
        end
    end
end

@inline function _peebles_recomb_cell(rho, e, hii_m, h2_m, dt, z, hz, fh, gamma,
                                      f_alpha, xe_mean, nsm_mass, nsm_neutral, fudge, gauss, dtfrac, itcap)
    R = typeof(e)
    mh = R(ChemistryKernels.MH)
    kb = R(ChemistryKernels.KBOLTZ)
    tiny = R(1e-30)
    fhd = max(R(fh) * rho, tiny)
    # Species are mass densities here.  A fixed 1e-30 g/cm^3 floor creates a
    # spurious H2 fraction as the cosmological density falls; zero is safe
    # because number-density denominators below retain their own unitful floor.
    h2 = min(max(h2_m, zero(R)), fhd)
    hii = min(max(hii_m, zero(R)), max(fhd - h2, zero(R)))
    nHeI = max((one(R) - R(fh)) * rho / (R(4) * mh), tiny)
    tleft = dt
    iter = 0
    while tleft > zero(R) && iter < itcap
        iter += 1
        nHII = max(hii / mh, tiny)
        nH2I = max(h2 / mh, tiny)
        nHI = max((fhd - hii - h2) / mh, tiny)
        ne = nHII
        ntot = max(nHI + nHII + ne + nHeI + R(0.5) * nH2I, tiny)
        Tgas = max((R(gamma) - one(R)) * rho * e / (kb * ntot), one(R))
        Trad = R(2.725) * (one(R) + R(z))
        nsm_h = nsm_neutral == 1 ? max(nsm_mass / mh, tiny) :
                max(nsm_mass * R(fh) / mh, tiny) * max(one(R) - R(xe_mean), zero(R))
        nHIeff = (one(R) - R(f_alpha)) * nHI + R(f_alpha) * nsm_h
        k2 = ChemistryKernels.peebles_k2_mixing(
            Tgas, nHI, nHIeff, hz; fudge=fudge, gauss=gauss, Trad=Trad)
        beta = ChemistryKernels.beta1s_freq(Trad) * k2 / (ChemistryKernels.recfast_alpha(Tgas) * R(1.0e6))
        kion = ChemistryKernels.k1(Tgas)
        nHtot = max((fhd - h2) / mh, tiny)
        ioncoef = beta + kion * ne
        dn = -k2 * nHII * ne + ioncoef * nHI
        dtit = min(tleft, max(R(dtfrac) * nHII / max(abs(dn), tiny), R(1e-30)))

        # Scalar backward-Euler solve for y=nHII with nHI=nHtot-y:
        #   dy/dt = -k2*y^2 + beta*(N-y) + kion*y*(N-y)
        # Rates/T are frozen within this substep, but electron density is not.
        A = dtit * (k2 + kion)
        B = one(R) + dtit * beta - dtit * kion * nHtot
        C = -(nHII + dtit * beta * nHtot)
        disc = max(B * B - R(4) * A * C, zero(R))
        sqd = sqrt(disc)
        nHII_new = A > tiny ? (-R(2) * C) / max(B + sqd, tiny) :
                   (nHII + beta * nHtot * dtit) / (one(R) + beta * dtit)
        nHII_new = min(max(nHII_new, tiny), nHtot)
        hii = nHII_new * mh

        # Stiff Compton energy relaxation at frozen ionization over this substep.
        c1 = ChemistryKernels.comp1_cmb(R(z))
        Tnow = max((R(gamma) - one(R)) * rho * e / (kb * ntot), one(R))
        edot_c = -c1 * (Tnow - Trad) * nHII_new
        Kc = c1 * nHII_new * (Tnow / max(e, tiny)) / rho
        if Kc * dtit > R(1e-4)
            B = c1 * nHII_new * Trad / rho
            e = (e + B * dtit) / (one(R) + Kc * dtit)
        else
            e = e + (edot_c / rho) * dtit
        end
        e = max(e, tiny)
        tleft -= dtit
    end
    return e, hii, h2
end

@kernel function _chem_peebles_k!(rho, eint, HII, H2I, @Const(nsm),
                                  nsm_scalar, nsm_is_scalar::Int, nsm_neutral::Int,
                                  du, vu2, tu, dt, z, hz, fh, gamma,
                                  f_alpha, xe_mean, fudge, gauss, dtfrac, itcap::Int)
    c = @index(Global)
    @inbounds begin
        R = eltype(eint)
        nsm_mass = nsm_is_scalar == 1 ? nsm_scalar : nsm[c]
        e, h, h2 = _peebles_recomb_cell(rho[c] * du, eint[c] * vu2,
                                        HII[c] * du, H2I[c] * du, dt * tu,
                                        z, hz, fh, gamma, f_alpha, xe_mean,
                                        nsm_mass * du, nsm_neutral, fudge, gauss, dtfrac, itcap)
        eint[c] = e / vu2
        HII[c] = h / du
        H2I[c] = h2 / du
    end
end

@kernel function _chem_ionized_compton_k!(rho, eint, HII, H2I,
                                          du, vu2, dt, z, fh, gamma)
    c = @index(Global)
    @inbounds begin
        R = eltype(eint)
        mh = R(ChemistryKernels.MH)
        kb = R(ChemistryKernels.KBOLTZ)
        tiny = R(1e-30)
        r_code = max(rho[c], tiny)
        r = r_code * du
        fhd = max(R(fh) * r, tiny)
        h2 = min(max(H2I[c] * du, tiny), fhd)
        hii = max(fhd - h2, tiny)
        nHII = hii / mh
        nH2I = h2 / mh
        nHI = tiny
        nHeI = max((one(R) - R(fh)) * r / (R(4) * mh), tiny)
        ne = nHII
        ntot = max(nHI + nHII + ne + nHeI + R(0.5) * nH2I, tiny)
        e = max(eint[c] * vu2, tiny)
        Trad = R(2.725) * (one(R) + R(z))
        Tnow = max((R(gamma) - one(R)) * r * e / (kb * ntot), one(R))
        c1 = ChemistryKernels.comp1_cmb(R(z))
        Kc = c1 * nHII * (Tnow / max(e, tiny)) / r
        if Kc * dt > R(1e-4)
            B = c1 * nHII * Trad / r
            e = (e + B * dt) / (one(R) + Kc * dt)
        else
            edot_c = -c1 * (Tnow - Trad) * nHII
            e = e + (edot_c / r) * dt
        end
        eint[c] = max(e, tiny) / vu2
        HII[c] = hii / du
        H2I[c] = h2 / du
    end
end

@kernel function _chem_ionized_compton_helium_k!(rho, eint, HII, H2I, HeII,
                                                 du, vu2, dt, z, fh, gamma,
                                                 xe_total, heii_mass_fraction)
    c = @index(Global)
    @inbounds begin
        R = eltype(eint)
        mh = R(ChemistryKernels.MH)
        kb = R(ChemistryKernels.KBOLTZ)
        tiny = R(1e-30)
        r_code = max(rho[c], tiny)
        r = r_code * du
        fhd = max(R(fh) * r, tiny)
        h2 = min(max(H2I[c] * du, tiny), fhd)
        hii = max(fhd - h2, tiny)
        nH = fhd / mh
        nH2I = h2 / mh
        nHII = hii / mh
        nHI = tiny
        nHe = max((one(R) - R(fh)) * r / (R(4) * mh), tiny)
        ne = max(R(xe_total) * nH, tiny)
        ntot = max(nHI + nHII + ne + nHe + R(0.5) * nH2I, tiny)
        e = max(eint[c] * vu2, tiny)
        Trad = R(2.725) * (one(R) + R(z))
        Tnow = max((R(gamma) - one(R)) * r * e / (kb * ntot), one(R))
        c1 = ChemistryKernels.comp1_cmb(R(z))
        Kc = c1 * ne * (Tnow / max(e, tiny)) / r
        if Kc * dt > R(1e-4)
            B = c1 * ne * Trad / r
            e = (e + B * dt) / (one(R) + Kc * dt)
        else
            edot_c = -c1 * (Tnow - Trad) * ne
            e = e + (edot_c / r) * dt
        end
        eint[c] = max(e, tiny) / vu2
        HII[c] = hii / du
        H2I[c] = h2 / du
        HeII[c] = R(heii_mass_fraction) * r_code
    end
end

function _parse_list(name, default)
    parse.(Float64, split(get(ENV, name, default), ","))
end

function _parse_optional_list(name)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return Float64[]
    return parse.(Float64, split(raw, ","))
end

function _interp_table(x::Float64, xs::Vector{Float64}, ys::Vector{Float64}, default::Float64)
    isempty(xs) && return default
    length(xs) == length(ys) || error("table length mismatch: $(length(xs)) z values vs $(length(ys)) values")
    p = sortperm(xs)
    xss = xs[p]; yss = ys[p]
    x <= xss[1] && return yss[1]
    x >= xss[end] && return yss[end]
    i = searchsortedfirst(xss, x)
    t = (x - xss[i-1]) / (xss[i] - xss[i-1])
    return yss[i-1] * (1 - t) + yss[i] * t
end

function _density_cos_mode_norm(N::Int, kmode::Int, harmonic::Int)
    k = 2π * harmonic * kmode
    norm1 = 0.0
    @inbounds for i in 1:N
        x = (Float64(i) - 0.5) / Float64(N)
        w = cos(k * x)
        norm1 += w * w
    end
    return Float64(N) * Float64(N) * norm1
end

function _fourier_mode_norm(N::Int, kmode::Int, quadrature::Int)
    k = 2π * kmode
    norm1 = 0.0
    @inbounds for i in 1:N
        x = (Float64(i) - 0.5) / Float64(N)
        w = quadrature == 1 ? cos(k * x) : sin(k * x)
        norm1 += w * w
    end
    return Float64(N) * Float64(N) * norm1
end

function _device_complex_mode!(be, scratch, src, N::Int, kmode::Int;
                               mean=zero(eltype(src)))
    n = length(src)
    mean_device = eltype(scratch)(mean)
    values = ntuple(2) do quadrature
        _scalar_fourier_mode_diag_k!(be)(scratch, src, Int(N), Int(kmode),
                                              quadrature, mean_device; ndrange=n)
        KA.synchronize(be)
        norm = _fourier_mode_norm(N, kmode, quadrature)
        norm > eps(Float64) ? Float64(sum(scratch)) / norm : 0.0
    end
    return complex(values[1], -values[2])
end

function _device_complex_velocity_mode!(be, scratch, momentum, rho,
                                        N::Int, kmode::Int, smallr)
    n = length(rho)
    values = ntuple(2) do quadrature
        _velocity_fourier_mode_diag_k!(be)(scratch, momentum, rho, Int(N),
                                                Int(kmode), quadrature, smallr;
                                                ndrange=n)
        KA.synchronize(be)
        norm = _fourier_mode_norm(N, kmode, quadrature)
        norm > eps(Float64) ? Float64(sum(scratch)) / norm : 0.0
    end
    return complex(values[1], -values[2])
end

function _device_complex_slip_mode!(be, scratch, photon_velocity, momentum, rho,
                                    N::Int, kmode::Int, smallr)
    n = length(rho)
    values = ntuple(2) do quadrature
        _slip_fourier_mode_diag_k!(be)(scratch, photon_velocity, momentum, rho,
                                            Int(N), Int(kmode), quadrature, smallr;
                                            ndrange=n)
        KA.synchronize(be)
        norm = _fourier_mode_norm(N, kmode, quadrature)
        norm > eps(Float64) ? Float64(sum(scratch)) / norm : 0.0
    end
    return complex(values[1], -values[2])
end

function _device_complex_species_mode!(be, scratch, species, rho,
                                       N::Int, kmode::Int, fh)
    n = length(rho)
    mean_fraction = _device_species_mean!(be, scratch, rho, species, fh)
    values = ntuple(2) do quadrature
        _species_fourier_mode_diag_k!(be)(scratch, species, rho, Int(N),
                                               Int(kmode), quadrature,
                                               eltype(rho)(fh),
                                               eltype(rho)(mean_fraction);
                                               ndrange=n)
        KA.synchronize(be)
        norm = _fourier_mode_norm(N, kmode, quadrature)
        norm > eps(Float64) ? Float64(sum(scratch)) / norm : 0.0
    end
    return complex(values[1], -values[2])
end

function _device_complex_temperature_fraction_mode!(be, scratch, temperature_scratch,
        s, HII, H2I, HeII, N::Int, kmode::Int, velocity_unit2, z, fh,
        hubble, omega_b)
    n = length(s.U[1])
    if HeII === nothing
        _gas_temperature_diag_k!(be)(
            temperature_scratch, s.U[1], s.U[2], s.U[3], s.U[4], s.U[5],
            s.U[6], s.U[7], s.U[8], HII, H2I, eltype(s.U[1])(velocity_unit2),
            eltype(s.U[1])(fh), s.γ, s.smallr; ndrange=n,
        )
    else
        density_unit = _density_unit_cgs(z, hubble, omega_b)
        _gas_temperature_helium_diag_k!(be)(
            temperature_scratch, s.U[1], s.U[2], s.U[3], s.U[4], s.U[5],
            s.U[6], s.U[7], s.U[8], HII, H2I, HeII,
            eltype(s.U[1])(density_unit), eltype(s.U[1])(velocity_unit2),
            eltype(s.U[1])(z),
            eltype(s.U[1])(fh), s.γ, s.smallr; ndrange=n,
        )
    end
    KA.synchronize(be)
    mean_temperature = Float64(sum(temperature_scratch)) / Float64(n)
    value = _device_complex_mode!(
        be, scratch, temperature_scratch, N, kmode; mean=mean_temperature
    )
    value / max(mean_temperature, eps(Float64))
end

function _device_complex_thermal_fraction_mode!(be, scratch, thermal_scratch,
        s, N::Int, kmode::Int, mode::Int)
    n = length(s.U[1])
    _thermal_diag_k!(be)(
        thermal_scratch, s.U[1], s.U[2], s.U[3], s.U[4], s.U[5],
        s.U[6], s.U[7], s.U[8], eltype(s.U[1])(s.γ),
        eltype(s.U[1])(s.smallr), eltype(s.U[1])(1e-30), mode;
        ndrange=n,
    )
    KA.synchronize(be)
    mean_value = Float64(sum(thermal_scratch)) / Float64(n)
    value = _device_complex_mode!(
        be, scratch, thermal_scratch, N, kmode; mean=mean_value
    )
    value / max(abs(mean_value), eps(Float64))
end

function _device_brms!(be, s, scratch)
    n = length(s.U[1])
    _b2_diag_k!(be)(scratch, s.U[6], s.U[7], s.U[8]; ndrange=n)
    KA.synchronize(be)
    return sqrt(Float64(sum(scratch)) / Float64(n))
end

function _device_divb_norm!(be, s, scratch, br::Float64)
    n = length(s.U[1])
    N = Int(s.dims[1])
    _divb_abs_periodic_diag_k!(be)(scratch, s.U[6], s.U[7], s.U[8], N, T(0.5) / T(s.dx);
                                   ndrange=n)
    KA.synchronize(be)
    return Float64(maximum(scratch)) * Float64(s.dx) / max(br, eps(Float64))
end

function _device_component_rms!(be, scratch, src)
    n = length(src)
    _component2_diag_k!(be)(scratch, src; ndrange=n)
    KA.synchronize(be)
    return sqrt(Float64(sum(scratch)) / Float64(n))
end

function _device_delta_rms!(be, scratch, src; mean=one(T))
    n = length(src)
    _delta2_diag_k!(be)(scratch, src, mean; ndrange=n)
    KA.synchronize(be)
    return sqrt(Float64(sum(scratch)) / Float64(n))
end

function _device_delta_rms!(be, scratch, src, mean)
    n = length(src)
    _delta2_diag_k!(be)(scratch, src, mean; ndrange=n)
    KA.synchronize(be)
    return sqrt(Float64(sum(scratch)) / Float64(n))
end

function _device_density_cos_mode_amp!(be,scratch,rho,N::Int,kmode::Int,
                                       harmonic::Int;mean=one(eltype(rho)))
    n = length(rho)
    _density_cos_mode_diag_k!(be)(scratch,rho,Int(N),Int(kmode),Int(harmonic),mean;
                                  ndrange=n)
    KA.synchronize(be)
    norm = _density_cos_mode_norm(N, kmode, harmonic)
    return norm > 0 ? Float64(sum(scratch)) / norm : NaN
end

function _device_species_stats!(be, scratch, rho, HII, H2I, HeII, fh;
                                z=NaN, h=0.674, Ob=0.049)
    if HII === nothing
        return ntuple(_ -> NaN, 12)
    end
    n = length(rho)
    _species_fraction_diag_k!(be)(scratch, rho, HII, T(fh); ndrange=n)
    KA.synchronize(be)
    xhmean = Float64(sum(scratch)) / Float64(n)
    xhmin = Float64(minimum(scratch))
    xhmax = Float64(maximum(scratch))

    _species_fraction_diag_k!(be)(scratch, rho, H2I, T(fh); ndrange=n)
    KA.synchronize(be)
    x2mean = Float64(sum(scratch)) / Float64(n)
    x2min = Float64(minimum(scratch))
    x2max = Float64(maximum(scratch))
    if HeII === nothing || !isfinite(z)
        return (xhmean, xhmin, xhmax, x2mean, x2min, x2max,
                NaN, NaN, NaN, NaN, NaN, NaN)
    end
    _species_fraction_diag_k!(be)(scratch, rho, HeII, T(4) * T(fh);
                                  ndrange=n)
    KA.synchronize(be)
    xhemean = Float64(sum(scratch)) / Float64(n)
    xhemin = Float64(minimum(scratch))
    xhemax = Float64(maximum(scratch))

    density_unit = _density_unit_cgs(z, h, Ob)
    _total_electron_fraction_diag_k!(be)(
        scratch, rho, HII, HeII, T(density_unit), T(z), T(fh); ndrange=n)
    KA.synchronize(be)
    xtmean = Float64(sum(scratch)) / Float64(n)
    xtmin = Float64(minimum(scratch))
    xtmax = Float64(maximum(scratch))
    return (xhmean, xhmin, xhmax, x2mean, x2min, x2max,
            xhemean, xhemin, xhemax, xtmean, xtmin, xtmax)
end

function _device_species_mean!(be, scratch, rho, species, fh)
    species === nothing && return NaN
    _species_fraction_diag_k!(be)(scratch, rho, species, T(fh); ndrange=length(rho))
    KA.synchronize(be)
    return Float64(sum(scratch)) / Float64(length(rho))
end

function _device_thermal_stats!(be, s, scratch)
    n = length(s.U[1])
    _thermal_diag_k!(be)(scratch, s.U[1], s.U[2], s.U[3], s.U[4], s.U[5],
                         s.U[6], s.U[7], s.U[8], T(s.γ), T(s.smallr), T(1e-30), 1;
                         ndrange=n)
    KA.synchronize(be)
    pmean = Float64(sum(scratch)) / Float64(n)
    pmin = Float64(minimum(scratch))
    pmax = Float64(maximum(scratch))

    _thermal_diag_k!(be)(scratch, s.U[1], s.U[2], s.U[3], s.U[4], s.U[5],
                         s.U[6], s.U[7], s.U[8], T(s.γ), T(s.smallr), T(1e-30), 2;
                         ndrange=n)
    KA.synchronize(be)
    emean = Float64(sum(scratch)) / Float64(n)
    emin = Float64(minimum(scratch))
    emax = Float64(maximum(scratch))
    return (pmean, pmin, pmax, emean, emin, emax)
end

function _device_temperature_stats!(be, s, scratch, HII, H2I, HeII,
                                    density_unit, velocity_unit2, z, fh)
    HII === nothing && return (NaN, NaN, NaN)
    n = length(s.U[1])
    if HeII === nothing
        _gas_temperature_diag_k!(be)(
            scratch, s.U[1], s.U[2], s.U[3], s.U[4], s.U[5],
            s.U[6], s.U[7], s.U[8], HII, H2I, T(velocity_unit2), T(fh),
            s.γ, s.smallr; ndrange=n)
    else
        _gas_temperature_helium_diag_k!(be)(
            scratch, s.U[1], s.U[2], s.U[3], s.U[4], s.U[5],
            s.U[6], s.U[7], s.U[8], HII, H2I, HeII, T(density_unit),
            T(velocity_unit2), T(z), T(fh), s.γ, s.smallr; ndrange=n)
    end
    KA.synchronize(be)
    (Float64(sum(scratch)) / Float64(n), Float64(minimum(scratch)),
     Float64(maximum(scratch)))
end

function _device_pressure_mean!(be, s, scratch)
    n = length(s.U[1])
    _thermal_diag_k!(be)(scratch, s.U[1], s.U[2], s.U[3], s.U[4], s.U[5],
                         s.U[6], s.U[7], s.U[8], T(s.γ), T(s.smallr), T(1e-30), 1;
                         ndrange=n)
    KA.synchronize(be)
    return Float64(sum(scratch)) / Float64(n)
end

function _final_device_stats!(be,s,ρdm,scratch,HII,H2I,HeII,fh;
                              density_perturbation=nothing, z=NaN,
                              h=0.674, Ob=0.049)
    br = _device_brms!(be, s, scratch)
    divn = _device_divb_norm!(be, s, scratch, br)
    δb=density_perturbation === nothing ?
        _device_delta_rms!(be,scratch,s.U[1],one(T)) :
        _device_delta_rms!(be,scratch,density_perturbation,zero(T))
    δdm = _device_delta_rms!(be, scratch, ρdm, one(T))
    chemstats = _device_species_stats!(be, scratch, s.U[1], HII, H2I, HeII, fh;
                                       z=z, h=h, Ob=Ob)
    return br, divn, δb, δdm, chemstats
end

function _adaptive_diag_watch!(be,s,scratch,HII,H2I,HeII,fh;
                               density_perturbation=nothing, z=NaN,
                               h=0.674, Ob=0.049)
    δb=density_perturbation === nothing ?
        _device_delta_rms!(be,scratch,s.U[1],one(T)) :
        _device_delta_rms!(be,scratch,density_perturbation,zero(T))
    chemstats = _device_species_stats!(be, scratch, s.U[1], HII, H2I, HeII, fh;
                                       z=z, h=h, Ob=Ob)
    return δb, chemstats[1]
end

function _rel_changed(now::Float64, old::Float64, frac::Float64)
    isfinite(now) || return false
    isfinite(old) || return true
    scale = max(abs(old), eps(Float64))
    return abs(now - old) / scale >= frac
end

function _state_summary(s)
    _b2_diag_k!(s.be)(s.scratch[1], s.U[6], s.U[7], s.U[8]; ndrange=length(s.U[1]))
    KA.synchronize(s.be)
    ρmin = minimum(s.U[1])
    ρmax = maximum(s.U[1])
    Emin = minimum(s.U[5])
    Emax = maximum(s.U[5])
    B2max = maximum(s.scratch[1])
    finite = isfinite(Float64(ρmin)) && isfinite(Float64(ρmax)) &&
             isfinite(Float64(Emin)) && isfinite(Float64(Emax)) &&
             isfinite(Float64(B2max))
    return (finite = finite,
            ρmin = ρmin, ρmax = ρmax,
            Emin = Emin, Emax = Emax,
            Bmax = isfinite(Float64(B2max)) && Float64(B2max) >= 0 ?
                   sqrt(Float64(B2max)) : Float64(B2max))
end

function _assert_finite_step(s, label, cyc, a, dt, smax)
    if !(isfinite(dt) && dt > 0 && isfinite(Float64(smax)) && Float64(smax) > 0)
        ss = _state_summary(s)
        error(@sprintf("%s cycle=%d invalid timestep dt=%.6e smax=%.6e a=%.8e z=%.6f rho=[%.6e,%.6e] E=[%.6e,%.6e] Bmax=%.6e finite=%s",
                       label, cyc, dt, Float64(smax), a, isfinite(a) ? a_to_z(a) : NaN,
                       Float64(ss.ρmin), Float64(ss.ρmax), Float64(ss.Emin), Float64(ss.Emax),
                       ss.Bmax, string(ss.finite)))
    end
end

function _assert_finite_state(s, label, cyc, a, dt, smax)
    ss = _state_summary(s)
    if !(ss.finite && isfinite(Float64(ss.ρmin)) && Float64(ss.ρmin) > 0 &&
         isfinite(Float64(ss.Emin)) && Float64(ss.Emin) > 0)
        error(@sprintf("%s cycle=%d invalid state dt=%.6e smax=%.6e a=%.8e z=%.6f rho=[%.6e,%.6e] E=[%.6e,%.6e] Bmax=%.6e finite=%s",
                       label, cyc, dt, Float64(smax), a, isfinite(a) ? a_to_z(a) : NaN,
                       Float64(ss.ρmin), Float64(ss.ρmax), Float64(ss.Emin), Float64(ss.Emax),
                       ss.Bmax, string(ss.finite)))
    end
    return ss
end

function _time_phase(f, be1, be2)
    t = @elapsed begin
        f()
        KA.synchronize(be1)
        be2 === be1 || KA.synchronize(be2)
    end
    return t
end

function _log_line(msg::AbstractString)
    @printf("[%s] %s\n", Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"), msg)
    flush(stdout)
end

function _device_memory_gib()
    if BACKEND_NAME === :metal
        Metal.synchronize()
        stats = Metal.alloc_stats
        live = max(0, stats.alloc_bytes - stats.free_bytes) / 2.0^30
        return (live=live, allocated=stats.alloc_bytes / 2.0^30,
                freed=stats.free_bytes / 2.0^30)
    elseif BACKEND_NAME === :cuda
        CUDA.synchronize()
        live = (CUDA.total_memory() - CUDA.available_memory()) / 2.0^30
        return (live=live, allocated=NaN, freed=NaN)
    end
    return (live=NaN, allocated=NaN, freed=NaN)
end

function _log_device_memory(label::AbstractString)
    get(ENV, "MHD_MEMPROBE", "0") == "1" || return nothing
    stats = _device_memory_gib()
    @printf("[%s] MEMORY %-28s live_device=%.3f GiB allocated=%.3f GiB freed=%.3f GiB\n",
            Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"), label,
            stats.live, stats.allocated, stats.freed)
    flush(stdout)
    return nothing
end

function _stage(label::AbstractString, f)
    _log_line("START " * label)
    result = nothing
    t = @elapsed begin
        result = f()
    end
    _log_line(@sprintf("DONE  %s %.3f s", label, t))
    _log_device_memory(label)
    return result
end

_stage(f, label::AbstractString) = _stage(label, f)

function _fft_mode(name::Symbol, env::AbstractString)
    s = Symbol(lowercase(env))
    if s === :auto
        return name === :metal ? :mps : :ka
    elseif s in (:mps, :native, :rfft)
        return :mps
    elseif s in (:ka, :ka_rfft)
        return :ka
    elseif s in (:c2c, :ka_c2c)
        return :c2c
    else
        error("MHD_FFT must be auto, mps/native/rfft, ka, or c2c; got '$env'")
    end
end

function _chem_mode(env::AbstractString)
    s = Symbol(lowercase(env))
    s in (:off, :none, :false, :0) && return :off
    s in (:peebles, :recfast, :full, :on, :true, :1) && return :peebles
    s in (:dense, :stiff, :network) && return :dense
    s in (:mixing, :lya, :lyalpha) && return :mixing
    s in (:analytic_mixing, :mixing_analytic, :fast_mixing, :analytic, :analytic_h2, :fast) &&
        return :analytic_mixing
    error("MHD_CHEM must be off, peebles/recfast, dense, mixing, or analytic_mixing; got '$env'")
end

function _chem_smooth_mode(env::AbstractString)
    s = Symbol(lowercase(env))
    s in (:local, :none, :cell) && return :local
    s in (:global, :mean, :box) && return :global
    s in (:gaussian, :fft, :fft_gaussian, :rfft, :rfft_gaussian) && return :gaussian
    error("MHD_CHEM_SMOOTH must be local, global, or gaussian; got '$env'")
end

function _chem_coupling_mode(env::AbstractString)
    s = Symbol(lowercase(env))
    s in (:lie, :split, :post, :operator_split) && return :lie
    s in (:strang, :midpoint, :half, :halfstep, :symmetric) && return :strang
    error("MHD_CHEM_COUPLING must be lie/split or strang/midpoint; got '$env'")
end

function _chem_expansion_enabled(env::AbstractString, zrun::Bool, chem_mode::Symbol)
    s = lowercase(strip(env))
    s in ("auto", "default") && return zrun && chem_mode !== :off
    s in ("on", "true", "yes", "1") && return true
    s in ("off", "false", "no", "0") && return false
    error("MHD_CHEM_EXPANSION must be auto, on/1, or off/0; got '$env'")
end

function _run_pmf_chemistry_step!(be_p, be_mhd, s, eint, HII, H2I, HeII, ρtot, φ;
                                  N::Int, ncells::Int, zrun::Bool,
                                  a0::Float64, a1::Float64, dt_code::Float64,
                                  hub::Float64, Om::Float64, OL::Float64, Or::Float64,
                                  Ob::Float64, boxsize::Float64,
                                  chem_mode::Symbol, chem_smooth::Symbol,
                                  chem_density_fixed::AbstractString,
                                  chem_vunit::Float64, chem_time_unit::Float64,
                                  chem_ionized_fast_zmin::Float64,
                                  chem_fused_analytic::Bool,
                                  chem_fused_mixing::Bool,
                                  chem_helium::Bool,
                                  chem_expansion::Bool,
                                  chem_profile_enabled::Bool, measuring::Bool,
                                  refresh_mean::Bool,
                                  chem_phase, chem_falpha::Float64,
                                  chem_falpha_zs, chem_falpha_vals,
                                  chem_nsmean::Float64, chem_smooth_sigma::Float64,
                                  chem_xe_mean::Float64, evolving_xe_mean::Float64,
                                  evolving_xheii_mean::Float64,
                                  chem_recfast_hswitch::Bool, chem_fudge::Float64,
                                  chem_rate_tables, chem_dtfrac::Float64,
                                  chem_itcap::Int, fh,
                                  chem_nonlocal_table=nothing,
                                  chem_nonlocal_scale::Float64=1.0,
                                  chem_nonlocal_before::Bool=false,
                                  radiation_workspace=nothing,
                                  terminal_lbox_cm::Float64=NaN,
                                  cosmological_mhd_aref::Float64=NaN)
    chem_mode === :off && return 0.0, evolving_xe_mean, evolving_xheii_mean
    T = eltype(s.U[1])
    xe_mean = evolving_xe_mean
    xheii_mean = evolving_xheii_mean
    tc = _time_phase(be_p, be_mhd) do
        profile_chem = chem_profile_enabled && measuring
        zchem = zrun ? a_to_z(0.5 * (a0 + a1)) : 0.0
        dtphys = zrun ? _cosmic_dt_seconds(a0, a1, hub, Om, OL, Or) :
                 dt_code * chem_time_unit
        amid = 0.5 * (a0 + a1)
        velocity_unit2 = chem_vunit^2 *
            (isfinite(cosmological_mhd_aref) ? cosmological_mhd_aref / amid : 1.0)
        dunit = isempty(chem_density_fixed) ? _density_unit_cgs(zchem, hub, Ob) :
                parse(Float64, chem_density_fixed)
        chem_hubble_expansion = chem_expansion && zrun
        chem_adot_over_a = chem_hubble_expansion ? _hubble_s(zchem, hub, Om, OL, Or) : NaN
        function run_nonlocal!()
            chem_nonlocal_table === nothing && return nothing
            chem_nonlocal_scale > 0 || return nothing
            radiation_workspace === nothing && error(
                "nonlocal chemistry requires the reusable radiation FFT workspace")
            isfinite(terminal_lbox_cm) && terminal_lbox_cm > 0 || error(
                "nonlocal chemistry requires a positive comoving MHD_TERMINAL_LBOX_CM")
            hz = _hubble_s(zchem, hub, Om, OL, Or)
            theta_conversion = sqrt(velocity_unit2) * boxsize /
                (terminal_lbox_cm * amid * hz)
            nonlocal_block = () -> MHDKernels.apply_nonlocal_recombination!(
                HII, H2I, s, radiation_workspace, chem_nonlocal_table;
                redshift=zchem, xe_mean=xe_mean,
                dt_seconds=dtphys * chem_nonlocal_scale,
                fundamental_k_mpc=2pi * _PMF_MPC_CM / terminal_lbox_cm,
                theta_over_aH_conversion=theta_conversion,
                hydrogen_mass_fraction=fh,
                velocity_unit2=velocity_unit2,
                temperature_reference=_PMF_T_CMB * (1 + zchem))
            if profile_chem
                chem_phase[:nonlocal] += _time_phase(nonlocal_block, be_p, be_mhd)
            else
                nonlocal_block()
            end
            nothing
        end
        chem_nonlocal_before && run_nonlocal!()
        ionized_fast = (chem_mode === :peebles || chem_mode === :mixing ||
                        chem_mode === :analytic_mixing) &&
                       chem_ionized_fast_zmin > 0 && zchem >= chem_ionized_fast_zmin
        use_fused_analytic = chem_fused_analytic && chem_mode === :analytic_mixing &&
                             !ionized_fast
        use_fused_mixing = chem_fused_mixing && chem_mode === :mixing && !ionized_fast
        chem_helium && chem_mode !== :mixing && error(
            "MHD_CHEM_HELIUM=1 currently requires MHD_CHEM=mixing")
        chem_helium && !ionized_fast && !use_fused_mixing && error(
            "MHD_CHEM_HELIUM=1 requires the fused full mixing kernel")
        use_fused_chem = use_fused_analytic || use_fused_mixing
        if profile_chem && !use_fused_chem
            chem_phase[:unpack] += _time_phase(be_p, be_mhd) do
                _mhd_to_eint_k!(be_p)(eint, s.U[1], s.U[2], s.U[3], s.U[4],
                                      s.U[5], s.U[6], s.U[7], s.U[8],
                                      T(s.smallr); ndrange=ncells)
            end
        elseif !use_fused_chem
            _mhd_to_eint_k!(be_p)(eint, s.U[1], s.U[2], s.U[3], s.U[4],
                                  s.U[5], s.U[6], s.U[7], s.U[8],
                                  T(s.smallr); ndrange=ncells)
            KA.synchronize(be_p)
        end
        if ionized_fast
            helium_state = chem_helium ?
                _equilibrium_helium_state(1.0, zchem, hub, Ob, Float64(fh)) : nothing
            if profile_chem
                chem_phase[:solve] += _time_phase(be_p, be_mhd) do
                    if chem_helium
                        _chem_ionized_compton_helium_k!(be_p)(
                            s.U[1], eint, HII, H2I, HeII,
                            T(dunit), T(velocity_unit2), T(dtphys), T(zchem), fh,
                            T(s.γ), T(helium_state.xe_total),
                            T(helium_state.heii_mass_fraction); ndrange=ncells)
                    else
                        _chem_ionized_compton_k!(be_p)(s.U[1], eint, HII, H2I,
                                                       T(dunit), T(velocity_unit2),
                                                       T(dtphys), T(zchem), fh, T(s.γ);
                                                       ndrange=ncells)
                    end
                end
            else
                if chem_helium
                    _chem_ionized_compton_helium_k!(be_p)(
                        s.U[1], eint, HII, H2I, HeII,
                        T(dunit), T(velocity_unit2), T(dtphys), T(zchem), fh,
                        T(s.γ), T(helium_state.xe_total),
                        T(helium_state.heii_mass_fraction); ndrange=ncells)
                else
                    _chem_ionized_compton_k!(be_p)(s.U[1], eint, HII, H2I,
                                                   T(dunit), T(velocity_unit2),
                                                   T(dtphys), T(zchem), fh, T(s.γ);
                                                   ndrange=ncells)
                end
            end
        elseif chem_mode === :peebles || chem_mode === :mixing ||
               chem_mode === :analytic_mixing
            f_alpha_step = clamp(_interp_table(Float64(zchem), chem_falpha_zs,
                                               chem_falpha_vals, chem_falpha),
                                 0.0, 1.0)
            nsm_arg = s.U[1]
            nsm_scalar = T(0)
            nsm_is_scalar = 0
            nsm_neutral = 0
            if f_alpha_step != 0
                if chem_smooth === :global
                    nsm_scalar = T(chem_nsmean)
                    nsm_is_scalar = 1
                elseif chem_smooth === :gaussian
                    if profile_chem
                        chem_phase[:smooth] += _time_phase(be_p, be_mhd) do
                            _neutral_h_mass_k!(be_p)(ρtot, s.U[1], HII, H2I, fh, Int(N);
                                                     ndrange=ncells)
                            PoissonKernels.rfft_smooth_gaussian!(φ, ρtot;
                                sigma_cells=chem_smooth_sigma, boxsize=boxsize)
                        end
                    else
                        _neutral_h_mass_k!(be_p)(ρtot, s.U[1], HII, H2I, fh, Int(N);
                                                 ndrange=ncells)
                        PoissonKernels.rfft_smooth_gaussian!(φ, ρtot;
                            sigma_cells=chem_smooth_sigma, boxsize=boxsize)
                    end
                    nsm_arg = φ
                    nsm_neutral = 1
                else
                    nsm_arg = s.U[1]
                end
            end
            gauss = chem_recfast_hswitch ? ChemistryKernels.recfast_gauss_factor(zchem) : 1.0
            xmean = f_alpha_step == 0 ? 0.0 :
                    (isfinite(chem_xe_mean) ? chem_xe_mean : xe_mean)
            solve_block = () -> begin
                if chem_mode === :mixing
                    if use_fused_mixing
                        if chem_helium
                            _chem_mixing_helium_mhd_k!(be_p)(
                                s.U[1], s.U[2], s.U[3], s.U[4], s.U[5],
                                s.U[6], s.U[7], s.U[8], HII, H2I, HeII, nsm_arg,
                                nsm_scalar, T(dunit), T(velocity_unit2), T(1.0),
                                T(dtphys), T(zchem), T(f_alpha_step), T(xmean),
                                T(chem_fudge), T(gauss), T(100hub), T(Om), T(OL), fh,
                                chem_hubble_expansion, chem_rate_tables,
                                T(chem_dtfrac), chem_itcap, T(s.smallr),
                                Val(nsm_is_scalar == 1),
                                Val(nsm_neutral == 1); ndrange=ncells)
                        else
                            _chem_mixing_mhd_k!(be_p)(
                                s.U[1], s.U[2], s.U[3], s.U[4], s.U[5],
                                s.U[6], s.U[7], s.U[8], HII, H2I, nsm_arg,
                                nsm_scalar, T(dunit), T(velocity_unit2), T(1.0),
                                T(dtphys), T(zchem), T(f_alpha_step), T(xmean),
                                T(chem_fudge), T(gauss), T(100hub), T(Om), T(OL), fh,
                                chem_hubble_expansion, chem_rate_tables,
                                T(chem_dtfrac), chem_itcap, T(s.smallr),
                                Val(nsm_is_scalar == 1),
                                Val(nsm_neutral == 1); ndrange=ncells)
                        end
                    else
                        nsm_source = nsm_is_scalar == 1 ? nsm_scalar : nsm_arg
                        ChemistryKernels.solve_chem_mixing_device!(
                            s.U[1], eint, HII, H2I, nsm_source;
                            a_value=1/(1+zchem), dt=dtphys,
                            density_units=dunit, length_units=sqrt(velocity_unit2), time_units=1,
                            f_alpha=f_alpha_step, Xe_mean=xmean,
                            smoothed_is_neutral=nsm_neutral == 1,
                            recfast_fudge=chem_fudge,
                            recfast_hswitch=chem_recfast_hswitch,
                            hubble_expansion=chem_hubble_expansion,
                            hubble=100hub, Om=Om, OL=OL, fh=Float64(fh),
                            rate_tables=chem_rate_tables, dtfrac=chem_dtfrac,
                            itcap=chem_itcap, backend=BACKEND_NAME, precision=T)
                    end
                elseif chem_mode === :analytic_mixing
                    if use_fused_analytic
                        _chem_analytic_mixing_mhd_k!(be_p)(
                            s.U[1], s.U[2], s.U[3], s.U[4], s.U[5],
                            s.U[6], s.U[7], s.U[8], HII, H2I, nsm_arg,
                            nsm_scalar, T(dunit), T(velocity_unit2), T(1.0),
                            T(dtphys), T(zchem), T(f_alpha_step), T(xmean),
                            T(chem_fudge), T(gauss), T(100hub), T(Om), T(OL),
                            fh, chem_hubble_expansion, T(chem_adot_over_a),
                            chem_rate_tables, T(chem_dtfrac), chem_itcap, T(s.smallr),
                            Val(nsm_is_scalar == 1),
                            Val(nsm_neutral == 1); ndrange=ncells)
                    else
                        nsm_source = nsm_is_scalar == 1 ? nsm_scalar : nsm_arg
                        ChemistryKernels.solve_chem_analytic_mixing_device!(
                            s.U[1], eint, HII, H2I, nsm_source;
                            a_value=1/(1+zchem), dt=dtphys,
                            density_units=dunit, length_units=sqrt(velocity_unit2), time_units=1,
                            f_alpha=f_alpha_step, Xe_mean=xmean,
                            smoothed_is_neutral=nsm_neutral == 1,
                            recfast_fudge=chem_fudge,
                            recfast_hswitch=chem_recfast_hswitch,
                            hubble_expansion=chem_hubble_expansion,
                            adot_over_a=chem_adot_over_a,
                            hubble=100hub, Om=Om, OL=OL, fh=Float64(fh),
                            rate_tables=chem_rate_tables, dtfrac=chem_dtfrac,
                            itcap=chem_itcap, backend=BACKEND_NAME, precision=T)
                    end
                else
                    hz = _hubble_s(zchem, hub, Om, OL, Or)
                    _chem_peebles_k!(be_p)(s.U[1], eint, HII, H2I, nsm_arg,
                                           nsm_scalar, nsm_is_scalar, nsm_neutral,
                                           T(dunit), T(velocity_unit2), T(1.0),
                                           T(dtphys), T(zchem), T(hz), fh, T(s.γ),
                                           T(f_alpha_step), T(xmean), T(chem_fudge),
                                           T(gauss), T(chem_dtfrac), chem_itcap;
                                           ndrange=ncells)
                end
            end
            if profile_chem
                chem_phase[:solve] += _time_phase(solve_block, be_p, be_mhd)
            else
                solve_block()
            end
        else
            if profile_chem
                chem_phase[:solve] += _time_phase(be_p, be_mhd) do
                    ChemistryKernels.solve_chem_device!(s.U[1], eint, HII, H2I;
                        a_value=1/(1+zchem), dt=dtphys, density_units=dunit,
                        length_units=sqrt(velocity_unit2), time_units=1,
                        hubble=100hub, Om=Om, OL=OL, fh=Float64(fh), deuterium=false,
                        hubble_expansion=chem_hubble_expansion,
                        adot_over_a=chem_adot_over_a,
                        dtfrac=chem_dtfrac, itcap=chem_itcap, backend=BACKEND_NAME,
                        precision=T)
                end
            else
                ChemistryKernels.solve_chem_device!(s.U[1], eint, HII, H2I;
                    a_value=1/(1+zchem), dt=dtphys, density_units=dunit,
                    length_units=sqrt(velocity_unit2), time_units=1,
                    hubble=100hub, Om=Om, OL=OL, fh=Float64(fh), deuterium=false,
                    hubble_expansion=chem_hubble_expansion,
                    adot_over_a=chem_adot_over_a,
                    dtfrac=chem_dtfrac, itcap=chem_itcap, backend=BACKEND_NAME,
                    precision=T)
            end
        end
        chem_nonlocal_before || run_nonlocal!()
        if profile_chem
            if !use_fused_chem
                chem_phase[:clamp] += _time_phase(be_p, be_mhd) do
                    _clamp_reduced_chem_k!(be_p)(HII, H2I, s.U[1], fh, Int(N);
                                                ndrange=ncells)
                    chem_helium && _clamp_reduced_helium_k!(be_p)(
                        HeII, s.U[1], fh, Int(N); ndrange=ncells)
                end
            end
            if refresh_mean
                chem_phase[:mean] += @elapsed begin
                    xe_mean = _device_species_mean!(be_p, s.scratch[1], s.U[1], HII, fh)
                    chem_helium && (xheii_mean = _device_species_mean!(
                        be_p, s.scratch[1], s.U[1], HeII, 4T(fh)))
                end
            end
            if !use_fused_chem
                chem_phase[:repack] += _time_phase(be_p, be_mhd) do
                    _eint_to_mhd_k!(be_p)(s.U[5], eint, s.U[1], s.U[2], s.U[3],
                                          s.U[4], s.U[6], s.U[7], s.U[8],
                                          T(s.smallr); ndrange=ncells)
                end
            end
        else
            if !use_fused_chem
                _clamp_reduced_chem_k!(be_p)(HII, H2I, s.U[1], fh, Int(N);
                                            ndrange=ncells)
                chem_helium && _clamp_reduced_helium_k!(be_p)(
                    HeII, s.U[1], fh, Int(N); ndrange=ncells)
                KA.synchronize(be_p)
            end
            if refresh_mean
                xe_mean = _device_species_mean!(be_p, s.scratch[1], s.U[1], HII, fh)
                chem_helium && (xheii_mean = _device_species_mean!(
                    be_p, s.scratch[1], s.U[1], HeII, 4T(fh)))
            end
            if !use_fused_chem
                _eint_to_mhd_k!(be_p)(s.U[5], eint, s.U[1], s.U[2], s.U[3],
                                      s.U[4], s.U[6], s.U[7], s.U[8],
                                      T(s.smallr); ndrange=ncells)
            end
        end
    end
    return tc, xe_mean, xheii_mean
end

function _hubble_s(z, h, Om, OL, Or)
    H0 = 100.0 * h * 1.0e5 / 3.0857e24
    Ok = 1.0 - Om - OL - Or
    H0 * sqrt(Om * (1 + z)^3 + Ok * (1 + z)^2 + Or * (1 + z)^4 + OL)
end

function _advance_a_rk2(Om, OL, Ok, Or, a, dt, a_end)
    k1 = dadtau(Om, OL, Ok, Or, a)
    amid = a + 0.5 * k1 * dt
    k2 = dadtau(Om, OL, Ok, Or, amid)
    return min(a + k2 * dt, a_end)
end

function _cosmic_dt_seconds(a0, a1, h, Om, OL, Or)
    amid = 0.5 * (a0 + a1)
    zmid = a_to_z(amid)
    max(a1 - a0, 0.0) / (amid * _hubble_s(zmid, h, Om, OL, Or))
end

function _compton_drag_rate_s(z, xe, fh)
    (4 / 3) * _PMF_ARAD * (_PMF_T_CMB * (1 + z))^4 *
    max(xe, 0.0) * fh * _PMF_SIGT / (_PMF_CL * ChemistryKernels.MH)
end

function _drag_electron_fraction(xhii, z, h, Ob, fh, include_helium)
    include_helium || return max(Float64(xhii), 0.0)
    nH = _density_unit_cgs(z, h, Ob) * fh / ChemistryKernels.MH
    return ChemistryKernels.total_electron_fraction(
        Float64(xhii), nH, _PMF_T_CMB * (1 + z); fh=Float64(fh))
end

function _drag_electron_fraction(xhii, xheii, z, h, Ob, fh, include_helium)
    include_helium || return max(Float64(xhii), 0.0)
    nH = _density_unit_cgs(z, h, Ob) * fh / ChemistryKernels.MH
    return ChemistryKernels.total_electron_fraction(
        Float64(xhii), Float64(xheii), nH, _PMF_T_CMB * (1 + z);
        fh=Float64(fh))
end

function _density_unit_cgs(z, h, Ob)
    Ob * 1.8788e-29 * h^2 * (1 + z)^3
end

function _equilibrium_helium_state(xhii, z, h, Ob, fh)
    nH = _density_unit_cgs(z, h, Ob) * Float64(fh) / ChemistryKernels.MH
    Trad = _PMF_T_CMB * (1 + Float64(z))
    xe = ChemistryKernels.total_electron_fraction(
        Float64(xhii), nH, Trad; fh=Float64(fh))
    s1, s2 = ChemistryKernels.helium_saha_pair(Trad)
    ne = max(xe * nH, 1e-300)
    r1 = s1 / ne
    r2 = s2 / ne
    denominator = 1 + r1 + r1 * r2
    fHe = (1 - Float64(fh)) / (4Float64(fh))
    xheii = fHe * r1 / denominator
    return (
        xe_total=xe,
        xheii=xheii,
        heii_mass_fraction=4Float64(fh) * xheii,
    )
end

function _radiation_closure(z,xe,h,Ob,fh,vunit,lbox_comoving_cm,
                            gas_sound_speed;bridge_q2=5,
                            longitudinal_viscosity=4/3,
                            nyquist_guard_fraction=1,
                            response_model=:angular)
    isfinite(lbox_comoving_cm) && lbox_comoving_cm>0 ||
        error("spectral radiation coupling needs MHD_TERMINAL_LBOX_CKPC or MHD_TERMINAL_LBOX_CM")
    rho_b=_density_unit_cgs(z,h,Ob)
    Tcmb=_PMF_T_CMB*(1+z)
    rho_gamma_energy=_PMF_ARAD*Tcmb^4
    R=3*rho_b*_PMF_CL^2/(4*rho_gamma_energy)
    ne=rho_b*max(xe,eps(Float64))*Float64(fh)/ChemistryKernels.MH
    lambda_phys=1/(ne*_PMF_SIGT)
    lphys=lbox_comoving_cm/(1+z)
    return MHDKernels.RadiationClosure(
        inertia_ratio=R,
        mean_free_path=lambda_phys/lphys,
        photon_speed=_PMF_CL/vunit,
        gas_sound_speed=gas_sound_speed,
        bridge_q2=bridge_q2,
        longitudinal_viscosity=longitudinal_viscosity,
        nyquist_guard_fraction=nyquist_guard_fraction,
        response_model=Symbol(response_model))
end

function _radiation_scale_diagnostics(rc, n)
    kfund = 2pi
    knyquist = pi * n
    fund = MHDKernels.radiation_mode_coefficients(kfund, 0, rc)
    nyquist = MHDKernels.radiation_mode_coefficients(knyquist, 0, rc)
    return (qfund=rc.mean_free_path * kfund,
            qnyquist=rc.mean_free_path * knyquist,
            stream_fund=fund.streaming_weight,
            stream_nyquist=nyquist.streaming_weight,
            crad_fund=sqrt(max(fund.radiation_sound_speed2, 0)),
            crad_nyquist=sqrt(max(nyquist.radiation_sound_speed2, 0)))
end

function _cosmological_ionized_p0(z, fh, vunit, h, Ob)
    z > -1 || error("automatic MHD_P0 requires a valid start redshift")
    vunit > 0 || error("automatic MHD_P0 requires positive MHD_CHEM_VUNIT")
    nH = _density_unit_cgs(z, h, Ob) * Float64(fh) / ChemistryKernels.MH
    temperature = _PMF_T_CMB * (1 + Float64(z))
    xe = ChemistryKernels.total_electron_fraction(
        1.0, nH, temperature; fh=Float64(fh))
    # Hydrogen nuclei + helium nuclei + all free electrons, per baryon mass.
    particles_per_h_mass = Float64(fh) * (1 + xe) +
                           (1 - Float64(fh)) / 4
    return particles_per_h_mass * ChemistryKernels.KBOLTZ * temperature /
           (ChemistryKernels.MH * Float64(vunit)^2)
end

function _pmf_code_brms_from_comoving_ng(b0_ng, zref, h, Ob, vunit)
    zref > -1 || error("MHD_PMF_B0_NG requires a valid redshift reference")
    vunit > 0 || error("MHD_CHEM_VUNIT must be positive for MHD_PMF_B0_NG")
    ρ = _density_unit_cgs(zref, h, Ob)
    bphys = Float64(b0_ng) * 1e-9 * (1 + zref)^2
    bunit = Float64(vunit) * sqrt(4π * ρ)
    return bphys / bunit
end

function _terminal_gamma_code(gamma_drag, dtphys, dtcode, vunit, lbox_comoving_cm, amid,
                              magnetic_a_ref=NaN)
    if !(isfinite(lbox_comoving_cm) && lbox_comoving_cm > 0)
        return gamma_drag * dtphys / max(dtcode, eps(Float64))
    end
    lphys = lbox_comoving_cm * amid
    gamma_code = gamma_drag * lphys / max(vunit, eps(Float64))
    # B code units are fixed when a comoving physical amplitude is mapped at
    # a_ref. The terminal force clock otherwise acquires a spurious a/a_ref.
    return isfinite(magnetic_a_ref) ? gamma_code * amid / magnetic_a_ref : gamma_code
end

function _hydro_time_per_tau(a, h, vunit, lbox_comoving_cm,
                             cosmological_mhd_aref=NaN)
    if !(isfinite(lbox_comoving_cm) && lbox_comoving_cm > 0)
        return 1.0
    end
    H0 = 100.0 * h * 1.0e5 / 3.0857e24
    scale = isfinite(cosmological_mhd_aref) ?
            sqrt(a * cosmological_mhd_aref) : a
    return scale * vunit / (H0 * lbox_comoving_cm)
end

function _hydro_code_dt(dtphys, dtau, vunit, lbox_comoving_cm, amid,
                        cosmological_mhd_aref=NaN)
    if !(isfinite(lbox_comoving_cm) && lbox_comoving_cm > 0)
        return Float64(dtau)
    end
    if isfinite(cosmological_mhd_aref)
        return Float64(dtphys) * vunit * sqrt(cosmological_mhd_aref) /
               (lbox_comoving_cm * amid^(3 / 2))
    end
    return Float64(dtphys) * vunit / (lbox_comoving_cm * amid)
end

@inline function _cosmological_velocity_unit(vunit, amid, cosmological_mhd_aref)
    return isfinite(cosmological_mhd_aref) ?
           vunit * sqrt(cosmological_mhd_aref / amid) : vunit
end

@inline function _hubble_momentum_decay(a0, a1, cosmological_mhd_aref)
    return isfinite(cosmological_mhd_aref) ? sqrt(a0 / a1) : 1.0
end

@inline function _hydro_subinterval(dt_hydro, dtau, sub_dtau)
    return Float64(dt_hydro) * Float64(sub_dtau) /
           max(Float64(dtau), eps(Float64))
end

function _gas_gravity_kick_dt(dtau, a, h, vunit, lbox_comoving_cm,
                              cosmological_mhd_aref=NaN)
    return Float64(dtau) /
           _hydro_time_per_tau(a, h, vunit, lbox_comoving_cm,
                               cosmological_mhd_aref)
end

@inline function _drag_impulse(gamma_drag, dtphys, dtcode, rate_code)
    return isfinite(rate_code) ? Float64(rate_code) * Float64(dtcode) :
           Float64(gamma_drag) * Float64(dtphys)
end

function _parse_z_outputs(env::AbstractString)
    isempty(strip(env)) && return Float64[]
    return sort(parse.(Float64, split(env, ",")); rev=true)
end

@inline function _reached_output_z(znow::Real, ztarget::Real)
    tol = 64eps(Float64) * max(1.0, abs(Float64(znow)), abs(Float64(ztarget)))
    return Float64(znow) <= Float64(ztarget) + tol
end

"""Write a central density slice and line-of-sight maximum projection.

Frames are deliberately reduced on the host after one reusable device-to-host
copy. This keeps science-movie output to O(N^2) storage per redshift rather
than retaining full O(N^3) density states.
"""
function _write_density_frame!(frame_dir::AbstractString, hostrho, hostslice, hostproj,
                               rho, N::Int, z::Real, cycle::Int)
    isempty(frame_dir) && return
    mkpath(frame_dir)
    copyto!(hostrho, rho)
    mid = fld(N, 2) + 1
    @views copyto!(hostslice, hostrho[:, :, mid])
    @inbounds for j in 1:N, i in 1:N
        vmax = hostrho[i, j, 1]
        for k in 2:N
            vmax = max(vmax, hostrho[i, j, k])
        end
        hostproj[i, j] = vmax
    end
    tag = @sprintf("z%011.5f", Float64(z))
    slice_name = "density_slice_$(tag).f32"
    proj_name = "density_maxproj_$(tag).f32"
    open(joinpath(frame_dir, slice_name), "w") do io
        write(io, hostslice)
    end
    open(joinpath(frame_dir, proj_name), "w") do io
        write(io, hostproj)
    end
    manifest = joinpath(frame_dir, "frames.tsv")
    exists = isfile(manifest)
    open(manifest, "a") do io
        exists || println(io, "cycle\tz\tN\tslice_f32\tmaxproj_f32\tlayout")
        println(io, join((cycle, Float64(z), N, slice_name, proj_name, "julia-column-major"), '\t'))
    end
    return nothing
end

function _write_f32_array_atomic(path::AbstractString, array)
    host = Array{Float32}(undef, size(array))
    copyto!(host, array)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        write(io, host)
    end
    mv(tmp, path; force=true)
    host = nothing
    GC.gc()
    return nothing
end

function _write_f64_array_atomic(path::AbstractString, array)
    host = Array{Float64}(undef, size(array))
    copyto!(host, array)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        write(io, host)
    end
    mv(tmp, path; force=true)
    return nothing
end

function _read_f32_array_into!(array, path::AbstractString, dims)
    expected = prod(dims)
    filesize(path) == expected * sizeof(Float32) ||
        error("bad raw Float32 size for $path: got $(filesize(path)) bytes, expected $(expected * sizeof(Float32))")
    host = Array{Float32}(undef, dims)
    open(path, "r") do io
        read!(io, host)
        eof(io) || error("extra trailing bytes in $path")
    end
    copyto!(array, host)
    host = nothing
    GC.gc()
    return nothing
end

function _read_c32_array_into!(array, path::AbstractString, dims)
    expected = prod(dims)
    filesize(path) == expected * sizeof(ComplexF32) ||
        error("bad raw ComplexF32 size for $path: got $(filesize(path)) bytes, expected $(expected * sizeof(ComplexF32))")
    host = Array{ComplexF32}(undef, dims)
    open(path, "r") do io
        read!(io, host)
        eof(io) || error("extra trailing bytes in $path")
    end
    copyto!(array, host)
    host = nothing
    GC.gc()
    return nothing
end

function _import_initial_mhd_state!(be, dir::AbstractString, s; N::Int)
    isempty(dir) && return nothing
    for i in eachindex(s.U)
        _read_f32_array_into!(s.U[i], joinpath(dir, "mhd_U_$i.f32"), (N, N, N))
    end
    rho_min = Float64(minimum(s.U[1]))
    energy_min = Float64(minimum(s.U[5]))
    isfinite(rho_min) && rho_min > 0 ||
        error("imported MHD state has invalid minimum density $rho_min")
    isfinite(energy_min) && energy_min > 0 ||
        error("imported MHD state has invalid minimum energy $energy_min")
    brms = _device_brms!(be, s, s.scratch[1])
    @printf("INITIAL_STATE_MHD_IMPORT path=%s n=%d brms=%.8e rho_min=%.8e E_min=%.8e\n",
            dir, N, brms, rho_min, energy_min)
    flush(stdout)
    return Float32(brms)
end

function _import_initial_aux_state!(dir::AbstractString; N::Int,
                                    particles, chemistry, photons,
                                    density_perturbation=nothing)
    isempty(dir) && return nothing
    particle_names = ("particle_px", "particle_py", "particle_pz",
                      "particle_vx", "particle_vy", "particle_vz")
    particle_present = map(name -> isfile(joinpath(dir, "$name.f32")), particle_names)
    all(particle_present) || !any(particle_present) ||
        error("initial-state particle import must provide all six particle arrays")
    if all(particle_present)
        for (array, name) in zip(particles, particle_names)
            _read_f32_array_into!(array, joinpath(dir, "$name.f32"), (N^3,))
        end
    end
    if chemistry !== nothing
        chem_names = length(chemistry) == 4 ?
            ("chem_eint", "chem_HII", "chem_H2I", "chem_HeII") :
            ("chem_eint", "chem_HII", "chem_H2I")
        length(chemistry) in (3, 4) ||
            error("initial-state chemistry import expects three H fields or four H+He fields")
        chem_present = map(name -> isfile(joinpath(dir, "$name.f32")), chem_names)
        all(chem_present) || !any(chem_present) ||
            error("initial-state chemistry import must provide all requested chemistry fields")
        if all(chem_present)
            for (array, name) in zip(chemistry, chem_names)
                _read_f32_array_into!(array, joinpath(dir, "$name.f32"), (N^3,))
            end
        end
    end
    photon_path = joinpath(dir, "photon_hat_batch.c32")
    if isfile(photon_path)
        photons === nothing && error("initial state contains photon moments but response_model is not moments")
        _read_c32_array_into!(photons.hat_batch, photon_path, (N ÷ 2 + 1, N, N, 4))
    elseif photons !== nothing
        error("response_model=moments initial-state import requires photon_hat_batch.c32")
    end
    density_path = joinpath(dir, "gas_delta.f32")
    if isfile(density_path)
        density_perturbation === nothing && error(
            "initial state contains gas_delta.f32 but radiation density tracking is inactive")
        _read_f32_array_into!(density_perturbation, density_path, (N^3,))
    end
    @printf("INITIAL_STATE_AUX_IMPORT path=%s particles=%s chemistry=%s photons=%s gas_delta=%s\n",
            dir, string(all(particle_present)), string(chemistry !== nothing &&
            isfile(joinpath(dir, "chem_eint.f32"))), string(isfile(photon_path)),
            string(isfile(density_path)))
    flush(stdout)
    return nothing
end

@kernel function _reset_imported_pmf_state_k!(rho, mx, my, mz, E,
                                              @Const(Bx), @Const(By), @Const(Bz),
                                              gamma, p0)
    c = @index(Global)
    @inbounds begin
        T = eltype(E)
        rho[c] = one(T)
        mx[c] = zero(T)
        my[c] = zero(T)
        mz[c] = zero(T)
        E[c] = T(p0) / (T(gamma) - one(T)) +
               T(0.5) * (Bx[c] * Bx[c] + By[c] * By[c] + Bz[c] * Bz[c])
    end
end

@kernel function _scale_imported_b_k!(Bx, By, Bz, factor)
    c = @index(Global)
    @inbounds begin
        Bx[c] *= factor
        By[c] *= factor
        Bz[c] *= factor
    end
end

function _import_initial_b_field!(be, dir::AbstractString, s; N::Int, p0,
                                  target_brms=nothing)
    isempty(dir) && return nothing
    _read_f32_array_into!(s.U[6], joinpath(dir, "bx.f32"), (N, N, N))
    _read_f32_array_into!(s.U[7], joinpath(dir, "by.f32"), (N, N, N))
    _read_f32_array_into!(s.U[8], joinpath(dir, "bz.f32"), (N, N, N))
    T = eltype(s.U[5])
    imported_brms = _device_brms!(be, s, s.scratch[1])
    if target_brms !== nothing
        target_brms > 0 || error("MHD imported target Brms must be positive")
        imported_brms > 0 || error("cannot rescale an imported zero magnetic field")
        factor = T(target_brms / imported_brms)
        _scale_imported_b_k!(be)(s.U[6], s.U[7], s.U[8], factor; ndrange=N^3)
        KA.synchronize(be)
    end
    _reset_imported_pmf_state_k!(be)(s.U[1], s.U[2], s.U[3], s.U[4], s.U[5],
                                     s.U[6], s.U[7], s.U[8], T(s.γ), T(p0);
                                     ndrange=N^3)
    KA.synchronize(be)
    brms = _device_brms!(be, s, s.scratch[1])
    @printf("INITIAL_B_IMPORT path=%s n=%d raw_brms=%.8e brms=%.8e rescaled=%s\n",
            dir, N, imported_brms, brms, string(target_brms !== nothing))
    flush(stdout)
    return Float32(brms)
end

function _write_initial_b_field!(dir::AbstractString, s, st; N::Int, boxsize,
                                 zstart, zend, seed, kcut, kmax,
                                 mode_lock_n, high_mode_lock_n, brms,
                                 normalization_z, target_comoving_ng,
                                 reference_comoving_ng, total_comoving_ng)
    isempty(dir) && return nothing
    mkpath(dir)
    _write_f32_array_atomic(joinpath(dir, "bx.f32"), s.U[6])
    _write_f32_array_atomic(joinpath(dir, "by.f32"), s.U[7])
    _write_f32_array_atomic(joinpath(dir, "bz.f32"), s.U[8])
    open(joinpath(dir, "metadata.csv.tmp"), "w") do io
        println(io, "n,boxsize,zstart,zend,seed,kcut,kmax,mode_lock_n,high_mode_lock_n,normalization_kind,normalization_z,target_comoving_ng,reference_comoving_ng,total_comoving_ng,brms,brms_measured")
        println(io, join((N, boxsize, zstart, zend, seed, kcut,
                          kmax === nothing ? "" : kmax,
                          mode_lock_n === nothing ? "" : mode_lock_n,
                          high_mode_lock_n === nothing ? "" : high_mode_lock_n,
                          st.normalization_kind, normalization_z,
                          target_comoving_ng, reference_comoving_ng,
                          total_comoving_ng,
                          brms, st.brms_measured), ','))
    end
    mv(joinpath(dir, "metadata.csv.tmp"), joinpath(dir, "metadata.csv"); force=true)
    @printf("INITIAL_B_EXPORT path=%s n=%d brms=%.8e\n", dir, N, st.brms_measured)
    flush(stdout)
    return nothing
end

function _write_final_b_field!(dir::AbstractString, s; N::Int, z, cycle, brms)
    isempty(dir) && return nothing
    mkpath(dir)
    suffix = eltype(s.U[6]) === Float64 ? "f64" : "f32"
    writer = eltype(s.U[6]) === Float64 ? _write_f64_array_atomic : _write_f32_array_atomic
    writer(joinpath(dir, "bx.$suffix"), s.U[6])
    writer(joinpath(dir, "by.$suffix"), s.U[7])
    writer(joinpath(dir, "bz.$suffix"), s.U[8])
    open(joinpath(dir, "metadata.csv.tmp"), "w") do io
        println(io, "n,z,cycle,brms,precision")
        println(io, join((N, z, cycle, brms, suffix), ','))
    end
    mv(joinpath(dir, "metadata.csv.tmp"), joinpath(dir, "metadata.csv"); force=true)
    @printf("FINAL_B_EXPORT path=%s n=%d z=%.8e brms=%.8e\n",
            dir, N, z, brms)
    flush(stdout)
    return nothing
end

function _write_history!(path, cycle, measured, a, br, divn, δb, δdm;
                         btrans=NaN, btrans0=NaN, delta2k=NaN,
                         thermalstats=(NaN, NaN, NaN, NaN, NaN, NaN),
                         temperaturestats=(NaN, NaN, NaN))
    isempty(path) && return
    exists = isfile(path)
    mkpath(dirname(path))
    open(path, "a") do io
        exists || println(io, join(("cycle","measured","z","a","brms","divBdx_over_brms",
                                    "delta_b_rms","delta_dm_rms",
                                    "xHII_mean","xHII_min","xHII_max",
                                    "fH2_mean","fH2_min","fH2_max",
                                    "xHeII_mean","xHeII_min","xHeII_max",
                                    "xe_total_mean","xe_total_min","xe_total_max",
                                    "btrans_rms","btrans_ratio",
                                    "delta2k_amp",
                                    "p_mean","p_min","p_max",
                                    "eint_mean","eint_min","eint_max",
                                    "temperature_mean","temperature_min",
                                    "temperature_max"), '\t'))
        println(io, join((cycle, measured, a_to_z(a), a, br, divn, δb, δdm,
                          ntuple(_ -> NaN, 12)...,
                          btrans, btrans / btrans0, delta2k,
                          thermalstats..., temperaturestats...), '\t'))
    end
end

function _write_history!(path, cycle, measured, a, br, divn, δb, δdm, chemstats;
                         btrans=NaN, btrans0=NaN, delta2k=NaN,
                         thermalstats=(NaN, NaN, NaN, NaN, NaN, NaN),
                         temperaturestats=(NaN, NaN, NaN))
    isempty(path) && return
    exists = isfile(path)
    mkpath(dirname(path))
    open(path, "a") do io
        exists || println(io, join(("cycle","measured","z","a","brms","divBdx_over_brms",
                                    "delta_b_rms","delta_dm_rms",
                                    "xHII_mean","xHII_min","xHII_max",
                                    "fH2_mean","fH2_min","fH2_max",
                                    "xHeII_mean","xHeII_min","xHeII_max",
                                    "xe_total_mean","xe_total_min","xe_total_max",
                                    "btrans_rms","btrans_ratio",
                                    "delta2k_amp",
                                    "p_mean","p_min","p_max",
                                    "eint_mean","eint_min","eint_max",
                                    "temperature_mean","temperature_min",
                                    "temperature_max"), '\t'))
        println(io, join((cycle, measured, a_to_z(a), a, br, divn, δb, δdm,
                          chemstats..., btrans, btrans / btrans0, delta2k,
                          thermalstats..., temperaturestats...), '\t'))
    end
end

function _write_pkmu!(path, label, result; cycle, measured, a, nmu)
    isempty(path) && return
    exists = isfile(path)
    mkpath(dirname(path))
    open(path, "a") do io
        exists || println(io, join(("field","cycle","measured","z","a","kbin","mubin",
                                    "k","mu_lo","mu_hi","P","Nmodes"), '\t'))
        for b in eachindex(result.k), m in 1:nmu
            μlo = (m - 1) / nmu
            μhi = m / nmu
            println(io, join((label, cycle, measured, a_to_z(a), a, b, m, result.k[b],
                              μlo, μhi, result.P[b,m], result.Nmodes[b,m]), '\t'))
        end
    end
end

function _write_density_pdf!(path, label, counts_dev, rho; be, cycle, measured, a,
                             loglo::Real, loghi::Real)
    isempty(path) && return
    nbins = length(counts_dev)
    fill!(counts_dev, Int32(0))
    width = (Float64(loghi) - Float64(loglo)) / Float64(nbins)
    invw = 1.0 / width
    _logrho_hist_k!(be)(counts_dev, rho, Int(nbins), T(loglo), T(invw); ndrange=length(rho))
    KA.synchronize(be)
    counts = Vector{Int32}(undef, nbins)
    copyto!(counts, counts_dev)

    exists = isfile(path)
    mkpath(dirname(path))
    norm = Float64(length(rho)) * width
    open(path, "a") do io
        exists || println(io, join(("field","cycle","measured","z","a","bin",
                                    "logrho_lo","logrho_hi","rho_mid","count","pdf"), '\t'))
        for b in 1:nbins
            lo = Float64(loglo) + (b - 1) * width
            hi = lo + width
            mid = 10.0 ^ (0.5 * (lo + hi))
            pdf = Float64(counts[b]) / norm
            println(io, join((label, cycle, measured, a_to_z(a), a, b,
                              lo, hi, mid, counts[b], pdf), '\t'))
        end
    end
end

function _write_linear_mode!(path; be, s, rho_dm, scratch, N, gravity,
                             cycle, measured, a, kmode, boxsize, hubble,
                             omega_b=0.049,
                             gas_delta=nothing, HII=nothing, H2I=nothing,
                             HeII=nothing,
                             fh=0.76f0, photon_real_batch=nothing,
                             temperature_velocity_unit2=NaN)
    isempty(path) && return
    0 < kmode < N ÷ 2 ||
        error("complex mode diagnostics require 0 < MHD_PMF_KMODE < N/2")
    rows = Pair{String,ComplexF64}[]
    for (name, src) in (("B_x", s.U[6]), ("B_y", s.U[7]), ("B_z", s.U[8]))
        mean = Float64(sum(src)) / Float64(length(src))
        push!(rows, name => _device_complex_mode!(be, scratch, src, N, kmode;
                                                   mean=mean))
    end
    gas_source = gas_delta === nothing ? s.U[1] : gas_delta
    gas_mean = gas_delta === nothing ? one(eltype(gas_source)) : zero(eltype(gas_source))
    push!(rows, "gas_delta" => _device_complex_mode!(be, scratch, gas_source, N,
                                                       kmode; mean=gas_mean))
    for (name, momentum) in (("gas_v_x", s.U[2]), ("gas_v_y", s.U[3]),
                             ("gas_v_z", s.U[4]))
        push!(rows, name => _device_complex_velocity_mode!(be, scratch, momentum,
                                                            s.U[1], N, kmode,
                                                            s.smallr))
    end
    MHDKernels.terminal_magnetic_force!(s, s.scratch[1], s.scratch[2],
                                        s.scratch[3])
    KA.synchronize(s.be)
    for d in 1:3
        axis = ("x", "y", "z")[d]
        push!(rows, "lorentz_accel_$axis" =>
                    _device_complex_velocity_mode!(be, scratch, s.scratch[d],
                                                     s.U[1], N, kmode,
                                                     s.smallr))
    end
    if gravity
        push!(rows, "dm_delta" => _device_complex_mode!(be, scratch, rho_dm, N,
                                                          kmode; mean=one(eltype(rho_dm))))
    end
    if photon_real_batch !== nothing
        photon_delta = reshape(view(photon_real_batch, :, :, :, 1), :)
        push!(rows, "photon_delta" => _device_complex_mode!(be, scratch,
                                                              photon_delta, N,
                                                              kmode))
        for d in 1:3
            photon_velocity = reshape(view(photon_real_batch, :, :, :, d + 1), :)
            axis = ("x", "y", "z")[d]
            push!(rows, "photon_v_$axis" => _device_complex_mode!(be, scratch,
                                                                    photon_velocity,
                                                                    N, kmode))
            push!(rows, "photon_baryon_slip_$axis" =>
                        _device_complex_slip_mode!(be, scratch, photon_velocity,
                                                   s.U[d + 1], s.U[1], N,
                                                   kmode, s.smallr))
        end
    end
    HII === nothing || push!(rows, "x_HII" =>
        _device_complex_species_mode!(be, scratch, HII, s.U[1], N, kmode, fh))
    H2I === nothing || push!(rows, "x_H2" =>
        _device_complex_species_mode!(be, scratch, H2I, s.U[1], N, kmode, fh))
    HeII === nothing || push!(rows, "x_HeII" =>
        _device_complex_species_mode!(be, scratch, HeII, s.U[1], N, kmode,
                                      4 * fh))
    push!(rows, "gas_pressure_fraction" =>
        _device_complex_thermal_fraction_mode!(
            be, scratch, s.scratch[4], s, N, kmode, 1,
        ))
    push!(rows, "gas_specific_internal_energy_fraction" =>
        _device_complex_thermal_fraction_mode!(
            be, scratch, s.scratch[4], s, N, kmode, 2,
        ))
    if HII !== nothing && H2I !== nothing && isfinite(temperature_velocity_unit2)
        push!(rows, "gas_temperature_fraction" =>
            _device_complex_temperature_fraction_mode!(
                be, scratch, s.scratch[4], s, HII, H2I, HeII, N, kmode,
                temperature_velocity_unit2, a_to_z(a), fh, hubble, omega_b,
            ))
    end

    k_box = 2π * Float64(kmode) / Float64(boxsize)
    k_mpc = 1000.0 * Float64(hubble) * k_box
    exists = isfile(path)
    mkpath(dirname(path))
    open(path, "a") do io
        exists || println(io, join(("field", "cycle", "measured", "z", "a",
                                    "kmode", "k_h_ckpc", "k_comoving_mpc_inv",
                                    "real", "imag", "amplitude", "phase"), '\t'))
        for (field, value) in rows
            println(io, join((field, cycle, measured, a_to_z(a), a, kmode,
                              k_box, k_mpc, real(value), imag(value), abs(value),
                              angle(value)), '\t'))
        end
    end
    return nothing
end

function _emit_diagnostics!(; be_p, s, ρdm, scratch, px, py, pz, vx, vy, vz,
                            N, ncells, gravity, cycle, measured, a, pk_path,
                            history_path, pdf_path, pdf_counts, pdf_loglo, pdf_loghi,
                            nmu, nbins, boxsize, axis, kmax=nothing,
                            lattice_displacements=false,
                            HII=nothing, H2I=nothing, HeII=nothing, fh=0.76f0,
                            btrans0=NaN, density_kmode=1,
                            mode_history_path="", mode_kmode=density_kmode,
                            hubble=0.674,
                            omega_b=0.049,
                            radiation_workspace=nothing,
                            radiation_photon_state=nothing,
                            radiation_drag_rate=NaN,
                            temperature_velocity_unit2=NaN)
    T = eltype(s.U[1])
    if gravity
        _deposit_particles!(ρdm, px, py, pz, vx, vy, vz;
                            N=N, disp=0, lattice_displacements=lattice_displacements)
        KA.synchronize(be_p)
    end
    br = _device_brms!(be_p, s, scratch)
    divn = _device_divb_norm!(be_p, s, scratch, br)
    gas_delta=radiation_workspace === nothing ? nothing :
              radiation_workspace.density_perturbation
    δb = gas_delta === nothing ? _device_delta_rms!(be_p,scratch,s.U[1],one(T)) :
                                 _device_delta_rms!(be_p,scratch,gas_delta,zero(T))
    δdm = gravity ? _device_delta_rms!(be_p, scratch, ρdm, one(T)) : NaN
    chemstats = _device_species_stats!(
        be_p, scratch, s.U[1], HII, H2I, HeII, fh;
        z=a_to_z(a), h=hubble, Ob=omega_b)
    thermalstats = _device_thermal_stats!(be_p, s, scratch)
    temperaturestats = isfinite(temperature_velocity_unit2) ?
        _device_temperature_stats!(be_p, s, scratch, HII, H2I, HeII,
                                   _density_unit_cgs(a_to_z(a), hubble, omega_b),
                                   temperature_velocity_unit2, a_to_z(a), fh) :
        (NaN, NaN, NaN)
    btrans = _device_component_rms!(be_p, scratch, s.U[7])
    delta2k = gas_delta === nothing ?
        _device_density_cos_mode_amp!(be_p,scratch,s.U[1],N,density_kmode,2) :
        _device_density_cos_mode_amp!(be_p,scratch,gas_delta,N,density_kmode,2;
                                      mean=zero(T))
    _write_history!(history_path, cycle, measured, a, br, divn, δb, δdm, chemstats;
                    btrans=btrans, btrans0=btrans0, delta2k=delta2k,
                    thermalstats=thermalstats,
                    temperaturestats=temperaturestats)

    photon_real_ready = false
    if !isempty(mode_history_path)
        if radiation_workspace !== nothing && radiation_photon_state !== nothing
            MHDKernels._radiation_inverse_batch!(radiation_workspace.real_batch,
                                                 radiation_photon_state.hat_batch)
            photon_real_ready = true
        end
        _write_linear_mode!(mode_history_path; be=be_p, s=s, rho_dm=ρdm,
                            scratch=scratch, N=N, gravity=gravity, cycle=cycle,
                            measured=measured, a=a, kmode=mode_kmode,
                            boxsize=boxsize, hubble=hubble, omega_b=omega_b,
                            gas_delta=gas_delta,
                            HII=HII, H2I=H2I, HeII=HeII, fh=fh,
                            temperature_velocity_unit2=temperature_velocity_unit2,
                            photon_real_batch=photon_real_ready ?
                                radiation_workspace.real_batch : nothing)
    end

    if pdf_counts !== nothing && !isempty(pdf_path)
        _write_density_pdf!(pdf_path, "gas_rho", pdf_counts, s.U[1];
                            be=be_p, cycle=cycle, measured=measured, a=a,
                            loglo=pdf_loglo, loghi=pdf_loghi)
        if gravity
            _write_density_pdf!(pdf_path, "dm_rho", pdf_counts, ρdm;
                                be=be_p, cycle=cycle, measured=measured, a=a,
                                loglo=pdf_loglo, loghi=pdf_loghi)
        end
    end

    isempty(pk_path) && return
    isotropic_nmu = 1
    if gas_delta === nothing
        _delta_from_density_k!(be_p)(scratch,s.U[1],Int(N);ndrange=ncells)
        KA.synchronize(be_p)
    else
        copyto!(scratch,gas_delta)
    end
    _write_pkmu!(pk_path, "gas_delta",
                 PoissonKernels.power_spectrum_aniso_gpu(scratch; boxsize=boxsize,
                                                          nmu=isotropic_nmu,
                                                          nbins=nbins, axis=axis,
                                                          kmax=kmax);
                 cycle=cycle, measured=measured, a=a, nmu=isotropic_nmu)
    if gravity
        _delta_from_density_k!(be_p)(scratch, ρdm, Int(N); ndrange=ncells)
        KA.synchronize(be_p)
        _write_pkmu!(pk_path, "dm_delta",
                     PoissonKernels.power_spectrum_aniso_gpu(scratch; boxsize=boxsize,
                                                              nmu=isotropic_nmu,
                                                              nbins=nbins, axis=axis,
                                                              kmax=kmax);
                     cycle=cycle, measured=measured, a=a, nmu=isotropic_nmu)
    end
    Bx = reshape(s.U[6], N, N, N)
    By = reshape(s.U[7], N, N, N)
    Bz = reshape(s.U[8], N, N, N)
    # Magnetic damping is isotropic in these campaigns. One mu bin preserves
    # the shared P(k,mu) schema while avoiding unnecessary angular output.
    _write_pkmu!(pk_path, "B_vector",
                 PoissonKernels.power_spectrum_aniso_gpu((Bx, By, Bz);
                                                          boxsize=boxsize,
                                                          nmu=isotropic_nmu,
                                                          nbins=nbins, axis=axis,
                                                          kmax=kmax);
                 cycle=cycle, measured=measured, a=a, nmu=isotropic_nmu)
    pressure_force = ntuple(d -> reshape(s.scratch[d], N, N, N), 3)
    tension_force = ntuple(d -> reshape(s.scratch[d + 3], N, N, N), 3)
    lorentz_force = ntuple(d -> reshape(s.scratch[d + 6], N, N, N), 3)
    MHDKernels.terminal_magnetic_force_split!(
        s, pressure_force..., tension_force...)
    MHDKernels.terminal_magnetic_force!(s, lorentz_force...)
    KA.synchronize(s.be)
    for (label, force) in (("B_pressure_force", pressure_force),
                           ("B_tension_force", tension_force),
                           ("B_lorentz_force", lorentz_force))
        _write_pkmu!(pk_path, label,
                     PoissonKernels.power_spectrum_aniso_gpu(force;
                         boxsize=boxsize, nmu=isotropic_nmu, nbins=nbins,
                         axis=axis, kmax=kmax);
                     cycle=cycle, measured=measured, a=a, nmu=isotropic_nmu)
    end
    if radiation_workspace !== nothing && radiation_photon_state !== nothing
        photon_real_ready || MHDKernels._radiation_inverse_batch!(
            radiation_workspace.real_batch, radiation_photon_state.hat_batch)
        photon_delta=reshape(view(radiation_workspace.real_batch,:,:,:,1),:)
        copyto!(scratch,photon_delta)
        _write_pkmu!(pk_path,"photon_delta",
            PoissonKernels.power_spectrum_aniso_gpu(scratch;boxsize=boxsize,
                nmu=isotropic_nmu,nbins=nbins,axis=axis,kmax=kmax);
            cycle=cycle,measured=measured,a=a,nmu=isotropic_nmu)

        photon_velocity=ntuple(d->reshape(s.scratch[d],N,N,N),3)
        for d in 1:3
            copyto!(photon_velocity[d],view(radiation_workspace.real_batch,:,:,:,d+1))
        end
        _write_pkmu!(pk_path,"photon_velocity",
            PoissonKernels.power_spectrum_aniso_gpu(photon_velocity;boxsize=boxsize,
                nmu=nmu,nbins=nbins,axis=axis,kmax=kmax);
            cycle=cycle,measured=measured,a=a,nmu=nmu)

        _photon_slip_diag_k!(s.be)(s.scratch[1],s.scratch[2],s.scratch[3],
            s.scratch[1],s.scratch[2],s.scratch[3],
            s.U[1],s.U[2],s.U[3],s.U[4],s.smallr;ndrange=ncells)
        KA.synchronize(s.be)
        slip=ntuple(d->reshape(s.scratch[d],N,N,N),3)
        _write_pkmu!(pk_path,"photon_baryon_slip",
            PoissonKernels.power_spectrum_aniso_gpu(slip;boxsize=boxsize,
                nmu=nmu,nbins=nbins,axis=axis,kmax=kmax);
            cycle=cycle,measured=measured,a=a,nmu=nmu)

        pressure_accel=ntuple(d->reshape(s.scratch[d+3],N,N,N),3)
        hydro_accel=ntuple(d->reshape(s.scratch[d+6],N,N,N),3)
        _gas_force_balance_diag_k!(s.be)(pressure_accel...,hydro_accel...,
            s.U[1],s.U[2],s.U[3],s.U[4],s.U[5],s.U[6],s.U[7],s.U[8],
            Int(N),T(0.5)/s.dx,s.γ,s.smallr,T(1e-30);ndrange=ncells)
        KA.synchronize(s.be)
        _write_pkmu!(pk_path,"gas_pressure_acceleration",
            PoissonKernels.power_spectrum_aniso_gpu(pressure_accel;boxsize=boxsize,
                nmu=isotropic_nmu,nbins=nbins,axis=axis,kmax=kmax);
            cycle=cycle,measured=measured,a=a,nmu=isotropic_nmu)
        _write_pkmu!(pk_path,"gas_hydro_acceleration",
            PoissonKernels.power_spectrum_aniso_gpu(hydro_accel;boxsize=boxsize,
                nmu=isotropic_nmu,nbins=nbins,axis=axis,kmax=kmax);
            cycle=cycle,measured=measured,a=a,nmu=isotropic_nmu)
        if isfinite(radiation_drag_rate)
            _gas_total_accel_diag_k!(s.be)(pressure_accel...,
                hydro_accel...,slip...,T(radiation_drag_rate);ndrange=ncells)
            KA.synchronize(s.be)
            _write_pkmu!(pk_path,"gas_total_acceleration",
                PoissonKernels.power_spectrum_aniso_gpu(pressure_accel;boxsize=boxsize,
                    nmu=isotropic_nmu,nbins=nbins,axis=axis,kmax=kmax);
                cycle=cycle,measured=measured,a=a,nmu=isotropic_nmu)
        end

        gas_velocity=ntuple(d->reshape(s.scratch[d],N,N,N),3)
        _gas_velocity_diag_k!(s.be)(gas_velocity...,
            s.U[1],s.U[2],s.U[3],s.U[4],s.smallr;ndrange=ncells)
        KA.synchronize(s.be)
        _write_pkmu!(pk_path,"gas_velocity",
            PoissonKernels.power_spectrum_aniso_gpu(gas_velocity;boxsize=boxsize,
                nmu=isotropic_nmu,nbins=nbins,axis=axis,kmax=kmax);
            cycle=cycle,measured=measured,a=a,nmu=isotropic_nmu)
        _mass_flux_divergence_diag_k!(s.be)(s.scratch[4],
            s.U[2],s.U[3],s.U[4],Int(N),T(0.5)/s.dx;ndrange=ncells)
        KA.synchronize(s.be)
        _write_pkmu!(pk_path,"gas_mass_flux_divergence",
            PoissonKernels.power_spectrum_aniso_gpu(reshape(s.scratch[4],N,N,N);
                boxsize=boxsize,nmu=isotropic_nmu,nbins=nbins,axis=axis,kmax=kmax);
            cycle=cycle,measured=measured,a=a,nmu=isotropic_nmu)
    end
end

function main()
    N = parse(Int, get(ENV, "MHD_PMF_N", "128"))
    steps = parse(Int, get(ENV, "MHD_STEPS", "8"))
    tfinal_env = get(ENV, "MHD_TFINAL", "")
    tfinal = isempty(tfinal_env) ? nothing : parse(Float64, tfinal_env)
    zstart_env = get(ENV, "MHD_ZSTART", "")
    zrun = !isempty(zstart_env)
    # A benchmark warmup advances every state array but intentionally does not
    # advance cosmological time. It is therefore valid only for fixed-time
    # throughput tests, never as the default for a physical redshift trajectory.
    warmup = parse(Int, get(ENV, "MHD_WARMUP", zrun ? "0" : "1"))
    profile_skip_cycles = parse(Int, get(ENV, "MHD_PROFILE_SKIP_CYCLES", "0"))
    profile_skip_cycles >= 0 || error("MHD_PROFILE_SKIP_CYCLES must be non-negative")
    zstart = zrun ? parse(Float64, zstart_env) : NaN
    zend = zrun ? parse(Float64, get(ENV, "MHD_ZEND", "20")) : NaN
    hub = parse(Float64, get(ENV, "MHD_H", "0.674"))
    Om = parse(Float64, get(ENV, "MHD_OM", "0.315"))
    OL = parse(Float64, get(ENV, "MHD_OL", "0.685"))
    Or = parse(Float64, get(ENV, "MHD_OR", string(4.15e-5 / hub^2)))
    Ok = 1.0 - Om - OL - Or
    Ob = parse(Float64, get(ENV, "MHD_OB", string(0.17037 * Om)))
    maxexp = parse(Float64, get(ENV, "MHD_MAXEXP", "0.01"))
    # Source factor 1 is the corrected-clock science reference.  Larger factors
    # are explicit performance experiments and must pass the cadence gate.
    source_max_dt_factor = parse(Float64, get(ENV, "MHD_SOURCE_MAX_DT_FACTOR", "1.0"))
    source_max_dt_factor >= 1 || error("MHD_SOURCE_MAX_DT_FACTOR must be >= 1")
    maxcycles = parse(Int, get(ENV, "MHD_MAX_CYCLES", "100000000"))
    debug_stop_cycle = parse(Int, get(ENV, "MHD_DEBUG_STOP_CYCLE", "0"))
    print_every = parse(Int, get(ENV, "MHD_PRINT_EVERY", "100"))
    cycle_print_every = parse(Int, get(ENV, "MHD_CYCLE_PRINT_EVERY", "1"))
    check_every = parse(Int, get(ENV, "MHD_CHECK_EVERY", zrun ? "25" : "0"))
    cfl = parse(Float64, get(ENV, "MHD_CFL", "0.4"))
    p0_env = lowercase(strip(get(ENV, "MHD_P0", "1.0")))
    p0_auto = p0_env in ("auto", "cosmology", "cmb")
    p0 = p0_auto ? NaN : parse(Float64, p0_env)
    pfloor = parse(Float64, get(ENV, "MHD_PFLOOR", "1e-7"))
    llf_dmin = parse(Float64, get(ENV, "MHD_LLF_DMIN", "1e-4"))
    llf_pmin = parse(Float64, get(ENV, "MHD_LLF_PMIN", "0"))
    drag_enabled = get(ENV, "MHD_COMPTON_DRAG", "0") in ("1", "true", "TRUE", "yes", "on")
    # Source-aware full MHD is the science default. The terminal closure did not
    # pass the density-response validation and must remain an explicit legacy
    # comparison mode rather than an automatically selected production regime.
    default_drag_dt_mode = drag_enabled && zrun ? "radiation" :
                           (drag_enabled ? "godunov" : "explicit")
    drag_dt_mode = Symbol(lowercase(get(ENV, "MHD_DRAG_DT_MODE", default_drag_dt_mode)))
    drag_dt_mode in (:explicit, :boost, :subcycle, :semiimplicit, :aware,
                     :godunov, :radiation, :terminal, :hybrid) ||
        error("MHD_DRAG_DT_MODE must be explicit, boost, subcycle, semiimplicit, aware, godunov, radiation, terminal, or hybrid; got $(drag_dt_mode)")
    drag_dt_mode in (:godunov, :radiation, :terminal, :hybrid) && !drag_enabled &&
        error("MHD_DRAG_DT_MODE=$(drag_dt_mode) requires MHD_COMPTON_DRAG=1")
    radiation_bridge_q2=parse(Float64,get(ENV,"MHD_RADIATION_BRIDGE_Q2","5.0"))
    radiation_long_viscosity=parse(Float64,get(ENV,"MHD_RADIATION_LONG_VISCOSITY",string(4/3)))
    radiation_max_dlna=parse(Float64,get(ENV,"MHD_RADIATION_MAX_DLNA","2.5e-4"))
    radiation_highz_max_dlna=parse(Float64,get(ENV,"MHD_RADIATION_HIGHZ_MAX_DLNA",
                                                    string(radiation_max_dlna)))
    radiation_highz_zmin=parse(Float64,get(ENV,"MHD_RADIATION_HIGHZ_ZMIN","Inf"))
    radiation_nyquist_guard=parse(Float64,get(ENV,"MHD_RADIATION_NYQUIST_GUARD","1"))
    radiation_response_model=Symbol(lowercase(get(ENV,"MHD_RADIATION_RESPONSE_MODEL","moments")))
    radiation_nonlinear_default=radiation_response_model in (:angular,:transport) ? "0.4" : "Inf"
    radiation_nonlinear_cfl=parse(Float64,get(ENV,"MHD_RADIATION_NONLINEAR_CFL",
                                                   radiation_nonlinear_default))
    radiation_energy_ledger_every=parse(Int,get(ENV,"MHD_RADIATION_ENERGY_LEDGER_EVERY","0"))
    radiation_free_stream_tolerance=parse(Float64,
        get(ENV,"MHD_RADIATION_FREE_STREAM_TOL","5e-3"))
    radiation_positivity_fallback=get(ENV,"MHD_RADIATION_POSITIVITY_FALLBACK","1") in
        ("1","true","TRUE","yes","on")
    radiation_positivity_max_fallback_passes=parse(Int,
        get(ENV,"MHD_RADIATION_POSITIVITY_MAX_FALLBACK_PASSES","8"))
    radiation_positivity_wave_growth_limit=parse(Float64,
        get(ENV,"MHD_RADIATION_POSITIVITY_WAVE_GROWTH_LIMIT","4"))
    radiation_positivity_max_subdivisions=parse(Int,
        get(ENV,"MHD_RADIATION_POSITIVITY_MAX_SUBDIVISIONS","8"))
    radiation_checkpoint_fallback=get(ENV,
        "MHD_RADIATION_CHECKPOINT_FALLBACK","0") in
        ("1","true","TRUE","yes","on")
    radiation_bridge_q2>0 || error("MHD_RADIATION_BRIDGE_Q2 must be positive")
    radiation_long_viscosity>0 || error("MHD_RADIATION_LONG_VISCOSITY must be positive")
    radiation_max_dlna>0 || error("MHD_RADIATION_MAX_DLNA must be positive")
    radiation_highz_max_dlna>0 || error("MHD_RADIATION_HIGHZ_MAX_DLNA must be positive")
    radiation_highz_zmin>=0 || error("MHD_RADIATION_HIGHZ_ZMIN must be non-negative")
    radiation_nonlinear_cfl>0 || error("MHD_RADIATION_NONLINEAR_CFL must be positive")
    0 < radiation_nyquist_guard <= 1 || error("MHD_RADIATION_NYQUIST_GUARD must be in (0,1]")
    radiation_response_model in (:bridge,:angular,:transport,:moments,:photon_moments,
                                  :moments_exponential,
                                  :photon_moments_exponential) ||
        error("MHD_RADIATION_RESPONSE_MODEL must be bridge, angular, moments, or " *
              "moments_exponential")
    radiation_energy_ledger_every>=0 ||
        error("MHD_RADIATION_ENERGY_LEDGER_EVERY must be non-negative")
    radiation_free_stream_tolerance>=0 ||
        error("MHD_RADIATION_FREE_STREAM_TOL must be non-negative")
    radiation_positivity_max_fallback_passes>0 ||
        error("MHD_RADIATION_POSITIVITY_MAX_FALLBACK_PASSES must be positive")
    radiation_positivity_wave_growth_limit>1 ||
        error("MHD_RADIATION_POSITIVITY_WAVE_GROWTH_LIMIT must exceed one")
    radiation_positivity_max_subdivisions>=0 ||
        error("MHD_RADIATION_POSITIVITY_MAX_SUBDIVISIONS must be non-negative")
    drag_dt_mode===:radiation && !zrun &&
        error("MHD_DRAG_DT_MODE=radiation requires a cosmological redshift run")
    allow_uncertified_terminal = get(ENV, "MHD_ALLOW_UNCERTIFIED_TERMINAL", "0") in
                                 ("1", "true", "TRUE", "yes", "on")
    drag_dt_mode in (:terminal, :hybrid) && !allow_uncertified_terminal &&
        error("MHD_DRAG_DT_MODE=$(drag_dt_mode) uses the uncertified terminal closure; " *
              "production runs must use godunov. Set MHD_ALLOW_UNCERTIFIED_TERMINAL=1 " *
              "only for legacy comparison experiments.")
    drag_dt_boost = parse(Float64, get(ENV, "MHD_DRAG_DT_BOOST", "1.0"))
    drag_dt_boost >= 1 || error("MHD_DRAG_DT_BOOST must be >= 1")
    drag_max_impulse = parse(Float64, get(ENV, "MHD_DRAG_MAX_IMPULSE", "Inf"))
    drag_max_impulse > 0 || error("MHD_DRAG_MAX_IMPULSE must be positive")
    drag_xe_env = lowercase(get(ENV, "MHD_DRAG_XE", "auto"))
    drag_xe_fixed = drag_xe_env == "auto" ? NaN : parse(Float64, drag_xe_env)
    drag_helium_electrons = get(ENV, "MHD_DRAG_HELIUM_ELECTRONS", "1") in
                            ("1", "true", "TRUE", "yes", "on")
    drag_rate_code = parse(Float64, get(ENV, "MHD_DRAG_RATE_CODE", "NaN"))
    (isnan(drag_rate_code) || drag_rate_code >= 0) ||
        error("MHD_DRAG_RATE_CODE must be non-negative or NaN")
    init_terminal_momenta = get(ENV, "MHD_INIT_TERMINAL_MOMENTA", "0") in
                            ("1", "true", "TRUE", "yes", "on")
    init_terminal_momenta && !drag_enabled &&
        error("MHD_INIT_TERMINAL_MOMENTA=1 requires MHD_COMPTON_DRAG=1")
    init_terminal_momenta && !zrun && !isfinite(drag_rate_code) &&
        error("automatic terminal-momentum initialization requires MHD_ZSTART")
    drag_vx0 = parse(Float64, get(ENV, "MHD_DRAG_VX0", "0.0"))
    drag_vy0 = parse(Float64, get(ENV, "MHD_DRAG_VY0", "0.0"))
    drag_vz0 = parse(Float64, get(ENV, "MHD_DRAG_VZ0", "0.0"))
    terminal_vcap = parse(Float64, get(ENV, "MHD_TERMINAL_VCAP", "Inf"))
    terminal_gamma_scale = parse(Float64, get(ENV, "MHD_TERMINAL_GAMMA_SCALE", "1.0"))
    terminal_gamma_code_override = parse(Float64, get(ENV, "MHD_TERMINAL_GAMMA_CODE", "NaN"))
    terminal_split = get(ENV, "MHD_TERMINAL_SPLIT", "0") != "0"
    terminal_vsmooth = parse(Float64, get(ENV, "MHD_TERMINAL_VSMOOTH", "0.0"))
    0 <= terminal_vsmooth <= 1 || error("MHD_TERMINAL_VSMOOTH must be in [0,1]")
    terminal_dedicated_scratch = get(ENV, "MHD_TERMINAL_DEDICATED_SCRATCH", "0") in
                                 ("1", "true", "TRUE", "yes", "on")
    terminal_lbox_cm_env = get(ENV, "MHD_TERMINAL_LBOX_CM", "")
    terminal_lbox_ckpc_env = get(ENV, "MHD_TERMINAL_LBOX_CKPC", "")
    terminal_lbox_cm = !isempty(strip(terminal_lbox_cm_env)) ? parse(Float64, terminal_lbox_cm_env) :
                       (!isempty(strip(terminal_lbox_ckpc_env)) ? parse(Float64, terminal_lbox_ckpc_env) * _PMF_KPC_CM : NaN)
    terminal_max_disp_cells = parse(Float64, get(ENV, "MHD_TERMINAL_MAX_DISP_CELLS", "0.1"))
    terminal_diff_cfl = parse(Float64, get(ENV, "MHD_TERMINAL_DIFF_CFL", "Inf"))
    terminal_diff_cfl > 0 || error("MHD_TERMINAL_DIFF_CFL must be positive")
    terminal_accuracy_cfl = parse(Float64, get(ENV, "MHD_TERMINAL_ACCURACY_CFL", "Inf"))
    terminal_accuracy_cfl > 0 || error("MHD_TERMINAL_ACCURACY_CFL must be positive")
    terminal_accuracy_pressure_env = lowercase(strip(
        get(ENV, "MHD_TERMINAL_ACCURACY_PRESSURE", "auto")))
    terminal_accuracy_pressure_auto = terminal_accuracy_pressure_env in
                                      ("auto", "initial", "p0")
    terminal_accuracy_pressure = terminal_accuracy_pressure_auto ? NaN :
                                 parse(Float64, terminal_accuracy_pressure_env)
    terminal_implicit_diffuse = get(ENV, "MHD_TERMINAL_IMPLICIT_DIFFUSE", "0") in
                                ("1", "true", "TRUE", "yes", "on")
    terminal_implicit_diffuse_fac = parse(Float64, get(ENV, "MHD_TERMINAL_IMPLICIT_DIFFUSE_FAC", "1.0"))
    terminal_implicit_diffuse_fac >= 0 || error("MHD_TERMINAL_IMPLICIT_DIFFUSE_FAC must be non-negative")
    terminal_pressure_implicit = get(ENV, "MHD_TERMINAL_PRESSURE_IMPLICIT", "0") in
                                 ("1", "true", "TRUE", "yes", "on")
    terminal_pressure_coeff_env = lowercase(strip(get(ENV, "MHD_TERMINAL_PRESSURE_COEFF", "auto")))
    terminal_pressure_coeff_auto = terminal_pressure_coeff_env == "auto"
    terminal_pressure_coeff = terminal_pressure_coeff_auto ? NaN :
                              parse(Float64, terminal_pressure_coeff_env)
    (terminal_pressure_coeff_auto || terminal_pressure_coeff >= 0) ||
        error("MHD_TERMINAL_PRESSURE_COEFF must be auto or non-negative")
    terminal_rho_rel_limit = parse(Float64, get(ENV, "MHD_TERMINAL_RHO_REL_LIMIT", "Inf"))
    terminal_rho_rel_limit > 0 || error("MHD_TERMINAL_RHO_REL_LIMIT must be positive")
    terminal_b_rel_limit = parse(Float64, get(ENV, "MHD_TERMINAL_B_REL_LIMIT", "Inf"))
    terminal_b_rel_limit > 0 || error("MHD_TERMINAL_B_REL_LIMIT must be positive")
    terminal_induction = Symbol(lowercase(get(ENV, "MHD_TERMINAL_INDUCTION", "centered")))
    terminal_induction in (:centered, :ssprk2, :ct) ||
        error("MHD_TERMINAL_INDUCTION must be centered, ssprk2, or ct")
    terminal_induction_cfl = parse(Float64, get(ENV, "MHD_TERMINAL_INDUCTION_CFL", "0.4"))
    0 < terminal_induction_cfl <= 1 || error("MHD_TERMINAL_INDUCTION_CFL must be in (0,1]")
    terminal_induction_dissipation = parse(Float64, get(ENV, "MHD_TERMINAL_INDUCTION_DISSIPATION", "0.05"))
    0 <= terminal_induction_dissipation <= 1 ||
        error("MHD_TERMINAL_INDUCTION_DISSIPATION must be in [0,1]")
    terminal_induction_dissipation_order = parse(Int, get(ENV, "MHD_TERMINAL_INDUCTION_DISSIPATION_ORDER", "4"))
    terminal_induction_dissipation_order in (2, 4) ||
        error("MHD_TERMINAL_INDUCTION_DISSIPATION_ORDER must be 2 or 4")
    terminal_project_b = get(ENV, "MHD_TERMINAL_PROJECT_B", terminal_induction === :ssprk2 ? "1" : "0") in
                         ("1", "true", "TRUE", "yes", "on")
    terminal_induction_max_subcycles = parse(Int, get(ENV, "MHD_TERMINAL_INDUCTION_MAX_SUBCYCLES", "256"))
    terminal_induction_max_subcycles >= 1 ||
        error("MHD_TERMINAL_INDUCTION_MAX_SUBCYCLES must be positive")
    hybrid_min_va_cs = parse(Float64, get(ENV, "MHD_HYBRID_MIN_VA_CS", "0.0"))
    hybrid_min_va_cs >= 0 || error("MHD_HYBRID_MIN_VA_CS must be non-negative")
    hybrid_exit_drag_omega = parse(Float64,
        get(ENV, "MHD_HYBRID_EXIT_DRAG_OVER_OMEGA",
            get(ENV, "MHD_HYBRID_MIN_DRAG_OVER_OMEGA", "1.0")))
    hybrid_enter_drag_omega = parse(Float64,
        get(ENV, "MHD_HYBRID_ENTER_DRAG_OVER_OMEGA", string(2 * hybrid_exit_drag_omega)))
    0 < hybrid_exit_drag_omega < hybrid_enter_drag_omega ||
        error("hybrid drag hysteresis requires 0 < exit < enter")
    hybrid_check_every = parse(Int, get(ENV, "MHD_HYBRID_CHECK_EVERY", "1"))
    hybrid_check_every >= 1 || error("MHD_HYBRID_CHECK_EVERY must be positive")
    hybrid_confirm_checks = parse(Int, get(ENV, "MHD_HYBRID_CONFIRM_CHECKS", "2"))
    hybrid_confirm_checks >= 1 || error("MHD_HYBRID_CONFIRM_CHECKS must be positive")
    hybrid_handoff_steps = parse(Int, get(ENV, "MHD_HYBRID_HANDOFF_STEPS", "0"))
    hybrid_handoff_steps >= 0 || error("MHD_HYBRID_HANDOFF_STEPS must be non-negative")
    hybrid_handoff_alpha = parse(Float64, get(ENV, "MHD_HYBRID_HANDOFF_ALPHA", "0.0"))
    0 <= hybrid_handoff_alpha <= 1 || error("MHD_HYBRID_HANDOFF_ALPHA must be in [0,1]")
    hybrid_handoff_error_tol = parse(Float64, get(ENV, "MHD_HYBRID_HANDOFF_ERROR_TOL", "Inf"))
    hybrid_handoff_error_tol > 0 || error("MHD_HYBRID_HANDOFF_ERROR_TOL must be positive")
    hybrid_handoff_displacement_tol = parse(Float64,
        get(ENV, "MHD_HYBRID_HANDOFF_DISPLACEMENT_TOL", "0.02"))
    hybrid_handoff_displacement_tol > 0 ||
        error("MHD_HYBRID_HANDOFF_DISPLACEMENT_TOL must be positive")
    hybrid_handoff_diag_every = parse(Int, get(ENV, "MHD_HYBRID_HANDOFF_DIAG_EVERY", "8"))
    hybrid_handoff_diag_every >= 1 || error("MHD_HYBRID_HANDOFF_DIAG_EVERY must be positive")
    hybrid_reenter_terminal = get(ENV, "MHD_HYBRID_REENTER_TERMINAL", "0") in
                              ("1", "true", "TRUE", "yes", "on")
    hybrid_aware_dt_factor = parse(Float64, get(ENV, "MHD_HYBRID_AWARE_DT_FACTOR", "1.0"))
    hybrid_aware_dt_factor > 0 || error("MHD_HYBRID_AWARE_DT_FACTOR must be positive")
    brms = parse(Float64, get(ENV, "MHD_PMF_BRMS", "1e-3"))
    pmf_b0_ng_env = get(ENV, "MHD_PMF_B0_NG", "")
    initial_state_dir = strip(get(ENV, "MHD_INITIAL_STATE_DIR", ""))
    pmf_normalization_z_env = strip(get(ENV, "MHD_PMF_NORMALIZATION_Z", ""))
    pmf_normalization_z = isempty(pmf_normalization_z_env) ? zstart :
                          parse(Float64, pmf_normalization_z_env)
    if zrun && isempty(initial_state_dir) &&
       !isapprox(pmf_normalization_z, zstart; rtol=0,
                 atol=8eps(Float64) * max(1, zstart))
        error("MHD_PMF_NORMALIZATION_Z must equal MHD_ZSTART unless MHD_INITIAL_STATE_DIR supplies a pre-evolved state")
    end
    pmf_init = Symbol(lowercase(get(ENV, "MHD_PMF_INIT", "batchelor")))
    pmf_kmode = parse(Int, get(ENV, "MHD_PMF_KMODE", "1"))
    meanfield_theta = parse(Float64, get(ENV, "MHD_MEANFIELD_THETA",
                                                string(π / 4)))
    0 <= meanfield_theta <= π / 2 ||
        error("MHD_MEANFIELD_THETA must be in [0,pi/2]")
    meanfield_polarization = Symbol(lowercase(
        get(ENV, "MHD_MEANFIELD_POLARIZATION", "magnetosonic")))
    meanfield_polarization_code =
        meanfield_polarization in (:magnetosonic, :compressive, :inplane) ? 1 :
        meanfield_polarization in (:alfven, :alfvenic, :outofplane) ? 2 :
        error("MHD_MEANFIELD_POLARIZATION must be magnetosonic or alfven")
    meanfield_pert_frac = parse(Float64,
        get(ENV, "MHD_MEANFIELD_PERT_FRAC", "1.0e-4"))
    meanfield_pert_frac > 0 || error("MHD_MEANFIELD_PERT_FRAC must be positive")
    pmf_single_mode_normalization =
        Symbol(lowercase(get(ENV, "MHD_PMF_SINGLE_MODE_NORMALIZATION", "amplitude")))
    pmf_single_mode_normalization in (:amplitude, :rms) ||
        error("MHD_PMF_SINGLE_MODE_NORMALIZATION must be amplitude or rms")
    boxsize = parse(Float64, get(ENV, "MHD_BOXSIZE", "1.0"))
    boxsize > 0 || error("MHD_BOXSIZE must be positive")
    pmf_tile_box_env = strip(get(ENV, "MHD_PMF_TILE_BOXSIZE", ""))
    pmf_tile_box = isempty(pmf_tile_box_env) ? nothing : parse(Float64, pmf_tile_box_env)
    pmf_tile_box === nothing || (pmf_tile_box > 0 && pmf_tile_box <= boxsize) ||
        error("MHD_PMF_TILE_BOXSIZE must be positive and no larger than MHD_BOXSIZE")
    pmf_tile_n_env = strip(get(ENV, "MHD_PMF_TILE_N", ""))
    pmf_tile_n = isempty(pmf_tile_n_env) ? nothing : parse(Int, pmf_tile_n_env)
    pmf_tile_n === nothing || pmf_tile_n > 0 || error("MHD_PMF_TILE_N must be positive")
    kcut = parse(Float64, get(ENV, "MHD_PMF_KCUT", string(0.625π*N / boxsize)))
    kmax_env = strip(get(ENV, "MHD_PMF_KMAX", ""))
    pmf_kmax = isempty(kmax_env) ? nothing : parse(Float64, kmax_env)
    pmf_kmax === nothing || pmf_kmax > 0 || error("MHD_PMF_KMAX must be positive")
    pmf_kmin = parse(Float64, get(ENV, "MHD_PMF_KMIN", "0"))
    pmf_kmin >= 0 || error("MHD_PMF_KMIN must be non-negative")
    pmf_kmax === nothing || pmf_kmin < pmf_kmax ||
        error("MHD_PMF_KMIN must be smaller than MHD_PMF_KMAX")
    pmf_required_cells_env = strip(get(ENV, "MHD_PMF_MIN_CELLS_PER_WAVELENGTH", ""))
    pmf_required_cells = isempty(pmf_required_cells_env) ? nothing :
                         parse(Float64, pmf_required_cells_env)
    pmf_required_cells === nothing || pmf_required_cells > 0 ||
        error("MHD_PMF_MIN_CELLS_PER_WAVELENGTH must be positive")
    pmf_required_cells === nothing || pmf_kmax !== nothing ||
        error("MHD_PMF_KMAX is required when enforcing PMF wavelength resolution")
    preserve_low_kmax_env = strip(get(ENV, "MHD_PMF_PRESERVE_LOW_KMAX", ""))
    pmf_preserve_low_kmax = isempty(preserve_low_kmax_env) ? nothing : parse(Float64, preserve_low_kmax_env)
    pmf_preserve_low_kmax === nothing || pmf_preserve_low_kmax > 0 ||
        error("MHD_PMF_PRESERVE_LOW_KMAX must be positive")
    preserve_low_kcut_env = strip(get(ENV, "MHD_PMF_PRESERVE_LOW_KCUT", ""))
    pmf_preserve_low_kcut = isempty(preserve_low_kcut_env) ? nothing : parse(Float64, preserve_low_kcut_env)
    pmf_preserve_low_kcut === nothing || pmf_preserve_low_kcut > 0 ||
        error("MHD_PMF_PRESERVE_LOW_KCUT must be positive")
    pmf_normalization_kind = Symbol(lowercase(get(ENV, "MHD_PMF_NORMALIZATION_KIND", "auto")))
    pmf_normalization_kind = pmf_normalization_kind in (:total, :total_band) ?
                             :total_band_rms : pmf_normalization_kind
    pmf_normalization_kind = pmf_normalization_kind in (:reference, :reference_band, :low_band) ?
                             :reference_band_rms : pmf_normalization_kind
    pmf_normalization_kind in (:auto, :total_band_rms, :reference_band_rms) ||
        error("MHD_PMF_NORMALIZATION_KIND must be auto, total_band_rms, or reference_band_rms")
    pmf_normalization_kind === :reference_band_rms && pmf_preserve_low_kmax === nothing &&
        error("reference_band_rms requires MHD_PMF_PRESERVE_LOW_KMAX")
    pmf_normalization_kind_resolved = pmf_normalization_kind === :auto ?
        (pmf_preserve_low_kmax === nothing ? :total_band_rms : :reference_band_rms) :
        pmf_normalization_kind
    pmf_fixed_mode_power = get(ENV, "MHD_PMF_FIXED_MODE_POWER", "0") in
                           ("1", "true", "TRUE", "yes", "on")
    pmf_mode_realization = Symbol(lowercase(get(ENV, "MHD_PMF_MODE_REALIZATION",
                                                   "legacy")))
    pmf_mode_realization in (:legacy, :physical_hash, :physical_hash_zero_phase) ||
        error("MHD_PMF_MODE_REALIZATION must be legacy, physical_hash, or " *
              "physical_hash_zero_phase")
    realization_box_env = strip(get(ENV, "MHD_PMF_REALIZATION_BOX", ""))
    pmf_realization_box = isempty(realization_box_env) ? boxsize :
                          parse(Float64, realization_box_env)
    pmf_realization_box > 0 || error("MHD_PMF_REALIZATION_BOX must be positive")
    pmf_curl_symbol = Symbol(lowercase(get(ENV, "MHD_PMF_CURL_SYMBOL", "centered")))
    pmf_curl_symbol in (:centered, :continuum, :spectral) ||
        error("MHD_PMF_CURL_SYMBOL must be centered or continuum")
    pmf_curl_symbol === :spectral && (pmf_curl_symbol = :continuum)
    curl_reference_env = strip(get(ENV, "MHD_PMF_CURL_REFERENCE_N", ""))
    pmf_curl_reference_n = isempty(curl_reference_env) ? nothing :
                           parse(Int, curl_reference_env)
    if pmf_curl_reference_n !== nothing
        pmf_curl_symbol === :centered ||
            error("MHD_PMF_CURL_REFERENCE_N requires centered curl")
        0 < pmf_curl_reference_n <= N ||
            error("MHD_PMF_CURL_REFERENCE_N must be positive and no larger than MHD_PMF_N")
    end
    mode_lock_env = strip(get(ENV, "MHD_PMF_MODE_LOCK_N", ""))
    pmf_mode_lock_n = isempty(mode_lock_env) ? nothing : parse(Int, mode_lock_env)
    high_mode_lock_env = strip(get(ENV, "MHD_PMF_HIGH_MODE_LOCK_N", ""))
    pmf_high_mode_lock_n = isempty(high_mode_lock_env) ? nothing :
                           parse(Int, high_mode_lock_env)
    pmf_shortest_k = pmf_kmax === nothing ? kcut : pmf_kmax
    pmf_min_cells_per_wavelength = 2π * N / (pmf_shortest_k * boxsize)
    if pmf_mode_lock_n !== nothing
        pmf_mode_lock_n <= N || error("MHD_PMF_MODE_LOCK_N cannot exceed MHD_PMF_N")
        lock_limit = π * pmf_mode_lock_n / boxsize
        if pmf_preserve_low_kmax === nothing
            pmf_shortest_k <= lock_limit ||
                error("fixed-band PMF exceeds the mode-lock Nyquist frequency")
        else
            pmf_preserve_low_kmax <= lock_limit ||
                error("preserved low-k PMF band exceeds the mode-lock Nyquist frequency")
        end
    end
    if pmf_high_mode_lock_n !== nothing
        pmf_high_mode_lock_n <= N ||
            error("MHD_PMF_HIGH_MODE_LOCK_N cannot exceed MHD_PMF_N")
        pmf_shortest_k <= π * pmf_high_mode_lock_n / boxsize ||
            error("PMF band exceeds the high-mode-lock Nyquist frequency")
    end
    seed = parse(Int, get(ENV, "MHD_PMF_SEED", "42"))
    recon = Symbol(lowercase(get(ENV, "MHD_RECON", "ppm")))
    debug_recon_override = strip(get(ENV, "MHD_DEBUG_RECON_OVERRIDE", ""))
    isempty(debug_recon_override) ||
        (recon = Symbol(lowercase(debug_recon_override)))
    riemann = Symbol(lowercase(get(ENV, "MHD_RIEMANN", "hlld")))
    debug_riemann_override = strip(get(ENV, "MHD_DEBUG_RIEMANN_OVERRIDE", ""))
    isempty(debug_riemann_override) ||
        (riemann = Symbol(lowercase(debug_riemann_override)))
    glm_ch_fac = parse(Float64, get(ENV, "MHD_GLM_CH_FAC", "2.0"))
    glm_cr = parse(Float64, get(ENV, "MHD_GLM_CR", "1.0"))
    glm_project_every = parse(Int, get(ENV, "MHD_GLM_PROJECT_EVERY",
        drag_dt_mode === :radiation ? "1" : "0"))
    glm_project_every >= 0 || error("MHD_GLM_PROJECT_EVERY must be non-negative")
    fb = T(parse(Float64, get(ENV, "MHD_FB", "0.17037")))
    fdm = one(T) - fb
    gravity = parse(Bool, get(ENV, "MHD_GRAVITY", "true"))
    gravity_gas_kick = get(ENV, "MHD_GRAVITY_GAS_KICK", "1") in
                           ("1", "true", "TRUE", "yes", "on")
    gravity_subcycles_fixed = parse(Int, get(ENV, "MHD_GRAVITY_SUBCYCLES", "1"))
    gravity_subcycles_fixed >= 1 || error("MHD_GRAVITY_SUBCYCLES must be >= 1")
    gravity_dt_boost = parse(Float64, get(ENV, "MHD_GRAVITY_DT_BOOST", "0.0"))
    (gravity_dt_boost == 0 || gravity_dt_boost >= 1) ||
        error("MHD_GRAVITY_DT_BOOST must be 0 or >= 1")
    gravity_source = Symbol(lowercase(get(ENV, "MHD_GRAVITY_SOURCE", "total")))
    gravity_source in (:total, :gas, :dm) ||
        error("MHD_GRAVITY_SOURCE must be total, gas, or dm")
    grav_gas_weight = gravity_source === :dm ? zero(T) : fb
    grav_dm_weight = gravity_source === :gas ? zero(T) : fdm
    particle_max_disp_cells = parse(Float64, get(ENV, "MHD_PARTICLE_MAX_DISP_CELLS", "Inf"))
    particle_max_disp_cells > 0 || error("MHD_PARTICLE_MAX_DISP_CELLS must be positive")
    particle_wrap = get(ENV, "MHD_PARTICLE_WRAP", "1") in ("1", "true", "TRUE", "yes", "on")
    lattice_displacements = get(ENV, "MHD_PARTICLE_LATTICE_DISPLACEMENTS", "1") in
                                ("1", "true", "TRUE", "yes", "on")
    gravity_phi_interp = get(ENV, "MHD_GRAVITY_PHI_INTERP", "0") in ("1", "true", "TRUE", "yes", "on")
    gravity_midpoint_source = get(ENV, "MHD_GRAVITY_MIDPOINT_SOURCE", "1") in
                                 ("1", "true", "TRUE", "yes", "on")
    gravity_phi_interp && gravity_midpoint_source &&
        error("choose either MHD_GRAVITY_PHI_INTERP=1 or MHD_GRAVITY_MIDPOINT_SOURCE=1")
    gravity_phi_interp && lattice_displacements &&
        error("potential interpolation is not implemented for lattice-displacement particles")
    fft_mode = _fft_mode(BACKEND_NAME, get(ENV, "MHD_FFT", "auto"))
    if BACKEND_NAME === :metal && fft_mode !== :mps &&
       !(get(ENV, "MHD_ALLOW_NON_MPS_FFT", "0") in ("1", "true", "TRUE", "yes", "on"))
        error("Metal PMF runs use the MPS rfft Poisson path by default; set MHD_ALLOW_NON_MPS_FFT=1 to benchmark MHD_FFT=$(fft_mode)")
    end
    gravity_1buf = get(ENV, "MHD_GRAVITY_1BUF",
                       (BACKEND_NAME === :metal && fft_mode === :mps) ? "1" : "0") in
                   ("1", "true", "TRUE", "yes", "on")
    # Cosmological PMF production uses the fused full mixing network.
    # Non-cosmological hydro/MHD benchmarks retain chemistry-off.
    chem_mode = _chem_mode(get(ENV, "MHD_CHEM", zrun ? "mixing" : "off"))
    chem_helium = get(ENV, "MHD_CHEM_HELIUM",
                       chem_mode === :mixing ? "1" : "0") in
                   ("1", "true", "TRUE", "yes", "on")
    chem_helium && chem_mode !== :mixing && error(
        "MHD_CHEM_HELIUM=1 currently requires the full MHD_CHEM=mixing network")
    chem_advect_species = chem_mode !== :off &&
        get(ENV, "MHD_CHEM_ADVECT_SPECIES", "1") in
            ("1", "true", "TRUE", "yes", "on")
    chem_coupling_default = zrun && chem_mode !== :off ? "strang" : "lie"
    chem_coupling = _chem_coupling_mode(get(ENV, "MHD_CHEM_COUPLING",
                                                chem_coupling_default))
    chem_expansion = _chem_expansion_enabled(get(ENV, "MHD_CHEM_EXPANSION", "auto"),
                                             zrun, chem_mode)
    fh = T(parse(Float64, get(ENV, "MHD_CHEM_FH", "0.76")))
    xhii_default = zrun && zstart > 1500 ? "1.0" : "2e-4"
    xhii0 = T(parse(Float64, get(ENV, "MHD_XHII_INIT", xhii_default)))
    xh20 = T(parse(Float64, get(ENV, "MHD_XH2_INIT", "1e-12")))
    chem_len_unit = parse(Float64, get(ENV, "MHD_CHEM_LENGTH_UNITS", "1.0"))
    chem_time_unit = parse(Float64, get(ENV, "MHD_CHEM_TIME_UNITS", "1.0"))
    chem_vunit_default = zrun ? "8e5" : string(chem_len_unit / chem_time_unit)
    chem_vunit = parse(Float64, get(ENV, "MHD_CHEM_VUNIT", chem_vunit_default))
    chem_time_unit = chem_len_unit / chem_vunit
    if p0_auto
        zrun || error("MHD_P0=auto requires MHD_ZSTART")
        p0 = _cosmological_ionized_p0(zstart, fh, chem_vunit, hub, Ob)
    end
    isfinite(p0) && p0 > 0 || error("MHD_P0 must be positive and finite")
    terminal_accuracy_pressure_auto && (terminal_accuracy_pressure = p0)
    terminal_accuracy_pressure > 0 ||
        error("MHD_TERMINAL_ACCURACY_PRESSURE must be positive")
    pmf_b0_ng = isempty(strip(pmf_b0_ng_env)) ? NaN : parse(Float64, pmf_b0_ng_env)
    pmf_comoving_clock = get(ENV, "MHD_PMF_COMOVING_CLOCK", "1") in
                         ("1", "true", "TRUE", "yes", "on")
    pmf_b_zref = zrun ? pmf_normalization_z :
                 parse(Float64, get(ENV, "MHD_PMF_B0_ZREF", "3000"))
    pmf_b_aref = isfinite(pmf_b0_ng) && pmf_comoving_clock ? z_to_a(pmf_b_zref) : NaN
    cosmological_mhd_scaling = Symbol(lowercase(strip(
        get(ENV, "MHD_COSMOLOGICAL_MHD_SCALING", "legacy"))))
    cosmological_mhd_scaling in (:legacy, :jedamzik) ||
        error("MHD_COSMOLOGICAL_MHD_SCALING must be legacy or jedamzik")
    cosmological_mhd_scaling === :jedamzik && !zrun &&
        error("MHD_COSMOLOGICAL_MHD_SCALING=jedamzik requires a redshift run")
    cosmological_mhd_scaling === :jedamzik && !isfinite(pmf_b_aref) &&
        error("MHD_COSMOLOGICAL_MHD_SCALING=jedamzik requires MHD_PMF_B0_NG and the comoving clock")
    cosmological_mhd_aref = cosmological_mhd_scaling === :jedamzik ? pmf_b_aref : NaN
    if isfinite(pmf_b0_ng)
        brms = _pmf_code_brms_from_comoving_ng(pmf_b0_ng, pmf_b_zref, hub, Ob, chem_vunit)
    end
    meanfield_bguide = parse(Float64,
        get(ENV, "MHD_MEANFIELD_BGUIDE", string(brms)))
    meanfield_bpert = parse(Float64, get(ENV, "MHD_MEANFIELD_BPERT",
        string(max(abs(meanfield_bguide), eps(Float64)) * meanfield_pert_frac)))
    meanfield_bguide > 0 || error("MHD_MEANFIELD_BGUIDE must be positive")
    meanfield_bpert > 0 || error("MHD_MEANFIELD_BPERT must be positive")
    alfven_bguide = parse(Float64, get(ENV, "MHD_ALFVEN_BGUIDE", string(brms)))
    alfven_bpert_frac = parse(Float64, get(ENV, "MHD_ALFVEN_BPERT_FRAC", "1.0e-3"))
    alfven_bpert = parse(Float64, get(ENV, "MHD_ALFVEN_BPERT",
                                      string(max(abs(alfven_bguide), eps(Float64)) * alfven_bpert_frac)))
    chem_density_fixed = get(ENV, "MHD_CHEM_DENSITY_UNITS", "")
    mixing_chem = chem_mode === :mixing || chem_mode === :analytic_mixing
    chem_dtfrac = parse(Float64, get(ENV, "MHD_CHEM_DTFRAC",
                                     zrun && mixing_chem ? "1.0" : "0.1"))
    # Boosted terminal steps can span the entire stiff recombination interval.
    # The loop exits early after consuming dt, so a high ceiling is cheap once
    # converged; lower ceilings silently left ionized cells partially evolved.
    chem_itcap_default = chem_mode === :mixing ? "1024" :
                         chem_mode === :analytic_mixing ? "64" :
                         chem_mode === :peebles ? "64" : "64"
    chem_itcap = parse(Int, get(ENV, "MHD_CHEM_ITCAP", chem_itcap_default))
    chem_falpha = parse(Float64, get(ENV, "MHD_CHEM_FALPHA",
                                     zrun && mixing_chem ? "1.0" : "0.0"))
    chem_falpha_zs = _parse_optional_list("MHD_CHEM_FALPHA_ZS")
    chem_falpha_vals = _parse_optional_list("MHD_CHEM_FALPHA_VALS")
    chem_smooth = _chem_smooth_mode(get(ENV, "MHD_CHEM_SMOOTH",
                                            zrun && mixing_chem ? "gaussian" : "local"))
    chem_allow_local_mixing = get(ENV, "MHD_CHEM_ALLOW_LOCAL_MIXING", "0") in
                              ("1", "true", "TRUE", "yes", "on")
    chem_nsmean = parse(Float64, get(ENV, "MHD_CHEM_NSMEAN", "1.0"))
    chem_smooth_sigma = parse(Float64, get(ENV, "MHD_CHEM_SMOOTH_SIGMA_CELLS", "16.0"))
    chem_nonlocal_table_path = strip(get(ENV, "MHD_CHEM_NONLOCAL_TABLE", ""))
    chem_nonlocal_scale = parse(Float64, get(ENV, "MHD_CHEM_NONLOCAL_SCALE", "1.0"))
    chem_nonlocal_scale >= 0 || error("MHD_CHEM_NONLOCAL_SCALE must be non-negative")
    !isempty(chem_nonlocal_table_path) && chem_nonlocal_scale == 0 && error(
        "remove MHD_CHEM_NONLOCAL_TABLE rather than configuring a zero response scale")
    chem_nonlocal_enabled = !isempty(chem_nonlocal_table_path) && chem_nonlocal_scale > 0
    chem_nonlocal_enabled && !zrun && error(
        "MHD_CHEM_NONLOCAL_TABLE is supported only for cosmological redshift runs")
    chem_nonlocal_enabled && chem_mode === :off && error(
        "MHD_CHEM_NONLOCAL_TABLE requires MHD_CHEM to be enabled")
    chem_nonlocal_enabled && chem_falpha != 0 && error(
        "nonlocal response tables require MHD_CHEM_FALPHA=0 to avoid double-counting mixing")
    chem_nonlocal_enabled && chem_smooth !== :local && error(
        "nonlocal response tables require MHD_CHEM_SMOOTH=local")
    chem_xe_mean = parse(Float64, get(ENV, "MHD_CHEM_XE_MEAN", "NaN"))
    chem_ionized_fast_zmin = parse(Float64, get(ENV, "MHD_CHEM_IONIZED_FAST_ZMIN",
        zrun && chem_helium ? "3000" : "0"))
    chem_helium && 0 < chem_ionized_fast_zmin < 3000 && error(
        "helium freeze-out must be evolved explicitly: set " *
        "MHD_CHEM_IONIZED_FAST_ZMIN=0 or >=3000")
    chem_recfast_hswitch = get(ENV, "MHD_CHEM_RECFAST_HSWITCH", "1") == "1"
    chem_rate_tables_enabled = get(ENV, "MHD_CHEM_RATE_TABLES", "1") in
                               ("1", "true", "TRUE", "yes", "on")
    chem_profile_enabled = get(ENV, "MHD_CHEM_PROFILE", "0") in
                           ("1", "true", "TRUE", "yes", "on")
    chem_fused_analytic = get(ENV, "MHD_CHEM_FUSED_ANALYTIC",
                              chem_mode === :analytic_mixing ? "1" : "0") in
                          ("1", "true", "TRUE", "yes", "on")
    chem_fused_mixing = get(ENV, "MHD_CHEM_FUSED_MIXING",
                            chem_mode === :mixing ? "1" : "0") in
                        ("1", "true", "TRUE", "yes", "on")
    chem_helium && !chem_fused_mixing && error(
        "MHD_CHEM_HELIUM=1 requires MHD_CHEM_FUSED_MIXING=1")
    chem_mean_every = parse(Int, get(ENV, "MHD_CHEM_MEAN_EVERY", "1"))
    chem_mean_every >= 1 || error("MHD_CHEM_MEAN_EVERY must be positive")
    chem_fudge = parse(Float64, get(ENV, "MHD_CHEM_RECFAST_FUDGE",
                                    chem_recfast_hswitch ? "1.125" : "1.0"))
    final_diag = get(ENV, "MHD_FINAL_DIAG", "1") != "0"
    history_path = get(ENV, "MHD_HISTORY_TSV", "")
    initial_history = get(ENV, "MHD_INITIAL_HISTORY", "1") in
                      ("1", "true", "TRUE", "yes", "on")
    initial_pk = get(ENV, "MHD_INITIAL_PK", "0") in
                 ("1", "true", "TRUE", "yes", "on")
    pk_path = get(ENV, "MHD_PK_TSV", "")
    mode_history_path = get(ENV, "MHD_MODE_HISTORY_TSV", "")
    if pmf_init in (:meanfield, :meanfield_linear, :sv25_mode) ||
       !isempty(mode_history_path)
        0 < pmf_kmode < N ÷ 2 ||
            error("matched linear modes require 0 < MHD_PMF_KMODE < N/2")
    end
    pdf_path = get(ENV, "MHD_PDF_TSV", "")
    adaptive_diag_rel = parse(Float64, get(ENV, "MHD_ADAPTIVE_DIAG_REL", "0"))
    adaptive_diag_rel >= 0 || error("MHD_ADAPTIVE_DIAG_REL must be non-negative")
    adaptive_diag_zmin = parse(Float64, get(ENV, "MHD_ADAPTIVE_DIAG_ZMIN", "-Inf"))
    adaptive_diag_zmax = parse(Float64, get(ENV, "MHD_ADAPTIVE_DIAG_ZMAX", "Inf"))
    adaptive_diag_zmax >= adaptive_diag_zmin ||
        error("MHD_ADAPTIVE_DIAG_ZMAX must be >= MHD_ADAPTIVE_DIAG_ZMIN")
    pdf_nbins = parse(Int, get(ENV, "MHD_PDF_NBINS", "128"))
    pdf_nbins > 0 || error("MHD_PDF_NBINS must be positive")
    pdf_loglo = parse(Float64, get(ENV, "MHD_PDF_LOGRHO_MIN", "-4"))
    pdf_loghi = parse(Float64, get(ENV, "MHD_PDF_LOGRHO_MAX", "4"))
    pdf_loghi > pdf_loglo || error("MHD_PDF_LOGRHO_MAX must exceed MHD_PDF_LOGRHO_MIN")
    pk_every = parse(Int, get(ENV, "MHD_PK_EVERY", "0"))
    pk_zs = _parse_z_outputs(get(ENV, "MHD_PK_ZS", ""))
    pk_next = Ref(1)
    diag_clear_fft = get(ENV, "MHD_DIAG_CLEAR_FFT", "0") in
                     ("1", "true", "TRUE", "yes", "on")
    frame_dir = get(ENV, "MHD_DENSITY_FRAME_DIR", "")
    frame_zs = _parse_z_outputs(get(ENV, "MHD_DENSITY_FRAME_ZS", ""))
    !isempty(frame_zs) && isempty(frame_dir) &&
        error("MHD_DENSITY_FRAME_ZS requires MHD_DENSITY_FRAME_DIR")
    frame_next = Ref(1)
    pk_nmu = parse(Int, get(ENV, "MHD_PK_NMU", "4"))
    pk_nbins = parse(Int, get(ENV, "MHD_PK_NBINS", string(max(1, N ÷ 2))))
    pk_axis = parse(Int, get(ENV, "MHD_PK_AXIS", "1"))
    pk_kmax_env = strip(get(ENV, "MHD_PK_KMAX", ""))
    pk_kmax = isempty(pk_kmax_env) ? nothing : parse(Float64, pk_kmax_env)
    pk_kmax === nothing || pk_kmax > 0 || error("MHD_PK_KMAX must be positive")
    debug_div_every = parse(Int, get(ENV, "MHD_DEBUG_DIV_EVERY", "0"))
    debug_div_every >= 0 || error("MHD_DEBUG_DIV_EVERY must be non-negative")
    debug_phase_finite = get(ENV, "MHD_DEBUG_PHASE_FINITE", "0") in
                         ("1", "true", "TRUE", "yes", "on")
    debug_phase_cycle_min = parse(Int, get(ENV, "MHD_DEBUG_PHASE_CYCLE_MIN", "0"))
    debug_phase_cycle_min >= 0 || error("MHD_DEBUG_PHASE_CYCLE_MIN must be non-negative")
    debug_phase_zmax = parse(Float64, get(ENV, "MHD_DEBUG_PHASE_ZMAX", "Inf"))
    debug_phase_rho_min = parse(Float64, get(ENV, "MHD_DEBUG_PHASE_RHO_MIN", "0"))
    debug_phase_rho_min >= 0 || error("MHD_DEBUG_PHASE_RHO_MIN must be non-negative")
    debug_radiation_rho_min = parse(Float64,
        get(ENV, "MHD_DEBUG_RADIATION_RHO_MIN", "0"))
    debug_radiation_cycle_min = parse(Int,
        get(ENV, "MHD_DEBUG_RADIATION_CYCLE_MIN", "0"))
    debug_radiation_correction_cfl = parse(Float64,
        get(ENV, "MHD_DEBUG_RADIATION_CORRECTION_CFL", "Inf"))
    debug_radiation_rho_min >= 0 ||
        error("MHD_DEBUG_RADIATION_RHO_MIN must be non-negative")
    debug_radiation_cycle_min >= 0 ||
        error("MHD_DEBUG_RADIATION_CYCLE_MIN must be non-negative")
    checkpoint_path = strip(get(ENV, "MHD_CHECKPOINT_PATH", ""))
    checkpoint_read_path = strip(get(ENV, "MHD_CHECKPOINT_READ_PATH", checkpoint_path))
    checkpoint_write_path = strip(get(ENV, "MHD_CHECKPOINT_WRITE_PATH", checkpoint_path))
    checkpoint_every = parse(Int, get(ENV, "MHD_CHECKPOINT_EVERY", "0"))
    checkpoint_at_cycle = parse(Int, get(ENV, "MHD_CHECKPOINT_AT_CYCLE", "0"))
    checkpoint_every >= 0 || error("MHD_CHECKPOINT_EVERY must be non-negative")
    checkpoint_at_cycle >= 0 || error("MHD_CHECKPOINT_AT_CYCLE must be non-negative")
    checkpoint_resume = get(ENV, "MHD_CHECKPOINT_RESUME", "1") in
                        ("1", "true", "TRUE", "yes", "on")
    checkpoint_final = get(ENV, "MHD_CHECKPOINT_FINAL", "1") in
                       ("1", "true", "TRUE", "yes", "on")
    checkpoint_disable_writes = get(ENV, "MHD_CHECKPOINT_DISABLE_WRITES", "0") in
                                ("1", "true", "TRUE", "yes", "on")
    checkpoint_every > 0 && isempty(checkpoint_path) &&
        error("MHD_CHECKPOINT_EVERY requires MHD_CHECKPOINT_PATH")
    adaptive_last_delta_b = Ref(NaN)
    adaptive_last_xhii = Ref(NaN)
    if (chem_mode === :mixing || chem_mode === :analytic_mixing) &&
       chem_falpha <= 0 && !chem_nonlocal_enabled && !chem_allow_local_mixing
        chem_falpha = 1.0
    end
    if (chem_mode === :mixing || chem_mode === :analytic_mixing) &&
       chem_smooth === :local && !chem_nonlocal_enabled && !chem_allow_local_mixing
        error("MHD_CHEM=$(chem_mode) needs MHD_CHEM_SMOOTH=global or gaussian")
    end

    be_mhd = MHDKernels.backend(BACKEND_NAME)
    be_p = PoissonKernels.backend(BACKEND_NAME)
    pdf_counts = isempty(pdf_path) ? nothing : PoissonKernels.device_zeros(be_p, Int32, (pdf_nbins,))
    frame_hostrho = isempty(frame_dir) ? nothing : Array{Float32}(undef, N, N, N)
    frame_hostslice = isempty(frame_dir) ? nothing : Array{Float32}(undef, N, N)
    frame_hostproj = isempty(frame_dir) ? nothing : Array{Float32}(undef, N, N)
    default_mhd_integrator = BACKEND_NAME === :cpu ? "ref" : "cube"
    mhd_integrator = Symbol(lowercase(get(ENV, "MHD_INTEGRATOR",
                                              default_mhd_integrator)))
    mhd_integrator in (:ref, :cube) ||
        error("MHD_INTEGRATOR must be ref or cube; got $mhd_integrator")
    if BACKEND_NAME === :metal && mhd_integrator === :cube
        f16max = Float64(floatmax(Float16))
        p0 < f16max || error(
            "MHD_P0=$(p0) overflows the Metal cube's Float16 primitive staging; " *
            "choose physical code units such as MHD_CHEM_VUNIT=1e8")
        abs(brms) < f16max / 8 || error(
            "MHD Brms=$(brms) leaves insufficient Float16 staging headroom; " *
            "increase MHD_CHEM_VUNIT")
    end
    drag_dt_mode in (:godunov,:radiation) && mhd_integrator === :cube &&
        any(x -> !iszero(x), (drag_vx0, drag_vy0, drag_vz0)) &&
        error("Metal/CUDA source-aware Godunov drag is specialized for the CMB frame; set MHD_DRAG_VX0/VY0/VZ0=0")
    ncells = N^3
    _log_line(@sprintf("CONFIG backend=%s N=%d ncells=%.3fM zrun=%s cosmology=(h=%.6g Om=%.6g OL=%.6g Or=%.6g) chem=%s chem_coupling=%s chem_expansion=%s rate_tables=%s gravity=%s gas_kick=%s source=%s phi_interp=%s midpoint_source=%s grav1buf=%s fft=%s particle_wrap=%s lattice_displacements=%s",
                       String(BACKEND_NAME), N, ncells/1e6,
                       zrun ? @sprintf("%.1f->%.1f", zstart, zend) : "off",
                       hub, Om, OL, Or, String(chem_mode), String(chem_coupling),
                       string(chem_expansion),
                       string(chem_rate_tables_enabled),
                       string(gravity), string(gravity_gas_kick), String(gravity_source),
                       string(gravity_phi_interp), string(gravity_midpoint_source),
                       string(gravity_1buf), String(fft_mode),
                       string(particle_wrap), string(lattice_displacements)))
    _log_line(@sprintf("PMF_MODES boxsize=%.9g kcut=%.9g kmin=%.9g kmax=%s hard_cutoff=%s mode_lock_n=%s high_mode_lock_n=%s curl_symbol=%s curl_reference_n=%s fixed_mode_power=%s mode_realization=%s realization_box=%.9g tile_box=%s tile_n=%s preserve_low_kmax=%s preserve_low_kcut=%s minimum_cells_per_wavelength=%.6g required_minimum_cells=%s",
                       boxsize,
                       kcut, pmf_kmin,
                       pmf_kmax === nothing ? "none" : @sprintf("%.9g", pmf_kmax),
                       string(pmf_kmax !== nothing),
                       pmf_mode_lock_n === nothing ? "none" : string(pmf_mode_lock_n),
                       pmf_high_mode_lock_n === nothing ? "none" : string(pmf_high_mode_lock_n),
                       String(pmf_curl_symbol),
                       pmf_curl_reference_n === nothing ? "none" : string(pmf_curl_reference_n),
                       string(pmf_fixed_mode_power),
                       String(pmf_mode_realization), pmf_realization_box,
                       pmf_tile_box === nothing ? "none" : @sprintf("%.9g", pmf_tile_box),
                       pmf_tile_n === nothing ? "auto" : string(pmf_tile_n),
                       pmf_preserve_low_kmax === nothing ? "none" : @sprintf("%.9g", pmf_preserve_low_kmax),
                       pmf_preserve_low_kcut === nothing ? "none" : @sprintf("%.9g", pmf_preserve_low_kcut),
                       pmf_min_cells_per_wavelength,
                       pmf_required_cells === nothing ? "none" : @sprintf("%.6g", pmf_required_cells)))

    s = _stage("allocate_mhd_state") do
        MHDKernels.allocate_state(be_mhd, T, (N,N,N);
                                  dx=1/N, gamma=5/3, riemann=riemann, recon=recon,
                                  pfl=pfloor, llf_dmin=llf_dmin,
                                  llf_pmin=llf_pmin)
    end
    st = _stage("init_pmf_" * String(pmf_init)) do
        if pmf_init === :batchelor
            base = if pmf_tile_box === nothing
                MHDKernels.init_pmf_batchelor!(s; brms=brms, rho0=1, p0=p0,
                                               box=boxsize, seed=seed, kcut=kcut,
                                               kmin=pmf_kmin, kmax=pmf_kmax,
                                               mode_lock_n=pmf_mode_lock_n,
                                               high_mode_lock_n=pmf_high_mode_lock_n,
                                               preserve_low_kmax=pmf_preserve_low_kmax,
                                               preserve_low_kcut=pmf_preserve_low_kcut,
                                               normalization_kind=pmf_normalization_kind_resolved,
                                               curl_symbol=pmf_curl_symbol,
                                               curl_reference_n=pmf_curl_reference_n,
                                               fixed_mode_power=pmf_fixed_mode_power,
                                               mode_realization=pmf_mode_realization,
                                               realization_box=pmf_realization_box,
                                               minimum_cells_per_wavelength=pmf_required_cells)
            else
                MHDKernels.init_pmf_batchelor_tiled!(s; brms=brms, rho0=1, p0=p0,
                                                     box=boxsize, tile_box=pmf_tile_box,
                                                     tile_n=pmf_tile_n,
                                                     seed=seed, kcut=kcut,
                                                     kmin=pmf_kmin, kmax=pmf_kmax,
                                                     mode_lock_n=pmf_mode_lock_n,
                                                     high_mode_lock_n=pmf_high_mode_lock_n,
                                                     preserve_low_kmax=pmf_preserve_low_kmax,
                                                     preserve_low_kcut=pmf_preserve_low_kcut,
                                                     normalization_kind=pmf_normalization_kind_resolved,
                                                     curl_symbol=pmf_curl_symbol,
                                                     curl_reference_n=pmf_curl_reference_n,
                                                     fixed_mode_power=pmf_fixed_mode_power,
                                                     mode_realization=pmf_mode_realization,
                                                     realization_box=pmf_realization_box,
                                                     minimum_cells_per_wavelength=pmf_required_cells)
            end
            merge(base, (btrans0=Float32(NaN), alfven_bguide=Float32(NaN),
                         alfven_bpert=Float32(NaN)))
        elseif pmf_init in (:meanfield, :meanfield_linear, :sv25_mode)
            _init_meanfield_linear_k!(be_mhd)(s.U..., Int(N), T(s.γ), T(p0),
                T(meanfield_bguide), T(meanfield_bpert), T(meanfield_theta),
                meanfield_polarization_code, pmf_kmode; ndrange=ncells)
            KA.synchronize(be_mhd)
            measured_brms = _device_brms!(be_mhd, s, s.scratch[1])
            btrans0 = abs(meanfield_bpert) / sqrt(2)
            (brms_measured=Float32(measured_brms),
             total_brms_measured=Float32(measured_brms),
             reference_band_brms_measured=Float32(measured_brms),
             normalization_brms_measured=Float32(meanfield_bguide),
             normalization_kind="mean_field_amplitude", seed=seed, kcut=kcut,
             btrans0=Float32(btrans0), alfven_bguide=Float32(NaN),
             alfven_bpert=Float32(NaN))
        elseif pmf_init in (:alfven, :alfven_linear, :linear_alfven, :guide_alfven)
            _init_alfven_linear_k!(be_mhd)(s.U..., Int(N), T(s.γ), T(p0),
                                           T(alfven_bguide), T(alfven_bpert), pmf_kmode;
                                           ndrange=ncells)
            KA.synchronize(be_mhd)
            measured_brms = _device_brms!(be_mhd, s, s.scratch[1])
            btrans0 = _device_component_rms!(be_mhd, s.scratch[1], s.U[7])
            (brms_measured=Float32(measured_brms),
             total_brms_measured=Float32(measured_brms),
             reference_band_brms_measured=Float32(measured_brms),
             normalization_brms_measured=Float32(measured_brms),
             normalization_kind="total_band_rms", seed=seed, kcut=kcut,
             btrans0=Float32(btrans0), alfven_bguide=Float32(alfven_bguide),
             alfven_bpert=Float32(alfven_bpert))
        else
            mode_code = _single_b_mode(String(pmf_init))
            single_b_amplitude = mode_code == 2 && pmf_single_mode_normalization === :rms ?
                                 sqrt(2) * brms : brms
            _init_single_b_k!(be_mhd)(s.U..., Int(N), T(s.γ), T(p0),
                                      T(single_b_amplitude), mode_code, pmf_kmode;
                                      ndrange=ncells)
            KA.synchronize(be_mhd)
            measured_brms = _device_brms!(be_mhd, s, s.scratch[1])
            (brms_measured=Float32(measured_brms),
             total_brms_measured=Float32(measured_brms),
             reference_band_brms_measured=Float32(measured_brms),
             normalization_brms_measured=Float32(measured_brms),
             normalization_kind="total_band_rms", seed=seed, kcut=kcut,
             btrans0=Float32(NaN), alfven_bguide=Float32(NaN),
             alfven_bpert=Float32(NaN))
        end
    end
    import_rescale_to_pmf = get(ENV, "MHD_INITIAL_B_RESCALE_TO_PMF", "0") in
                            ("1", "true", "TRUE", "yes", "on")
    initial_b_import_dir = strip(get(ENV, "MHD_INITIAL_B_IMPORT_DIR", ""))
    isempty(initial_state_dir) || isempty(initial_b_import_dir) ||
        error("MHD_INITIAL_STATE_DIR and MHD_INITIAL_B_IMPORT_DIR are mutually exclusive")
    imported_brms = _stage("import_initial_b") do
        _import_initial_b_field!(be_mhd, initial_b_import_dir, s;
                                 N=N, p0=p0,
                                 target_brms=import_rescale_to_pmf ? brms : nothing)
    end
    imported_brms === nothing || (st = merge(st, (
        brms_measured=imported_brms, total_brms_measured=imported_brms,
        reference_band_brms_measured=Float32(NaN),
        normalization_brms_measured=imported_brms,
        normalization_kind="imported_total_rms", btrans0=Float32(NaN))))
    imported_state_brms = _stage("import_initial_mhd_state") do
        _import_initial_mhd_state!(be_mhd, initial_state_dir, s; N=N)
    end
    imported_state_brms === nothing || (st = merge(st, (
        brms_measured=imported_state_brms, total_brms_measured=imported_state_brms,
        reference_band_brms_measured=Float32(NaN),
        normalization_brms_measured=imported_state_brms,
        normalization_kind="imported_joint_state", btrans0=Float32(NaN))))
    if imported_state_brms !== nothing
        p0 = _stage("measure_imported_pressure") do
            _device_pressure_mean!(be_mhd, s, s.scratch[1])
        end
        terminal_accuracy_pressure_auto && (terminal_accuracy_pressure = p0)
        @printf("INITIAL_STATE_PRESSURE p_mean=%.8e source=imported_energy\n", p0)
        flush(stdout)
    end
    pmf_ng_per_code = isfinite(pmf_b0_ng) ? pmf_b0_ng / brms : NaN
    pmf_initial_reference_comoving_ng =
        Float64(st.reference_band_brms_measured) * pmf_ng_per_code
    pmf_initial_total_comoving_ng = Float64(st.total_brms_measured) * pmf_ng_per_code
    _stage("export_initial_b") do
        _write_initial_b_field!(get(ENV, "MHD_INITIAL_B_DIR", ""), s, st;
                                N=N, boxsize=boxsize, zstart=zstart, zend=zend,
                                seed=seed, kcut=kcut, kmax=pmf_kmax,
                                mode_lock_n=pmf_mode_lock_n,
                                high_mode_lock_n=pmf_high_mode_lock_n, brms=brms,
                                normalization_z=pmf_normalization_z,
                                target_comoving_ng=pmf_b0_ng,
                                reference_comoving_ng=pmf_initial_reference_comoving_ng,
                                total_comoving_ng=pmf_initial_total_comoving_ng)
    end

    px, py, pz, vx, vy, vz, ρdm, ρtot, φ, φprev = _stage("allocate_particle_gravity_arrays") do
        px = PoissonKernels.device_zeros(be_p, T, (ncells,))
        py = similar(px); pz = similar(px)
        vx = similar(px); vy = similar(px); vz = similar(px)
        ρdm = PoissonKernels.device_zeros(be_p, T, (ncells,))
        ρtot = PoissonKernels.device_zeros(be_p, T, (N,N,N))
        φ = gravity_1buf ? ρtot : similar(ρtot)
        φprev = gravity && gravity_source === :gas && gravity_phi_interp ? similar(ρtot) : nothing
        (px, py, pz, vx, vy, vz, ρdm, ρtot, φ, φprev)
    end
    eint, HII, H2I, HeII = _stage("allocate_chemistry_arrays") do
        eint = chem_mode === :off ? nothing : PoissonKernels.device_zeros(be_p, T, (ncells,))
        HII = chem_mode === :off ? nothing : PoissonKernels.device_zeros(be_p, T, (ncells,))
        H2I = chem_mode === :off ? nothing : PoissonKernels.device_zeros(be_p, T, (ncells,))
        HeII = chem_helium ? PoissonKernels.device_zeros(be_p, T, (ncells,)) : nothing
        (eint, HII, H2I, HeII)
    end
    chem_rate_tables = _stage("build_chemistry_rate_tables") do
        (chem_mode === :mixing || chem_mode === :analytic_mixing) && chem_rate_tables_enabled ?
            ChemistryKernels.build_rate_tables(; backend=BACKEND_NAME, precision=T) : nothing
    end
    chem_nonlocal_table = _stage("load_nonlocal_chemistry_table") do
        chem_nonlocal_enabled ? MHDKernels.load_nonlocal_recombination_table(
            chem_nonlocal_table_path, be_mhd; precision=T) : nothing
    end
    terminal_tvx, terminal_tvy, terminal_tvz = _stage("allocate_terminal_scratch") do
        if terminal_dedicated_scratch
            (PoissonKernels.device_zeros(be_p, T, (ncells,)),
             PoissonKernels.device_zeros(be_p, T, (ncells,)),
             PoissonKernels.device_zeros(be_p, T, (ncells,)))
        else
            (nothing, nothing, nothing)
        end
    end

    initial_helium_state = chem_helium ?
        _equilibrium_helium_state(Float64(xhii0), zstart, hub, Ob, Float64(fh)) :
        (xe_total=Float64(xhii0), xheii=0.0, heii_mass_fraction=0.0)
    _stage("init_particles_and_chemistry") do
        if lattice_displacements
            MHDKernels.initialize_lattice_displacements!(px, py, pz, vx, vy, vz)
        else
            _init_lattice_k!(be_p)(px, py, pz, vx, vy, vz, Int(N); ndrange=ncells)
        end
        if chem_mode !== :off
            _init_reduced_chem_k!(be_p)(HII, H2I, s.U[1], xhii0, xh20, fh, Int(N); ndrange=ncells)
            chem_helium && _init_reduced_helium_k!(be_p)(
                HeII, s.U[1], T(initial_helium_state.heii_mass_fraction), Int(N);
                ndrange=ncells)
        end
        KA.synchronize(be_p)
        nothing
    end
    if init_terminal_momenta && isempty(initial_state_dir)
        _stage("init_terminal_momenta") do
            gamma_init = if isfinite(drag_rate_code)
                drag_rate_code
            else
                drag_xe_init = isfinite(drag_xe_fixed) ? drag_xe_fixed :
                               _drag_electron_fraction(
                                   Float64(xhii0), initial_helium_state.xheii,
                                   zstart, hub, Ob, Float64(fh),
                                   drag_helium_electrons)
                gamma_drag_init = _compton_drag_rate_s(zstart, drag_xe_init, Float64(fh))
                terminal_gamma_scale *
                    _terminal_gamma_code(gamma_drag_init, 1.0, 1.0, chem_vunit,
                                         terminal_lbox_cm, z_to_a(zstart), pmf_b_aref)
            end
            isfinite(terminal_gamma_code_override) &&
                (gamma_init = terminal_gamma_code_override)
            gamma_init > 0 || error("terminal-momentum initialization needs positive drag")
            _terminal_handoff_momenta!(be_mhd, s, gamma_init, terminal_vcap, 1.0)
            @printf("INITIAL_TERMINAL_MOMENTA gamma_code=%.8e mode=%s\n", gamma_init,
                    isfinite(drag_rate_code) ? "fixed" : "physical")
            flush(stdout)
            nothing
        end
    end

    terminal_rho_mean = Float64(sum(s.U[1])) / Float64(ncells)
    terminal_pressure_coeff_step = terminal_pressure_coeff_auto ?
        Float64(s.γ) * p0 / max(terminal_rho_mean, eps(Float64)) : terminal_pressure_coeff
    radiation_initial_xe = isfinite(drag_xe_fixed) ? drag_xe_fixed :
        _drag_electron_fraction(xhii0, initial_helium_state.xheii,
                                zstart, hub, Ob, Float64(fh),
                                drag_helium_electrons)
    radiation_initial_closure = drag_dt_mode === :radiation ?
        _radiation_closure(zstart, radiation_initial_xe, hub, Ob, fh,
            chem_vunit, terminal_lbox_cm,
            sqrt(max(terminal_pressure_coeff_step, 0.0));
            bridge_q2=radiation_bridge_q2,
            longitudinal_viscosity=radiation_long_viscosity,
            nyquist_guard_fraction=radiation_nyquist_guard,
            response_model=radiation_response_model) : nothing
    radiation_needs_fft = chem_nonlocal_enabled ||
        (drag_dt_mode === :radiation &&
         (radiation_response_model in (:moments, :photon_moments,
                                       :moments_exponential,
                                       :photon_moments_exponential) ||
          radiation_response_model !== :bridge ||
          radiation_initial_closure.mean_free_path^2 * (2pi)^2 <
          radiation_initial_closure.bridge_q2))
    radiation_workspace = radiation_needs_fft ?
        MHDKernels.allocate_radiation_workspace(s) : nothing
    radiation_workspace_bytes = radiation_needs_fft ?
        5ncells*sizeof(T)+4*(N÷2+1)*N*N*sizeof(ComplexF32) : 0
    radiation_photon_state = drag_dt_mode === :radiation && radiation_response_model in (
        :moments,:photon_moments,:moments_exponential,
        :photon_moments_exponential) ?
        MHDKernels.allocate_photon_moment_state(s) : nothing
    if radiation_photon_state !== nothing
        MHDKernels.initialize_photon_moments!(radiation_photon_state,
            radiation_workspace,s;
            adiabatic_density=parse(Bool,get(ENV,"MHD_RADIATION_PHOTON_ADIABATIC","true")),
            comoving_velocity=parse(Bool,get(ENV,"MHD_RADIATION_PHOTON_COMOVING","true")))
    end
    _stage("import_initial_aux_state") do
        _import_initial_aux_state!(initial_state_dir; N=N,
            particles=(px, py, pz, vx, vy, vz),
            chemistry=chem_mode === :off ? nothing :
                (chem_helium ? (eint, HII, H2I, HeII) : (eint, HII, H2I)),
            photons=radiation_photon_state,
            density_perturbation=radiation_workspace === nothing ? nothing :
                                 radiation_workspace.density_perturbation)
        radiation_workspace === nothing ||
            MHDKernels.reconstruct_radiation_density!(radiation_workspace,s)
    end
    radiation_photon_bytes = radiation_photon_state === nothing ? 0 :
        4*(N÷2+1)*N*N*sizeof(ComplexF32)
    radiation_energy_ledger = drag_dt_mode === :radiation &&
                              radiation_energy_ledger_every > 0 ?
                              MHDKernels.RadiationEnergyLedger() : nothing

    phase = Dict(:cfl => 0.0, :mhd => 0.0, :radiation => 0.0,
                 :glm_projection => 0.0,
                 :drag => 0.0, :deposit => 0.0, :assemble => 0.0,
                 :fft => 0.0, :gas_gravity => 0.0, :push => 0.0, :chem => 0.0)
    radiation_step_timing = MHDKernels.RadiationStepTiming()
    radiation_phase = Dict(:predictor => 0.0, :corrector => 0.0, :ledger => 0.0)
    chem_phase = Dict(:unpack => 0.0, :smooth => 0.0, :solve => 0.0,
                      :nonlocal => 0.0,
                      :clamp => 0.0, :mean => 0.0, :repack => 0.0)
    glm_projection_steps = 0
    terminal_phase = Dict(:force => 0.0, :pressure_fft => 0.0, :finalize => 0.0,
                          :induction => 0.0, :projection => 0.0)
    terminal_measured = 0
    measured = 0
    measured_subcycles = 0
    last_dt = 0.0
    last_dt_explicit = 0.0
    last_drag_gammaH = NaN
    last_drag_f = NaN
    last_drag_xe = isfinite(drag_xe_fixed) ? drag_xe_fixed : Float64(xhii0)
    last_radiation_closure = radiation_initial_closure
    evolving_xe_mean = Float64(xhii0)
    evolving_xheii_mean = initial_helium_state.xheii
    last_hybrid_va_cs = NaN
    last_hybrid_drag_omega = NaN
    last_hybrid_regime = "n/a"
    was_terminal = false
    hybrid_terminal_active = false
    hybrid_controller = MHDKernels.DragRegimeController(
        exit_drag_over_omega=hybrid_exit_drag_omega,
        enter_drag_over_omega=hybrid_enter_drag_omega,
        min_va_over_cs=hybrid_min_va_cs,
        confirm_checks=hybrid_confirm_checks,
        reenter_terminal=hybrid_reenter_terminal)
    handoff_remaining = 0
    handoff_events = 0
    handoff_steps_done = 0
    last_handoff_remaining = 0
    last_handoff_alpha = 0.0
    last_handoff_error = NaN
    max_handoff_error = NaN
    max_handoff_settled_error = NaN
    last_handoff_delta_v_rms = NaN
    last_handoff_terminal_v_rms = NaN
    last_handoff_full_v_rms = NaN
    last_handoff_displacement = NaN
    max_handoff_displacement = 0.0
    cumulative_handoff_displacement = 0.0
    source_limited_steps = 0
    drag_impulse_limited_steps = 0
    last_terminal_diffuse_coeff = 0.0
    last_terminal_vmax = NaN
    last_induction_nsub = 0
    induction_subcycles_total = 0
    last_nsub = 1
    subcycles_total = 0
    last_gravity_nsub = gravity ? 1 : 0
    gravity_subcycles_total = 0
    last_particle_vmax = NaN
    last_particle_gmax = NaN
    particle_accel_max_dt = Inf
    boosted_steps = 0
    elapsed_time = 0.0
    a = zrun ? z_to_a(zstart) : NaN
    a_end = zrun ? z_to_a(zend) : NaN

    checkpoint_config = (
        backend=BACKEND_NAME, precision=string(T), N=N, zstart=zstart, zend=zend,
        boxsize=boxsize, pmf_init=pmf_init, pmf_b0_ng=pmf_b0_ng,
        pmf_kmode=pmf_kmode,
        meanfield_theta=meanfield_theta,
        meanfield_polarization=meanfield_polarization,
        meanfield_bguide=meanfield_bguide,
        meanfield_bpert=meanfield_bpert,
        pmf_seed=seed, pmf_kcut=kcut, pmf_kmin=pmf_kmin, pmf_kmax=pmf_kmax,
        pmf_mode_lock_n=pmf_mode_lock_n,
        pmf_high_mode_lock_n=pmf_high_mode_lock_n, cfl=cfl,
        pmf_normalization_z=pmf_normalization_z,
        cosmological_mhd_scaling=cosmological_mhd_scaling,
        initial_state_imported=!isempty(initial_state_dir),
        pmf_curl_reference_n=pmf_curl_reference_n,
        recon=recon, riemann=riemann, llf_dmin=llf_dmin, llf_pmin=llf_pmin,
        mhd_integrator=mhd_integrator, drag_dt_mode=drag_dt_mode,
        radiation_response_model=radiation_response_model,
        radiation_nonlinear_cfl=radiation_nonlinear_cfl,
        radiation_free_stream_tolerance=radiation_free_stream_tolerance,
        radiation_max_dlna=radiation_max_dlna,
        radiation_highz_max_dlna=radiation_highz_max_dlna,
        chem_mode=chem_mode, chem_coupling=chem_coupling,
        chem_helium=chem_helium,
        chem_expansion=chem_expansion,
        chem_nonlocal_table=chem_nonlocal_table_path,
        chem_nonlocal_scale=chem_nonlocal_scale, gravity=gravity,
        gravity_source=gravity_source, gravity_gas_kick=gravity_gas_kick,
        gravity_midpoint_source=gravity_midpoint_source,
        lattice_displacements=lattice_displacements, fft_mode=fft_mode,
    )
    function _checkpoint_arrays()
        arrays = Pair{String,Any}[]
        # The integrator swaps U and scratch after every accepted step, so this
        # tuple must be resolved at checkpoint time rather than captured once.
        for i in eachindex(s.U)
            push!(arrays, "mhd_U_$i" => s.U[i])
        end
        for (name, array) in (("particle_px", px), ("particle_py", py),
                              ("particle_pz", pz), ("particle_vx", vx),
                              ("particle_vy", vy), ("particle_vz", vz))
            push!(arrays, name => array)
        end
        if chem_mode !== :off
            push!(arrays, "chem_eint" => eint)
            push!(arrays, "chem_HII" => HII)
            push!(arrays, "chem_H2I" => H2I)
            chem_helium && push!(arrays, "chem_HeII" => HeII)
        end
        if radiation_photon_state !== nothing
            push!(arrays, "photon_hat_batch" => radiation_photon_state.hat_batch)
        end
        if radiation_workspace !== nothing
            push!(arrays, "gas_delta" => radiation_workspace.density_perturbation)
        end
        return arrays
    end

    restart_cycle = 0
    resumed_from_checkpoint = checkpoint_resume && !isempty(checkpoint_read_path) &&
                              isfile(checkpoint_read_path)
    if resumed_from_checkpoint
        _log_line("START restore_checkpoint path=$checkpoint_read_path")
        restore_seconds = @elapsed begin
            checkpoint_allowed_changes = Symbol[]
            get(ENV, "MHD_CHECKPOINT_ALLOW_CFL_CHANGE", "0") in
                ("1", "true", "TRUE", "yes", "on") &&
                append!(checkpoint_allowed_changes,
                        (:cfl, :radiation_nonlinear_cfl))
            get(ENV, "MHD_CHECKPOINT_ALLOW_NONLOCAL_TABLE_CHANGE", "0") in
                ("1", "true", "TRUE", "yes", "on") &&
                push!(checkpoint_allowed_changes, :chem_nonlocal_table)
            runtime = _restore_pmf_checkpoint!(
                checkpoint_read_path, checkpoint_config, _checkpoint_arrays();
                allowed_config_changes=Tuple(checkpoint_allowed_changes),
                allowed_config_validator=
                    _checkpoint_nonlocal_table_change_is_safe)
            restart_cycle = runtime.cycle
            measured = runtime.measured
            measured_subcycles = runtime.measured_subcycles
            elapsed_time = runtime.elapsed_time
            a = runtime.a
            evolving_xe_mean = runtime.evolving_xe_mean
            evolving_xheii_mean = runtime.evolving_xheii_mean
            merge!(phase, runtime.phase)
            merge!(chem_phase, runtime.chem_phase)
            glm_projection_steps = runtime.glm_projection_steps
            source_limited_steps = runtime.source_limited_steps
            boosted_steps = runtime.boosted_steps
            subcycles_total = runtime.subcycles_total
            gravity_subcycles_total = runtime.gravity_subcycles_total
            induction_subcycles_total = runtime.induction_subcycles_total
            terminal_measured = runtime.terminal_measured
            merge!(terminal_phase, runtime.terminal_phase)
            pk_next[] = runtime.pk_next
            frame_next[] = runtime.frame_next
            adaptive_last_delta_b[] = runtime.adaptive_last_delta_b
            adaptive_last_xhii[] = runtime.adaptive_last_xhii
            last_radiation_closure = runtime.last_radiation_closure
            reset_energy_ledger = get(
                ENV, "MHD_CHECKPOINT_RESET_ENERGY_LEDGER", "0") in
                ("1", "true", "TRUE", "yes", "on")
            if radiation_energy_ledger !== nothing && runtime.radiation_energy !== nothing &&
               !reset_energy_ledger
                energy = runtime.radiation_energy
                radiation_energy_ledger.samples = energy.samples
                radiation_energy_ledger.total_change = energy.total_change
                radiation_energy_ledger.kinetic_change = energy.kinetic_change
                radiation_energy_ledger.internal_change = energy.internal_change
                radiation_energy_ledger.magnetic_change = energy.magnetic_change
                radiation_energy_ledger.photon_change = energy.photon_change
                radiation_energy_ledger.resolved_coupled_change = energy.resolved_coupled_change
                radiation_energy_ledger.component_residual = energy.component_residual
                radiation_energy_ledger.component_residual_abs = energy.component_residual_abs
            elseif radiation_energy_ledger !== nothing && reset_energy_ledger
                @info "resetting radiation energy ledger after checkpoint restore"
            end
            KA.synchronize(be_mhd)
            KA.synchronize(be_p)
        end
        _log_line(@sprintf("DONE  restore_checkpoint %.3f s cycle=%d z=%.6f",
                           restore_seconds, restart_cycle,
                           zrun ? a_to_z(a) : NaN))
    end

    if φprev !== nothing
        _stage("init_previous_gas_potential") do
            if radiation_workspace === nothing
                _assemble_total_delta_k!(be_p)(ρtot,s.U[1],ρdm,
                    grav_gas_weight,grav_dm_weight,Int(N);ndrange=ncells)
            else
                _assemble_total_from_gas_delta_k!(be_p)(ρtot,
                    radiation_workspace.density_perturbation,ρdm,
                    grav_gas_weight,grav_dm_weight,Int(N);ndrange=ncells)
            end
            gravcoef0 = zrun ? 1.5 * Om * a : 1.0
            if fft_mode === :mps
                PoissonKernels.fft_poisson_rfft!(φprev, ρtot; G=gravcoef0, a=1, boxsize=1)
            elseif fft_mode === :ka
                PoissonKernels.fft_poisson_rfft_ka!(φprev, ρtot; G=gravcoef0, a=1, boxsize=1)
            else
                PoissonKernels.fft_poisson_root_gpu!(φprev, ρtot; G=gravcoef0, a=1, boxsize=1)
            end
            KA.synchronize(be_p)
            nothing
        end
    end

    cs_floor = sqrt(Float64(s.γ) * max(p0, pfloor))
    va_rms = Float64(st.brms_measured)
    terminal_keff = parse(Float64, get(ENV, "MHD_TERMINAL_KEFF",
                                       pmf_init === :batchelor ? string(max(1.0, kcut * boxsize / (2π))) :
                                       string(max(1, pmf_kmode))))
    hybrid_va_rms = va_rms
    if isfinite(pmf_b0_ng)
        @printf("PMF physical normalization: kind=%s epoch=initial_handoff z=%.3f target=%.6g comoving_nG reference_band=%.6g comoving_nG total_band=%.6g comoving_nG brms_code=%.6e vunit=%.6e cm/s\n",
                st.normalization_kind, pmf_normalization_z, pmf_b0_ng,
                pmf_initial_reference_comoving_ng, pmf_initial_total_comoving_ng,
                brms, chem_vunit)
        @printf("PMF comoving terminal clock: enabled=%s a_ref=%.9e\n",
                string(pmf_comoving_clock), z_to_a(pmf_b_zref))
        @printf("PMF cosmological MHD scaling: mode=%s velocity_aref=%.9e hubble_momentum_drag=%s\n",
                string(cosmological_mhd_scaling), cosmological_mhd_aref,
                string(cosmological_mhd_scaling === :jedamzik))
        flush(stdout)
    end
    if pmf_init in (:meanfield, :meanfield_linear, :sv25_mode)
        @printf("PMF matched mean-field mode: theta=%.9e polarization=%s guide_code=%.9e perturbation_code=%.9e fraction=%.9e\n",
                meanfield_theta, String(meanfield_polarization), meanfield_bguide,
                meanfield_bpert, meanfield_bpert / meanfield_bguide)
        flush(stdout)
    end
    p0_auto && @printf("PMF cosmological pressure: Tcmb(zstart)=%.6e K -> p0_code=%.9e\n",
                       _PMF_T_CMB * (1 + zstart), p0)
    @printf("PMF+DM lattice bench: backend=%s N=%d steps=%d warmup=%d profile_skip=%d tfinal=%s zrun=%s source_max_dt_factor=%.3g recon=%s riemann=%s chfac=%.2f cr=%.2f init=%s kmode=%d brms=%.3e p0=%.3e pfloor=%.3e va_rms/cs_floor=%.3e kcut=%.3f kmax=%s mode_lock_n=%s gravity=%s gas_kick=%s grav_source=%s phi_interp=%s midpoint_source=%s grav1buf=%s grav_sub=%d grav_dt_boost=%.2f particle_max_disp=%.3g particle_wrap=%s lattice_displacements=%s fft=%s chem=%s chem_coupling=%s chem_expansion=%s smooth=%s falpha=%.3g xHII0=%.3e drag=%s drag_dt=%s boost=%.2f drag_xe=%s terminal_split=%s terminal_vsmooth=%.3f terminal_gamma_scale=%.3e terminal_vcap=%.3e terminal_lbox_ckpc=%.3e terminal_diff_cfl=%.3e terminal_accuracy_cfl=%.3e terminal_accuracy_pressure=%.3e terminal_implicit_diffuse=%s terminal_implicit_fac=%.3e terminal_pressure_implicit=%s terminal_pressure_coeff=%.3e terminal_pressure_coeff_mode=%s terminal_rho_rel_limit=%.3e terminal_b_rel_limit=%.3e hybrid_handoff_steps=%d hybrid_handoff_alpha_max=%.3f hybrid_handoff_ramp=cosine hybrid_handoff_error_tol=%.3e hybrid_handoff_displacement_tol=%.3e hybrid_reenter=%s hybrid_aware_dt_factor=%.3f\n",
            String(BACKEND_NAME), N, steps, warmup, profile_skip_cycles, tfinal === nothing ? "cycle-count" : string(tfinal),
            zrun ? @sprintf("%.1f->%.1f", zstart, zend) : "off", source_max_dt_factor,
            String(recon), String(riemann), glm_ch_fac, glm_cr, String(pmf_init), pmf_kmode, brms, p0, pfloor,
            va_rms / max(cs_floor, eps(Float64)), kcut,
            pmf_kmax === nothing ? "none" : @sprintf("%.6g", pmf_kmax),
            pmf_mode_lock_n === nothing ? "none" : string(pmf_mode_lock_n),
            string(gravity), string(gravity_gas_kick), String(gravity_source),
            string(gravity_phi_interp), string(gravity_midpoint_source),
            string(gravity_1buf), gravity_subcycles_fixed,
            gravity_dt_boost, particle_max_disp_cells, string(particle_wrap),
            string(lattice_displacements), String(fft_mode),
            String(chem_mode), String(chem_coupling), string(chem_expansion),
            String(chem_smooth), chem_falpha,
            Float64(xhii0), string(drag_enabled),
            String(drag_dt_mode), drag_dt_boost, drag_xe_env, string(terminal_split), terminal_vsmooth,
            terminal_gamma_scale, terminal_vcap,
            isfinite(terminal_lbox_cm) ? terminal_lbox_cm / _PMF_KPC_CM : NaN,
            terminal_diff_cfl, terminal_accuracy_cfl, terminal_accuracy_pressure,
            string(terminal_implicit_diffuse), terminal_implicit_diffuse_fac,
            string(terminal_pressure_implicit), terminal_pressure_coeff_step,
            terminal_pressure_coeff_auto ? "auto" : "fixed",
            terminal_rho_rel_limit, terminal_b_rel_limit,
            hybrid_handoff_steps, hybrid_handoff_alpha, hybrid_handoff_error_tol,
            hybrid_handoff_displacement_tol,
            string(hybrid_reenter_terminal),
            hybrid_aware_dt_factor)
    @printf("  terminal induction=%s cfl=%.3f dissipation=%.3f/order%d project_b=%s max_subcycles=%d\n",
            string(terminal_induction), terminal_induction_cfl, terminal_induction_dissipation,
            terminal_induction_dissipation_order, string(terminal_project_b),
            terminal_induction_max_subcycles)
    @printf("  chemistry coupling=%s fused_analytic=%s fused_mixing=%s helium=%s ionized_fast_zmin=%.6g mean_every=%d profile=%s rate_tables=%s itcap=%d dtfrac=%.3g\n",
            String(chem_coupling), string(chem_fused_analytic), string(chem_fused_mixing),
            string(chem_helium), chem_ionized_fast_zmin, chem_mean_every,
            string(chem_profile_enabled), string(chem_rate_tables_enabled),
            chem_itcap, chem_dtfrac)
    @printf("  chemistry species transport=%s scheme=bounded_fraction_backtrace scratch=reused_mhd\n",
            string(chem_advect_species))
    drag_dt_mode === :hybrid &&
        @printf("  hybrid hysteresis exit=%.3f enter=%.3f check_every=%d confirm=%d min_va/cs=%.3f\n",
                hybrid_exit_drag_omega, hybrid_enter_drag_omega, hybrid_check_every,
                hybrid_confirm_checks, hybrid_min_va_cs)
    drag_dt_mode === :radiation &&
        @printf("  radiation full-MHD response=%s bridge_q2=%.6g longitudinal_viscosity=%.6g max_dlna=%.6g highz_max_dlna=%.6g highz_zmin=%.6g nonlinear_cfl=%.6g free_stream_tol=%.6g fft_active=%s workspace_bytes=%d photon_state_bytes=%d\n",
                string(radiation_response_model),
                radiation_bridge_q2,radiation_long_viscosity,radiation_max_dlna,
                radiation_highz_max_dlna,radiation_highz_zmin,
                radiation_nonlinear_cfl,radiation_free_stream_tolerance,
                string(radiation_needs_fft),radiation_workspace_bytes,
                radiation_photon_bytes)
    chem_nonlocal_enabled &&
        @printf("  nonlocal Lyman-alpha table=%s scale=%.6g workspace=reused-radiation\n",
                chem_nonlocal_table_path, chem_nonlocal_scale)
    isfinite(drag_rate_code) &&
        @printf("  fixed dimensionless drag rate Gamma_code=%.6e\n", drag_rate_code)
    flush(stdout)

    function _debug_div(label, cyc_now, a_now)
        debug_div_every == 0 && return nothing
        if cyc_now != 0 && cyc_now % debug_div_every != 0
            return nothing
        end
        br_dbg = _device_brms!(be_mhd, s, s.scratch[1])
        div_dbg = _device_divb_norm!(be_mhd, s, s.scratch[1], br_dbg)
        @printf("  divdbg cycle=%d phase=%s z=%.6f brms=%.8e divBdx_over_brms=%.8e\n",
                cyc_now, label, zrun ? a_to_z(a_now) : NaN, br_dbg, div_dbg)
        flush(stdout)
        return nothing
    end
    function _debug_phase_state(label, cyc_now, a_now, dt_now, smax_now)
        debug_phase_finite || return nothing
        cyc_now < debug_phase_cycle_min && return nothing
        z_now = zrun ? a_to_z(a_now) : NaN
        zrun && z_now > debug_phase_zmax && return nothing
        ss = _assert_finite_state(s, label, cyc_now, a_now, dt_now, smax_now)
        if debug_phase_rho_min > 0 && Float64(ss.ρmin) < debug_phase_rho_min
            error(@sprintf("%s cycle=%d density threshold crossed z=%.6f rho_min=%.8e threshold=%.8e rho_max=%.8e E=[%.8e,%.8e] Bmax=%.8e",
                           label, cyc_now, z_now, Float64(ss.ρmin),
                           debug_phase_rho_min, Float64(ss.ρmax),
                           Float64(ss.Emin), Float64(ss.Emax), ss.Bmax))
        end
        return nothing
    end
    _debug_div("init", 0, zrun ? a : 1.0)

    function _chem_step_interval!(a0_step, a1_step, dt_code_step, refresh_mean,
                                  measuring_now, xe_mean_in, xheii_mean_in,
                                  nonlocal_before::Bool=false)
        return _run_pmf_chemistry_step!(
            be_p, be_mhd, s, eint, HII, H2I, HeII, ρtot, φ;
            N=N, ncells=ncells, zrun=zrun,
            a0=Float64(a0_step), a1=Float64(a1_step), dt_code=Float64(dt_code_step),
            hub=hub, Om=Om, OL=OL, Or=Or, Ob=Ob, boxsize=boxsize,
            chem_mode=chem_mode, chem_smooth=chem_smooth,
            chem_density_fixed=chem_density_fixed,
            chem_vunit=chem_vunit, chem_time_unit=chem_time_unit,
            chem_ionized_fast_zmin=chem_ionized_fast_zmin,
            chem_fused_analytic=chem_fused_analytic,
            chem_fused_mixing=chem_fused_mixing,
            chem_helium=chem_helium,
            chem_expansion=chem_expansion,
            chem_profile_enabled=chem_profile_enabled, measuring=measuring_now,
            refresh_mean=refresh_mean, chem_phase=chem_phase,
            chem_falpha=chem_falpha, chem_falpha_zs=chem_falpha_zs,
            chem_falpha_vals=chem_falpha_vals, chem_nsmean=chem_nsmean,
            chem_smooth_sigma=chem_smooth_sigma, chem_xe_mean=chem_xe_mean,
            evolving_xe_mean=xe_mean_in,
            evolving_xheii_mean=xheii_mean_in,
            chem_recfast_hswitch=chem_recfast_hswitch, chem_fudge=chem_fudge,
            chem_rate_tables=chem_rate_tables, chem_dtfrac=chem_dtfrac,
            chem_itcap=chem_itcap, fh=fh,
            chem_nonlocal_table=chem_nonlocal_table,
            chem_nonlocal_scale=chem_nonlocal_scale,
            chem_nonlocal_before=nonlocal_before,
            radiation_workspace=radiation_workspace,
            terminal_lbox_cm=terminal_lbox_cm,
            cosmological_mhd_aref=cosmological_mhd_aref)
    end

    function _save_checkpoint!(cycle_now)
        (isempty(checkpoint_write_path) || checkpoint_disable_writes) && return nothing
        energy = radiation_energy_ledger === nothing ? nothing : (
            samples=radiation_energy_ledger.samples,
            total_change=radiation_energy_ledger.total_change,
            kinetic_change=radiation_energy_ledger.kinetic_change,
            internal_change=radiation_energy_ledger.internal_change,
            magnetic_change=radiation_energy_ledger.magnetic_change,
            photon_change=radiation_energy_ledger.photon_change,
            resolved_coupled_change=radiation_energy_ledger.resolved_coupled_change,
            component_residual=radiation_energy_ledger.component_residual,
            component_residual_abs=radiation_energy_ledger.component_residual_abs,
        )
        runtime = (
            cycle=cycle_now, measured=measured,
            measured_subcycles=measured_subcycles,
            elapsed_time=elapsed_time, a=a,
            evolving_xe_mean=evolving_xe_mean,
            evolving_xheii_mean=evolving_xheii_mean,
            phase=copy(phase), chem_phase=copy(chem_phase),
            glm_projection_steps=glm_projection_steps,
            source_limited_steps=source_limited_steps,
            boosted_steps=boosted_steps,
            subcycles_total=subcycles_total,
            gravity_subcycles_total=gravity_subcycles_total,
            induction_subcycles_total=induction_subcycles_total,
            terminal_measured=terminal_measured,
            terminal_phase=copy(terminal_phase),
            pk_next=pk_next[], frame_next=frame_next[],
            adaptive_last_delta_b=adaptive_last_delta_b[],
            adaptive_last_xhii=adaptive_last_xhii[],
            last_radiation_closure=last_radiation_closure,
            radiation_energy=energy,
        )
        KA.synchronize(be_mhd)
        KA.synchronize(be_p)
        bytes = _write_pmf_checkpoint(checkpoint_write_path, checkpoint_config,
                                      runtime, _checkpoint_arrays())
        @printf("  checkpoint cycle=%d z=%.6f bytes=%d path=%s\n",
                cycle_now, zrun ? a_to_z(a) : NaN, bytes, checkpoint_write_path)
        flush(stdout)
        return nothing
    end

    # Establish the actual initialized state before any MHD/drag source step.
    # Batchelor ICs deliberately start with uniform gas density; recording this
    # row prevents the first post-step diagnostic from being mistaken for an IC.
    if !resumed_from_checkpoint &&
       ((initial_history && (!isempty(history_path) || !isempty(mode_history_path))) ||
        (initial_pk && !isempty(pk_path)))
        _emit_diagnostics!(be_p=be_p, s=s, ρdm=ρdm, scratch=ρtot,
                           px=px, py=py, pz=pz, vx=vx, vy=vy, vz=vz,
                           N=N, ncells=ncells, gravity=gravity, cycle=0,
                           measured=0, a=zrun ? a : 1.0,
                           pk_path=initial_pk ? pk_path : "",
                           history_path=initial_history ? history_path : "", pdf_path="",
                           pdf_counts=nothing, pdf_loglo=pdf_loglo, pdf_loghi=pdf_loghi,
                           nmu=pk_nmu, nbins=pk_nbins, boxsize=boxsize, axis=pk_axis,
                           kmax=pk_kmax,
                           lattice_displacements=lattice_displacements,
                           HII=HII, H2I=H2I, HeII=HeII, fh=fh,
                           btrans0=Float64(st.btrans0), density_kmode=pmf_kmode,
                           mode_history_path=initial_history ? mode_history_path : "",
                           mode_kmode=pmf_kmode, hubble=hub, omega_b=Ob,
                           radiation_workspace=radiation_workspace,
                           radiation_photon_state=radiation_photon_state,
                           temperature_velocity_unit2=chem_mode === :off ? NaN :
                               chem_vunit^2 * (isfinite(cosmological_mhd_aref) ?
                               cosmological_mhd_aref / a : 1.0))
        @printf("INITIAL_DIAG emitted at z=%.6f before the first MHD/drag step (history=%s pk=%s)\n",
                zrun ? a_to_z(a) : NaN, string(initial_history), string(initial_pk))
        flush(stdout)
    end

    if !resumed_from_checkpoint && zrun && frame_next[] <= length(frame_zs) &&
       _reached_output_z(a_to_z(a), frame_zs[frame_next[]])
        _write_density_frame!(frame_dir, frame_hostrho, frame_hostslice, frame_hostproj,
                              s.U[1], Int(N), a_to_z(a), 0)
        while frame_next[] <= length(frame_zs) &&
              _reached_output_z(a_to_z(a), frame_zs[frame_next[]])
            frame_next[] += 1
        end
        @printf("INITIAL_FRAME emitted at z=%.6f\n", a_to_z(a))
        flush(stdout)
    end

    cyc = restart_cycle
    while true
        cyc += 1
        if debug_stop_cycle > 0 && cyc > debug_stop_cycle
            error("DEBUG stop cycle=$debug_stop_cycle")
        end
        dt0_ref = Ref(zero(T))
        smax_ref = Ref(zero(T))
        tcfl = _time_phase(be_mhd, be_p) do
            dt_tmp, smax_tmp = MHDKernels.compute_dt!(s.scratch[1], s; cfl=cfl)
            dt0_ref[] = dt_tmp
            smax_ref[] = smax_tmp
            nothing
        end
        dt0 = dt0_ref[]
        smax = smax_ref[]
        ch = T(glm_ch_fac) * T(smax)
        dt_hydro_explicit = Float64(dt0) * Float64(smax) / Float64(max(T(smax), ch))
        hydro_per_tau = zrun ? _hydro_time_per_tau(a, hub, chem_vunit,
                                                   terminal_lbox_cm,
                                                   cosmological_mhd_aref) : 1.0
        dt_explicit = dt_hydro_explicit / hydro_per_tau
        dt = dt_explicit
        if drag_enabled && drag_dt_mode in (:boost, :subcycle, :semiimplicit, :aware,
                                            :godunov, :radiation, :terminal, :hybrid) && drag_dt_boost > 1
            dt *= drag_dt_boost
            boosted_steps += 1
        end
        if zrun
            if !(isfinite(a) && a > 0 && a <= a_end)
                error(@sprintf("cycle=%d invalid scale factor a=%.8e z=%.6f", cyc, a, isfinite(a) ? a_to_z(a) : NaN))
            end
            if isapprox(a, a_end; rtol=0, atol=8eps(Float64) * max(1.0, a_end))
                a = a_end
                break
            end
            radiation_tight_now = drag_dt_mode === :radiation &&
                last_radiation_closure !== nothing &&
                last_radiation_closure.mean_free_path^2 * (2pi)^2 <
                last_radiation_closure.bridge_q2
            radiation_dlna_now = a_to_z(a) > radiation_highz_zmin ?
                                 radiation_highz_max_dlna : radiation_max_dlna
            maxexp_step = radiation_tight_now ? min(maxexp, radiation_dlna_now) : maxexp
            dt = min(dt, dtau_for_dlna(Om, OL, Ok, Or, a, maxexp_step))
            if radiation_tight_now && isfinite(radiation_nonlinear_cfl)
                # Linear photon-acoustic propagation is integrated by the
                # spectral propagator. Limit only the finite-volume MHD/GLM
                # displacement; using c_rad here restores the linear CFL and
                # makes the exact propagator unusable across recombination.
                nonlinear_signal = max(Float64(smax), Float64(ch))
                dt_radiation = radiation_nonlinear_cfl * Float64(s.dx) /
                               (max(nonlinear_signal, eps(Float64)) * hydro_per_tau)
                dt = min(dt, dt_radiation)
            end
            dt_end = dtau_for_dlna(Om, OL, Ok, Or, a, max(0.0, log(a_end / a)))
            if dt_end <= 8eps(Float64) * max(1.0, abs(dt))
                a = a_end
                break
            end
            dt = min(dt, dt_end)
        end
        if tfinal !== nothing && cyc > warmup && elapsed_time + dt > tfinal
            dt = tfinal - elapsed_time
        end
        source_dt = _limit_source_dt(dt, dt_explicit, source_max_dt_factor)
        dt = source_dt.dt
        source_limited_steps += source_dt.limited
        _assert_finite_step(s, "pre-step", cyc, zrun ? a : NaN, dt, smax)
        last_dt = dt
        last_dt_explicit = dt_explicit
        measuring = cyc > warmup + profile_skip_cycles
        nsub = 1
        if drag_enabled && drag_dt_mode in (:subcycle, :semiimplicit, :aware,
                                            :godunov, :radiation, :hybrid) && dt > dt_explicit
            nsub = max(1, ceil(Int, dt / max(dt_explicit, eps(Float64))))
        end
        last_nsub = nsub

        a_before = zrun ? a : 1.0
        a_after_step = zrun ? _advance_a_rk2(Om, OL, Ok, Or, a, dt, a_end) : 1.0
        zsource = zrun ? a_to_z(0.5 * (a_before + a_after_step)) : 0.0
        dtphys_source = zrun ? _cosmic_dt_seconds(a_before, a_after_step, hub, Om, OL, Or) :
                        dt * chem_time_unit
        dt_hydro = zrun ? _hydro_code_dt(dtphys_source, dt, chem_vunit,
                                         terminal_lbox_cm,
                                         0.5 * (a_before + a_after_step),
                                         cosmological_mhd_aref) : dt
        tdrag = 0.0
        tm = 0.0
        trad = 0.0
        tgp = 0.0
        radiation_timing_before = (
            predictor=radiation_step_timing.predictor_s,
            corrector=radiation_step_timing.corrector_s,
            ledger=radiation_step_timing.ledger_s,
        )
        terminal_timed = false
        terminal_step_timing = (force_s=0.0, pressure_fft_s=0.0, finalize_s=0.0,
                                induction_s=0.0, projection_s=0.0)
        gamma_drag = 0.0
        drag_impulse = 0.0
        if drag_enabled
            drag_xe = isfinite(drag_xe_fixed) ? drag_xe_fixed :
                      _drag_electron_fraction(evolving_xe_mean, evolving_xheii_mean,
                                              zsource, hub, Ob,
                                              Float64(fh), drag_helium_electrons)
            last_drag_xe = drag_xe
            gamma_drag = _compton_drag_rate_s(zsource, drag_xe, Float64(fh))
            last_drag_gammaH = isfinite(drag_rate_code) ? NaN :
                               zrun ? gamma_drag / _hubble_s(zsource, hub, Om, OL, Or) : NaN
            drag_impulse = _drag_impulse(gamma_drag, dtphys_source, dt, drag_rate_code)
            last_drag_f = exp(-drag_impulse)
        end
        if drag_enabled && drag_dt_mode in (:aware, :semiimplicit) &&
           isfinite(drag_max_impulse) && drag_impulse > drag_max_impulse
            drag_impulse_limited_steps += 1
            for _ in 1:2
                drag_impulse <= drag_max_impulse * (1 + 1e-6) && break
                dt *= drag_max_impulse / drag_impulse
                last_dt = dt
                a_after_step = zrun ? _advance_a_rk2(Om, OL, Ok, Or, a, dt, a_end) : 1.0
                zsource = zrun ? a_to_z(0.5 * (a_before + a_after_step)) : 0.0
                dtphys_source = zrun ?
                    _cosmic_dt_seconds(a_before, a_after_step, hub, Om, OL, Or) :
                    dt * chem_time_unit
                dt_hydro = zrun ?
                    _hydro_code_dt(dtphys_source, dt, chem_vunit, terminal_lbox_cm,
                                   0.5 * (a_before + a_after_step),
                                   cosmological_mhd_aref) : dt
                drag_xe = isfinite(drag_xe_fixed) ? drag_xe_fixed :
                          _drag_electron_fraction(evolving_xe_mean, evolving_xheii_mean,
                                                  zsource, hub, Ob,
                                                  Float64(fh), drag_helium_electrons)
                last_drag_xe = drag_xe
                gamma_drag = _compton_drag_rate_s(zsource, drag_xe, Float64(fh))
                last_drag_gammaH = isfinite(drag_rate_code) ? NaN :
                                   zrun ? gamma_drag / _hubble_s(zsource, hub, Om, OL, Or) : NaN
                drag_impulse = _drag_impulse(gamma_drag, dtphys_source, dt, drag_rate_code)
                last_drag_f = exp(-drag_impulse)
            end
            nsub = max(1, ceil(Int, dt / max(dt_explicit, eps(Float64))))
            last_nsub = nsub
        end
        gamma_code = drag_enabled ?
                     (isfinite(drag_rate_code) ? drag_rate_code : terminal_gamma_scale *
                      _terminal_gamma_code(gamma_drag, dtphys_source, dt, chem_vunit,
                                           terminal_lbox_cm, 0.5 * (a_before + a_after_step),
                                           pmf_b_aref)) : 0.0
        isfinite(terminal_gamma_code_override) && (gamma_code = terminal_gamma_code_override)
        use_terminal = drag_dt_mode === :terminal
        if drag_dt_mode === :hybrid
            hybrid_check = cyc == 1 || ((cyc - 1) % hybrid_check_every == 0)
            if hybrid_check
                hybrid_va_rms = _device_brms!(be_mhd, s, s.scratch[1])
            end
            last_hybrid_va_cs = hybrid_va_rms / max(cs_floor, eps(Float64))
            omega_A = 2π * terminal_keff * max(hybrid_va_rms, eps(Float64))
            last_hybrid_drag_omega = gamma_code / max(omega_A, eps(Float64))
            if hybrid_check
                decision = MHDKernels.update_drag_regime!(hybrid_controller,
                                                          last_hybrid_drag_omega,
                                                          last_hybrid_va_cs)
                hybrid_terminal_active = decision.terminal
            end
            use_terminal = hybrid_terminal_active
            last_hybrid_regime = use_terminal ? "terminal" : "aware"
            if use_terminal
                nsub = 1
                last_nsub = nsub
            else
                aware_step = _limit_hybrid_aware_dt(dt, dt_explicit,
                                                     hybrid_aware_dt_factor)
                nsub = aware_step.nsub
                last_nsub = nsub
            end
            if !use_terminal && aware_step.limited
                dt = aware_step.dt
                last_dt = dt
                a_after_step = zrun ? _advance_a_rk2(Om, OL, Ok, Or, a, dt, a_end) : 1.0
                zsource = zrun ? a_to_z(0.5 * (a_before + a_after_step)) : 0.0
                dtphys_source = zrun ? _cosmic_dt_seconds(a_before, a_after_step, hub, Om, OL, Or) :
                                dt * chem_time_unit
                dt_hydro = zrun ? _hydro_code_dt(dtphys_source, dt, chem_vunit,
                                                 terminal_lbox_cm,
                                                 0.5 * (a_before + a_after_step),
                                                 cosmological_mhd_aref) : dt
                if drag_enabled
                    drag_xe = isfinite(drag_xe_fixed) ? drag_xe_fixed :
                              _drag_electron_fraction(evolving_xe_mean,
                                                      evolving_xheii_mean,
                                                      zsource, hub, Ob,
                                                      Float64(fh), drag_helium_electrons)
                    last_drag_xe = drag_xe
                    gamma_drag = _compton_drag_rate_s(zsource, drag_xe, Float64(fh))
                    last_drag_gammaH = isfinite(drag_rate_code) ? NaN :
                                       zrun ? gamma_drag / _hubble_s(zsource, hub, Om, OL, Or) : NaN
                    drag_impulse = _drag_impulse(gamma_drag, dtphys_source, dt, drag_rate_code)
                    last_drag_f = exp(-drag_impulse)
                    gamma_code = isfinite(drag_rate_code) ? drag_rate_code : terminal_gamma_scale *
                                 _terminal_gamma_code(gamma_drag, dtphys_source, dt, chem_vunit,
                                                      terminal_lbox_cm, 0.5 * (a_before + a_after_step),
                                                      pmf_b_aref)
                    isfinite(terminal_gamma_code_override) && (gamma_code = terminal_gamma_code_override)
                    last_hybrid_drag_omega = gamma_code / max(omega_A, eps(Float64))
                end
            end
        elseif drag_dt_mode === :terminal
            last_hybrid_regime = "terminal"
        elseif drag_dt_mode in (:aware, :subcycle, :semiimplicit)
            last_hybrid_regime = "aware"
        elseif drag_dt_mode === :godunov
            last_hybrid_regime = "godunov"
        elseif drag_dt_mode === :radiation
            last_hybrid_regime = "radiation"
        elseif drag_dt_mode === :boost
            last_hybrid_regime = "boost"
        else
            last_hybrid_regime = "explicit"
        end
        if (use_terminal && terminal_pressure_coeff_auto) || drag_dt_mode===:radiation
            pmean = _device_pressure_mean!(be_mhd, s, s.scratch[1])
            terminal_pressure_coeff_step = Float64(s.γ) * pmean /
                                               max(terminal_rho_mean, eps(Float64))
        end
        if drag_dt_mode === :radiation
            radiation_velocity_unit = _cosmological_velocity_unit(
                chem_vunit, 0.5 * (a_before + a_after_step),
                cosmological_mhd_aref)
            last_radiation_closure = _radiation_closure(
                zsource, last_drag_xe, hub, Ob, fh, radiation_velocity_unit,
                terminal_lbox_cm, sqrt(max(terminal_pressure_coeff_step, 0.0));
                bridge_q2=radiation_bridge_q2,
                longitudinal_viscosity=radiation_long_viscosity,
                nyquist_guard_fraction=radiation_nyquist_guard,
                response_model=radiation_response_model)
        end
        if drag_dt_mode === :hybrid && was_terminal && !use_terminal
            handoff_events += 1
            cumulative_handoff_displacement = 0.0
            @printf("  hybrid handoff start cycle=%d z=%.6f G/omA=%.3e va/cs=%.3e steps=%d alpha_max=%.3f ramp=cosine error_tol=%.3e displacement_tol=%.3e\n",
                    cyc, zrun ? a_to_z(a_before) : NaN, last_hybrid_drag_omega,
                    last_hybrid_va_cs, hybrid_handoff_steps, hybrid_handoff_alpha,
                    hybrid_handoff_error_tol, hybrid_handoff_displacement_tol)
            flush(stdout)
            if hybrid_handoff_steps > 0
                handoff_remaining = max(handoff_remaining, hybrid_handoff_steps)
            end
        end
        if drag_dt_mode === :hybrid && !was_terminal && use_terminal && cyc > 1
            @printf("  hybrid terminal re-entry cycle=%d z=%.6f G/omA=%.3e va/cs=%.3e\n",
                    cyc, zrun ? a_to_z(a_before) : NaN, last_hybrid_drag_omega,
                    last_hybrid_va_cs)
            flush(stdout)
        end
        if use_terminal && (isfinite(terminal_max_disp_cells) ||
                            isfinite(terminal_diff_cfl) || isfinite(terminal_accuracy_cfl))
            for _limiter_iter in 1:2
                dt_terminal_limit = Inf
                if isfinite(terminal_max_disp_cells) && terminal_max_disp_cells > 0
                    terminal_vmax = MHDKernels.terminal_velocity_max!(s;
                        gamma_drag=gamma_code, pressure_coeff=terminal_pressure_coeff_step,
                        vcap=terminal_vcap)
                    last_terminal_vmax = terminal_vmax
                    dt_disp = terminal_max_disp_cells * Float64(s.dx) /
                              max(terminal_vmax, eps(Float64))
                    dt_terminal_limit = min(dt_terminal_limit, dt_disp)
                end
                if isfinite(terminal_diff_cfl)
                    dt_diff = terminal_diff_cfl * max(gamma_code, eps(Float64)) *
                              Float64(s.dx)^2 / max(Float64(smax)^2, eps(Float64))
                    dt_terminal_limit = min(dt_terminal_limit, dt_diff)
                end
                if isfinite(terminal_accuracy_cfl)
                    kacc = 2π * max(terminal_keff, eps(Float64))
                    dt_acc = terminal_accuracy_cfl * max(gamma_code, eps(Float64)) /
                             (4 * terminal_accuracy_pressure * max(kacc * kacc, eps(Float64)))
                    dt_terminal_limit = min(dt_terminal_limit, dt_acc)
                end
                if dt_hydro <= dt_terminal_limit * (1 + 1e-6)
                    break
                end
                dt *= dt_terminal_limit / dt_hydro
                dt = max(dt, 8eps(Float64) * max(1.0, abs(dt)))
                last_dt = dt
                nsub = 1
                last_nsub = nsub
                a_after_step = zrun ? _advance_a_rk2(Om, OL, Ok, Or, a, dt, a_end) : 1.0
                zsource = zrun ? a_to_z(0.5 * (a_before + a_after_step)) : 0.0
                dtphys_source = zrun ? _cosmic_dt_seconds(a_before, a_after_step, hub, Om, OL, Or) :
                                dt * chem_time_unit
                dt_hydro = zrun ? _hydro_code_dt(dtphys_source, dt, chem_vunit,
                                                 terminal_lbox_cm,
                                                 0.5 * (a_before + a_after_step),
                                                 cosmological_mhd_aref) : dt
                if drag_enabled
                    drag_xe = isfinite(drag_xe_fixed) ? drag_xe_fixed :
                              _drag_electron_fraction(evolving_xe_mean,
                                                      evolving_xheii_mean,
                                                      zsource, hub, Ob,
                                                      Float64(fh), drag_helium_electrons)
                    last_drag_xe = drag_xe
                    gamma_drag = _compton_drag_rate_s(zsource, drag_xe, Float64(fh))
                    last_drag_gammaH = isfinite(drag_rate_code) ? NaN :
                                       zrun ? gamma_drag / _hubble_s(zsource, hub, Om, OL, Or) : NaN
                    drag_impulse = _drag_impulse(gamma_drag, dtphys_source, dt, drag_rate_code)
                    last_drag_f = exp(-drag_impulse)
                    gamma_code = isfinite(drag_rate_code) ? drag_rate_code : terminal_gamma_scale *
                                 _terminal_gamma_code(gamma_drag, dtphys_source, dt, chem_vunit,
                                                      terminal_lbox_cm, 0.5 * (a_before + a_after_step),
                                                      pmf_b_aref)
                    isfinite(terminal_gamma_code_override) && (gamma_code = terminal_gamma_code_override)
                    if drag_dt_mode === :hybrid
                        omega_A = 2π * terminal_keff * max(hybrid_va_rms, eps(Float64))
                        last_hybrid_drag_omega = gamma_code / max(omega_A, eps(Float64))
                    end
                end
            end
        end
        was_terminal = use_terminal
        last_handoff_remaining = handoff_remaining
        last_handoff_alpha = 0.0
        subcycles_total += nsub
        tc = 0.0
        chem_refresh_mean = cyc <= warmup || chem_mean_every == 1 ||
                            (cyc % chem_mean_every == 0)
        a_chem_mid = zrun ? _advance_a_rk2(Om, OL, Ok, Or, a_before, 0.5 * dt,
                                           a_after_step) : 1.0
        if chem_mode !== :off && chem_coupling === :strang
            tchem, evolving_xe_mean, evolving_xheii_mean =
                _chem_step_interval!(a_before, a_chem_mid, 0.5 * dt,
                                     chem_refresh_mean, measuring, evolving_xe_mean,
                                     evolving_xheii_mean,
                                     false)
            tc += tchem
            if drag_enabled
                drag_xe = isfinite(drag_xe_fixed) ? drag_xe_fixed :
                          _drag_electron_fraction(evolving_xe_mean, evolving_xheii_mean,
                                                  zsource, hub, Ob,
                                                  Float64(fh), drag_helium_electrons)
                last_drag_xe = drag_xe
                gamma_drag = _compton_drag_rate_s(zsource, drag_xe, Float64(fh))
                last_drag_gammaH = isfinite(drag_rate_code) ? NaN :
                                   zrun ? gamma_drag / _hubble_s(zsource, hub, Om, OL, Or) : NaN
                drag_impulse = _drag_impulse(gamma_drag, dtphys_source, dt, drag_rate_code)
                last_drag_f = exp(-drag_impulse)
                gamma_code = isfinite(drag_rate_code) ? drag_rate_code : terminal_gamma_scale *
                             _terminal_gamma_code(gamma_drag, dtphys_source, dt, chem_vunit,
                                                  terminal_lbox_cm,
                                                  0.5 * (a_before + a_after_step), pmf_b_aref)
                isfinite(terminal_gamma_code_override) && (gamma_code = terminal_gamma_code_override)
                if drag_dt_mode === :hybrid
                    omega_A = 2π * terminal_keff * max(hybrid_va_rms, eps(Float64))
                    last_hybrid_drag_omega = gamma_code / max(omega_A, eps(Float64))
                end
            end
            _debug_div("after_chem_half_pre", cyc, zrun ? a_before : 1.0)
            _debug_phase_state("after_chem_half_pre", cyc, a_before, dt, smax)
        end
        if use_terminal
            vcap_step = terminal_vcap
            if chem_advect_species
                tm += _time_phase(be_mhd, be_p) do
                    MHDKernels.conserved_species_to_fractions!(
                        HII, H2I, s.U[1]; maximum_total=fh)
                    chem_helium && MHDKernels.conserved_species_to_fraction!(
                        HeII, s.U[1]; maximum_total=one(T) - fh)
                end
            end
            tm += _time_phase(be_mhd, be_p) do
                if terminal_pressure_implicit
                    result = _terminal_pressure_implicit_step!(be_mhd, s, dt_hydro, gamma_code,
                                                               vcap_step, terminal_pressure_coeff_step,
                                                               terminal_rho_rel_limit,
                                                               terminal_b_rel_limit;
                                                               induction=terminal_induction,
                                                               induction_cfl=terminal_induction_cfl,
                                                               induction_dissipation=terminal_induction_dissipation,
                                                               induction_dissipation_order=terminal_induction_dissipation_order,
                                                               project_b=terminal_project_b,
                                                               max_induction_subcycles=terminal_induction_max_subcycles)
                    last_induction_nsub = result.nsub
                    induction_subcycles_total += result.nsub
                    terminal_step_timing = result
                    terminal_timed = true
                elseif terminal_split && (terminal_dedicated_scratch || !(φ === ρtot && eint === nothing))
                    tvx = terminal_dedicated_scratch ? terminal_tvx : ρdm
                    tvy = terminal_dedicated_scratch ? terminal_tvy : ρtot
                    tvz = terminal_dedicated_scratch ? terminal_tvz :
                          (φ === ρtot && eint !== nothing) ? eint : φ
                    _terminal_velocity_step_cached!(be_mhd, s, dt_hydro, gamma_code, vcap_step,
                                                    tvx, tvy, tvz, terminal_vsmooth)
                else
                    _terminal_velocity_step!(be_mhd, s, dt_hydro, gamma_code, vcap_step)
                end
            end
            if terminal_implicit_diffuse
                tm += _time_phase(be_mhd, be_p) do
                    br_step = _device_brms!(be_mhd, s, s.scratch[1])
                    coeff = terminal_implicit_diffuse_fac * dt_hydro * br_step^2 /
                            max(gamma_code, eps(Float64)) / max(Float64(s.dx)^2, eps(Float64))
                    last_terminal_diffuse_coeff = coeff
                    _terminal_implicit_diffuse_B!(s, coeff)
                end
            else
                last_terminal_diffuse_coeff = 0.0
            end
            if chem_advect_species
                tm += _time_phase(be_mhd, be_p) do
                    MHDKernels.advect_species_fractions!(
                        HII, H2I, s.scratch[5], s.scratch[9], s, dt_hydro;
                        maximum_total=fh)
                    chem_helium && MHDKernels.advect_species_fraction!(
                        HeII, eint, s, dt_hydro; maximum_total=one(T) - fh)
                end
            end
        else
            handoff_active_this_step = handoff_remaining > 0
            handoff_alpha_step = 0.0
            if handoff_remaining > 0
                vcap_step = terminal_vcap
                handoff_alpha_step = _hybrid_handoff_alpha(hybrid_handoff_alpha,
                                                           handoff_remaining,
                                                           hybrid_handoff_steps)
                last_handoff_alpha = handoff_alpha_step
                tdrag += _time_phase(be_mhd, be_p) do
                    _terminal_handoff_momenta!(be_mhd, s, gamma_code, vcap_step,
                                               handoff_alpha_step)
                end
            end
            hubble_half_decay = sqrt(_hubble_momentum_decay(
                a_before, a_after_step, cosmological_mhd_aref)^(1 / nsub))
            for _ in 1:nsub
                dtsub = dt_hydro / nsub
                source_aware_drag = drag_enabled && drag_dt_mode === :godunov
                radiation_drag = drag_enabled && drag_dt_mode === :radiation
                exponential_drag = drag_enabled &&
                                   drag_dt_mode in (:aware, :semiimplicit, :hybrid)
                if hubble_half_decay < 1
                    tdrag += _time_phase(be_mhd, be_p) do
                        MHDKernels.apply_linear_drag!(s, hubble_half_decay)
                    end
                end
                if drag_enabled && !exponential_drag && !source_aware_drag && !radiation_drag
                    fhalf = exp(-0.5 * drag_impulse / nsub)
                    tdrag += _time_phase(be_mhd, be_p) do
                        _apply_compton_drag_mhd!(be_mhd, s, fhalf, drag_vx0, drag_vy0, drag_vz0)
                    end
                end
                godunov_before = radiation_step_timing.godunov_s
                if chem_advect_species
                    tm += _time_phase(be_mhd, be_p) do
                        MHDKernels.conserved_species_to_fractions!(
                            HII, H2I, s.U[1]; maximum_total=fh)
                        chem_helium && MHDKernels.conserved_species_to_fraction!(
                            HeII, s.U[1]; maximum_total=one(T) - fh)
                    end
                end
                step_elapsed = _time_phase(be_mhd, be_p) do
                    if radiation_drag
                        debug_radiation_now = debug_radiation_rho_min > 0 &&
                                              cyc >= debug_radiation_cycle_min
                        MHDKernels.step_radiation_godunov!(
                            s,radiation_workspace,dtsub;
                            drag_impulse=drag_impulse/nsub,
                            closure=last_radiation_closure,
                            ch=ch,glm_cr=glm_cr,integrator=mhd_integrator,
                            photon_state=radiation_photon_state,
                            debug_rho_min=debug_radiation_now ?
                                          debug_radiation_rho_min : 0.0,
                            debug_context=debug_radiation_now ?
                                          "cycle=$cyc z=$(a_to_z(a_before))" : "",
                            debug_correction_cfl=cyc >= debug_radiation_cycle_min ?
                                debug_radiation_correction_cfl : Inf,
                            timing=radiation_step_timing,
                            free_streaming_tolerance=radiation_free_stream_tolerance,
                            positivity_fallback=radiation_positivity_fallback,
                            positivity_max_fallback_passes=
                                radiation_positivity_max_fallback_passes,
                            positivity_wave_growth_limit=
                                radiation_positivity_wave_growth_limit,
                            positivity_max_subdivisions=
                                radiation_positivity_max_subdivisions,
                            checkpoint_fallback_on_rejection=
                                radiation_checkpoint_fallback,
                            energy_ledger=radiation_energy_ledger !== nothing &&
                                cyc % radiation_energy_ledger_every == 0 ?
                                radiation_energy_ledger : nothing)
                    elseif source_aware_drag
                        MHDKernels.step_drag_godunov!(
                            s, dtsub; drag_impulse=drag_impulse / nsub,
                            ch=ch, glm_cr=glm_cr, integrator=mhd_integrator,
                            target_velocity=(drag_vx0, drag_vy0, drag_vz0))
                    else
                        MHDKernels.step!(s, dtsub; ch=ch, glm_cr=glm_cr,
                                         integrator=mhd_integrator)
                    end
                end
                if radiation_drag
                    godunov_delta = radiation_step_timing.godunov_s-godunov_before
                    tm += godunov_delta
                    trad += max(step_elapsed-godunov_delta,0.0)
                else
                    tm += step_elapsed
                end
                if exponential_drag
                    tdrag += _time_phase(be_mhd, be_p) do
                        MHDKernels.apply_exponential_drag_increment!(
                            s, s.scratch, drag_impulse / nsub;
                            target_velocity=(drag_vx0, drag_vy0, drag_vz0))
                    end
                elseif drag_enabled && !source_aware_drag && !radiation_drag
                    fhalf = exp(-0.5 * drag_impulse / nsub)
                    tdrag += _time_phase(be_mhd, be_p) do
                        _apply_compton_drag_mhd!(be_mhd, s, fhalf, drag_vx0, drag_vy0, drag_vz0)
                    end
                end
                if hubble_half_decay < 1
                    tdrag += _time_phase(be_mhd, be_p) do
                        MHDKernels.apply_linear_drag!(s, hubble_half_decay)
                    end
                end
                if chem_advect_species
                    tm += _time_phase(be_mhd, be_p) do
                        MHDKernels.advect_species_fractions!(
                            HII, H2I, s.scratch[5], s.scratch[9], s, dtsub;
                            maximum_total=fh)
                        chem_helium && MHDKernels.advect_species_fraction!(
                            HeII, eint, s, dtsub; maximum_total=one(T) - fh)
                    end
                end
            end
            if glm_project_every > 0 && cyc % glm_project_every == 0
                tgp += _time_phase(be_mhd, be_p) do
                    _project_mhd_b!(be_mhd, s)
                end
                glm_projection_steps += 1
            end
            if handoff_active_this_step
                overlap = nothing
                tdrag += _time_phase(be_mhd, be_p) do
                    overlap = _terminal_handoff_momenta!(be_mhd, s, gamma_code, terminal_vcap,
                                                         0.0; measure=true)
                end
                last_handoff_error = overlap.relative
                max_handoff_error = isnan(max_handoff_error) ? overlap.relative :
                                    max(max_handoff_error, overlap.relative)
                max_handoff_settled_error = isnan(max_handoff_settled_error) ? overlap.relative :
                                            max(max_handoff_settled_error, overlap.relative)
                last_handoff_delta_v_rms = overlap.delta_v_rms
                last_handoff_terminal_v_rms = overlap.terminal_v_rms
                last_handoff_full_v_rms = overlap.full_v_rms
                handoff_index = handoff_steps_done + 1
                raw_handoff_displacement = overlap.delta_v_rms * dt_hydro / Float64(s.dx)
                last_handoff_displacement = handoff_alpha_step * raw_handoff_displacement
                max_handoff_displacement = max(max_handoff_displacement,
                                               last_handoff_displacement)
                cumulative_handoff_displacement += last_handoff_displacement
                if handoff_index == 1 || handoff_remaining == 1 ||
                   handoff_index % hybrid_handoff_diag_every == 0
                    @printf("  hybrid overlap step=%d remaining=%d alpha=%.6f rel=%.6e dv_rms=%.6e vt_rms=%.6e vf_rms=%.6e raw_dcell=%.6e correction_dcell=%.6e sum_correction_dcell=%.6e\n",
                            handoff_index, handoff_remaining, handoff_alpha_step, overlap.relative,
                            overlap.delta_v_rms, overlap.terminal_v_rms, overlap.full_v_rms,
                            raw_handoff_displacement, last_handoff_displacement,
                            cumulative_handoff_displacement)
                    flush(stdout)
                end
                overlap.relative <= hybrid_handoff_error_tol ||
                    error("hybrid handoff velocity mismatch $(overlap.relative) exceeds " *
                          "MHD_HYBRID_HANDOFF_ERROR_TOL=$(hybrid_handoff_error_tol)")
                cumulative_handoff_displacement <= hybrid_handoff_displacement_tol ||
                    error("hybrid handoff accumulated correction displacement " *
                          "$(cumulative_handoff_displacement) cells exceeds " *
                          "MHD_HYBRID_HANDOFF_DISPLACEMENT_TOL=" *
                          "$(hybrid_handoff_displacement_tol)")
                handoff_remaining -= 1
                handoff_steps_done += 1
            end
        end
        _debug_div("after_mhd", cyc, zrun ? a_before : 1.0)
        _debug_phase_state("after_mhd", cyc, a_before, dt, smax)
        if chem_mode !== :off
            if chem_coupling === :strang
                tchem, evolving_xe_mean, evolving_xheii_mean =
                    _chem_step_interval!(a_chem_mid, a_after_step, 0.5 * dt,
                                         chem_refresh_mean, measuring, evolving_xe_mean,
                                         evolving_xheii_mean,
                                         true)
                tc += tchem
            else
                tchem, evolving_xe_mean, evolving_xheii_mean =
                    _chem_step_interval!(a_before, a_after_step, dt,
                                         chem_refresh_mean, measuring, evolving_xe_mean,
                                         evolving_xheii_mean,
                                         false)
                tc += tchem
            end
        end
        _debug_div("after_chem", cyc, zrun ? a_before : 1.0)
        _debug_phase_state("after_chem", cyc, a_before, dt, smax)
        td = ta = tf = tg = tp = 0.0
        if gravity
            ngrav = gravity_subcycles_fixed
            if gravity_dt_boost > 0
                grav_max_dt = max(gravity_dt_boost * dt_explicit, eps(Float64))
                ngrav = max(ngrav, ceil(Int, dt / grav_max_dt))
            end
            if isfinite(particle_max_disp_cells)
                # Reuse the previous potential's acceleration bound rather than
                # adding a preflight deposit/FFT to the total-source path.
                if isfinite(particle_accel_max_dt) && particle_accel_max_dt > 0
                    ngrav = max(ngrav, ceil(Int, dt / particle_accel_max_dt))
                end
                _particle_speed2_k!(be_p)(ρdm, vx, vy, vz; ndrange=ncells)
                KA.synchronize(be_p)
                last_particle_vmax = sqrt(max(0.0, Float64(maximum(ρdm))))
                if last_particle_vmax > 0
                    particle_max_dt = particle_max_disp_cells / (Float64(N) * last_particle_vmax)
                    ngrav = max(ngrav, ceil(Int, dt / max(particle_max_dt, eps(Float64))))
                end
            end
            last_gravity_nsub = max(1, ngrav)
            dtdm = dt / last_gravity_nsub
            if grav_dm_weight == 0
                ta += _time_phase(be_p, be_mhd) do
                    if gravity_midpoint_source
                        if radiation_workspace === nothing
                            MHDKernels.predict_density_backward!(ρtot,s,0.5*dt_hydro)
                            _assemble_total_delta_k!(be_p)(ρtot,ρtot,ρdm,
                                grav_gas_weight,grav_dm_weight,Int(N);ndrange=ncells)
                        else
                            MHDKernels.predict_density_perturbation_backward!(ρtot,
                                radiation_workspace.density_perturbation,
                                s.U[2],s.U[3],s.U[4],s.dims;
                                dx=s.dx,dt_back=0.5*dt_hydro)
                            _assemble_total_from_gas_delta_k!(be_p)(ρtot,ρtot,ρdm,
                                grav_gas_weight,grav_dm_weight,Int(N);ndrange=ncells)
                        end
                    else
                        if radiation_workspace === nothing
                            _assemble_total_delta_k!(be_p)(ρtot,s.U[1],ρdm,
                                grav_gas_weight,grav_dm_weight,Int(N);ndrange=ncells)
                        else
                            _assemble_total_from_gas_delta_k!(be_p)(ρtot,
                                radiation_workspace.density_perturbation,ρdm,
                                grav_gas_weight,grav_dm_weight,Int(N);ndrange=ncells)
                        end
                    end
                end
                old_single_step = !gravity_midpoint_source && last_gravity_nsub == 1 &&
                                  gravity_subcycles_fixed == 1 && gravity_dt_boost == 0
                agrav = old_single_step ? a : (zrun ? 0.5 * (a_before + a_after_step) : 1.0)
                tf += _time_phase(be_p, be_mhd) do
                    gravcoef = zrun ? 1.5 * Om * agrav : 1.0
                    if fft_mode === :mps
                        PoissonKernels.fft_poisson_rfft!(φ, ρtot; G=gravcoef, a=1, boxsize=1)
                    elseif fft_mode === :ka
                        PoissonKernels.fft_poisson_rfft_ka!(φ, ρtot; G=gravcoef, a=1, boxsize=1)
                    else
                        PoissonKernels.fft_poisson_root_gpu!(φ, ρtot; G=gravcoef, a=1, boxsize=1)
                    end
                end
                if gravity_gas_kick
                    dtgas = zrun ? _gas_gravity_kick_dt(
                        dt, agrav, hub, chem_vunit, terminal_lbox_cm,
                        cosmological_mhd_aref) : dt
                    tg += _time_phase(be_mhd, be_p) do
                        MHDKernels.apply_cell_center_gravity!(s, φ, dtgas)
                    end
                end
                if isfinite(particle_max_disp_cells)
                    _grid_accel2_k!(be_p)(ρdm, φ, Int(N); ndrange=ncells)
                    KA.synchronize(be_p)
                    last_particle_gmax = sqrt(max(0.0, Float64(maximum(ρdm))))
                    maxdisp = particle_max_disp_cells / Float64(N)
                    particle_max_dt = _particle_dt_limit(last_particle_vmax, last_particle_gmax, maxdisp)
                    if isfinite(particle_max_dt) && particle_max_dt > 0
                        last_gravity_nsub = max(last_gravity_nsub,
                                                ceil(Int, dt / max(particle_max_dt, eps(Float64))))
                        dtdm = dt / last_gravity_nsub
                    end
                end
                for ig in 1:last_gravity_nsub
                    tp += _time_phase(be_p, be_mhd) do
                        if φprev === nothing
                            _push_particles!(px, py, pz, vx, vy, vz, φ;
                                             dtau=dtdm, N=N, wrap=particle_wrap,
                                             lattice_displacements=lattice_displacements)
                        else
                            θ = (ig - 0.5) / last_gravity_nsub
                            _push_particles_fused_global_phi_mix!(px, py, pz, vx, vy, vz, φprev, φ;
                                                                  theta=θ, dtau=dtdm, nc=(N,N,N), wrap=1)
                        end
                    end
                end
                if φprev !== nothing
                    tf += _time_phase(be_p, be_mhd) do
                        copyto!(φprev, φ)
                    end
                end
            else
                for ig in 1:last_gravity_nsub
                    td += _time_phase(be_p, be_mhd) do
                        midpoint_disp = gravity_midpoint_source ? 0.5 * dtdm : 0.0
                        _deposit_particles!(ρdm, px, py, pz, vx, vy, vz;
                                            N=N, disp=midpoint_disp,
                                            lattice_displacements=lattice_displacements)
                    end
                    ta += _time_phase(be_p, be_mhd) do
                        if gravity_midpoint_source && grav_gas_weight != 0
                            dt_back = dt - (ig - 0.5) * dtdm
                            dt_back_hydro = _hydro_subinterval(dt_hydro, dt, dt_back)
                            if radiation_workspace === nothing
                                MHDKernels.predict_density_backward!(ρtot,s,dt_back_hydro)
                                _assemble_total_delta_k!(be_p)(ρtot,ρtot,ρdm,
                                    grav_gas_weight,grav_dm_weight,Int(N);ndrange=ncells)
                            else
                                MHDKernels.predict_density_perturbation_backward!(ρtot,
                                    radiation_workspace.density_perturbation,
                                    s.U[2],s.U[3],s.U[4],s.dims;
                                    dx=s.dx,dt_back=dt_back_hydro)
                                _assemble_total_from_gas_delta_k!(be_p)(ρtot,ρtot,ρdm,
                                    grav_gas_weight,grav_dm_weight,Int(N);ndrange=ncells)
                            end
                        else
                            if radiation_workspace === nothing
                                _assemble_total_delta_k!(be_p)(ρtot,s.U[1],ρdm,
                                    grav_gas_weight,grav_dm_weight,Int(N);ndrange=ncells)
                            else
                                _assemble_total_from_gas_delta_k!(be_p)(ρtot,
                                    radiation_workspace.density_perturbation,ρdm,
                                    grav_gas_weight,grav_dm_weight,Int(N);ndrange=ncells)
                            end
                        end
                    end
                    old_single_step = last_gravity_nsub == 1 &&
                                      gravity_subcycles_fixed == 1 &&
                                      gravity_dt_boost == 0 &&
                                      !gravity_midpoint_source
                    agrav = old_single_step ? a :
                            (zrun ? a_before + (a_after_step - a_before) * ((ig - 0.5) / last_gravity_nsub) : 1.0)
                    tf += _time_phase(be_p, be_mhd) do
                        gravcoef = zrun ? 1.5 * Om * agrav : 1.0
                        if fft_mode === :mps
                            PoissonKernels.fft_poisson_rfft!(φ, ρtot; G=gravcoef, a=1, boxsize=1)
                        elseif fft_mode === :ka
                            PoissonKernels.fft_poisson_rfft_ka!(φ, ρtot; G=gravcoef, a=1, boxsize=1)
                        else
                            PoissonKernels.fft_poisson_root_gpu!(φ, ρtot; G=gravcoef, a=1, boxsize=1)
                        end
                    end
                    if isfinite(particle_max_disp_cells) && ig == 1
                        _grid_accel2_k!(be_p)(ρdm, φ, Int(N); ndrange=ncells)
                        KA.synchronize(be_p)
                        last_particle_gmax = sqrt(max(0.0, Float64(maximum(ρdm))))
                        maxdisp = particle_max_disp_cells / Float64(N)
                        particle_accel_max_dt = _particle_dt_limit(last_particle_vmax,
                                                                    last_particle_gmax, maxdisp)
                    end
                    if gravity_gas_kick
                        dtgas = zrun ? _gas_gravity_kick_dt(
                            dtdm, agrav, hub, chem_vunit, terminal_lbox_cm,
                            cosmological_mhd_aref) : dtdm
                        tg += _time_phase(be_mhd, be_p) do
                            MHDKernels.apply_cell_center_gravity!(s, φ, dtgas)
                        end
                    end
                    tp += _time_phase(be_p, be_mhd) do
                        _push_particles!(px, py, pz, vx, vy, vz, φ;
                                         dtau=dtdm, N=N, wrap=particle_wrap,
                                         lattice_displacements=lattice_displacements)
                    end
                end
            end
        else
            last_gravity_nsub = 0
        end
        _debug_div("after_gravity", cyc, zrun ? a_before : 1.0)
        _debug_phase_state("after_gravity", cyc, a_before, dt, smax)

        if measuring
            measured += 1
            measured_subcycles += nsub
            gravity_subcycles_total += last_gravity_nsub
            elapsed_time += dt
            phase[:cfl] += tcfl
            phase[:mhd] += tm
            phase[:radiation] += trad
            phase[:glm_projection] += tgp
            phase[:drag] += tdrag
            phase[:deposit] += td
            phase[:assemble] += ta
            phase[:fft] += tf
            phase[:gas_gravity] += tg
            phase[:push] += tp
            phase[:chem] += tc
            radiation_phase[:predictor] +=
                radiation_step_timing.predictor_s-radiation_timing_before.predictor
            radiation_phase[:corrector] +=
                radiation_step_timing.corrector_s-radiation_timing_before.corrector
            radiation_phase[:ledger] +=
                radiation_step_timing.ledger_s-radiation_timing_before.ledger
            if terminal_timed
                terminal_measured += 1
                terminal_phase[:force] += terminal_step_timing.force_s
                terminal_phase[:pressure_fft] += terminal_step_timing.pressure_fft_s
                terminal_phase[:finalize] += terminal_step_timing.finalize_s
                terminal_phase[:induction] += terminal_step_timing.induction_s
                terminal_phase[:projection] += terminal_step_timing.projection_s
            end
        end
        if cycle_print_every > 0 && (cyc <= warmup || cyc % cycle_print_every == 0)
            @printf("cycle %3d%s dt=%.3e hdt=%.3e pdt_s=%.3e expdt=%.3e nsub=%d gsub=%d pvmax=%.3e pgmax=%.3e regime=%s handoff=%d herr=%.3e va/cs=%.3e G/omA=%.3e idiff=%.3e drag %.4f f=%.6f G/H=%.3e  cfl %.4f mhd %.4f rad %.4f chem %.4f dep %.4f asm %.4f fft %.4f ggas %.4f push %.4f\n",
                    cyc, measuring ? "" : " warm", dt, dt_hydro, dtphys_source,
                    dt_explicit, nsub, last_gravity_nsub,
                    last_particle_vmax, last_particle_gmax, last_hybrid_regime,
                    last_handoff_remaining, last_handoff_error,
                    last_hybrid_va_cs, last_hybrid_drag_omega,
                    last_terminal_diffuse_coeff, tdrag, last_drag_f, last_drag_gammaH,
                    tcfl, tm, trad, tc, td, ta, tf, tg, tp)
            flush(stdout)
            if drag_dt_mode === :radiation
                rd = _radiation_scale_diagnostics(last_radiation_closure, N)
                free_bound=MHDKernels.radiation_free_streaming_bound(
                    2pi,drag_impulse/max(dt_hydro,eps(Float64)),
                    last_radiation_closure)
                @printf("  radiation R=%.6e lambda/L=%.6e q=[%.3e,%.3e] stream=[%.3e,%.3e] crad/cs=[%.3e,%.3e] free_bound=%.3e fast=%d spectral=%d\n",
                        last_radiation_closure.inertia_ratio,
                        last_radiation_closure.mean_free_path,
                        rd.qfund, rd.qnyquist, rd.stream_fund,
                        rd.stream_nyquist,
                        rd.crad_fund / max(last_radiation_closure.gas_sound_speed,
                                          eps(Float64)),
                        rd.crad_nyquist / max(last_radiation_closure.gas_sound_speed,
                                              eps(Float64)),
                        free_bound,radiation_step_timing.fast_steps,
                        radiation_step_timing.spectral_steps)
                flush(stdout)
            end
        end
        if check_every > 0 && (cyc % check_every == 0)
            ss = _assert_finite_state(s, "post-step", cyc, zrun ? a : NaN, dt, smax)
            @printf("  check rho=[%.6e, %.6e] E=[%.6e, %.6e] Bmax=%.6e\n",
                    Float64(ss.ρmin), Float64(ss.ρmax), Float64(ss.Emin), Float64(ss.Emax), ss.Bmax)
            flush(stdout)
        end
        if zrun && cyc > warmup
            a = a_after_step
            if measuring && print_every > 0 &&
               (measured % print_every == 0 || a >= a_end)
                @printf("  z=%.3f a=%.6e measured=%d\n", a_to_z(a), a, measured)
                flush(stdout)
            end
            if measuring && frame_next[] <= length(frame_zs) &&
               _reached_output_z(a_to_z(a), frame_zs[frame_next[]])
                _write_density_frame!(frame_dir, frame_hostrho, frame_hostslice, frame_hostproj,
                                      s.U[1], Int(N), a_to_z(a), cyc)
                while frame_next[] <= length(frame_zs) &&
                      _reached_output_z(a_to_z(a), frame_zs[frame_next[]])
                    frame_next[] += 1
                end
                @printf("  frame z=%.3f wrote density slice and maximum projection\n", a_to_z(a))
                flush(stdout)
            end
        end
        if measuring && (!isempty(history_path) || !isempty(mode_history_path) ||
                         !isempty(pk_path) || !isempty(pdf_path))
            znow = zrun ? a_to_z(a) : NaN
            due_every = pk_every > 0 && measured % pk_every == 0
            due_z = zrun && pk_next[] <= length(pk_zs) &&
                    _reached_output_z(znow, pk_zs[pk_next[]])
            due_adaptive = false
            adaptive_delta_b = NaN
            adaptive_xhii = NaN
            in_adaptive_window = !zrun ||
                                 (isfinite(znow) && adaptive_diag_zmin <= znow <= adaptive_diag_zmax)
            if adaptive_diag_rel > 0 && in_adaptive_window && !isempty(history_path)
                adaptive_delta_b,adaptive_xhii=_adaptive_diag_watch!(
                    be_p,s,ρtot,HII,H2I,HeII,fh;
                    z=znow,h=hub,Ob=Ob,
                    density_perturbation=radiation_workspace === nothing ? nothing :
                                         radiation_workspace.density_perturbation)
                due_adaptive = _rel_changed(adaptive_delta_b, adaptive_last_delta_b[], adaptive_diag_rel) ||
                               _rel_changed(adaptive_xhii, adaptive_last_xhii[], adaptive_diag_rel)
            end
            if due_every || due_z || due_adaptive
                full_diag = due_every || due_z
                tdiag = @elapsed begin
                    _emit_diagnostics!(be_p=be_p, s=s, ρdm=ρdm, scratch=ρtot,
                                       px=px, py=py, pz=pz, vx=vx, vy=vy, vz=vz,
                                       N=N, ncells=ncells, gravity=gravity, cycle=cyc,
                                       measured=measured, a=zrun ? a : 1.0,
                                       pk_path=full_diag ? pk_path : "", history_path=history_path,
                                       pdf_path=full_diag ? pdf_path : "", pdf_counts=pdf_counts,
                                       pdf_loglo=pdf_loglo, pdf_loghi=pdf_loghi,
                                       nmu=pk_nmu, nbins=pk_nbins, boxsize=boxsize,
                                       axis=pk_axis, kmax=pk_kmax,
                                       lattice_displacements=lattice_displacements,
                                       HII=HII, H2I=H2I, HeII=HeII, fh=fh,
                                       btrans0=Float64(st.btrans0),
                                       density_kmode=pmf_kmode,
                                       mode_history_path=mode_history_path,
                                       mode_kmode=pmf_kmode, hubble=hub, omega_b=Ob,
                                       radiation_workspace=radiation_workspace,
                                       radiation_photon_state=radiation_photon_state,
                                       radiation_drag_rate=drag_impulse /
                                           max(dt_hydro,eps(Float64)),
                                       temperature_velocity_unit2=chem_mode === :off ? NaN :
                                           chem_vunit^2 *
                                           (isfinite(cosmological_mhd_aref) ?
                                            cosmological_mhd_aref / a : 1.0))
                    KA.synchronize(be_mhd)
                    KA.synchronize(be_p)
                    # MPSGraph diagnostics create independent rFFT plans. This
                    # opt-in Metal barrier isolates them from the next terminal
                    # pressure/chemistry FFT step.
                    if full_diag && diag_clear_fft
                        PoissonKernels.clear_fft_scratch!(be_p)
                    end
                end
                if adaptive_diag_rel > 0
                    adaptive_last_delta_b[] = isfinite(adaptive_delta_b) ? adaptive_delta_b :
                                             (radiation_workspace === nothing ?
                                              _device_delta_rms!(be_p,ρtot,s.U[1],one(T)) :
                                              _device_delta_rms!(be_p,ρtot,
                                                  radiation_workspace.density_perturbation,
                                                  zero(T)))
                    adaptive_last_xhii[] = isfinite(adaptive_xhii) ? adaptive_xhii :
                                           _device_species_stats!(
                                               be_p, ρtot, s.U[1], HII, H2I,
                                               HeII, fh; z=znow,h=hub,Ob=Ob)[1]
                end
                if full_diag && !isempty(pk_path)
                    reason = due_z ? "z" : due_every ? "every" : due_adaptive ? "adaptive" : "manual"
                    @printf("  diag z=%.3f wrote P(k,mu) in %.3f s (%s)\n",
                            zrun ? znow : NaN, tdiag, reason)
                    flush(stdout)
                end
                _debug_div("after_diag", cyc, zrun ? a : 1.0)
                while zrun && pk_next[] <= length(pk_zs) &&
                      _reached_output_z(znow, pk_zs[pk_next[]])
                    pk_next[] += 1
                end
            end
        end
        if (checkpoint_every > 0 && measuring && measured % checkpoint_every == 0) ||
           (checkpoint_at_cycle > 0 && cyc == checkpoint_at_cycle)
            _save_checkpoint!(cyc)
        end
        if tfinal === nothing
            if zrun
                (a >= a_end || measured >= maxcycles) && break
            else
                cyc >= warmup + steps && break
            end
        else
            elapsed_time >= tfinal * (1 - 1e-9) && break
        end
    end

    if checkpoint_final && !isempty(checkpoint_write_path) && !checkpoint_disable_writes
        _save_checkpoint!(cyc)
    end

    if gravity && final_diag
        _deposit_particles!(ρdm, px, py, pz, vx, vy, vz;
                            N=N, disp=0,
                            lattice_displacements=lattice_displacements)
        KA.synchronize(be_p)
    end
    # Loop diagnostics are evaluated before advancing a. Emit a final target
    # after the last accepted step so z=zend P(k,mu) and PDF products exist.
    final_pk_due = zrun && pk_next[] <= length(pk_zs) &&
                   _reached_output_z(a_to_z(a), pk_zs[pk_next[]])
    if final_diag && final_pk_due
        _emit_diagnostics!(be_p=be_p, s=s, ρdm=ρdm, scratch=ρtot,
                           px=px, py=py, pz=pz, vx=vx, vy=vy, vz=vz,
                           N=N, ncells=ncells, gravity=gravity, cycle=cyc,
                           measured=measured, a=a,
                           pk_path=pk_path, history_path=history_path, pdf_path=pdf_path,
                           pdf_counts=pdf_counts, pdf_loglo=pdf_loglo, pdf_loghi=pdf_loghi,
                           nmu=pk_nmu, nbins=pk_nbins, boxsize=boxsize, axis=pk_axis,
                           kmax=pk_kmax,
                           lattice_displacements=lattice_displacements,
                           HII=HII, H2I=H2I, HeII=HeII, fh=fh,
                           btrans0=Float64(st.btrans0), density_kmode=pmf_kmode,
                           mode_history_path=mode_history_path,
                           mode_kmode=pmf_kmode, hubble=hub, omega_b=Ob,
                           radiation_workspace=radiation_workspace,
                           radiation_photon_state=radiation_photon_state,
                           radiation_drag_rate=drag_impulse /
                               max(dt_hydro,eps(Float64)),
                           temperature_velocity_unit2=chem_mode === :off ? NaN :
                               chem_vunit^2 * (isfinite(cosmological_mhd_aref) ?
                               cosmological_mhd_aref / a : 1.0))
        KA.synchronize(be_mhd)
        KA.synchronize(be_p)
        @printf("FINAL_DIAG emitted at z=%.6f\n", a_to_z(a))
        flush(stdout)
    end
    _log_device_memory("final")
    memory_stats = get(ENV, "MHD_MEMPROBE", "0") == "1" ? _device_memory_gib() :
                   (live=NaN, allocated=NaN, freed=NaN)
    total = sum(values(phase))
    mcells = ncells * measured / total / 1e6
    substep_mcells = ncells * measured_subcycles / total / 1e6
    br, divn, δb, δdm, chemstats =
        final_diag ? _final_device_stats!(be_p,s,ρdm,ρtot,HII,H2I,HeII,fh;
            density_perturbation=radiation_workspace === nothing ? nothing :
                                 radiation_workspace.density_perturbation,
            z=zrun ? a_to_z(a) : NaN,h=hub,Ob=Ob) :
        (NaN, NaN, NaN, NaN, ntuple(_ -> NaN, 12))
    gravity || (δdm = NaN)
    _write_final_b_field!(get(ENV, "MHD_FINAL_B_DIR", ""), s;
                          N=N, z=zrun ? a_to_z(a) : NaN,
                          cycle=measured, brms=br)
    delta2k_final = final_diag && pmf_init === :magpressure ?
                    (radiation_workspace === nothing ?
                     _device_density_cos_mode_amp!(be_p,ρtot,s.U[1],N,pmf_kmode,2) :
                     _device_density_cos_mode_amp!(be_p,ρtot,
                         radiation_workspace.density_perturbation,N,pmf_kmode,2;
                         mean=zero(T))) : NaN

    @printf("\nMEASURED cycles=%d subcycles=%d cells=%.3fM total=%.4f s throughput=%.1f macro-Mcell/s substep=%.1f Mcell/s last_dt=%.3e\n",
            measured, measured_subcycles, ncells/1e6, total, mcells, substep_mcells, last_dt)
    @printf("EVOLVED measured_time=%.6e\n", elapsed_time)
    zrun && @printf("COSMO final_z=%.6f final_a=%.8e cycles=%d\n", a_to_z(a), a, measured)
    @printf("PHASE s/cycle: cfl %.4f | mhd %.4f | radiation %.4f | glm_project %.4f | drag %.4f | chem %.4f | deposit %.4f | assemble %.4f | fft %.4f | gas_gravity %.4f | push %.4f | total %.4f\n",
            phase[:cfl]/measured, phase[:mhd]/measured, phase[:radiation]/measured,
            phase[:glm_projection]/measured,
            phase[:drag]/measured, phase[:chem]/measured, phase[:deposit]/measured,
            phase[:assemble]/measured, phase[:fft]/measured, phase[:gas_gravity]/measured,
            phase[:push]/measured, total/measured)
    if chem_profile_enabled && measured > 0
        chem_profile_sum = sum(values(chem_phase))
        @printf("CHEM_PROFILE s/cycle: unpack %.5f | smooth %.5f | solve %.5f | nonlocal %.5f | clamp %.5f | mean %.5f | repack %.5f | sum %.5f\n",
                chem_phase[:unpack]/measured, chem_phase[:smooth]/measured,
                chem_phase[:solve]/measured, chem_phase[:nonlocal]/measured,
                chem_phase[:clamp]/measured,
                chem_phase[:mean]/measured, chem_phase[:repack]/measured,
                chem_profile_sum/measured)
    end
    @printf("RADIATION_PROFILE s/cycle: predictor %.5f | corrector %.5f | ledger %.5f | sum %.5f | fast_steps %d | spectral_steps %d | fallback_steps %d | plm_hll_fallback_steps %d | subdivision_steps %d | global_fallback_steps %d | density_limited_stages %d | density_min_theta %.6f | free_tol %.3e\n",
            radiation_phase[:predictor]/measured,
            radiation_phase[:corrector]/measured,
            radiation_phase[:ledger]/measured,
            phase[:radiation]/measured,radiation_step_timing.fast_steps,
            radiation_step_timing.spectral_steps,radiation_step_timing.fallback_steps,
            radiation_step_timing.plm_hll_fallback_steps,
            radiation_step_timing.subdivision_steps,
            radiation_step_timing.global_fallback_steps,
            radiation_step_timing.density_limited_stages,
            radiation_workspace===nothing ? 1.0 :
                radiation_workspace.density_limiter_min_theta[],
            radiation_free_stream_tolerance)
    @printf("SHARES: cfl %.0f%% | mhd %.0f%% | radiation %.0f%% | glm_project %.0f%% | drag %.0f%% | chem %.0f%% | deposit %.0f%% | assemble %.0f%% | fft %.0f%% | gas_gravity %.0f%% | push %.0f%%\n",
            100*phase[:cfl]/total, 100*phase[:mhd]/total, 100*phase[:radiation]/total,
            100*phase[:glm_projection]/total,
            100*phase[:drag]/total, 100*phase[:chem]/total, 100*phase[:deposit]/total,
            100*phase[:assemble]/total, 100*phase[:fft]/total,
            100*phase[:gas_gravity]/total, 100*phase[:push]/total)
    @printf("GLM: ch_fac=%.6g cr=%.6g project_every=%d projections=%d\n",
            glm_ch_fac, glm_cr, glm_project_every, glm_projection_steps)
    @printf("SOURCE_CADENCE: max_dt_factor=%.6g limited_steps=%d\n",
            source_max_dt_factor, source_limited_steps)
    terminal_denom = max(terminal_measured, 1)
    @printf("TERMINAL measured_cycles=%d s/terminal-cycle: force %.6f | pressure_fft %.6f | finalize %.6f | induction %.6f | projection %.6f\n",
            terminal_measured, terminal_phase[:force]/terminal_denom,
            terminal_phase[:pressure_fft]/terminal_denom, terminal_phase[:finalize]/terminal_denom,
            terminal_phase[:induction]/terminal_denom, terminal_phase[:projection]/terminal_denom)
    @printf("MEMORY device_live_gib=%.3f allocated_gib=%.3f freed_gib=%.3f (Metal allocator; use vmmap physical footprint for MPS/driver memory)\n",
            memory_stats.live, memory_stats.allocated, memory_stats.freed)
    @printf("DIAG: Brms %.3e -> %.3e | divB*dx/Brms %.3e | delta_b_rms %.3e | delta_dm_rms %.3e\n",
            st.brms_measured, br, divn, δb, δdm)
    radiation_component_scale = radiation_energy_ledger === nothing ? NaN :
        abs(radiation_energy_ledger.kinetic_change) +
        abs(radiation_energy_ledger.internal_change) +
        abs(radiation_energy_ledger.magnetic_change)
    radiation_component_residual_norm = radiation_energy_ledger === nothing ? NaN :
        abs(radiation_energy_ledger.component_residual) /
        max(radiation_component_scale, eps(Float64))
    radiation_component_residual_abs_norm = radiation_energy_ledger === nothing ? NaN :
        radiation_energy_ledger.component_residual_abs /
        max(radiation_component_scale, eps(Float64))
    pmf_init === :magpressure && @printf("MODE: delta2k_amp %.8e\n", delta2k_final)
    drag_enabled && @printf("DRAG: mode=%s regime=%s boost=%.3f boosted_steps=%d subcycles_total=%d last_nsub=%d max_impulse=%.6e impulse_limited_steps=%d last_explicit_dt=%.3e last_dt=%.3e last_f=%.8f last_Gamma_over_H=%.6e rate_code=%.6e xe_mode=%s last_xe=%.6e terminal_gamma_scale=%.3e terminal_vcap=%.3e terminal_vmax=%.3e terminal_pressure_coeff=%.6e terminal_pressure_coeff_mode=%s induction_subcycles=%d last_induction_nsub=%d terminal_idiff_coeff=%.3e hybrid_va_cs=%.3e hybrid_drag_over_omega=%.3e handoff_events=%d handoff_steps=%d handoff_remaining=%d handoff_alpha_max=%.3f handoff_ramp=cosine handoff_alpha_last=%.3f handoff_error_last=%.6e handoff_error_max=%.6e handoff_error_settled_max=%.6e handoff_dv_rms=%.6e handoff_vt_rms=%.6e handoff_vf_rms=%.6e handoff_error_tol=%.6e handoff_correction_dcell_last=%.6e handoff_correction_dcell_max=%.6e handoff_correction_dcell_sum=%.6e handoff_correction_dcell_tol=%.6e\n",
                            String(drag_dt_mode), last_hybrid_regime, drag_dt_boost, boosted_steps, subcycles_total, last_nsub,
                            drag_max_impulse, drag_impulse_limited_steps,
                            last_dt_explicit, last_dt, last_drag_f, last_drag_gammaH, drag_rate_code,
                            drag_xe_env, last_drag_xe,
                            terminal_gamma_scale, terminal_vcap, last_terminal_vmax,
                            terminal_pressure_coeff_step,
                            terminal_pressure_coeff_auto ? "auto" : "fixed",
                            induction_subcycles_total, last_induction_nsub, last_terminal_diffuse_coeff,
                            last_hybrid_va_cs, last_hybrid_drag_omega,
                            handoff_events, handoff_steps_done, handoff_remaining,
                            hybrid_handoff_alpha, last_handoff_alpha,
                            last_handoff_error, max_handoff_error, max_handoff_settled_error,
                            last_handoff_delta_v_rms,
                            last_handoff_terminal_v_rms, last_handoff_full_v_rms,
                            hybrid_handoff_error_tol, last_handoff_displacement,
                            max_handoff_displacement, cumulative_handoff_displacement,
                            hybrid_handoff_displacement_tol)
    if drag_dt_mode === :radiation
        rd = _radiation_scale_diagnostics(last_radiation_closure, N)
        @printf("RADIATION: response_model=%s R=%.8e lambda_over_box=%.8e q_fund=%.8e q_nyquist=%.8e stream_fund=%.8e stream_nyquist=%.8e crad_fund=%.8e crad_nyquist=%.8e gas_cs=%.8e bridge_q2=%.8e longitudinal_viscosity=%.8e max_dlna=%.8e highz_max_dlna=%.8e highz_zmin=%.8e nonlinear_cfl=%.8e workspace_bytes=%d\n",
                String(radiation_response_model),
                last_radiation_closure.inertia_ratio,
                last_radiation_closure.mean_free_path,
                rd.qfund, rd.qnyquist, rd.stream_fund,
                rd.stream_nyquist, rd.crad_fund, rd.crad_nyquist,
                last_radiation_closure.gas_sound_speed,
                last_radiation_closure.bridge_q2,
                last_radiation_closure.longitudinal_viscosity,
                radiation_max_dlna,radiation_highz_max_dlna,
                radiation_highz_zmin,radiation_nonlinear_cfl,
                radiation_workspace_bytes)
        if radiation_energy_ledger !== nothing
            @printf("RADIATION_ENERGY: samples=%d gas_total_change=%.8e kinetic_change=%.8e internal_change=%.8e magnetic_change=%.8e photon_change=%.8e resolved_coupled_change=%.8e component_residual=%.8e component_residual_abs=%.8e component_residual_norm=%.8e component_residual_abs_norm=%.8e photon_closure=%s\n",
                    radiation_energy_ledger.samples,
                    radiation_energy_ledger.total_change,
                    radiation_energy_ledger.kinetic_change,
                    radiation_energy_ledger.internal_change,
                    radiation_energy_ledger.magnetic_change,
                    radiation_energy_ledger.photon_change,
                    radiation_energy_ledger.resolved_coupled_change,
                    radiation_energy_ledger.component_residual,
                    radiation_energy_ledger.component_residual_abs,
                    radiation_component_residual_norm,
                    radiation_component_residual_abs_norm,
                    radiation_response_model in (
                        :moments,:photon_moments,:moments_exponential,
                        :photon_moments_exponential) ?
                        "moments" : "untracked")
        end
    end
    gravity && @printf("GRAVITY: source=%s gas_kick=%s weights=(gas %.6g, dm %.6g) midpoint_source=%s lattice_displacements=%s fixed_subcycles=%d dt_boost=%.3f particle_max_disp=%.6g last_particle_vmax=%.6e last_particle_gmax=%.6e subcycles_total=%d last_nsub=%d\n",
                       String(gravity_source), string(gravity_gas_kick),
                       Float64(grav_gas_weight), Float64(grav_dm_weight),
                       string(gravity_midpoint_source), string(lattice_displacements),
                       gravity_subcycles_fixed, gravity_dt_boost, particle_max_disp_cells,
                       last_particle_vmax, last_particle_gmax, gravity_subcycles_total, last_gravity_nsub)
    chem_mode !== :off && @printf("CHEM: xHII mean/min/max %.6e %.6e %.6e | fH2 mean/min/max %.6e %.6e %.6e | xHeII mean/min/max %.6e %.6e %.6e | xe_total mean/min/max %.6e %.6e %.6e\n",
                                  chemstats...)
    flush(stdout)

    out = get(ENV, "MHD_OUT_TSV", "")
    if !isempty(out)
        exists = isfile(out)
        open(out, "a") do io
            if !exists
                println(io, join(("timestamp","backend","N","recon","riemann","glm_ch_fac","glm_cr",
                                  "chem","chem_coupling","chem_expansion","chem_helium",
                                  "chem_rate_tables","chem_itcap","chem_dtfrac",
                                  "initial_p0","initial_p0_mode",
                                  "source_max_dt_factor","source_limited_steps",
                                  "hubble_h","omega_m","omega_l","omega_r",
                                  "pmf_kcut","pmf_kmax","pmf_hard_cutoff",
                                  "pmf_normalization_kind","pmf_normalization_z",
                                  "cosmological_mhd_scaling",
                                  "pmf_target_comoving_ng","pmf_initial_reference_comoving_ng",
                                  "pmf_initial_total_comoving_ng",
                                  "pmf_preserve_low_kmax","pmf_preserve_low_kcut",
                                  "pmf_mode_lock_n","pmf_high_mode_lock_n",
                                  "pmf_curl_symbol","pmf_curl_reference_n","pmf_fixed_mode_power",
                                  "pmf_mode_realization","pmf_realization_box",
                                  "pmf_seed","pmf_min_cells_per_wavelength",
                                  "boxsize","pk_nbins","pk_kmax",
                                  "glm_project_every","glm_projection_steps",
                                  "drag","drag_dt_mode","drag_dt_boost","drag_boosted_steps",
                                  "drag_subcycles_total","drag_last_nsub","drag_max_impulse","drag_impulse_limited_steps",
                                  "drag_last_f","drag_last_gamma_over_H","drag_xe",
                                  "drag_helium_electrons","drag_last_xe","drag_rate_code",
                                  "radiation_response_model","radiation_free_stream_tolerance",
                                  "radiation_fast_steps","radiation_spectral_steps",
                                  "radiation_fallback_steps","radiation_plm_hll_fallback_steps",
                                  "radiation_subdivision_steps","radiation_global_fallback_steps",
                                  "radiation_density_limited_stages","radiation_density_min_theta",
                                  "radiation_energy_samples",
                                  "radiation_total_change","radiation_kinetic_change",
                                  "radiation_internal_change","radiation_magnetic_change",
                                  "radiation_photon_change","radiation_resolved_coupled_change",
                                  "radiation_component_residual",
                                  "radiation_component_residual_abs",
                                  "radiation_component_residual_norm",
                                  "radiation_component_residual_abs_norm",
                                  "gravity","gravity_gas_kick","gravity_source",
                                  "gravity_phi_interp","gravity_midpoint_source",
                                  "particle_lattice_displacements",
                                  "gravity_gas_weight","gravity_dm_weight",
                                  "gravity_subcycles_fixed","gravity_dt_boost","particle_max_disp_cells",
                                  "particle_last_vmax","particle_last_gmax","gravity_subcycles_total","gravity_last_nsub",
                                  "terminal_gamma_scale","terminal_vcap","terminal_diff_cfl",
                                  "terminal_accuracy_cfl","terminal_accuracy_pressure",
                                  "terminal_implicit_diffuse","terminal_implicit_diffuse_fac","terminal_idiff_coeff",
                                  "terminal_pressure_implicit","terminal_pressure_coeff",
                                  "terminal_pressure_coeff_mode",
                                  "terminal_rho_rel_limit","terminal_b_rel_limit",
                                  "terminal_max_disp_cells","terminal_last_vmax",
                                  "terminal_induction","terminal_induction_cfl","terminal_induction_dissipation",
                                  "terminal_induction_dissipation_order","terminal_project_b",
                                  "terminal_induction_max_subcycles","terminal_induction_subcycles_total",
                                  "terminal_induction_last_nsub",
                                  "hybrid_regime","hybrid_va_cs","hybrid_drag_over_omega",
                                  "hybrid_exit_drag_over_omega","hybrid_enter_drag_over_omega",
                                  "hybrid_check_every","hybrid_confirm_checks","hybrid_reenter_terminal",
                                  "hybrid_handoff_events","hybrid_handoff_steps_done","hybrid_handoff_remaining",
                                  "hybrid_handoff_alpha_max","hybrid_handoff_ramp","hybrid_handoff_alpha_last",
                                  "hybrid_handoff_error_last","hybrid_handoff_error_max",
                                  "hybrid_handoff_error_settled_max",
                                  "hybrid_handoff_delta_v_rms","hybrid_handoff_terminal_v_rms",
                                  "hybrid_handoff_full_v_rms","hybrid_handoff_error_tol",
                                  "hybrid_handoff_correction_dcell_last","hybrid_handoff_correction_dcell_max",
                                  "hybrid_handoff_correction_dcell_sum","hybrid_handoff_correction_dcell_tol",
                                  "brms0","brms","divBdx_over_brms","delta_b_rms","delta_dm_rms","delta2k_amp",
                                  "xHII_mean","xHII_min","xHII_max","fH2_mean","fH2_min","fH2_max",
                                  "xHeII_mean","xHeII_min","xHeII_max",
                                  "xe_total_mean","xe_total_min","xe_total_max",
                                  "cycles","measured_subcycles","measured_time","zstart","zend","final_z","total_s",
                                  "throughput_macro_mcells_s","throughput_substep_mcells_s",
                                  "last_dt","last_explicit_dt",
                                  "cfl_s_cyc","mhd_s_cyc","radiation_s_cyc",
                                  "glm_projection_s_cyc",
                                  "drag_s_cyc","chem_s_cyc","deposit_s_cyc","assemble_s_cyc",
                                  "fft_s_cyc","gas_gravity_s_cyc","push_s_cyc","terminal_measured_cycles",
                                  "terminal_force_s_cyc","terminal_pressure_fft_s_cyc","terminal_finalize_s_cyc",
                                  "terminal_induction_s_cyc","terminal_projection_s_cyc",
                                  "device_live_gib","device_allocated_gib","device_freed_gib"),
                                 '\t'))
            end
            println(io, join((Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
                              String(BACKEND_NAME), N, String(recon), String(riemann),
                              glm_ch_fac, glm_cr, String(chem_mode), String(chem_coupling),
                              chem_expansion, chem_helium,
                              chem_rate_tables_enabled, chem_itcap,
                              chem_dtfrac, p0,
                              !isempty(initial_state_dir) ? "imported" :
                                  (p0_auto ? "cmb" : "fixed"),
                              source_max_dt_factor,
                              source_limited_steps, hub, Om, OL, Or,
                              kcut, pmf_kmax === nothing ? "" : pmf_kmax, pmf_kmax !== nothing,
                              st.normalization_kind, pmf_normalization_z,
                              cosmological_mhd_scaling,
                              pmf_b0_ng, pmf_initial_reference_comoving_ng,
                              pmf_initial_total_comoving_ng,
                              pmf_preserve_low_kmax === nothing ? "" : pmf_preserve_low_kmax,
                              pmf_preserve_low_kcut === nothing ? "" : pmf_preserve_low_kcut,
                              pmf_mode_lock_n === nothing ? 0 : pmf_mode_lock_n,
                              pmf_high_mode_lock_n === nothing ? 0 : pmf_high_mode_lock_n,
                              String(pmf_curl_symbol),
                              pmf_curl_reference_n === nothing ? 0 : pmf_curl_reference_n,
                              pmf_fixed_mode_power,
                              String(pmf_mode_realization), pmf_realization_box,
                              seed,
                              pmf_min_cells_per_wavelength,
                              boxsize, pk_nbins, pk_kmax === nothing ? "" : pk_kmax,
                              glm_project_every, glm_projection_steps,
                              drag_enabled, String(drag_dt_mode),
                              drag_dt_boost, boosted_steps, subcycles_total, last_nsub,
                              drag_max_impulse, drag_impulse_limited_steps,
                              last_drag_f, last_drag_gammaH, drag_xe_env, drag_helium_electrons,
                              last_drag_xe, drag_rate_code,
                              String(radiation_response_model),
                              radiation_free_stream_tolerance,
                              radiation_step_timing.fast_steps,
                              radiation_step_timing.spectral_steps,
                              radiation_step_timing.fallback_steps,
                              radiation_step_timing.plm_hll_fallback_steps,
                              radiation_step_timing.subdivision_steps,
                              radiation_step_timing.global_fallback_steps,
                              radiation_step_timing.density_limited_stages,
                              radiation_workspace === nothing ? 1.0 :
                                  radiation_workspace.density_limiter_min_theta[],
                              radiation_energy_ledger === nothing ? 0 : radiation_energy_ledger.samples,
                              radiation_energy_ledger === nothing ? NaN : radiation_energy_ledger.total_change,
                              radiation_energy_ledger === nothing ? NaN : radiation_energy_ledger.kinetic_change,
                              radiation_energy_ledger === nothing ? NaN : radiation_energy_ledger.internal_change,
                              radiation_energy_ledger === nothing ? NaN : radiation_energy_ledger.magnetic_change,
                              radiation_energy_ledger === nothing ? NaN : radiation_energy_ledger.photon_change,
                              radiation_energy_ledger === nothing ? NaN : radiation_energy_ledger.resolved_coupled_change,
                              radiation_energy_ledger === nothing ? NaN : radiation_energy_ledger.component_residual,
                              radiation_energy_ledger === nothing ? NaN : radiation_energy_ledger.component_residual_abs,
                              radiation_component_residual_norm,
                              radiation_component_residual_abs_norm,
                              gravity, gravity_gas_kick, String(gravity_source),
                              gravity_phi_interp, gravity_midpoint_source, lattice_displacements,
                              grav_gas_weight, grav_dm_weight, gravity_subcycles_fixed,
                              gravity_dt_boost, particle_max_disp_cells, last_particle_vmax,
                              last_particle_gmax, gravity_subcycles_total, last_gravity_nsub,
                              terminal_gamma_scale, terminal_vcap, terminal_diff_cfl,
                              terminal_accuracy_cfl, terminal_accuracy_pressure,
                              terminal_implicit_diffuse, terminal_implicit_diffuse_fac, last_terminal_diffuse_coeff,
                              terminal_pressure_implicit, terminal_pressure_coeff_step,
                              terminal_pressure_coeff_auto ? "auto" : "fixed",
                              terminal_rho_rel_limit, terminal_b_rel_limit,
                              terminal_max_disp_cells, last_terminal_vmax,
                              String(terminal_induction), terminal_induction_cfl,
                              terminal_induction_dissipation, terminal_induction_dissipation_order,
                              terminal_project_b, terminal_induction_max_subcycles,
                              induction_subcycles_total, last_induction_nsub,
                              last_hybrid_regime, last_hybrid_va_cs, last_hybrid_drag_omega,
                              hybrid_exit_drag_omega, hybrid_enter_drag_omega,
                              hybrid_check_every, hybrid_confirm_checks, hybrid_reenter_terminal,
                              handoff_events, handoff_steps_done, handoff_remaining,
                              hybrid_handoff_alpha, "cosine", last_handoff_alpha,
                              last_handoff_error, max_handoff_error, max_handoff_settled_error,
                              last_handoff_delta_v_rms,
                              last_handoff_terminal_v_rms, last_handoff_full_v_rms,
                              hybrid_handoff_error_tol, last_handoff_displacement,
                              max_handoff_displacement, cumulative_handoff_displacement,
                              hybrid_handoff_displacement_tol,
                              st.brms_measured, br, divn, δb, δdm, delta2k_final,
                              chemstats...,
                              measured, measured_subcycles, elapsed_time, zstart, zend, zrun ? a_to_z(a) : NaN,
                              total, mcells, substep_mcells,
                              last_dt, last_dt_explicit,
                              phase[:cfl]/measured, phase[:mhd]/measured,
                              phase[:radiation]/measured,
                              phase[:glm_projection]/measured, phase[:drag]/measured, phase[:chem]/measured,
                              phase[:deposit]/measured, phase[:assemble]/measured, phase[:fft]/measured,
                              phase[:gas_gravity]/measured, phase[:push]/measured, terminal_measured,
                              terminal_phase[:force]/terminal_denom,
                              terminal_phase[:pressure_fft]/terminal_denom,
                              terminal_phase[:finalize]/terminal_denom,
                              terminal_phase[:induction]/terminal_denom,
                              terminal_phase[:projection]/terminal_denom,
                              memory_stats.live, memory_stats.allocated, memory_stats.freed), '\t'))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
