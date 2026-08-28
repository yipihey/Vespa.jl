# ── Primordial magnetic field initial conditions ─────────────────────────────
# Non-helical Gaussian PMFs with a Batchelor magnetic-energy spectrum.  We build
# a random vector potential A(k), take B(k) = i s(k) × A(k), then inverse-transform
# B.  Here s(k)=sin(k dx)/dx is the central-difference derivative symbol, so the
# diagnostic divergence uses the same operator and s·B vanishes to roundoff.

export PMFSpectrumSpec, pmf_batchelor_band, pmf_batchelor_field,
       pmf_multiband_manifest, pmf_resolved_kmax,
       init_pmf_batchelor!, init_pmf_batchelor_tiled!

"""A resolution-independent definition of a physical Batchelor PMF spectrum."""
Base.@kwdef struct PMFSpectrumSpec
    box::Float64
    reference_brms::Float64
    normalization_kind::Symbol = :reference_band_rms
    normalization_redshift::Float64 = NaN
    seed::Int = 1
    kcut::Float64
    reference_kmax::Float64
    reference_kcut::Float64 = kcut
    energy_index::Float64 = 4.0
    cutoff_order::Float64 = 2.0
    minimum_cells_per_wavelength::Float64 = 16.0
    mode_lock_n::Int
    curl_symbol::Symbol = :centered
end

function _pmf_normalization_kind(kind, preserve_low::Bool)
    value = Symbol(lowercase(String(kind)))
    value = value in (:total, :total_band) ? :total_band_rms : value
    value = value in (:reference, :reference_band, :low_band) ?
            :reference_band_rms : value
    value === :auto && return preserve_low ? :reference_band_rms : :total_band_rms
    value in (:total_band_rms, :reference_band_rms) || error(
        "normalization_kind must be :total_band_rms or :reference_band_rms")
    value === :reference_band_rms && !preserve_low && error(
        "reference-band normalization requires preserve_low_kmax")
    return value
end

"""Largest physical wavenumber resolved at the requested cells per wavelength."""
function pmf_resolved_kmax(dims::NTuple{3,Int}, box::Real,
                           minimum_cells_per_wavelength::Real)
    cells = Float64(minimum_cells_per_wavelength)
    cells > 0 || error("minimum_cells_per_wavelength must be positive")
    return 2π * minimum(dims) / (Float64(box) * cells)
end

"""
    pmf_multiband_manifest(spec, edges)

Describe disjoint physical wavenumber bands and the smallest even cubic grid
that resolves each upper edge.  Every band uses the same reference low modes
and normalization from `spec`.
"""
function pmf_multiband_manifest(spec::PMFSpectrumSpec, edges)
    ks = Float64.(collect(edges))
    length(ks) >= 2 || error("multiband edges require at least two values")
    first(ks) >= 0 || error("multiband edges must be nonnegative")
    all(diff(ks) .> 0) || error("multiband edges must be strictly increasing")
    spec.reference_kmax <= ks[2] ||
        error("the first band must contain reference_kmax=$(spec.reference_kmax)")
    return [begin
        raw_n = ceil(Int, khi * spec.box * spec.minimum_cells_per_wavelength / (2π))
        required_n = iseven(raw_n) ? raw_n : raw_n + 1
        (band=i, kmin=klo, kmax=khi, required_n=required_n,
         cells_per_wavelength=2π * required_n / (spec.box * khi),
         box=spec.box, reference_brms=spec.reference_brms,
         normalization_kind=spec.normalization_kind,
         normalization_redshift=spec.normalization_redshift,
         reference_kmax=spec.reference_kmax, seed=spec.seed,
         mode_lock_n=spec.mode_lock_n)
    end for (i, (klo, khi)) in enumerate(zip(ks[1:end-1], ks[2:end]))]
end

"""Generate one band from a resolution-independent PMF spectrum definition."""
function pmf_batchelor_band(spec::PMFSpectrumSpec, dims::NTuple{3,Int};
                            kmin::Real=0, kmax::Real=spec.reference_kmax)
    minimum(dims) >= spec.mode_lock_n ||
        error("grid $dims is smaller than mode_lock_n=$(spec.mode_lock_n)")
    return pmf_batchelor_field(dims; box=spec.box, brms=spec.reference_brms,
        seed=spec.seed, kcut=spec.kcut, kmin=kmin, kmax=kmax,
        energy_index=spec.energy_index, cutoff_order=spec.cutoff_order,
        mode_lock_n=spec.mode_lock_n, preserve_low_kmax=spec.reference_kmax,
        preserve_low_kcut=spec.reference_kcut, curl_symbol=spec.curl_symbol,
        normalization_kind=spec.normalization_kind,
        minimum_cells_per_wavelength=spec.minimum_cells_per_wavelength)
end

@inline _fftfreq_i(i::Int, n::Int, L::Float64) =
    (i - 1 <= n ÷ 2 ? i - 1 : i - 1 - n) * (2π / L)

@inline _rfftfreq_i(i::Int, n::Int, L::Float64) = (i - 1) * (2π / L)

@inline _pmf_mode_i(i::Int, n::Int) = i - 1 <= n ÷ 2 ? i - 1 : i - 1 - n

function _random_vector_potential_k(dims::NTuple{3,Int}, seed::Integer)
    nx, ny, nz = dims
    rng = MersenneTwister(seed)
    a = randn(rng, Float32, nx, ny, nz)
    ax = FFTW.rfft(a, (1,2,3))
    randn!(rng, a)
    ay = FFTW.rfft(a, (1,2,3))
    randn!(rng, a)
    az = FFTW.rfft(a, (1,2,3))
    return ax, ay, az
end

@inline function _pmf_mix64(x::UInt64)
    z = x + 0x9e3779b97f4a7c15
    z = xor(z, z >> 30) * 0xbf58476d1ce4e5b9
    z = xor(z, z >> 27) * 0x94d049bb133111eb
    return xor(z, z >> 31)
end

@inline function _pmf_hash_word(seed::Integer, qx::Int, qy::Int, qz::Int,
                                component::Int, lane::Int)
    h = _pmf_mix64(reinterpret(UInt64, Int64(seed)))
    for value in (qx, qy, qz, component, lane)
        h = _pmf_mix64(xor(h, reinterpret(UInt64, Int64(value))))
    end
    return h
end

@inline function _pmf_hash_uniform(seed::Integer, qx::Int, qy::Int, qz::Int,
                                   component::Int, lane::Int)
    word = _pmf_hash_word(seed, qx, qy, qz, component, lane)
    return (Float64(word >> 11) + 0.5) * 0x1.0p-53
end

@inline function _pmf_hash_normal(seed::Integer, qx::Int, qy::Int, qz::Int,
                                  component::Int; real_only::Bool=false)
    u1 = _pmf_hash_uniform(seed, qx, qy, qz, component, 1)
    u2 = _pmf_hash_uniform(seed, qx, qy, qz, component, 2)
    radius = sqrt(-2 * log(u1))
    if real_only
        return ComplexF32(Float32(radius * cospi(2u2)), 0)
    end
    phase = 2π * u2
    return ComplexF32(Float32(radius * cos(phase)), Float32(radius * sin(phase)))
end

"""
Build a vector-potential realization whose random coefficients are keyed by
physical Fourier indices rather than array traversal.  Boxes whose lengths are
integer subdivisions of `reference_box` therefore receive identical phase and
polarization on every shared, non-Nyquist physical mode.
"""
function _physical_hash_vector_potential_k(dims::NTuple{3,Int}, seed::Integer,
                                           box::Float64, reference_box::Float64;
                                           zero_phase::Bool=false)
    nx, ny, nz = dims
    ratio = reference_box / box
    qscale = round(Int, ratio)
    qscale > 0 && isapprox(ratio, qscale; rtol=0,
                           atol=32eps(Float64) * max(1, abs(ratio))) || error(
        "physical-hash reference_box/box must be a positive integer; got " *
        "reference_box=$reference_box box=$box")
    nxh = nx ÷ 2 + 1
    fields = ntuple(_ -> Array{ComplexF32}(undef, nxh, ny, nz), 3)
    @inbounds for kk in 1:nz, jj in 1:ny, ii in 1:nxh
        mx = ii - 1
        my = _pmf_mode_i(jj, ny)
        mz = _pmf_mode_i(kk, nz)
        boundary_x = ii == 1 || ii == nxh
        conjugate_mode = false
        real_only = false
        key_my = my
        key_mz = mz
        if boundary_x
            pair_j = mod(-my, ny) + 1
            pair_k = mod(-mz, nz) + 1
            if (pair_j, pair_k) < (jj, kk)
                conjugate_mode = true
                key_my = _pmf_mode_i(pair_j, ny)
                key_mz = _pmf_mode_i(pair_k, nz)
            elseif pair_j == jj && pair_k == kk
                real_only = true
            end
        end
        qx = qscale * mx
        qy = qscale * key_my
        qz = qscale * key_mz
        for component in 1:3
            value = _pmf_hash_normal(seed, qx, qy, qz, component;
                                     real_only=zero_phase || real_only)
            fields[component][ii,jj,kk] = conjugate_mode ? conj(value) : value
        end
    end
    return fields
end

function _vector_potential_k(dims::NTuple{3,Int}, seed::Integer, box::Float64,
                             mode_realization::Symbol, realization_box::Float64)
    mode_realization === :legacy && return _random_vector_potential_k(dims, seed)
    mode_realization === :physical_hash && return _physical_hash_vector_potential_k(
        dims, seed, box, realization_box)
    mode_realization === :physical_hash_zero_phase &&
        return _physical_hash_vector_potential_k(
            dims, seed, box, realization_box; zero_phase=true)
    error("mode_realization must be :legacy, :physical_hash, or " *
          ":physical_hash_zero_phase")
end

function _rfft_zero_pad_complex(ak::Array{ComplexF32,3}, source_dims::NTuple{3,Int},
                                target_dims::NTuple{3,Int})
    ox, oy, oz = source_dims
    nx, ny, nz = target_dims
    out = zeros(ComplexF32, nx ÷ 2 + 1, ny, nz)
    scale = Float32((nx * ny * nz) / (ox * oy * oz))
    @inbounds for kk in 1:oz
        mz = _pmf_mode_i(kk, oz)
        tk = mz >= 0 ? mz + 1 : nz + mz + 1
        for jj in 1:oy
            my = _pmf_mode_i(jj, oy)
            tj = my >= 0 ? my + 1 : ny + my + 1
            for ii in 1:(ox ÷ 2 + 1)
                mx = ii - 1
                out[mx + 1, tj, tk] = scale * ak[ii,jj,kk]
            end
        end
    end
    return out
end

function _periodic_tile_field(a::Array{Float32,3}, dims::NTuple{3,Int})
    sx, sy, sz = size(a)
    nx, ny, nz = dims
    nx % sx == 0 || error("tile x-size $sx must divide target x-size $nx")
    ny % sy == 0 || error("tile y-size $sy must divide target y-size $ny")
    nz % sz == 0 || error("tile z-size $sz must divide target z-size $nz")
    out = Array{Float32}(undef, dims)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        out[i,j,k] = a[mod1(i, sx), mod1(j, sy), mod1(k, sz)]
    end
    return out
end

function _install_pmf_field!(s::MHDState{T}, Bx::Array{Float32,3}, By::Array{Float32,3},
                             Bz::Array{Float32,3}; rho0::Real = 1,
                             p0::Real = 1) where {T}
    nc = ncells(s)
    ρ = T(rho0); p = T(p0)
    Ehost = Vector{T}(undef, nc)
    Bxv = vec(Bx); Byv = vec(By); Bzv = vec(Bz)
    Threads.@threads for idx in 1:nc
        @inbounds begin
            bx = T(Bxv[idx]); by = T(Byv[idx]); bz = T(Bzv[idx])
            Ehost[idx] = p/(s.γ - one(T)) + T(0.5) * (bx*bx + by*by + bz*bz)
        end
    end
    fill!(s.U[1], ρ)
    fill!(s.U[2], zero(T)); fill!(s.U[3], zero(T)); fill!(s.U[4], zero(T))
    copyto!(s.U[5], Ehost)
    copyto!(s.U[6], Bxv); copyto!(s.U[7], Byv); copyto!(s.U[8], Bzv)
    fill!(s.U[9], zero(T))
    KA.synchronize(s.be)
    return nothing
end

function _pmf_fft_threads()
    env = get(ENV, "MHD_PMF_FFT_THREADS", get(ENV, "CIC_FFT_THREADS", ""))
    n = isempty(strip(env)) ? min(Sys.CPU_THREADS, 4) : parse(Int, env)
    return max(1, n)
end

function _pmf_kspace_threads()
    get(ENV, "MHD_PMF_KSPACE_THREADS", "0") in ("1", "true", "TRUE", "yes", "on")
end

"""
    pmf_batchelor_field(dims; box=1, brms=1, seed=1, kcut=nothing, kmax=nothing,
                        energy_index=4, cutoff_order=2, normalization_kind=:auto,
                        fixed_mode_power=false, mode_realization=:legacy)

Generate a non-helical, approximately Batchelor-spectrum magnetic field on a
periodic grid.  `energy_index=4` means shell magnetic energy `E_B(k) ∝ k^4`
below the exponential `kcut` taper.  Set `kmax` to impose an additional strict
physical band limit; every Fourier coefficient with `|k| > kmax` is exactly
zero before the inverse transform. `normalization_kind=:total_band_rms` makes
`brms` the target `sqrt(<B^2>)` over the generated band.
`normalization_kind=:reference_band_rms` instead normalizes the preserved
`0 < k <= preserve_low_kmax` reference band, so adding ultraviolet modes raises
the total rms. `:auto` retains the legacy choice: reference-band normalization
when `preserve_low_kmax` is present and total-band normalization otherwise.
The returned tuple is `(Bx, By, Bz, stats)`.

With `fixed_mode_power=true`, each nonzero Fourier mode is rescaled to the
analytic envelope while retaining its random phase and transverse direction.
This removes finite-volume amplitude scatter from matched-box spectrum tests.
With `mode_realization=:physical_hash`, phase and polarization are keyed by
physical mode number relative to `realization_box`; this is intended for
matched-box and fixed-mode convergence studies.
`mode_realization=:physical_hash_zero_phase` uses the same physical keys but
real Fourier coefficients, matching the zero-phase non-helical construction
used by Jedamzik, Abel, and Ali-Haimoud for their recombination calculations.
"""
function pmf_batchelor_field(dims::NTuple{3,Int}; box::Real = 1,
                             brms::Real = 1, seed::Integer = 1,
                             kcut = nothing, kmin::Real = 0,
                             kmax = nothing, energy_index::Real = 4,
                             cutoff_order::Real = 2, mode_lock_n = nothing,
                             high_mode_lock_n = nothing,
                             preserve_low_kmax = nothing,
                             preserve_low_kcut = nothing,
                             normalization_kind = :auto,
                             curl_symbol = :centered,
                             curl_reference_n = nothing,
                             fixed_mode_power::Bool = false,
                             mode_realization = :legacy,
                             realization_box = nothing,
                             minimum_cells_per_wavelength = nothing)
    nx, ny, nz = dims
    iseven(nx) || error("pmf_batchelor_field currently requires even nx for rfft packing")
    L = Float64(box)
    target = Float64(brms)
    target > 0 || error("pmf_batchelor_field requires brms > 0")
    realization_mode = Symbol(lowercase(String(mode_realization)))
    realization_mode in (:legacy, :physical_hash, :physical_hash_zero_phase) ||
        error("mode_realization must be :legacy, :physical_hash, or " *
              ":physical_hash_zero_phase")
    realization_L = realization_box === nothing ? L : Float64(realization_box)
    realization_L > 0 || error("realization_box must be positive")
    realization_L + 32eps(Float64) * max(1, realization_L) >= L || error(
        "realization_box cannot be smaller than box")
    kc = kcut === nothing ? (π * minimum(dims) / L) : Float64(kcut)
    kc > 0 || error("kcut must be positive")
    kmin_f = Float64(kmin)
    kmin_f >= 0 || error("kmin must be nonnegative")
    km = kmax === nothing ? Inf : Float64(kmax)
    km > 0 || error("kmax must be positive")
    kmin_f < km || error("kmin must be smaller than kmax")
    resolved_kmax = if minimum_cells_per_wavelength === nothing
        Inf
    else
        isfinite(km) || error("kmax is required when enforcing minimum cells per wavelength")
        resolved = pmf_resolved_kmax(dims, L, minimum_cells_per_wavelength)
        km <= resolved * (1 + 16eps(Float64)) || error(
            "kmax=$km is under-resolved on grid $dims: limit is $resolved " *
            "for $(minimum_cells_per_wavelength) cells per wavelength")
        resolved
    end
    low_km = preserve_low_kmax === nothing ? nothing : Float64(preserve_low_kmax)
    preserve_low = low_km !== nothing
    if preserve_low
        low_km > 0 || error("preserve_low_kmax must be positive")
        low_km <= km || error("preserve_low_kmax cannot exceed kmax")
    end
    low_kc = preserve_low_kcut === nothing ? kc : Float64(preserve_low_kcut)
    low_kc > 0 || error("preserve_low_kcut must be positive")
    norm_kind = _pmf_normalization_kind(normalization_kind, preserve_low)
    source_lock_n = high_mode_lock_n === nothing ?
                    (!preserve_low ? mode_lock_n : nothing) : high_mode_lock_n
    source_dims = dims
    if source_lock_n !== nothing && Int(source_lock_n) != minimum(dims)
        lock_n = Int(source_lock_n)
        lock_n > 0 || error("mode_lock_n must be positive")
        lock_n <= minimum(dims) ||
            error("high_mode_lock_n=$lock_n cannot exceed target dims $dims")
        km <= π * lock_n / L ||
            error("PMF band exceeds the high-mode-lock Nyquist frequency")
        source_dims = (lock_n, lock_n, lock_n)
    end
    amp_exp = 0.5 * (Float64(energy_index) - 4.0)
    curl_mode = Symbol(curl_symbol)
    curl_mode in (:centered, :continuum, :spectral) ||
        error("curl_symbol must be :centered or :continuum")
    curl_mode === :spectral && (curl_mode = :continuum)
    curl_ref_n = curl_reference_n === nothing ? 0 : Int(curl_reference_n)
    if curl_ref_n != 0
        curl_mode === :centered ||
            error("curl_reference_n is only supported with centered curl")
        0 < curl_ref_n <= minimum(dims) ||
            error("curl_reference_n must be positive and no larger than target dims")
    end
    FFTW.set_num_threads(_pmf_fft_threads())
    nxh = nx ÷ 2 + 1
    axk, ayk, azk = begin
        ax, ay, az = _vector_potential_k(source_dims, seed, L, realization_mode,
                                         realization_L)
        source_dims == dims ? (ax, ay, az) :
            (_rfft_zero_pad_complex(ax, source_dims, dims),
             _rfft_zero_pad_complex(ay, source_dims, dims),
             _rfft_zero_pad_complex(az, source_dims, dims))
    end
    low_source_dims = dims
    low_axk, low_ayk, low_azk = if preserve_low
        if mode_lock_n !== nothing && Int(mode_lock_n) != minimum(dims)
            lock_n = Int(mode_lock_n)
            lock_n > 0 || error("mode_lock_n must be positive")
            lock_n <= minimum(dims) ||
                error("mode_lock_n=$lock_n cannot exceed target dims $dims")
            low_km <= π * lock_n / L ||
                error("preserved low-k band exceeds the mode-lock Nyquist frequency")
            low_source_dims = (lock_n, lock_n, lock_n)
        end
        ax, ay, az = _vector_potential_k(low_source_dims, seed, L,
                                         realization_mode, realization_L)
        low_source_dims == dims ? (ax, ay, az) :
            (_rfft_zero_pad_complex(ax, low_source_dims, dims),
             _rfft_zero_pad_complex(ay, low_source_dims, dims),
             _rfft_zero_pad_complex(az, low_source_dims, dims))
    else
        (axk, ayk, azk)
    end
    # Zero-padding multiplies each source grid by Vtarget/Vsource. Match the
    # high-source white-noise variance to the embedded low-source realization
    # before joining the two spectral bands.
    high_source_scale = preserve_low ?
        Float32(sqrt(prod(source_dims) / prod(low_source_dims))) : 1.0f0
    bxk = similar(axk)
    byk = similar(axk)
    bzk = similar(axk)
    low_bxk = preserve_low ? zeros(eltype(bxk), size(bxk)) : bxk
    low_byk = preserve_low ? zeros(eltype(byk), size(byk)) : byk
    low_bzk = preserve_low ? zeros(eltype(bzk), size(bzk)) : bzk
    dx = L / nx
    im = ComplexF32(0, 1)
    kxv = [_rfftfreq_i(ii, nx, L) for ii in 1:nxh]
    kyv = [_fftfreq_i(jj, ny, L) for jj in 1:ny]
    kzv = [_fftfreq_i(kk, nz, L) for kk in 1:nz]
    sxv = Float32[curl_mode === :centered ? sin(kx * dx) / dx : kx for kx in kxv]
    syv = Float32[curl_mode === :centered ? sin(ky * dx) / dx : ky for ky in kyv]
    szv = Float32[curl_mode === :centered ? sin(kz * dx) / dx : kz for kz in kzv]
    reference_dx = curl_ref_n == 0 ? dx : L / curl_ref_n
    rxv = Float32[curl_mode === :centered ? sin(kx * reference_dx) / reference_dx : kx
                   for kx in kxv]
    ryv = Float32[curl_mode === :centered ? sin(ky * reference_dx) / reference_dx : ky
                   for ky in kyv]
    rzv = Float32[curl_mode === :centered ? sin(kz * reference_dx) / reference_dx : kz
                   for kz in kzv]
    project_reference_curl = curl_ref_n != 0 && curl_ref_n != minimum(dims)
    function fill_bk_slice!(kk)
        @inbounds for jj in 1:ny, ii in 1:nxh
            kx = kxv[ii]
            ky = kyv[jj]
            kz = kzv[kk]
            k2 = kx*kx + ky*ky + kz*kz
            if k2 == 0 || k2 > km * km
                bxk[ii,jj,kk] = 0
                byk[ii,jj,kk] = 0
                bzk[ii,jj,kk] = 0
            else
                kmag = sqrt(k2)
                # k-space coordinates are Float32. Treat a cutoff within a few
                # Float32 ulps of a lattice shell as inclusive, as documented.
                use_low = preserve_low &&
                          k2 <= Float32(low_km)^2 * (1f0 + 8f0 * eps(Float32))
                local_kc = use_low ? low_kc : kc
                amp = (kmag / local_kc)^amp_exp *
                      exp(-0.5 * (kmag / local_kc)^Float64(cutoff_order))
                src_ax = use_low ? low_axk[ii,jj,kk] : axk[ii,jj,kk]
                src_ay = use_low ? low_ayk[ii,jj,kk] : ayk[ii,jj,kk]
                src_az = use_low ? low_azk[ii,jj,kk] : azk[ii,jj,kk]
                source_scale = use_low ? 1.0f0 : high_source_scale
                weighted_ax = ComplexF32(amp * source_scale) * src_ax
                weighted_ay = ComplexF32(amp * source_scale) * src_ay
                weighted_az = ComplexF32(amp * source_scale) * src_az
                sx = sxv[ii]
                sy = syv[jj]
                sz = szv[kk]
                rx = rxv[ii]
                ry = ryv[jj]
                rz = rzv[kk]
                bx = im * (ry * weighted_az - rz * weighted_ay)
                by = im * (rz * weighted_ax - rx * weighted_az)
                bz = im * (rx * weighted_ay - ry * weighted_ax)
                if project_reference_curl
                    s2 = sx*sx + sy*sy + sz*sz
                    if s2 > 0
                        parallel = (sx*bx + sy*by + sz*bz) / s2
                        bx -= sx * parallel
                        by -= sy * parallel
                        bz -= sz * parallel
                    end
                end
                bxk[ii,jj,kk] = bx
                byk[ii,jj,kk] = by
                bzk[ii,jj,kk] = bz
                if use_low
                    low_bxk[ii,jj,kk] = bx
                    low_byk[ii,jj,kk] = by
                    low_bzk[ii,jj,kk] = bz
                end
                if kmin_f > 0 && k2 <= kmin_f * kmin_f
                    bxk[ii,jj,kk] = 0
                    byk[ii,jj,kk] = 0
                    bzk[ii,jj,kk] = 0
                end
            end
        end
    end
    if _pmf_kspace_threads()
        Threads.@threads for kk in 1:nz
            fill_bk_slice!(kk)
        end
    else
        for kk in 1:nz
            fill_bk_slice!(kk)
        end
    end
    if fixed_mode_power
        function fix_mode_power_slice!(kk)
            @inbounds for jj in 1:ny, ii in 1:nxh
                kx = kxv[ii]
                ky = kyv[jj]
                kz = kzv[kk]
                k2 = kx*kx + ky*ky + kz*kz
                (k2 == 0 || k2 > km * km || k2 <= kmin_f * kmin_f) && continue
                raw = abs2(bxk[ii,jj,kk]) + abs2(byk[ii,jj,kk]) +
                      abs2(bzk[ii,jj,kk])
                raw > 0 || continue
                kmag = sqrt(k2)
                use_low = preserve_low &&
                          k2 <= Float32(low_km)^2 * (1f0 + 8f0 * eps(Float32))
                local_kc = use_low ? low_kc : kc
                target_power = (kmag / local_kc)^(Float64(energy_index) - 2.0) *
                               exp(-(kmag / local_kc)^Float64(cutoff_order))
                scale_mode = Float32(sqrt(target_power / raw))
                bxk[ii,jj,kk] *= scale_mode
                byk[ii,jj,kk] *= scale_mode
                bzk[ii,jj,kk] *= scale_mode
                if use_low
                    low_bxk[ii,jj,kk] *= scale_mode
                    low_byk[ii,jj,kk] *= scale_mode
                    low_bzk[ii,jj,kk] *= scale_mode
                end
            end
        end
        if _pmf_kspace_threads()
            Threads.@threads for kk in 1:nz
                fix_mode_power_slice!(kk)
            end
        else
            for kk in 1:nz
                fix_mode_power_slice!(kk)
            end
        end
    end
    Bx = Array{Float32}(FFTW.irfft(bxk, nx, (1,2,3)))
    By = Array{Float32}(FFTW.irfft(byk, nx, (1,2,3)))
    Bz = Array{Float32}(FFTW.irfft(bzk, nx, (1,2,3)))
    rms0 = sqrt((sum(abs2, Bx) + sum(abs2, By) + sum(abs2, Bz)) / length(Bx))
    rms0 > 0 || error("generated zero PMF field; check kcut, kmax, and dims")
    reference_rms0 = if preserve_low
        lowsum = sum(abs2, FFTW.irfft(low_bxk, nx, (1,2,3))) +
                 sum(abs2, FFTW.irfft(low_byk, nx, (1,2,3))) +
                 sum(abs2, FFTW.irfft(low_bzk, nx, (1,2,3)))
        low_rms0 = sqrt(lowsum / length(Bx))
        low_rms0 > 0 || error("preserved low-k PMF field is zero")
        low_rms0
    else
        rms0
    end
    norm_rms0 = norm_kind === :reference_band_rms ? reference_rms0 : rms0
    scale = Float32(target / norm_rms0)
    Bx .*= scale; By .*= scale; Bz .*= scale
    total_brms = sqrt((sum(abs2, Bx) + sum(abs2, By) + sum(abs2, Bz)) / length(Bx))
    reference_brms = Float64(scale) * reference_rms0
    normalization_brms = norm_kind === :reference_band_rms ? reference_brms : total_brms
    stats = (brms = target,
             brms_measured = total_brms,
             total_brms_measured = total_brms,
             reference_band_brms_measured = reference_brms,
             normalization_brms_measured = normalization_brms,
             normalization_kind = String(norm_kind),
             box = L, kcut = kc, energy_index = Float64(energy_index), seed = Int(seed),
             kmin = kmin_f, kmax = km,
             minimum_cells_per_wavelength = minimum_cells_per_wavelength === nothing ?
                 NaN : Float64(minimum_cells_per_wavelength),
             actual_cells_per_wavelength = isfinite(km) ? 2π * minimum(dims) / (L * km) : NaN,
             resolved_kmax = resolved_kmax,
             mode_lock_n = mode_lock_n === nothing ? 0 : Int(mode_lock_n),
             high_mode_lock_n = high_mode_lock_n === nothing ? 0 : Int(high_mode_lock_n),
             curl_reference_n = curl_ref_n,
             curl_symbol = String(curl_mode),
             fixed_mode_power = fixed_mode_power,
             mode_realization = String(realization_mode),
             realization_box = realization_L,
             preserve_low_kmax = preserve_low ? low_km : NaN,
             preserve_low_kcut = preserve_low ? low_kc : NaN,
             preserve_high_source_scale = preserve_low ? Float64(high_source_scale) : 1.0,
             preserve_low_norm_rms0 = preserve_low ? reference_rms0 : NaN)
    return Bx, By, Bz, stats
end

"""
    init_pmf_batchelor!(s; brms=nothing, b0_pG=nothing, Bunit_pG=1,
                        rho0=1, p0=1, box=Nx*dx, seed=1, kcut=nothing,
                        kmax=nothing, energy_index=4)

Initialize `s` with uniform gas and a non-helical Batchelor PMF.  Use `brms`
directly in code units, or provide `b0_pG` with `Bunit_pG` to map present-day
comoving pico-Gauss into code units.  Velocities start at zero and `ψ=0`.
"""
function init_pmf_batchelor!(s::MHDState{T}; brms = nothing, b0_pG = nothing,
                             Bunit_pG::Real = 1, rho0::Real = 1, p0::Real = 1,
                             box::Real = s.dims[1] * s.dx, seed::Integer = 1,
                             kcut = nothing, kmin::Real = 0,
                             kmax = nothing, energy_index::Real = 4,
                             cutoff_order::Real = 2, mode_lock_n = nothing,
                             high_mode_lock_n = nothing,
                             preserve_low_kmax = nothing,
                             preserve_low_kcut = nothing,
                             normalization_kind = :auto,
                             curl_symbol = :centered,
                             curl_reference_n = nothing,
                             fixed_mode_power::Bool = false,
                             mode_realization = :legacy,
                             realization_box = nothing,
                             minimum_cells_per_wavelength = nothing) where {T}
    br = brms === nothing ? begin
        b0_pG === nothing && error("provide either brms code units or b0_pG")
        Float64(b0_pG) / Float64(Bunit_pG)
    end : Float64(brms)
    Bx, By, Bz, stats = pmf_batchelor_field(s.dims; box=box, brms=br, seed=seed,
                                            kcut=kcut, kmin=kmin, kmax=kmax,
                                            energy_index=energy_index,
                                            cutoff_order=cutoff_order,
                                            mode_lock_n=mode_lock_n,
                                            high_mode_lock_n=high_mode_lock_n,
                                            preserve_low_kmax=preserve_low_kmax,
                                            preserve_low_kcut=preserve_low_kcut,
                                            normalization_kind=normalization_kind,
                                            curl_symbol=curl_symbol,
                                            curl_reference_n=curl_reference_n,
                                            fixed_mode_power=fixed_mode_power,
                                            mode_realization=mode_realization,
                                            realization_box=realization_box,
                                            minimum_cells_per_wavelength=minimum_cells_per_wavelength)
    _install_pmf_field!(s, Bx, By, Bz; rho0=rho0, p0=p0)
    return merge(stats, (rho0=Float64(rho0), p0=Float64(p0), Bunit_pG=Float64(Bunit_pG)))
end

"""
    init_pmf_batchelor_tiled!(s; tile_box, tile_n=nothing, ...)

Initialize a Batchelor PMF on a smaller physical periodic tile and repeat that
tile through the full domain.  This is useful for box-size controls where a
large box must contain the exact same physical magnetic pattern as a smaller
box, with no additional wavelengths larger than `tile_box`.
"""
function init_pmf_batchelor_tiled!(s::MHDState{T}; tile_box::Real,
                                  tile_n = nothing,
                                  brms = nothing, b0_pG = nothing,
                                  Bunit_pG::Real = 1, rho0::Real = 1,
                                  p0::Real = 1, box::Real = s.dims[1] * s.dx,
                                  seed::Integer = 1, kcut = nothing, kmin::Real = 0,
                                  kmax = nothing, energy_index::Real = 4,
                                  cutoff_order::Real = 2, mode_lock_n = nothing,
                                  high_mode_lock_n = nothing,
                                  preserve_low_kmax = nothing,
                                  preserve_low_kcut = nothing,
                                  normalization_kind = :auto,
                                  curl_symbol = :centered,
                                  curl_reference_n = nothing,
                                  fixed_mode_power::Bool = false,
                                  mode_realization = :legacy,
                                  realization_box = nothing,
                                  minimum_cells_per_wavelength = nothing) where {T}
    br = brms === nothing ? begin
        b0_pG === nothing && error("provide either brms code units or b0_pG")
        Float64(b0_pG) / Float64(Bunit_pG)
    end : Float64(brms)
    L = Float64(box)
    Lt = Float64(tile_box)
    Lt > 0 || error("tile_box must be positive")
    Lt <= L || error("tile_box cannot exceed full box")
    n = s.dims[1]
    nt = tile_n === nothing ? round(Int, n * Lt / L) : Int(tile_n)
    nt > 0 || error("tile_n must be positive")
    nt <= n || error("tile_n cannot exceed target grid size")
    n % nt == 0 || error("tile_n=$nt must divide target grid size $n")
    isapprox(Lt / L, nt / n; rtol=0, atol=4eps(Float64) * max(1, Lt / L)) ||
        error("tile_box/full box must match tile_n/target_n")
    Bxt, Byt, Bzt, stats = pmf_batchelor_field((nt, nt, nt); box=Lt, brms=br,
                                                seed=seed, kcut=kcut, kmin=kmin, kmax=kmax,
                                                energy_index=energy_index,
                                                cutoff_order=cutoff_order,
                                                mode_lock_n=mode_lock_n,
                                                high_mode_lock_n=high_mode_lock_n,
                                                preserve_low_kmax=preserve_low_kmax,
                                                preserve_low_kcut=preserve_low_kcut,
                                                normalization_kind=normalization_kind,
                                                curl_symbol=curl_symbol,
                                                curl_reference_n=curl_reference_n,
                                                fixed_mode_power=fixed_mode_power,
                                                mode_realization=mode_realization,
                                                realization_box=realization_box,
                                                minimum_cells_per_wavelength=minimum_cells_per_wavelength)
    Bx = nt == n ? Bxt : _periodic_tile_field(Bxt, s.dims)
    By = nt == n ? Byt : _periodic_tile_field(Byt, s.dims)
    Bz = nt == n ? Bzt : _periodic_tile_field(Bzt, s.dims)
    _install_pmf_field!(s, Bx, By, Bz; rho0=rho0, p0=p0)
    return merge(stats, (rho0=Float64(rho0), p0=Float64(p0),
                         Bunit_pG=Float64(Bunit_pG), full_box=L,
                         tile_box=Lt, tile_n=nt))
end
