# ── Primordial magnetic field initial conditions ─────────────────────────────
# Non-helical Gaussian PMFs with a Batchelor magnetic-energy spectrum.  We build
# a random vector potential A(k), take B(k) = i s(k) × A(k), then inverse-transform
# B.  Here s(k)=sin(k dx)/dx is the central-difference derivative symbol, so the
# diagnostic divergence uses the same operator and s·B vanishes to roundoff.

export pmf_batchelor_field, init_pmf_batchelor!

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
                        energy_index=4, cutoff_order=2)

Generate a non-helical, approximately Batchelor-spectrum magnetic field on a
periodic grid.  `energy_index=4` means shell magnetic energy `E_B(k) ∝ k^4`
below the exponential `kcut` taper.  Set `kmax` to impose an additional strict
physical band limit; every Fourier coefficient with `|k| > kmax` is exactly
zero before the inverse transform.  `brms` is the target real-space rms
`sqrt(<B^2>)` in code units.  The returned tuple is `(Bx, By, Bz, stats)`.
"""
function pmf_batchelor_field(dims::NTuple{3,Int}; box::Real = 1,
                             brms::Real = 1, seed::Integer = 1,
                             kcut = nothing, kmax = nothing, energy_index::Real = 4,
                             cutoff_order::Real = 2, mode_lock_n = nothing)
    nx, ny, nz = dims
    iseven(nx) || error("pmf_batchelor_field currently requires even nx for rfft packing")
    L = Float64(box)
    target = Float64(brms)
    target > 0 || error("pmf_batchelor_field requires brms > 0")
    kc = kcut === nothing ? (π * minimum(dims) / L) : Float64(kcut)
    kc > 0 || error("kcut must be positive")
    km = kmax === nothing ? Inf : Float64(kmax)
    km > 0 || error("kmax must be positive")
    source_dims = dims
    if mode_lock_n !== nothing && Int(mode_lock_n) != minimum(dims)
        lock_n = Int(mode_lock_n)
        lock_n > 0 || error("mode_lock_n must be positive")
        lock_n <= minimum(dims) ||
            error("mode_lock_n=$lock_n cannot exceed target dims $dims")
        source_dims = (lock_n, lock_n, lock_n)
    end
    amp_exp = 0.5 * (Float64(energy_index) - 4.0)
    FFTW.set_num_threads(_pmf_fft_threads())
    nxh = nx ÷ 2 + 1
    axk, ayk, azk = begin
        ax, ay, az = _random_vector_potential_k(source_dims, seed)
        source_dims == dims ? (ax, ay, az) :
            (_rfft_zero_pad_complex(ax, source_dims, dims),
             _rfft_zero_pad_complex(ay, source_dims, dims),
             _rfft_zero_pad_complex(az, source_dims, dims))
    end
    bxk = similar(axk)
    byk = similar(axk)
    bzk = similar(axk)
    dx = L / nx
    im = ComplexF32(0, 1)
    kxv = [_rfftfreq_i(ii, nx, L) for ii in 1:nxh]
    kyv = [_fftfreq_i(jj, ny, L) for jj in 1:ny]
    kzv = [_fftfreq_i(kk, nz, L) for kk in 1:nz]
    sxv = Float32[sin(kx * dx) / dx for kx in kxv]
    syv = Float32[sin(ky * dx) / dx for ky in kyv]
    szv = Float32[sin(kz * dx) / dx for kz in kzv]
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
                amp = (kmag / kc)^amp_exp * exp(-0.5 * (kmag / kc)^Float64(cutoff_order))
                ax = ComplexF32(amp) * axk[ii,jj,kk]
                ay = ComplexF32(amp) * ayk[ii,jj,kk]
                az = ComplexF32(amp) * azk[ii,jj,kk]
                sx = sxv[ii]
                sy = syv[jj]
                sz = szv[kk]
                bxk[ii,jj,kk] = im * (sy * az - sz * ay)
                byk[ii,jj,kk] = im * (sz * ax - sx * az)
                bzk[ii,jj,kk] = im * (sx * ay - sy * ax)
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
    Bx = Array{Float32}(FFTW.irfft(bxk, nx, (1,2,3)))
    By = Array{Float32}(FFTW.irfft(byk, nx, (1,2,3)))
    Bz = Array{Float32}(FFTW.irfft(bzk, nx, (1,2,3)))
    rms0 = sqrt((sum(abs2, Bx) + sum(abs2, By) + sum(abs2, Bz)) / length(Bx))
    rms0 > 0 || error("generated zero PMF field; check kcut, kmax, and dims")
    scale = Float32(target / rms0)
    Bx .*= scale; By .*= scale; Bz .*= scale
    stats = (brms = target,
             brms_measured = sqrt((sum(abs2, Bx) + sum(abs2, By) + sum(abs2, Bz)) / length(Bx)),
             box = L, kcut = kc, energy_index = Float64(energy_index), seed = Int(seed),
             kmax = km,
             mode_lock_n = mode_lock_n === nothing ? 0 : Int(mode_lock_n))
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
                             kcut = nothing, kmax = nothing, energy_index::Real = 4,
                             cutoff_order::Real = 2, mode_lock_n = nothing) where {T}
    br = brms === nothing ? begin
        b0_pG === nothing && error("provide either brms code units or b0_pG")
        Float64(b0_pG) / Float64(Bunit_pG)
    end : Float64(brms)
    Bx, By, Bz, stats = pmf_batchelor_field(s.dims; box=box, brms=br, seed=seed,
                                            kcut=kcut, kmax=kmax, energy_index=energy_index,
                                            cutoff_order=cutoff_order,
                                            mode_lock_n=mode_lock_n)
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
    return merge(stats, (rho0=Float64(rho0), p0=Float64(p0), Bunit_pG=Float64(Bunit_pG)))
end
