# cic_deposit! — Cloud-In-Cell particle → grid mass deposit, KA, device-agnostic.
#
# One thread per particle scatters its mass to the 8 surrounding cells with the
# trilinear (CIC) weights, using atomic adds (f32 atomics verified on Metal). The
# grid is periodic. Positions are box-normalized to [0,1); a per-particle DRIFT
# (pos + disp·vel) and a constant SHIFT (in cells) are fused in:
#
#     g = mod(pos + disp·vel, 1)·N + shift
#
# This reproduces Enzo's GravitatingMassField particle deposit bit-for-bit when
# shift = -0.5 (edge→cell-centre registration) and disp = ½·dt/a (the When=0.5
# leapfrog drift PrepareDensityField applies) — verified corr=1.0, slope=1.0 vs
# `problem_get_gravitating_mass`. (The repic project carries an exact integer-CIC
# variant for reversibility; here we want the f32 GPU speed, not bit-reversibility.)

# NB: no `where {T}` on the @kernel — a parametric kernel signature makes KA box the
# type params (→ "call to gpu_malloc" InvalidIR on Metal). Element type flows in
# through the array args; `disp`/`shift` are converted to that type by the launcher.
@kernel function _cic_deposit_kernel!(ρ, @Const(px), @Const(py), @Const(pz),
                                      @Const(vx), @Const(vy), @Const(vz), @Const(mass),
                                      N::Int, disp, shift)
    p = @index(Global)
    @inbounds begin
        one_ = oneunit(px[p])
        gx = mod(px[p] + disp*vx[p], one_)*N + shift
        gy = mod(py[p] + disp*vy[p], one_)*N + shift
        gz = mod(pz[p] + disp*vz[p], one_)*N + shift
        fi = floor(gx); i0 = unsafe_trunc(Int, fi); fx = gx - fi
        fj = floor(gy); j0 = unsafe_trunc(Int, fj); fy = gy - fj
        fk = floor(gz); k0 = unsafe_trunc(Int, fk); fz = gz - fk
        m  = mass[p]
        # neighbour cell indices (periodic) and trilinear weights
        ia = mod(i0, N); ib = mod(i0+1, N); wxa = one_-fx; wxb = fx
        ja = mod(j0, N); jb = mod(j0+1, N); wya = one_-fy; wyb = fy
        ka = mod(k0, N); kb = mod(k0+1, N); wza = one_-fz; wzb = fz
        Nj = N; Nk = N*N
        KA.@atomic ρ[ia + Nj*ja + Nk*ka + 1] += m*wxa*wya*wza
        KA.@atomic ρ[ib + Nj*ja + Nk*ka + 1] += m*wxb*wya*wza
        KA.@atomic ρ[ia + Nj*jb + Nk*ka + 1] += m*wxa*wyb*wza
        KA.@atomic ρ[ib + Nj*jb + Nk*ka + 1] += m*wxb*wyb*wza
        KA.@atomic ρ[ia + Nj*ja + Nk*kb + 1] += m*wxa*wya*wzb
        KA.@atomic ρ[ib + Nj*ja + Nk*kb + 1] += m*wxb*wya*wzb
        KA.@atomic ρ[ia + Nj*jb + Nk*kb + 1] += m*wxa*wyb*wzb
        KA.@atomic ρ[ib + Nj*jb + Nk*kb + 1] += m*wxb*wyb*wzb
    end
end

# FIXED-POINT CIC: positions are UInt32 in [0, 2^32) spanning the box (U = N·2^k, so the
# cell index is the high log2(N) bits and the sub-cell fraction is the low k bits — EXACT,
# uniform sub-cell precision, no x·N cancellation; ~2 orders better than f32 at large N).
# Trilinear weights (f64, from the exact fraction) are quantized to 2^Q and accumulated in
# an Int64 grid → the SUM is EXACT and DETERMINISTIC (order-independent integer atomics),
# so the mean-subtraction keeps the tiny high-z δ that f32 loses.  shift=-0.5 cell (the
# Enzo GMF registration) is the half-unit subtract `2^{k-1}` (UInt32 wraps periodically).
# Q=30 default (per-cell Σ ~ Nppc·2^Q ≪ 2^63).  Drift omitted here (pre-drift positions).
@kernel function _cic_deposit_fixed_kernel!(ρ, @Const(Px), @Const(Py), @Const(Pz),
                                            N::Int, k::Int, scl::Float64)
    p = @index(Global)
    @inbounds begin
        mask = (UInt32(1) << k) - UInt32(1); half = UInt32(1) << (k - 1); kf = Float64(UInt32(1) << k)
        gx = Px[p] - half; gy = Py[p] - half; gz = Pz[p] - half
        ix = Int(gx >> k); fx = Float64(gx & mask) / kf
        iy = Int(gy >> k); fy = Float64(gy & mask) / kf
        iz = Int(gz >> k); fz = Float64(gz & mask) / kf
        ia = mod(ix, N); ib = mod(ix+1, N); ja = mod(iy, N); jb = mod(iy+1, N); ka = mod(iz, N); kb = mod(iz+1, N)
        Nj = N; Nk = N*N
        KA.@atomic ρ[ia + Nj*ja + Nk*ka + 1] += round(Int64, (1-fx)*(1-fy)*(1-fz)*scl)
        KA.@atomic ρ[ib + Nj*ja + Nk*ka + 1] += round(Int64, fx*(1-fy)*(1-fz)*scl)
        KA.@atomic ρ[ia + Nj*jb + Nk*ka + 1] += round(Int64, (1-fx)*fy*(1-fz)*scl)
        KA.@atomic ρ[ib + Nj*jb + Nk*ka + 1] += round(Int64, fx*fy*(1-fz)*scl)
        KA.@atomic ρ[ia + Nj*ja + Nk*kb + 1] += round(Int64, (1-fx)*(1-fy)*fz*scl)
        KA.@atomic ρ[ib + Nj*ja + Nk*kb + 1] += round(Int64, fx*(1-fy)*fz*scl)
        KA.@atomic ρ[ia + Nj*jb + Nk*kb + 1] += round(Int64, (1-fx)*fy*fz*scl)
        KA.@atomic ρ[ib + Nj*jb + Nk*kb + 1] += round(Int64, fx*fy*fz*scl)
    end
end

# f32 variant: UInt32 fixed-point positions (the precision-carrying part), but weights and
# accumulation in Float32 — as fast as the plain f32 deposit, yet ~100× more accurate at
# large N because the sub-cell fraction comes from the integer LOW BITS (frac→Float32 is
# EXACT for N≥256, k≤24; the ·2^-k is an exact exponent shift) — no `x·N` cancellation.
@kernel function _cic_deposit_fixed_f32_kernel!(ρ, @Const(Px), @Const(Py), @Const(Pz), N::Int, k::Int)
    p = @index(Global)
    @inbounds begin
        mask = (UInt32(1) << k) - UInt32(1); half = UInt32(1) << (k - 1); inv = 1f0 / Float32(UInt32(1) << k)
        gx = Px[p] - half; gy = Py[p] - half; gz = Pz[p] - half
        ix = Int(gx >> k); fx = Float32(gx & mask) * inv
        iy = Int(gy >> k); fy = Float32(gy & mask) * inv
        iz = Int(gz >> k); fz = Float32(gz & mask) * inv
        ia = mod(ix, N); ib = mod(ix+1, N); ja = mod(iy, N); jb = mod(iy+1, N); ka = mod(iz, N); kb = mod(iz+1, N)
        Nj = N; Nk = N*N
        KA.@atomic ρ[ia + Nj*ja + Nk*ka + 1] += (1f0-fx)*(1f0-fy)*(1f0-fz)
        KA.@atomic ρ[ib + Nj*ja + Nk*ka + 1] += fx*(1f0-fy)*(1f0-fz)
        KA.@atomic ρ[ia + Nj*jb + Nk*ka + 1] += (1f0-fx)*fy*(1f0-fz)
        KA.@atomic ρ[ib + Nj*jb + Nk*ka + 1] += fx*fy*(1f0-fz)
        KA.@atomic ρ[ia + Nj*ja + Nk*kb + 1] += (1f0-fx)*(1f0-fy)*fz
        KA.@atomic ρ[ib + Nj*ja + Nk*kb + 1] += fx*(1f0-fy)*fz
        KA.@atomic ρ[ia + Nj*jb + Nk*kb + 1] += (1f0-fx)*fy*fz
        KA.@atomic ρ[ib + Nj*jb + Nk*kb + 1] += fx*fy*fz
    end
end

# PURE-BITS CIC: no weights stored at all — the CIC overlap fractions ARE the low bits of
# the UInt32 position.  cell index = top log2(N) bits; the next 16 bits are the sub-cell
# fraction (rounded), used directly as fixed-point 1-D weights wlo=2^16−f, whi=f (exact
# partition of unity, Σ=2^16).  The trilinear weight is the integer product of three (≤2^16)
# → ≤2^48, accumulated EXACTLY in Int64 (deterministic, mean is exact ⇒ no ρ−mean
# cancellation).  shift=-0.5 cell (Enzo registration) = subtract 2^{k-1}; round-to-nearest
# the 16-bit fraction = add 2^{sh-1}.  Decode density = ρ / 2^48.
@kernel function _cic_deposit_bits_kernel!(ρ, @Const(Px), @Const(Py), @Const(Pz),
                                           N::Int, k::Int, sh::Int, dshift::Int)
    p = @index(Global)
    @inbounds begin
        T = eltype(ρ); m16 = UInt32(0xffff)
        bias = (UInt32(1) << (k-1)) - (sh > 0 ? (UInt32(1) << (sh-1)) : UInt32(0))  # −½cell + round
        gx = Px[p] - bias; gy = Py[p] - bias; gz = Pz[p] - bias
        ix = Int(gx >> k); fx = Int64((gx >> sh) & m16)
        iy = Int(gy >> k); fy = Int64((gy >> sh) & m16)
        iz = Int(gz >> k); fz = Int64((gz >> sh) & m16)
        W = Int64(65536)
        wxl=W-fx; wxh=fx; wyl=W-fy; wyh=fy; wzl=W-fz; wzh=fz
        ia=mod(ix,N);ib=mod(ix+1,N);ja=mod(iy,N);jb=mod(iy+1,N);ka=mod(iz,N);kb=mod(iz+1,N)
        Nj=N; Nk=N*N
        # trilinear product in Int64, ROUND-shifted to the accumulator's scale (round-to-nearest
        # removes the truncation/mass bias), then narrowed to T (no overflow while Σ/cell < typemax(T))
        rnd = dshift > 0 ? (Int64(1) << (dshift - 1)) : Int64(0)
        KA.@atomic ρ[ia+Nj*ja+Nk*ka+1] += (((wxl*wyl*wzl)+rnd) >> dshift) % T
        KA.@atomic ρ[ib+Nj*ja+Nk*ka+1] += (((wxh*wyl*wzl)+rnd) >> dshift) % T
        KA.@atomic ρ[ia+Nj*jb+Nk*ka+1] += (((wxl*wyh*wzl)+rnd) >> dshift) % T
        KA.@atomic ρ[ib+Nj*jb+Nk*ka+1] += (((wxh*wyh*wzl)+rnd) >> dshift) % T
        KA.@atomic ρ[ia+Nj*ja+Nk*kb+1] += (((wxl*wyl*wzh)+rnd) >> dshift) % T
        KA.@atomic ρ[ib+Nj*ja+Nk*kb+1] += (((wxh*wyl*wzh)+rnd) >> dshift) % T
        KA.@atomic ρ[ia+Nj*jb+Nk*kb+1] += (((wxl*wyh*wzh)+rnd) >> dshift) % T
        KA.@atomic ρ[ib+Nj*jb+Nk*kb+1] += (((wxh*wyh*wzh)+rnd) >> dshift) % T
    end
end

# right-shift of the 48-bit (16-bit/axis) trilinear product to fit each accumulator type,
# trading headroom for the unused high bits.  decode density = ρ / 2^(48−dshift).
_bits_dshift(::Type{<:Int64}) = 0    # scale 2^48, exact
_bits_dshift(::Type{<:Int32}) = 32   # scale 2^16, overflow only at ~2^15 part/cell
_bits_dshift(::Type{<:Int16}) = 34   # scale 2^14, overflow at ~2 part/cell (signed)
"Decode scale: density = ρ ./ cic_bits_scale(eltype(ρ))."
cic_bits_scale(::Type{T}) where {T<:Integer} = 2.0^(48 - _bits_dshift(T))

"""
    cic_deposit_bits!(ρ::AbstractVector{<:Integer}, Px,Py,Pz; N) -> ρ

Pure fixed-point CIC: the sub-cell overlap fractions ARE the position's low bits — no
weights stored or rounded in any float format.  `Px,Py,Pz` are `UInt32` device vectors
(positions ∈ [0,2^32) across the box); `ρ` is a flat `N³` integer device array.  16
sub-cell fraction bits/axis; the trilinear product is right-shifted to the accumulator's
scale (`Int64` exact 2^48, `Int32` 2^16 → ~2^15 particles/cell headroom, `Int16` 2^14 →
~2/cell).  Decode density as `ρ ./ cic_bits_scale(eltype(ρ))`.  `N` power of two ≤ 2^16.
Deterministic; overflow only when a cell exceeds the headroom.
"""
function cic_deposit_bits!(ρ::AbstractVector{T}, Px, Py, Pz; N::Integer) where {T<:Integer}
    be = KA.get_backend(ρ)
    k = 32 - round(Int, log2(N)); sh = k - 16
    sh >= 0 || error("cic_deposit_bits!: need ≥16 fraction bits (N ≤ 2^16); got k=$k")
    fill!(ρ, zero(T))
    _cic_deposit_bits_kernel!(be)(ρ, Px, Py, Pz, Int(N), Int(k), Int(sh), _bits_dshift(T);
                                  ndrange = length(Px))
    return ρ
end

"""
    cic_deposit_fixed!(ρi, Px,Py,Pz; N, qbits=30) -> ρi

Fixed-point CIC: `Px,Py,Pz` are `UInt32` device vectors holding positions as integers in
`[0,2^32)` across the box; `ρi` is a flat `N³` **`Int64`** device array (zeroed first).
Weights are quantized to `2^qbits` and accumulated exactly (deterministic integer atomics);
recover density as `ρi ./ 2^qbits`.  Same accuracy as f64 at 4-byte particle storage, with
uniform sub-cell precision (no f32 `x·N` cancellation).  `N` must be a power of two.
"""
function cic_deposit_fixed!(ρ::AbstractVector{<:Integer}, Px, Py, Pz; N::Integer, qbits::Integer=30)
    be = KA.get_backend(ρ)
    k = 32 - round(Int, log2(N))
    fill!(ρ, zero(eltype(ρ)))
    _cic_deposit_fixed_kernel!(be)(ρ, Px, Py, Pz, Int(N), Int(k), Float64(Int64(1) << qbits);
                                   ndrange = length(Px))
    return ρ
end

"""
    cic_deposit_fixed!(ρ::AbstractVector{<:AbstractFloat}, Px,Py,Pz; N) -> ρ

Float32 fixed-point CIC: UInt32 positions, Float32 weights + Float32 atomic accumulate —
**as fast as the plain f32 deposit** but ~100× more accurate at large N (the sub-cell
fraction is read from the integer low bits, exact, with no `x·N` cancellation).  The
density is in `ρ` directly (no `/2^qbits`).  Use the `Int64` method for an exact /
deterministic sum (mean-subtraction); use this when f32-accumulate precision suffices.
"""
function cic_deposit_fixed!(ρ::AbstractVector{<:AbstractFloat}, Px, Py, Pz; N::Integer)
    be = KA.get_backend(ρ)
    k = 32 - round(Int, log2(N))
    fill!(ρ, zero(eltype(ρ)))
    _cic_deposit_fixed_f32_kernel!(be)(ρ, Px, Py, Pz, Int(N), Int(k); ndrange = length(Px))
    return ρ
end

# DETERMINISTIC drop-in for `_cic_deposit_kernel!`: identical float positions / drift /
# shift / CIC weights, but each corner's contribution `m·w` is quantized to `round(Int64,
# m·w·qscale)` and accumulated by INTEGER atomic add.  Integer addition is associative ⇒
# the grid sum is independent of thread/atomic order ⇒ BIT-EXACTLY reproducible (run-to-run
# and across checkpoint/restart), unlike the float atomic.  This is the repic
# `deposit_det_kernel!` pattern (yipihey/repic).  `qscale = 2^qbits`; qbits=23 keeps the
# quantum at the Float32 ULP of `m·w` (no accuracy change vs the float deposit) while the
# per-cell Int64 sum (≤ Nppc·m·2^23) stays far below 2^63.  Host recovers ρ = ρi / qscale.
@kernel function _cic_deposit_det_kernel!(ρi, @Const(px), @Const(py), @Const(pz),
                                          @Const(vx), @Const(vy), @Const(vz), @Const(mass),
                                          N::Int, disp, shift, qscale)
    p = @index(Global)
    @inbounds begin
        one_ = oneunit(px[p])
        gx = mod(px[p] + disp*vx[p], one_)*N + shift
        gy = mod(py[p] + disp*vy[p], one_)*N + shift
        gz = mod(pz[p] + disp*vz[p], one_)*N + shift
        fi = floor(gx); i0 = unsafe_trunc(Int, fi); fx = gx - fi
        fj = floor(gy); j0 = unsafe_trunc(Int, fj); fy = gy - fj
        fk = floor(gz); k0 = unsafe_trunc(Int, fk); fz = gz - fk
        mq = mass[p]*qscale
        ia = mod(i0, N); ib = mod(i0+1, N); wxa = one_-fx; wxb = fx
        ja = mod(j0, N); jb = mod(j0+1, N); wya = one_-fy; wyb = fy
        ka = mod(k0, N); kb = mod(k0+1, N); wza = one_-fz; wzb = fz
        Nj = N; Nk = N*N
        KA.@atomic ρi[ia + Nj*ja + Nk*ka + 1] += round(eltype(ρi), mq*wxa*wya*wza)
        KA.@atomic ρi[ib + Nj*ja + Nk*ka + 1] += round(eltype(ρi), mq*wxb*wya*wza)
        KA.@atomic ρi[ia + Nj*jb + Nk*ka + 1] += round(eltype(ρi), mq*wxa*wyb*wza)
        KA.@atomic ρi[ib + Nj*jb + Nk*ka + 1] += round(eltype(ρi), mq*wxb*wyb*wza)
        KA.@atomic ρi[ia + Nj*ja + Nk*kb + 1] += round(eltype(ρi), mq*wxa*wya*wzb)
        KA.@atomic ρi[ib + Nj*ja + Nk*kb + 1] += round(eltype(ρi), mq*wxb*wya*wzb)
        KA.@atomic ρi[ia + Nj*jb + Nk*kb + 1] += round(eltype(ρi), mq*wxa*wyb*wzb)
        KA.@atomic ρi[ib + Nj*jb + Nk*kb + 1] += round(eltype(ρi), mq*wxb*wyb*wzb)
    end
end

"""
    cic_deposit_det!(ρi::AbstractVector{Int64}, px,py,pz, vx,vy,vz, mass; N, disp=0,
                     shift=-0.5, qbits=23) -> ρi

Deterministic CIC deposit — same registration as [`cic_deposit!`](@ref) (drift `disp·v`,
cell `shift`, periodic, Enzo-GMF `shift=-0.5`) but accumulated as quantized integers
(`round(Int64, m·w·2^qbits)`) with integer atomics, so the result is independent of atomic
order ⇒ bit-reproducible across runs and checkpoint/restart.  Recover the float density as
`ρi ./ 2.0^qbits`.  qbits=23 matches the Float32 ULP of the weights (no accuracy change).
"""
function cic_deposit_det!(ρi::AbstractVector{<:Integer},
                          px, py, pz, vx, vy, vz, mass;
                          N::Integer, disp::Real=0, shift::Real=-0.5, qbits::Integer=23)
    be = KA.get_backend(ρi)
    fill!(ρi, zero(eltype(ρi)))
    length(mass) == 0 && return ρi
    Tp = eltype(px)
    _cic_deposit_det_kernel!(be)(ρi, px, py, pz, vx, vy, vz, mass,
                                 Int(N), Tp(disp), Tp(shift), Tp(2.0^qbits); ndrange = length(mass))
    return ρi
end

# UNIFORM-MASS variants of the two deposits: when every particle carries the same mass
# (e.g. equal-mass DM), pass `mass` as a scalar `Real` instead of an N-vector — no per-
# particle mass array is allocated (saves 4 B/particle = a full N³ Float32 grid).  The
# scalar folds into the per-corner factor (`m` for float, `m·qscale` for the integer det).
@kernel function _cic_deposit_unif_kernel!(ρ, @Const(px), @Const(py), @Const(pz),
                                           @Const(vx), @Const(vy), @Const(vz),
                                           N::Int, disp, shift, m)
    p = @index(Global)
    @inbounds begin
        one_ = oneunit(px[p])
        gx = mod(px[p] + disp*vx[p], one_)*N + shift
        gy = mod(py[p] + disp*vy[p], one_)*N + shift
        gz = mod(pz[p] + disp*vz[p], one_)*N + shift
        fi = floor(gx); i0 = unsafe_trunc(Int, fi); fx = gx - fi
        fj = floor(gy); j0 = unsafe_trunc(Int, fj); fy = gy - fj
        fk = floor(gz); k0 = unsafe_trunc(Int, fk); fz = gz - fk
        ia = mod(i0, N); ib = mod(i0+1, N); wxa = one_-fx; wxb = fx
        ja = mod(j0, N); jb = mod(j0+1, N); wya = one_-fy; wyb = fy
        ka = mod(k0, N); kb = mod(k0+1, N); wza = one_-fz; wzb = fz
        Nj = N; Nk = N*N
        KA.@atomic ρ[ia + Nj*ja + Nk*ka + 1] += m*wxa*wya*wza
        KA.@atomic ρ[ib + Nj*ja + Nk*ka + 1] += m*wxb*wya*wza
        KA.@atomic ρ[ia + Nj*jb + Nk*ka + 1] += m*wxa*wyb*wza
        KA.@atomic ρ[ib + Nj*jb + Nk*ka + 1] += m*wxb*wyb*wza
        KA.@atomic ρ[ia + Nj*ja + Nk*kb + 1] += m*wxa*wya*wzb
        KA.@atomic ρ[ib + Nj*ja + Nk*kb + 1] += m*wxb*wya*wzb
        KA.@atomic ρ[ia + Nj*jb + Nk*kb + 1] += m*wxa*wyb*wzb
        KA.@atomic ρ[ib + Nj*jb + Nk*kb + 1] += m*wxb*wyb*wzb
    end
end

@kernel function _cic_deposit_det_unif_kernel!(ρi, @Const(px), @Const(py), @Const(pz),
                                               @Const(vx), @Const(vy), @Const(vz),
                                               N::Int, disp, shift, mq)
    p = @index(Global)
    @inbounds begin
        one_ = oneunit(px[p])
        gx = mod(px[p] + disp*vx[p], one_)*N + shift
        gy = mod(py[p] + disp*vy[p], one_)*N + shift
        gz = mod(pz[p] + disp*vz[p], one_)*N + shift
        fi = floor(gx); i0 = unsafe_trunc(Int, fi); fx = gx - fi
        fj = floor(gy); j0 = unsafe_trunc(Int, fj); fy = gy - fj
        fk = floor(gz); k0 = unsafe_trunc(Int, fk); fz = gz - fk
        ia = mod(i0, N); ib = mod(i0+1, N); wxa = one_-fx; wxb = fx
        ja = mod(j0, N); jb = mod(j0+1, N); wya = one_-fy; wyb = fy
        ka = mod(k0, N); kb = mod(k0+1, N); wza = one_-fz; wzb = fz
        Nj = N; Nk = N*N
        KA.@atomic ρi[ia + Nj*ja + Nk*ka + 1] += round(eltype(ρi), mq*wxa*wya*wza)
        KA.@atomic ρi[ib + Nj*ja + Nk*ka + 1] += round(eltype(ρi), mq*wxb*wya*wza)
        KA.@atomic ρi[ia + Nj*jb + Nk*ka + 1] += round(eltype(ρi), mq*wxa*wyb*wza)
        KA.@atomic ρi[ib + Nj*jb + Nk*ka + 1] += round(eltype(ρi), mq*wxb*wyb*wza)
        KA.@atomic ρi[ia + Nj*ja + Nk*kb + 1] += round(eltype(ρi), mq*wxa*wya*wzb)
        KA.@atomic ρi[ib + Nj*ja + Nk*kb + 1] += round(eltype(ρi), mq*wxb*wya*wzb)
        KA.@atomic ρi[ia + Nj*jb + Nk*kb + 1] += round(eltype(ρi), mq*wxa*wyb*wzb)
        KA.@atomic ρi[ib + Nj*jb + Nk*kb + 1] += round(eltype(ρi), mq*wxb*wyb*wzb)
    end
end

"Uniform-mass deterministic deposit: `mass` is a scalar (no N-vector). See [`cic_deposit_det!`](@ref)."
function cic_deposit_det!(ρi::AbstractVector{<:Integer},
                          px, py, pz, vx, vy, vz, mass::Real;
                          N::Integer, disp::Real=0, shift::Real=-0.5, qbits::Integer=23)
    be = KA.get_backend(ρi); fill!(ρi, zero(eltype(ρi)))
    length(px) == 0 && return ρi
    Tp = eltype(px)
    _cic_deposit_det_unif_kernel!(be)(ρi, px, py, pz, vx, vy, vz,
                                      Int(N), Tp(disp), Tp(shift), Tp(mass*2.0^qbits); ndrange = length(px))
    return ρi
end

"""
    cic_deposit!(ρ, px,py,pz, vx,vy,vz, mass; N, disp=0, shift=-0.5) -> ρ

Periodic CIC deposit of `length(mass)` particles (box-normalized positions in
[0,1)³, device vectors) onto the flat `N³` device array `ρ` (column-major,
`ρ[ic + N*jc + N²*kc + 1]`). `ρ` is zeroed first. `disp` drifts each particle by
`disp·v` before depositing; `shift` is a constant cell offset (−0.5 ⇒ Enzo's GMF
registration). Velocity vectors may be the same as positions when `disp=0`.
"""
function cic_deposit!(ρ::AbstractVector{T},
                      px, py, pz, vx, vy, vz, mass;
                      N::Integer, disp::Real=0, shift::Real=-0.5) where {T}
    be = KA.get_backend(ρ)
    fill!(ρ, zero(T))
    length(mass) == 0 && return ρ      # nothing to deposit (e.g. an empty AMR subgrid) — KA's
                                       # launch partition divides by ndrange, so guard ndrange=0
    Tp = eltype(px)
    _cic_deposit_kernel!(be)(ρ, px, py, pz, vx, vy, vz, mass,
                             Int(N), Tp(disp), Tp(shift); ndrange = length(mass))
    return ρ
end

"Uniform-mass float deposit: `mass` is a scalar (no N-vector). See [`cic_deposit!`](@ref)."
function cic_deposit!(ρ::AbstractVector{T},
                      px, py, pz, vx, vy, vz, mass::Real;
                      N::Integer, disp::Real=0, shift::Real=-0.5) where {T}
    be = KA.get_backend(ρ); fill!(ρ, zero(T))
    length(px) == 0 && return ρ
    Tp = eltype(px)
    _cic_deposit_unif_kernel!(be)(ρ, px, py, pz, vx, vy, vz,
                                  Int(N), Tp(disp), Tp(shift), T(mass); ndrange = length(px))
    return ρ
end
