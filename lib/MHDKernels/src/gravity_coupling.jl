# Time-center a gas density source without retaining the previous full grid.
# From continuity, rho(t-dt) = rho(t) + dt div(m)(t) + O(dt^2).

export predict_density_backward!, predict_density_perturbation_backward!,
       apply_cell_center_gravity!
export initialize_lattice_displacements!
export deposit_lattice_displacements!, deposit_lattice_displacement_delta!
export drift_lattice_displacements!

@kernel function _initialize_lattice_displacements_k!(dxp, dyp, dzp, vx, vy, vz)
    p = @index(Global)
    @inbounds begin
        dxp[p] = zero(eltype(dxp)); dyp[p] = zero(eltype(dyp)); dzp[p] = zero(eltype(dzp))
        vx[p] = zero(eltype(vx)); vy[p] = zero(eltype(vy)); vz[p] = zero(eltype(vz))
    end
end

"""
    initialize_lattice_displacements!(dxp, dyp, dzp, vx, vy, vz)

Initialize particles represented by displacements from an implicit regular
lattice. Particle `p` has an unperturbed centre reconstructed from `p` and the
grid size, so no absolute-position arrays are required.
"""
function initialize_lattice_displacements!(dxp, dyp, dzp, vx, vy, vz)
    n = length(dxp)
    all(length(a) == n for a in (dyp, dzp, vx, vy, vz)) ||
        throw(DimensionMismatch("particle arrays must have equal lengths"))
    be = KA.get_backend(dxp)
    _initialize_lattice_displacements_k!(be)(dxp, dyp, dzp, vx, vy, vz; ndrange=n)
    return dxp, dyp, dzp, vx, vy, vz
end

@kernel function _deposit_lattice_displacements_k!(rho,
                                                    @Const(dxp), @Const(dyp), @Const(dzp),
                                                    @Const(vx), @Const(vy), @Const(vz),
                                                    N::Int, disp, shift)
    p = @index(Global)
    @inbounds begin
        T = eltype(rho)
        q = p - 1
        i = q % N
        j = (q ÷ N) % N
        k = q ÷ (N * N)
        # Resolve the fractional coordinate before adding the integer cell.
        # Adding N*dxp to i rounds away early-time drifts for large i in f32.
        base = T(0.5) + shift
        ux = base + T(N) * (T(dxp[p]) + disp * T(vx[p]))
        uy = base + T(N) * (T(dyp[p]) + disp * T(vy[p]))
        uz = base + T(N) * (T(dzp[p]) + disp * T(vz[p]))
        ox = unsafe_trunc(Int, floor(ux))
        oy = unsafe_trunc(Int, floor(uy))
        oz = unsafe_trunc(Int, floor(uz))
        # Compute both weights from the local coordinate. The small member of
        # each pair then remains representable whether the drift is positive
        # or negative.
        wxa = T(ox + 1) - ux; wxb = ux - T(ox)
        wya = T(oy + 1) - uy; wyb = uy - T(oy)
        wza = T(oz + 1) - uz; wzb = uz - T(oz)
        ia = mod(i + ox, N); ib = mod(i + ox + 1, N)
        ja = mod(j + oy, N); jb = mod(j + oy + 1, N)
        ka = mod(k + oz, N); kb = mod(k + oz + 1, N)
        Nj = N; Nk = N * N
        KA.@atomic rho[ia + Nj*ja + Nk*ka + 1] += wxa*wya*wza
        KA.@atomic rho[ib + Nj*ja + Nk*ka + 1] += wxb*wya*wza
        KA.@atomic rho[ia + Nj*jb + Nk*ka + 1] += wxa*wyb*wza
        KA.@atomic rho[ib + Nj*jb + Nk*ka + 1] += wxb*wyb*wza
        KA.@atomic rho[ia + Nj*ja + Nk*kb + 1] += wxa*wya*wzb
        KA.@atomic rho[ib + Nj*ja + Nk*kb + 1] += wxb*wya*wzb
        KA.@atomic rho[ia + Nj*jb + Nk*kb + 1] += wxa*wyb*wzb
        KA.@atomic rho[ib + Nj*jb + Nk*kb + 1] += wxb*wyb*wzb
    end
end

@inline function _unit_product_delta(lx, ly, lz)
    # (1-lx)*(1-ly)*(1-lz) - 1, without subtracting two values near one.
    return -(lx + ly + lz) + lx*ly + lx*lz + ly*lz - lx*ly*lz
end

@kernel function _deposit_lattice_displacement_delta_k!(delta,
                                                        @Const(dxp), @Const(dyp),
                                                        @Const(dzp), @Const(vx),
                                                        @Const(vy), @Const(vz),
                                                        N::Int, disp, shift)
    p = @index(Global)
    @inbounds begin
        T = eltype(delta)
        q = p - 1
        i = q % N
        j = (q ÷ N) % N
        k = q ÷ (N * N)
        base = T(0.5) + shift
        ux = base + T(N) * (T(dxp[p]) + disp * T(vx[p]))
        uy = base + T(N) * (T(dyp[p]) + disp * T(vy[p]))
        uz = base + T(N) * (T(dzp[p]) + disp * T(vz[p]))
        ox = unsafe_trunc(Int, floor(ux))
        oy = unsafe_trunc(Int, floor(uy))
        oz = unsafe_trunc(Int, floor(uz))
        wxa = T(ox + 1) - ux; wxb = ux - T(ox)
        wya = T(oy + 1) - uy; wyb = uy - T(oy)
        wza = T(oz + 1) - uz; wzb = uz - T(oz)
        ia = mod(i + ox, N); ib = mod(i + ox + 1, N)
        ja = mod(j + oy, N); jb = mod(j + oy + 1, N)
        ka = mod(k + oz, N); kb = mod(k + oz + 1, N)
        Nj = N; Nk = N * N

        h000 = ia == i && ja == j && ka == k
        h100 = ib == i && ja == j && ka == k
        h010 = ia == i && jb == j && ka == k
        h110 = ib == i && jb == j && ka == k
        h001 = ia == i && ja == j && kb == k
        h101 = ib == i && ja == j && kb == k
        h011 = ia == i && jb == j && kb == k
        h111 = ib == i && jb == j && kb == k

        v000 = h000 ? _unit_product_delta(wxb, wyb, wzb) : wxa*wya*wza
        v100 = h100 ? _unit_product_delta(wxa, wyb, wzb) : wxb*wya*wza
        v010 = h010 ? _unit_product_delta(wxb, wya, wzb) : wxa*wyb*wza
        v110 = h110 ? _unit_product_delta(wxa, wya, wzb) : wxb*wyb*wza
        v001 = h001 ? _unit_product_delta(wxb, wyb, wza) : wxa*wya*wzb
        v101 = h101 ? _unit_product_delta(wxa, wyb, wza) : wxb*wya*wzb
        v011 = h011 ? _unit_product_delta(wxb, wya, wza) : wxa*wyb*wzb
        v111 = h111 ? _unit_product_delta(wxa, wya, wza) : wxb*wyb*wzb

        KA.@atomic delta[ia + Nj*ja + Nk*ka + 1] += v000
        KA.@atomic delta[ib + Nj*ja + Nk*ka + 1] += v100
        KA.@atomic delta[ia + Nj*jb + Nk*ka + 1] += v010
        KA.@atomic delta[ib + Nj*jb + Nk*ka + 1] += v110
        KA.@atomic delta[ia + Nj*ja + Nk*kb + 1] += v001
        KA.@atomic delta[ib + Nj*ja + Nk*kb + 1] += v101
        KA.@atomic delta[ia + Nj*jb + Nk*kb + 1] += v011
        KA.@atomic delta[ib + Nj*jb + Nk*kb + 1] += v111
        if !(h000 || h100 || h010 || h110 || h001 || h101 || h011 || h111)
            KA.@atomic delta[i + Nj*j + Nk*k + 1] -= one(T)
        end
    end
end

"""
    deposit_lattice_displacements!(rho, dxp, dyp, dzp, vx, vy, vz; N, disp=0, shift=-0.5)

CIC-deposit one unit-mass particle per cell from displacement coordinates. The
lattice centre and displacement are combined in cell coordinates, preserving
sub-ULP box displacements in Float32. `rho` is cleared before deposit.
"""
function deposit_lattice_displacements!(rho::AbstractArray{T}, dxp, dyp, dzp,
                                        vx, vy, vz; N::Integer,
                                        disp::Real=0, shift::Real=-0.5) where {T}
    length(rho) == N^3 || throw(DimensionMismatch("rho must contain N^3 cells"))
    length(dxp) == N^3 || throw(DimensionMismatch("lattice must contain N^3 particles"))
    fill!(rho, zero(T))
    be = KA.get_backend(rho)
    _deposit_lattice_displacements_k!(be)(rho, dxp, dyp, dzp, vx, vy, vz,
                                           Int(N), T(disp), T(shift);
                                           ndrange=length(dxp))
    return rho
end

"""
    deposit_lattice_displacement_delta!(delta, dxp, dyp, dzp, vx, vy, vz;
                                         N, disp=0, shift=-0.5)

Deposit the density contrast of one particle per implicit lattice cell. The
kernel accumulates the displaced CIC weights minus the undisplaced lattice
weight directly, so perturbations much smaller than `eps(Float32)` are not
lost by first constructing a density near one. No additional grid is needed.
"""
function deposit_lattice_displacement_delta!(delta::AbstractArray{T}, dxp, dyp,
                                             dzp, vx, vy, vz; N::Integer,
                                             disp::Real=0,
                                             shift::Real=-0.5) where {T}
    N >= 2 || throw(ArgumentError("lattice density-contrast deposit requires N >= 2"))
    length(delta) == N^3 ||
        throw(DimensionMismatch("delta must contain N^3 cells"))
    length(dxp) == N^3 ||
        throw(DimensionMismatch("lattice must contain N^3 particles"))
    fill!(delta, zero(T))
    be = KA.get_backend(delta)
    _deposit_lattice_displacement_delta_k!(be)(
        delta, dxp, dyp, dzp, vx, vy, vz, Int(N), T(disp), T(shift);
        ndrange=length(dxp),
    )
    return delta
end

@kernel function _drift_lattice_displacements_k!(dxp, dyp, dzp,
                                                  @Const(vx), @Const(vy), @Const(vz),
                                                  coef, dowrap::Int)
    p = @index(Global)
    @inbounds begin
        x = dxp[p] + coef * vx[p]
        y = dyp[p] + coef * vy[p]
        z = dzp[p] + coef * vz[p]
        if dowrap == 1
            x -= floor(x + oftype(x, 0.5))
            y -= floor(y + oftype(y, 0.5))
            z -= floor(z + oftype(z, 0.5))
        end
        dxp[p] = x; dyp[p] = y; dzp[p] = z
    end
end

"Accumulate a drift into zero-centred periodic lattice displacements."
function drift_lattice_displacements!(dxp::AbstractVector{T}, dyp, dzp, vx, vy, vz;
                                      coef::Real, wrap::Bool=true) where {T}
    be = KA.get_backend(dxp)
    _drift_lattice_displacements_k!(be)(dxp, dyp, dzp, vx, vy, vz,
                                         T(coef), wrap ? 1 : 0;
                                         ndrange=length(dxp))
    return dxp, dyp, dzp
end

@kernel function _predict_density_backward_k!(dst,
                                               @Const(rho), @Const(mx),
                                               @Const(my), @Const(mz),
                                               dt_back, inv2dx,
                                               nx::Int, ny::Int, nz::Int)
    c = @index(Global)
    @inbounds begin
        i = (c - 1) % nx + 1
        q = (c - 1) ÷ nx
        j = q % ny + 1
        k = q ÷ ny + 1
        im = i == 1  ? nx : i - 1
        ip = i == nx ? 1  : i + 1
        jm = j == 1  ? ny : j - 1
        jp = j == ny ? 1  : j + 1
        km = k == 1  ? nz : k - 1
        kp = k == nz ? 1  : k + 1

        cim = ((k - 1) * ny + (j - 1)) * nx + im
        cip = ((k - 1) * ny + (j - 1)) * nx + ip
        cjm = ((k - 1) * ny + (jm - 1)) * nx + i
        cjp = ((k - 1) * ny + (jp - 1)) * nx + i
        ckm = ((km - 1) * ny + (j - 1)) * nx + i
        ckp = ((kp - 1) * ny + (j - 1)) * nx + i
        divm = ((mx[cip] - mx[cim]) +
                (my[cjp] - my[cjm]) +
                (mz[ckp] - mz[ckm])) * inv2dx
        dst[c] = rho[c] + dt_back * divm
    end
end

"""
    predict_density_backward!(dst, rho, mx, my, mz, dims; dx, dt_back)

Predict density at `t - dt_back` from conserved state at `t` using the periodic
continuity equation. This second-order midpoint source reconstruction avoids a
previous-density grid in operator-split particle gravity.
"""
function predict_density_backward!(dst::AbstractArray{T}, rho, mx, my, mz,
                                   dims::NTuple{3,Int}; dx::Real,
                                   dt_back::Real) where {T}
    length(dst) == prod(dims) || throw(DimensionMismatch("dst does not match dims"))
    be = KA.get_backend(dst)
    _predict_density_backward_k!(be)(dst, rho, mx, my, mz,
                                      T(dt_back), T(0.5) / T(dx), dims...;
                                      ndrange=length(dst))
    return dst
end

function predict_density_backward!(dst::AbstractArray, s::MHDState,
                                   dt_back::Real)
    predict_density_backward!(dst, s.U[1], s.U[2], s.U[3], s.U[4], s.dims;
                              dx=s.dx, dt_back=dt_back)
end

"""
    predict_density_perturbation_backward!(dst, delta_rho, mx, my, mz, dims;
                                           dx, dt_back)

Time-center an authoritative density perturbation without adding and subtracting
the homogeneous background in Float32.
"""
function predict_density_perturbation_backward!(dst::AbstractArray{T},delta_rho,
                                                mx,my,mz,dims::NTuple{3,Int};
                                                dx::Real,dt_back::Real) where {T}
    length(dst)==prod(dims) || throw(DimensionMismatch("dst does not match dims"))
    be=KA.get_backend(dst)
    _predict_density_backward_k!(be)(dst,delta_rho,mx,my,mz,
        T(dt_back),T(0.5)/T(dx),dims...;ndrange=length(dst))
    dst
end

@kernel function _apply_cell_center_gravity_k!(rho, mx, my, mz, E,
                                                @Const(phi), dt, inv2dx,
                                                nx::Int, ny::Int, nz::Int)
    c = @index(Global)
    @inbounds begin
        i = (c - 1) % nx + 1
        q = (c - 1) ÷ nx
        j = q % ny + 1
        k = q ÷ ny + 1
        im = i == 1  ? nx : i - 1
        ip = i == nx ? 1  : i + 1
        jm = j == 1  ? ny : j - 1
        jp = j == ny ? 1  : j + 1
        km = k == 1  ? nz : k - 1
        kp = k == nz ? 1  : k + 1

        cim = ((k - 1) * ny + (j - 1)) * nx + im
        cip = ((k - 1) * ny + (j - 1)) * nx + ip
        cjm = ((k - 1) * ny + (jm - 1)) * nx + i
        cjp = ((k - 1) * ny + (jp - 1)) * nx + i
        ckm = ((km - 1) * ny + (j - 1)) * nx + i
        ckp = ((kp - 1) * ny + (j - 1)) * nx + i

        gx = -(phi[cip] - phi[cim]) * inv2dx
        gy = -(phi[cjp] - phi[cjm]) * inv2dx
        gz = -(phi[ckp] - phi[ckm]) * inv2dx
        r = rho[c]
        mx0 = mx[c]; my0 = my[c]; mz0 = mz[c]
        dmx = r * gx * dt
        dmy = r * gy * dt
        dmz = r * gz * dt
        mx[c] = mx0 + dmx
        my[c] = my0 + dmy
        mz[c] = mz0 + dmz
        # Exact kinetic-energy change for a constant acceleration over the kick.
        E[c] += dt * (mx0 * gx + my0 * gy + mz0 * gz) +
                (dt * dt * r) * (gx * gx + gy * gy + gz * gz) / oftype(r, 2)
    end
end

"""
    apply_cell_center_gravity!(state, phi, dt)

Apply the periodic cell-centred acceleration `-grad(phi)` to gas momentum and
total energy. The energy update is the exact kinetic-energy change of the kick,
so internal and magnetic energy are unchanged. The operation is allocation-free
on CPU, CUDA, and Metal backends.
"""
function apply_cell_center_gravity!(s::MHDState, phi::AbstractArray, dt::Real)
    length(phi) == prod(s.dims) || throw(DimensionMismatch("phi does not match state dimensions"))
    be = KA.get_backend(s.U[1])
    T = eltype(s.U[1])
    _apply_cell_center_gravity_k!(be)(s.U[1], s.U[2], s.U[3], s.U[4], s.U[5], phi,
                                      T(dt), T(0.5) / T(s.dx), s.dims...;
                                      ndrange=length(phi))
    return s
end
