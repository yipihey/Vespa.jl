export conserved_species_to_fraction!, conserved_species_to_fractions!
export advect_species_fraction!, advect_species_fractions!

@kernel function _conserved_species_to_fraction_k!(species, @Const(rho), maximum_total)
    c = @index(Global)
    @inbounds begin
        T = eltype(rho)
        density = max(rho[c], eps(T))
        species[c] = min(max(species[c] / density, zero(T)), maximum_total)
    end
end

@kernel function _conserved_species_to_fractions_k!(species1, species2,
        @Const(rho), maximum_total)
    c = @index(Global)
    @inbounds begin
        T = eltype(rho)
        density = max(rho[c], eps(T))
        q2 = min(max(species2[c] / density, zero(T)), maximum_total)
        q1 = min(max(species1[c] / density, zero(T)), maximum_total - q2)
        species1[c] = q1
        species2[c] = q2
    end
end

@inline function _passive_index(i::Int32, j::Int32, k::Int32, n::Int32)
    ii = mod(i, n)
    jj = mod(j, n)
    kk = mod(k, n)
    ii + n * (jj + n * kk) + Int32(1)
end

@inline function _passive_pair(species1, species2, i::Int32, j::Int32,
                               k::Int32, n::Int32)
    index = _passive_index(i, j, k, n)
    @inbounds species1[index], species2[index]
end

@inline _passive_lerp(left::T, right::T, fraction::T) where {T} =
    muladd(fraction, right - left, left)

@inline function _passive_trilinear_component(p000::T, p100::T, p010::T,
        p110::T, p001::T, p101::T, p011::T, p111::T,
        fx::T, fy::T, fz::T) where {T}
    x00 = _passive_lerp(p000, p100, fx)
    x10 = _passive_lerp(p010, p110, fx)
    x01 = _passive_lerp(p001, p101, fx)
    x11 = _passive_lerp(p011, p111, fx)
    y0 = _passive_lerp(x00, x10, fy)
    y1 = _passive_lerp(x01, x11, fy)
    _passive_lerp(y0, y1, fz)
end

@inline function _passive_trilinear(species1, species2, n::Int32,
        i0::Int32, j0::Int32, k0::Int32, fx::T, fy::T, fz::T) where {T}
    onei = Int32(1)
    p000 = _passive_pair(species1, species2, i0,        j0,        k0,        n)
    p100 = _passive_pair(species1, species2, i0 + onei, j0,        k0,        n)
    p010 = _passive_pair(species1, species2, i0,        j0 + onei, k0,        n)
    p110 = _passive_pair(species1, species2, i0 + onei, j0 + onei, k0,        n)
    p001 = _passive_pair(species1, species2, i0,        j0,        k0 + onei, n)
    p101 = _passive_pair(species1, species2, i0 + onei, j0,        k0 + onei, n)
    p011 = _passive_pair(species1, species2, i0,        j0 + onei, k0 + onei, n)
    p111 = _passive_pair(species1, species2, i0 + onei, j0 + onei, k0 + onei, n)
    (
        _passive_trilinear_component(
            p000[1], p100[1], p010[1], p110[1],
            p001[1], p101[1], p011[1], p111[1], fx, fy, fz,
        ),
        _passive_trilinear_component(
            p000[2], p100[2], p010[2], p110[2],
            p001[2], p101[2], p011[2], p111[2], fx, fy, fz,
        ),
    )
end

@kernel function _advect_species_fractions_k!(out1, out2,
        @Const(species1), @Const(species2), @Const(rho),
        @Const(mx), @Const(my), @Const(mz), n::Int32, dtdx, maximum_total)
    c = @index(Global)
    @inbounds begin
        T = eltype(rho)
        q = Int32(c - 1)
        i = q % n
        q = q ÷ n
        j = q % n
        k = q ÷ n
        density = max(rho[c], eps(T))
        x = T(i) - dtdx * mx[c] / density
        y = T(j) - dtdx * my[c] / density
        z = T(k) - dtdx * mz[c] / density
        i0 = unsafe_trunc(Int32, floor(x))
        j0 = unsafe_trunc(Int32, floor(y))
        k0 = unsafe_trunc(Int32, floor(z))
        sampled = _passive_trilinear(
            species1, species2, n, i0, j0, k0,
            x - T(i0), y - T(j0), z - T(k0),
        )
        q2 = min(max(sampled[2], zero(T)), maximum_total)
        q1 = min(max(sampled[1], zero(T)), maximum_total - q2)
        out1[c] = density * q1
        out2[c] = density * q2
    end
end

@kernel function _advect_species_fraction_k!(out,
        @Const(species), @Const(rho), @Const(mx), @Const(my), @Const(mz),
        n::Int32, dtdx, maximum_total)
    c = @index(Global)
    @inbounds begin
        T = eltype(rho)
        q = Int32(c - 1)
        i = q % n
        q = q ÷ n
        j = q % n
        k = q ÷ n
        density = max(rho[c], eps(T))
        x = T(i) - dtdx * mx[c] / density
        y = T(j) - dtdx * my[c] / density
        z = T(k) - dtdx * mz[c] / density
        i0 = unsafe_trunc(Int32, floor(x))
        j0 = unsafe_trunc(Int32, floor(y))
        k0 = unsafe_trunc(Int32, floor(z))
        onei = Int32(1)
        p000 = species[_passive_index(i0,        j0,        k0,        n)]
        p100 = species[_passive_index(i0 + onei, j0,        k0,        n)]
        p010 = species[_passive_index(i0,        j0 + onei, k0,        n)]
        p110 = species[_passive_index(i0 + onei, j0 + onei, k0,        n)]
        p001 = species[_passive_index(i0,        j0,        k0 + onei, n)]
        p101 = species[_passive_index(i0 + onei, j0,        k0 + onei, n)]
        p011 = species[_passive_index(i0,        j0 + onei, k0 + onei, n)]
        p111 = species[_passive_index(i0 + onei, j0 + onei, k0 + onei, n)]
        sampled = _passive_trilinear_component(
            p000, p100, p010, p110, p001, p101, p011, p111,
            x - T(i0), y - T(j0), z - T(k0),
        )
        out[c] = density * min(max(sampled, zero(T)), maximum_total)
    end
end

"""
    conserved_species_to_fraction!(species, rho; maximum_total=1)

Convert one conserved species mass-density field to a bounded gas mass fraction
in place. This is the single-field counterpart of
`conserved_species_to_fractions!`.
"""
function conserved_species_to_fraction!(species, rho; maximum_total::Real=1)
    length(species) == length(rho) || error("species field must match gas density")
    be = KA.get_backend(rho)
    T = eltype(rho)
    _conserved_species_to_fraction_k!(be)(
        species, rho, T(maximum_total); ndrange=length(rho)
    )
    KA.synchronize(be)
    species
end

"""
    conserved_species_to_fractions!(species1, species2, rho; maximum_total=1)

Convert two conserved species mass-density fields to bounded gas mass fractions
in place. Call immediately before the MHD step; the fields then remain valid
material labels while the gas density changes.
"""
function conserved_species_to_fractions!(species1, species2, rho;
                                          maximum_total::Real=1)
    be = KA.get_backend(rho)
    T = eltype(rho)
    _conserved_species_to_fractions_k!(be)(
        species1, species2, rho, T(maximum_total); ndrange=length(rho)
    )
    KA.synchronize(be)
    species1, species2
end

"""
    advect_species_fractions!(species1, species2, scratch1, scratch2, s, dt;
                              maximum_total=1)

Backtrace two mass fractions with the post-step cell velocity, restore their
conserved mass densities using the new gas density, and copy them back. The
bounded trilinear remap preserves a uniform composition exactly and uses caller
scratch, so it adds no full-grid allocation.
"""
function advect_species_fractions!(species1, species2, scratch1, scratch2,
                                   s::MHDState{T}, dt::Real;
                                   maximum_total::Real=1) where {T}
    length(species1) == ncells(s) == length(species2) ||
        error("species fields must match the MHD state")
    length(scratch1) == ncells(s) == length(scratch2) ||
        error("species scratch fields must match the MHD state")
    n = s.dims[1]
    all(==(n), s.dims) || error("species remap currently requires a cubic grid")
    dtdx = T(dt) / s.dx
    _advect_species_fractions_k!(s.be)(
        scratch1, scratch2, species1, species2,
        s.U[1], s.U[2], s.U[3], s.U[4], Int32(n), dtdx, T(maximum_total);
        ndrange=ncells(s),
    )
    KA.synchronize(s.be)
    copyto!(species1, scratch1)
    copyto!(species2, scratch2)
    KA.synchronize(s.be)
    species1, species2
end

"""
    advect_species_fraction!(species, scratch, s, dt; maximum_total=1)

Backtrace one bounded mass fraction with the post-step cell velocity and restore
its conserved mass density using the new gas density. Caller-owned scratch keeps
the remap allocation free.
"""
function advect_species_fraction!(species, scratch, s::MHDState{T}, dt::Real;
                                  maximum_total::Real=1) where {T}
    length(species) == ncells(s) || error("species field must match the MHD state")
    length(scratch) == ncells(s) || error("species scratch must match the MHD state")
    n = s.dims[1]
    all(==(n), s.dims) || error("species remap currently requires a cubic grid")
    dtdx = T(dt) / s.dx
    _advect_species_fraction_k!(s.be)(
        scratch, species, s.U[1], s.U[2], s.U[3], s.U[4], Int32(n), dtdx,
        T(maximum_total); ndrange=ncells(s),
    )
    KA.synchronize(s.be)
    copyto!(species, scratch)
    KA.synchronize(s.be)
    species
end
