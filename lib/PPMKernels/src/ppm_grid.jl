# ── Phase 4 — ppm_grid: 3-D directional-split PPM on a uniform grid ───────────
# Assembles the certified 1-D sweep (`ppm_sweep_1d!`) into a 3-D update by
# directional (Strang) splitting. Because directional splitting makes a single
# axis-sweep a BATCH of independent 1-D pencils, each `sweep_axis!` is exactly
# `ppm_sweep_1d_full!` applied to every pencil along that axis — which is how it
# is certified (pencil-wise; there is no 3-D Fortran reference).
#
# Layout: fields are flat column-major (nx,ny,nz) vectors, A[i,j,k] at
# i + nx(j-1) + nx·ny(k-1). The x-sweep is contiguous (no transpose); the y/z
# sweeps transpose the swept axis to the lead, run the batched sweep, transpose
# back. Velocities rotate cyclically per Enzo's xyz EulerSweeps:
#   x: (u,v,w)=(vx,vy,vz)   y: (vy,vz,vx)   z: (vz,vx,vy)

export sweep_axis!, ppm_step_3d!, total_mass

# ── 3-D axis-permuting gather (transpose) ────────────────────────────────────
@kernel function _gather3!(dst, @Const(src), m1::Int, m2::Int, sa::Int, sb::Int, sc::Int)
    g = @index(Global, Linear)
    a = (g - 1) % m1 + 1
    t = (g - 1) ÷ m1
    b = t % m2 + 1
    c = t ÷ m2 + 1
    @inbounds dst[g] = src[1 + (a - 1) * sa + (b - 1) * sb + (c - 1) * sc]
end

_invperm3(p::NTuple{3,Int}) = ntuple(q -> findfirst(==(q), p), 3)

"`transpose3(src, dims, perm)` → a fresh array whose axis k is `src` axis `perm[k]`."
function transpose3(src, dims::NTuple{3,Int}, perm::NTuple{3,Int})
    str = (1, dims[1], dims[1] * dims[2])
    m = (dims[perm[1]], dims[perm[2]], dims[perm[3]])
    dst = _scratch(src, prod(m); zero = false)        # fully written by the gather
    be = KA.get_backend(src)
    _gather3!(be)(dst, src, m[1], m[2], str[perm[1]], str[perm[2]], str[perm[3]];
                  ndrange = prod(m))
    return dst
end

# inverse transpose written DIRECTLY into `dst` (the original-layout state array):
# one gather pass, vs the old `dst .= transpose3(...)` which gathered into a temp
# and then copied. Saves a full-grid read+write per written field per y/z sweep.
function _untranspose_into!(dst, slab, dims::NTuple{3,Int}, perm::NTuple{3,Int})
    invp = _invperm3(perm)
    md = (dims[perm[1]], dims[perm[2]], dims[perm[3]])      # slab dims
    mstr = (1, md[1], md[1] * md[2])
    be = KA.get_backend(dst)
    _gather3!(be)(dst, slab, dims[1], dims[2], mstr[invp[1]], mstr[invp[2]], mstr[invp[3]];
                  ndrange = length(dst))
    return dst
end

# per-axis cyclic spatial permutation (swept axis first) and velocity roles
_axis_perm(axis) = axis == 1 ? (1, 2, 3) : axis == 2 ? (2, 1, 3) : (3, 1, 2)

"""
    sweep_axis!(d, e, ge, vx, vy, vz, p, gr, dims, ng, axis; dt, gamma, kw...)

One directional PPM update of the whole 3-D grid along `axis` (1=x,2=y,3=z),
mutating `d,e,ge,vx,vy,vz` IN PLACE. `dims=(nx,ny,nz)`, `ng` ghost zones each
side; the active region along the swept axis is `ng+1 .. dims[axis]-ng`, all
transverse pencils. `p` is the precomputed pressure, `gr` the acceleration. `kw`
are the `ppm_sweep_1d!` physics flags. Cell widths are uniform `dx` (kw `dx=`).
"""
function sweep_axis!(d, e, ge, vx, vy, vz, p, gr, dims::NTuple{3,Int}, ng::Int, axis::Int;
                     dt::Real, gamma::Real, dx::Real = 1.0, kw...)
    na = dims[axis]
    ntr = (dims[1] * dims[2] * dims[3]) ÷ na
    dxi = _axisdxi(d, na, dx)
    # cyclic velocity roles: (normal, transverse1, transverse2)
    vu, vv, vw = axis == 1 ? (vx, vy, vz) : axis == 2 ? (vy, vz, vx) : (vz, vx, vy)
    i1, i2 = ng + 1, na - ng

    if axis == 1                                   # x is contiguous — sweep in place
        ppm_sweep_1d!(d, e, ge, vu, vv, vw, p, gr, dxi;
                      idim = na, i1 = i1, i2 = i2, jdim = ntr, dt = dt, gamma = gamma, kw...)
    else
        perm = _axis_perm(axis)
        dT  = transpose3(d, dims, perm);  eT = transpose3(e, dims, perm)
        geT = transpose3(ge, dims, perm)
        uT  = transpose3(vu, dims, perm); vT = transpose3(vv, dims, perm); wT = transpose3(vw, dims, perm)
        pT  = transpose3(p, dims, perm);  grT = transpose3(gr, dims, perm)
        ppm_sweep_1d!(dT, eT, geT, uT, vT, wT, pT, grT, dxi;
                      idim = na, i1 = i1, i2 = i2, jdim = ntr, dt = dt, gamma = gamma, kw...)
        # scatter the mutated slabs back into the original-layout arrays (one pass)
        _untranspose_into!(d, dT, dims, perm);  _untranspose_into!(e, eT, dims, perm)
        _untranspose_into!(ge, geT, dims, perm)
        _untranspose_into!(vu, uT, dims, perm); _untranspose_into!(vv, vT, dims, perm)
        _untranspose_into!(vw, wT, dims, perm)
    end
    return nothing
end

# length-`na` uniform cell-width vector on the same backend as `proto`
function _axisdxi(proto, na::Int, dx::Real)
    a = _scratch(proto, na; zero = false); fill!(a, eltype(proto)(dx)); a
end

# recompute pressure over the WHOLE grid (purely local EOS), into `p`
function _recompute_pressure!(p, d, e, vx, vy, vz, gamma)
    N = length(p)
    pgas2d!(p, d, e, vx, vy, vz; idim = N, i1 = 1, i2 = N, j1 = 1, j2 = 1,
            gamma = gamma, pmin = eltype(p)(1e-20))
    return p
end

"""
    ppm_step_3d!(d, e, ge, vx, vy, vz, grx, gry, grz, dims, ng;
                 dt, gamma, order=(1,2,3), bc!=nothing, kw...)

A full directional-split timestep: pressure is recomputed before each axis sweep
and the three sweeps are applied in `order` (alternate `(1,2,3)`/`(3,2,1)` across
steps for second-order Strang accuracy). Gravity is DIRECTIONAL — each sweep uses
its own acceleration component (`grx`/`gry`/`grz`); pass zero arrays when gravity
is off. `bc!(d,e,ge,vx,vy,vz)` (optional) refills the ghost zones before EACH
directional sweep (e.g. a periodic wrap) — needed for a conservative standalone run;
omit it under a framework (e.g. Enzo) that sets the boundaries externally. Mutates
the state in place.
"""
function ppm_step_3d!(d, e, ge, vx, vy, vz, grx, gry, grz, dims::NTuple{3,Int}, ng::Int;
                      dt::Real, gamma::Real, order::NTuple{3,Int} = (1, 2, 3), bc! = nothing, kw...)
    p = similar(d)
    gr = (grx, gry, grz)
    be = KA.get_backend(d)
    for axis in order
        # the ~50 kernels of a sweep pipeline (no per-kernel syncs); ONE sync per
        # sweep ensures it has completed before its scratch is recycled.
        KA.synchronize(be)
        _pool_reset!()
        # directional split: refill the ghost zones consistent with the state the
        # PREVIOUS sweep just updated (e.g. a periodic wrap), else the inter-sweep
        # ghosts are stale and the boundary is non-conservative.
        bc! === nothing || bc!(d, e, ge, vx, vy, vz)
        _recompute_pressure!(p, d, e, vx, vy, vz, gamma)
        sweep_axis!(d, e, ge, vx, vy, vz, p, gr[axis], dims, ng, axis; dt = dt, gamma = gamma, kw...)
    end
    KA.synchronize(be)                    # flush the final sweep
    return nothing
end

"`total_mass(d, dims, ng, dx)` — Σ ρ·dV over the active (non-ghost) interior."
function total_mass(d, dims::NTuple{3,Int}, ng::Int, dx::Real)
    nx, ny, nz = dims
    h = to_host(d)
    T = eltype(h)
    dV = T(dx)^3
    s = zero(T)
    @inbounds for k in (ng + 1):(nz - ng), j in (ng + 1):(ny - ng), i in (ng + 1):(nx - ng)
        s += h[i + nx * (j - 1) + nx * ny * (k - 1)]
    end
    return s * dV
end
