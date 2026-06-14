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
    Tp = eltype(px)
    _cic_deposit_kernel!(be)(ρ, px, py, pz, vx, vy, vz, mass,
                             Int(N), Tp(disp), Tp(shift); ndrange = length(mass))
    return ρ
end
