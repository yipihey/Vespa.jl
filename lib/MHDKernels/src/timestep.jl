# ── CFL timestep: max over cells & axes of (|v_d| + c_fast,d) ─────────────────
export compute_dt, max_wavespeed

@kernel function _wavespeed_kernel!(w, @Const(a1),@Const(a2),@Const(a3),@Const(a4),@Const(a5),
        @Const(a6),@Const(a7),@Const(a8),@Const(a9), γ::T, smallr::T, pfl::T) where {T}
    c = @index(Global, Linear)
    @inbounds begin
        q = cons2prim((a1[c],a2[c],a3[c],a4[c],a5[c],a6[c],a7[c],a8[c],a9[c]), γ, smallr, pfl)
        sx = abs(q[2]) + fast_speed(q, γ, q[6])
        sy = abs(q[3]) + fast_speed(q, γ, q[7])
        sz = abs(q[4]) + fast_speed(q, γ, q[8])
        w[c] = max(sx, max(sy, sz))
    end
end

"`max_wavespeed(s)` — the maximum (|v|+c_fast) over the grid (max signal speed)."
function max_wavespeed(s::MHDState{T}) where {T}
    w = device_zeros(s.be, T, (ncells(s),))
    _wavespeed_kernel!(s.be, 256)(w, s.U..., s.γ, s.smallr, s.pfl; ndrange = ncells(s))
    KA.synchronize(s.be)
    smax = maximum(w)
    return smax
end

"""
    compute_dt(s; cfl=0.4) -> (dt, smax)

CFL timestep `dt = cfl·dx/smax` and the max signal speed `smax` (reused as the GLM
cleaning speed `ch`).
"""
function compute_dt(s::MHDState{T}; cfl::Real = 0.4) where {T}
    smax = max_wavespeed(s)
    dt = T(cfl) * s.dx / max(smax, eps(T))
    return dt, smax
end
