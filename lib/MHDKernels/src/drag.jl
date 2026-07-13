export apply_linear_drag!, apply_exponential_drag_increment!, step_drag_strang!,
       step_drag_exponential!, DragRegimeController, update_drag_regime!

mutable struct DragRegimeController
    terminal::Bool
    initialized::Bool
    consecutive::Int
    exit_drag_over_omega::Float64
    enter_drag_over_omega::Float64
    min_va_over_cs::Float64
    confirm_checks::Int
    reenter_terminal::Bool
end

function DragRegimeController(; exit_drag_over_omega::Real=32,
                              enter_drag_over_omega::Real=64,
                              min_va_over_cs::Real=0,
                              confirm_checks::Integer=2,
                              reenter_terminal::Bool=false)
    0 < exit_drag_over_omega < enter_drag_over_omega ||
        error("drag regime hysteresis requires 0 < exit < enter")
    min_va_over_cs >= 0 || error("min_va_over_cs must be non-negative")
    confirm_checks >= 1 || error("confirm_checks must be positive")
    return DragRegimeController(false, false, 0, Float64(exit_drag_over_omega),
                                Float64(enter_drag_over_omega), Float64(min_va_over_cs),
                                Int(confirm_checks), reenter_terminal)
end

"""
    update_drag_regime!(controller, drag_over_omega, va_over_cs)

Update the hysteretic terminal/full-MHD regime decision at one sampled check.
Returns `(terminal, transition)`, where `transition` is `:none`,
`:terminal_to_full`, or `:full_to_terminal`. By default the handoff is one-way;
opt-in reentry still requires the wider enter threshold and consecutive checks.
"""
function update_drag_regime!(controller::DragRegimeController,
                             drag_over_omega::Real, va_over_cs::Real)
    isfinite(drag_over_omega) && drag_over_omega >= 0 ||
        error("drag_over_omega must be finite and non-negative")
    isfinite(va_over_cs) && va_over_cs >= 0 ||
        error("va_over_cs must be finite and non-negative")
    ratio = Float64(drag_over_omega)
    magnetization = Float64(va_over_cs)
    if !controller.initialized
        controller.terminal = magnetization >= controller.min_va_over_cs &&
                              ratio >= controller.enter_drag_over_omega
        controller.initialized = true
        controller.consecutive = 0
        return (terminal=controller.terminal, transition=:none)
    end

    if controller.terminal
        wants_full = magnetization < controller.min_va_over_cs ||
                     ratio <= controller.exit_drag_over_omega
        controller.consecutive = wants_full ? controller.consecutive + 1 : 0
        if controller.consecutive >= controller.confirm_checks
            controller.terminal = false
            controller.consecutive = 0
            return (terminal=false, transition=:terminal_to_full)
        end
    elseif controller.reenter_terminal
        wants_terminal = magnetization >= controller.min_va_over_cs &&
                         ratio >= controller.enter_drag_over_omega
        controller.consecutive = wants_terminal ? controller.consecutive + 1 : 0
        if controller.consecutive >= controller.confirm_checks
            controller.terminal = true
            controller.consecutive = 0
            return (terminal=true, transition=:full_to_terminal)
        end
    else
        controller.consecutive = 0
    end
    return (terminal=controller.terminal, transition=:none)
end

@kernel function _linear_drag_k!(mx, my, mz, E, @Const(rho),
                                 vx0, vy0, vz0, decay, smallr, smallE)
    c = @index(Global)
    @inbounds begin
        T = eltype(E)
        r = max(rho[c], smallr)
        ox = mx[c]; oy = my[c]; oz = mz[c]
        tx = r * vx0; ty = r * vy0; tz = r * vz0
        ek0 = T(0.5) * (ox*ox + oy*oy + oz*oz) / r
        nx = tx + (ox - tx) * decay
        ny = ty + (oy - ty) * decay
        nz = tz + (oz - tz) * decay
        ek1 = T(0.5) * (nx*nx + ny*ny + nz*nz) / r
        mx[c] = nx; my[c] = ny; mz[c] = nz
        E[c] = max(E[c] - ek0 + ek1, smallE)
    end
end

"""
    apply_linear_drag!(s, decay; target_velocity=(0, 0, 0))

Apply the exact constant-coefficient solution for linear momentum drag over an
interval whose velocity decay factor is `decay = exp(-Gamma*dt)`. The kinetic
energy removed by drag is removed from total energy; thermal and magnetic energy
are unchanged. This operation is stable for arbitrarily large `Gamma*dt`.
"""
function apply_linear_drag!(s::MHDState{T}, decay::Real;
                            target_velocity=(0, 0, 0)) where {T}
    isfinite(decay) && 0 <= decay <= 1 ||
        error("linear drag decay must be finite and in [0,1]")
    vx0, vy0, vz0 = target_velocity
    _linear_drag_k!(s.be)(s.U[2], s.U[3], s.U[4], s.U[5], s.U[1],
                          T(vx0), T(vy0), T(vz0), T(decay), s.smallr,
                          T(1e-30); ndrange=ncells(s))
    KA.synchronize(s.be)
    return s
end

@inline function _drag_phi1(q::T) where {T}
    aq = abs(q)
    return aq < T(1e-4) ? one(T) - q / T(2) + q*q / T(6) : -expm1(-q) / q
end

@kernel function _exponential_drag_increment_k!(mx, my, mz, E,
                                                 @Const(rho),
                                                 @Const(old_rho),
                                                 @Const(old_mx), @Const(old_my),
                                                 @Const(old_mz),
                                                 @Const(Bx), @Const(By), @Const(Bz),
                                                 impulse, vx0, vy0, vz0,
                                                 smallr, smallE)
    c = @index(Global)
    @inbounds begin
        T = eltype(E)
        rn = max(rho[c], smallr)
        ro = max(old_rho[c], smallr)
        decay = exp(-impulse)
        phi1 = _drag_phi1(impulse)
        vox = old_mx[c] / ro; voy = old_my[c] / ro; voz = old_mz[c] / ro
        vtx = mx[c] / rn; vty = my[c] / rn; vtz = mz[c] / rn
        ek0 = T(0.5) * (mx[c]^2 + my[c]^2 + mz[c]^2) / rn
        eb = T(0.5) * (Bx[c]^2 + By[c]^2 + Bz[c]^2)
        nonkinetic = max(E[c] - ek0, eb + smallE)
        vnx = vx0 + decay * (vox - vx0) + phi1 * (vtx - vox)
        vny = vy0 + decay * (voy - vy0) + phi1 * (vty - voy)
        vnz = vz0 + decay * (voz - vz0) + phi1 * (vtz - voz)
        nmx = rn * vnx; nmy = rn * vny; nmz = rn * vnz
        ek1 = T(0.5) * (nmx*nmx + nmy*nmy + nmz*nmz) / rn
        mx[c] = nmx; my[c] = nmy; mz[c] = nmz
        E[c] = max(nonkinetic + ek1, eb + smallE)
    end
end

"""
    apply_exponential_drag_increment!(s, old_U, drag_impulse;
                                      target_velocity=(0, 0, 0))

Correct a no-drag trial update in `s.U` with the exponential solution of
`dv/dt = a - Gamma*(v-v_target)`, treating the trial velocity increment as the
constant-force impulse. `old_U` is the state before the trial update. Density,
magnetic field, and trial internal energy are retained. This has the correct
zero-drag limit and tends to the terminal velocity increment divided by
`Gamma*dt` for stiff drag.
"""
function apply_exponential_drag_increment!(s::MHDState{T}, old_U,
                                           drag_impulse::Real;
                                           target_velocity=(0, 0, 0)) where {T}
    isfinite(drag_impulse) && drag_impulse >= 0 ||
        error("drag impulse must be finite and non-negative")
    length(old_U) == NVAR || error("old_U must contain $NVAR conserved fields")
    vx0, vy0, vz0 = target_velocity
    _exponential_drag_increment_k!(s.be)(
        s.U[2], s.U[3], s.U[4], s.U[5], s.U[1],
        old_U[1], old_U[2], old_U[3], old_U[4],
        s.U[6], s.U[7], s.U[8], T(drag_impulse),
        T(vx0), T(vy0), T(vz0), s.smallr, T(1e-30);
        ndrange=ncells(s))
    KA.synchronize(s.be)
    return s
end

"""
    step_drag_strang!(s, dt; drag_impulse, ch, glm_cr=0.18,
                      integrator=:cube, target_velocity=(0, 0, 0))

Advance one full ideal-MHD step with exact linear-drag half steps on either side.
`drag_impulse` is the dimensionless integral `integral(Gamma dt)` over the full
step. For a time-dependent rate, callers should supply a midpoint or otherwise
second-order estimate of that integral.
"""
function step_drag_strang!(s::MHDState, dt::Real; drag_impulse::Real, ch::Real,
                           glm_cr::Real=0.18, integrator::Symbol=:cube,
                           target_velocity=(0, 0, 0))
    isfinite(drag_impulse) && drag_impulse >= 0 ||
        error("drag impulse must be finite and non-negative")
    half_decay = exp(-0.5 * Float64(drag_impulse))
    apply_linear_drag!(s, half_decay; target_velocity=target_velocity)
    step!(s, dt; ch=ch, glm_cr=glm_cr, integrator=integrator)
    apply_linear_drag!(s, half_decay; target_velocity=target_velocity)
    return s
end

"""
    step_drag_exponential!(s, dt; drag_impulse, ch, glm_cr=0.18,
                           integrator=:cube, target_velocity=(0, 0, 0))

Advance a no-drag MHD trial and integrate its velocity impulse together with
linear drag using the exact exponential constant-force update. Unlike Strang
splitting, the MHD force is weighted by `phi1(-Gamma*dt)` rather than being
exponentially erased when `Gamma*dt` is large. The ping-pong state left in
`s.scratch` by `step!` supplies the old state without another full-grid copy.
"""
function step_drag_exponential!(s::MHDState, dt::Real; drag_impulse::Real,
                                ch::Real, glm_cr::Real=0.18,
                                integrator::Symbol=:cube,
                                target_velocity=(0, 0, 0))
    isfinite(drag_impulse) && drag_impulse >= 0 ||
        error("drag impulse must be finite and non-negative")
    step!(s, dt; ch=ch, glm_cr=glm_cr, integrator=integrator)
    apply_exponential_drag_increment!(s, s.scratch, drag_impulse;
                                      target_velocity=target_velocity)
    return s
end
