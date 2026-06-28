# ── Time integration driver: step! / evolve! ─────────────────────────────────
# GLM cleaning schedule (Dedner+ 2002): the hyperbolic cleaning speed `ch` is the
# max signal speed (so the cleaning waves ride the CFL limit), and ψ is damped each
# step by `decay = exp(-cr·ch·dt/dx)` with cr≈0.18 (the mixed-GLM parabolic rate).
export step!, evolve!

const GLM_CR = 0.18

"""
    step!(s, dt; ch, integrator=:ref)

Advance the state one step of size `dt` with GLM cleaning speed `ch`. `integrator`
selects `:ref` (portable per-cell) or `:cube` (GPU shared-memory; added next).
"""
function step!(s::MHDState{T}, dt::Real; ch::Real, integrator::Symbol = :ref) where {T}
    decay = exp(-T(GLM_CR) * T(ch) * T(dt) / s.dx)
    if integrator === :ref
        step_ref!(s, dt; ch = ch, decay = decay)
    elseif integrator === :cube
        step_cube!(s, dt; ch = ch, decay = decay)
    else
        error("unknown integrator :$integrator (have :ref, :cube)")
    end
end

"""
    evolve!(s, tfinal; cfl=0.4, integrator=:ref, callback=nothing) -> (t, nsteps)

Integrate to `tfinal` with CFL-limited steps. `callback(s, t, n)` (if given) runs
after each step (diagnostics/output). Returns the final time and step count.
"""
function evolve!(s::MHDState{T}, tfinal::Real; cfl::Real = 0.4,
                 integrator::Symbol = :ref, callback = nothing) where {T}
    t = zero(Float64); n = 0; tf = Float64(tfinal)
    while t < tf * (1 - 1e-9)
        dt, smax = compute_dt(s; cfl = cfl)
        dtf = Float64(dt)
        if t + dtf > tf
            dtf = tf - t
        end
        step!(s, dtf; ch = smax, integrator = integrator)
        t += dtf; n += 1
        callback === nothing || callback(s, t, n)
    end
    return t, n
end
