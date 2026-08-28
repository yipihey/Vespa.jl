using Metal
using MHDKernels
using Printf

const N = parse(Int, get(ENV, "MHD_RADIATION_PROBE_N", "8"))
const NSTEPS = parse(Int, get(ENV, "MHD_RADIATION_PROBE_STEPS", "40000"))
const CHECK_EVERY = parse(Int, get(ENV, "MHD_RADIATION_PROBE_CHECK_EVERY", "250"))

function initial_state(n)
    gamma = 5f0 / 3f0
    host = allocate_state(backend(:cpu), Float32, (n, n, n); dx=1f0 / n,
                          gamma=gamma, riemann=:hll, recon=:ppm)
    amplitude = 4.942815f0 / sqrt(6f0)
    @inbounds for k in 1:n, j in 1:n, i in 1:n
        x = 2f0 * Float32(pi) * Float32(i - 1) / Float32(n)
        y = 2f0 * Float32(pi) * Float32(j - 1) / Float32(n)
        z = 2f0 * Float32(pi) * Float32(k - 1) / Float32(n)
        bx = amplitude * (sin(y) + sin(z))
        by = amplitude * (sin(z) + sin(x))
        bz = amplitude * (sin(x) + sin(y))
        c = linidx(n, n, i, j, k)
        host.U[1][c] = 1
        host.U[2][c] = 0
        host.U[3][c] = 0
        host.U[4][c] = 0
        host.U[5][c] = 2000f0 / (gamma - 1f0) + 0.5f0 * (bx^2 + by^2 + bz^2)
        host.U[6][c] = bx
        host.U[7][c] = by
        host.U[8][c] = bz
        host.U[9][c] = 0
    end
    device = allocate_state(backend(:metal), Float32, (n, n, n); dx=1f0 / n,
                            gamma=gamma, riemann=:hll, recon=:ppm)
    for field in 1:9
        copyto!(device.U[field], host.U[field])
    end
    return device
end

function diagnostics(s)
    h = fields_to_host(s)
    rho = h[1]
    energy = h[5]
    momentum_max = maximum(maximum(abs, h[d]) for d in 2:4)
    return extrema(rho), extrema(energy), momentum_max,
           all(all(isfinite, field) for field in h)
end

s = initial_state(N)
ws = allocate_radiation_workspace(s)
dt = 8.44f-8
drag_impulse = 1.07f5
rc = RadiationClosure(inertia_ratio=1.996264f-4,
    mean_free_path=1.475060f-4, photon_speed=3.688f4,
    gas_sound_speed=57.735f0, bridge_q2=5f0,
    longitudinal_viscosity=4f0 / 3f0)

for step in 1:NSTEPS
    step_radiation_godunov!(s, ws, dt; drag_impulse=drag_impulse,
                            closure=rc, ch=100f0, integrator=:cube)
    if step == 1 || step % CHECK_EVERY == 0
        rho, energy, momentum_max, finite = diagnostics(s)
        @printf("step=%d rho=[%.8e,%.8e] E=[%.8e,%.8e] mommax=%.8e finite=%s\n",
                step, rho..., energy..., momentum_max, string(finite))
        flush(stdout)
        finite || error("non-finite state at step $step")
    end
end
