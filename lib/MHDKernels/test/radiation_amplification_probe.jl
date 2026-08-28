using MHDKernels
using LinearAlgebra
using Printf

const PROBE_BACKEND = Symbol(get(ENV, "MHD_RADIATION_PROBE_BACKEND", "cpu"))
PROBE_BACKEND === :metal && (@eval using Metal)

function mode_amplitudes(s, mode)
    n = s.dims[1]
    h = fields_to_host(s)
    rho = reshape(h[1], s.dims)
    mx = reshape(h[2], s.dims)
    energy = reshape(h[5], s.dims)
    gamma = Float64(s.γ)
    arho = 0.0
    avx = 0.0
    ap = 0.0
    @inbounds for i in 1:n
        phase = 2pi * mode * (i - 1) / n
        r = Float64(rho[i, 1, 1])
        vx = Float64(mx[i, 1, 1]) / r
        p = (gamma - 1) * (Float64(energy[i, 1, 1]) -
             0.5 * Float64(mx[i, 1, 1])^2 / r)
        arho += (r - 1) * cos(phase)
        avx += vx * sin(phase)
        ap += (p - 2000) * cos(phase)
    end
    return 2 / n .* (arho, avx, ap)
end

function init_mode!(s, mode, variable, amplitude)
    n = s.dims[1]
    gamma = Float64(s.γ)
    @inbounds for k in 1:n, j in 1:n, i in 1:n
        phase = 2pi * mode * (i - 1) / n
        rho = 1.0 + (variable == 1 ? amplitude * cos(phase) : 0.0)
        vx = variable == 2 ? amplitude * sin(phase) : 0.0
        p = 2000.0 + (variable == 3 ? amplitude * cos(phase) : 0.0)
        c = linidx(n, n, i, j, k)
        s.U[1][c] = rho
        s.U[2][c] = rho * vx
        s.U[3][c] = 0
        s.U[4][c] = 0
        s.U[5][c] = p / (gamma - 1) + 0.5 * rho * vx^2
        s.U[6][c] = 0
        s.U[7][c] = 0
        s.U[8][c] = 0
        s.U[9][c] = 0
    end
    return s
end

function amplification_matrix(::Type{T}, n, mode; amplitude=1e-3) where T
    dt = T(8.44e-8)
    drag_impulse = T(1.07e5)
    rc = RadiationClosure(inertia_ratio=T(1.996264e-4),
        mean_free_path=T(1.475060e-4), photon_speed=T(3.688e4),
        gas_sound_speed=T(57.735), bridge_q2=T(5),
        longitudinal_viscosity=T(4 / 3))
    matrix = zeros(Float64, 3, 3)
    scales = (amplitude, amplitude, amplitude * 2000)
    for variable in 1:3
        s = allocate_state(backend(PROBE_BACKEND), T, (n, n, n); dx=1 / n,
                           gamma=5 / 3, riemann=:hll, recon=:ppm)
        if PROBE_BACKEND === :cpu
            init_mode!(s, mode, variable, scales[variable])
        else
            host = allocate_state(backend(:cpu), T, (n, n, n); dx=1 / n,
                                  gamma=5 / 3, riemann=:hll, recon=:ppm)
            init_mode!(host, mode, variable, scales[variable])
            for field in 1:9
                copyto!(s.U[field], host.U[field])
            end
        end
        ws = allocate_radiation_workspace(s)
        integrator = Symbol(get(ENV, "MHD_RADIATION_PROBE_INTEGRATOR", "ref"))
        step_radiation_godunov!(s, ws, dt; drag_impulse=drag_impulse,
                                closure=rc, ch=T(100), integrator=integrator)
        matrix[:, variable] .= mode_amplitudes(s, mode) ./ scales
    end
    return matrix
end

n = parse(Int, get(ENV, "MHD_RADIATION_PROBE_N", "16"))
precisions = PROBE_BACKEND === :cpu ? (Float64, Float32) : (Float32,)
for T in precisions
    println("precision=$T N=$n")
    for mode in 1:div(n, 2)
        matrix = amplification_matrix(T, n, mode)
        values = eigvals(matrix)
        @printf("mode=%2d spectral_radius=%.12e eig=%s\n", mode,
                maximum(abs, values), repr(values))
        get(ENV, "MHD_RADIATION_PROBE_MATRIX", "0") == "1" && display(matrix)
    end
end
