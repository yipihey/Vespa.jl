using MHDKernels, KernelAbstractions, Test

@testset "continuity midpoint density predictor" begin
    N = 256
    dims = (N, 1, 1)
    s = allocate_state(backend(:cpu), Float32, dims; dx=1 / N)
    amp = 0.1f0
    velocity = 0.3f0

    function prediction_error(dt)
        t1 = 0.2f0
        rho = Vector{Float32}(undef, N)
        mx = similar(rho)
        exact = similar(rho)
        for i in 1:N
            x = Float32((i - 0.5) / N)
            rho[i] = 1f0 + amp * sinpi(2f0 * (x - velocity * t1))
            mx[i] = velocity * rho[i]
            exact[i] = 1f0 + amp * sinpi(2f0 * (x - velocity * (t1 - 0.5f0 * dt)))
        end
        copyto!(s.U[1], rho)
        copyto!(s.U[2], mx)
        fill!(s.U[3], 0f0); fill!(s.U[4], 0f0)
        predict_density_backward!(s.scratch[1], s, 0.5f0 * dt)
        return sqrt(sum(abs2, Array(s.scratch[1]) .- exact) / sum(abs2, exact .- 1f0))
    end

    coarse = prediction_error(0.04f0)
    fine = prediction_error(0.02f0)
    @test fine < 3.0f-4
    @test coarse / fine > 3.5
end

@testset "cell-centred gas gravity kick" begin
    N = 64
    s = allocate_state(backend(:cpu), Float32, (N, 1, 1); dx=1 / N)
    phi = Vector{Float32}(undef, N)
    fill!(s.U[1], 2f0)
    fill!(s.U[2], 0.3f0); fill!(s.U[3], -0.2f0); fill!(s.U[4], 0.1f0)
    fill!(s.U[6], 0.04f0); fill!(s.U[7], -0.03f0); fill!(s.U[8], 0.02f0)
    kinetic0 = (0.3f0^2 + 0.2f0^2 + 0.1f0^2) / 4f0
    magnetic = 0.5f0 * (0.04f0^2 + 0.03f0^2 + 0.02f0^2)
    fill!(s.U[5], 1.7f0 + kinetic0 + magnetic)
    for i in 1:N
        phi[i] = 0.07f0 * sinpi(2f0 * (Float32(i) - 0.5f0) / Float32(N))
    end
    h0 = fields_to_host(s)
    dt = 0.013f0
    apply_cell_center_gravity!(s, phi, dt)
    h1 = fields_to_host(s)
    inv2dx = 0.5f0 * N
    for i in 1:N
        im = i == 1 ? N : i - 1
        ip = i == N ? 1 : i + 1
        gx = -(phi[ip] - phi[im]) * inv2dx
        @test h1[2][i] ≈ h0[2][i] + 2f0 * gx * dt rtol=3f-6 atol=2f-7
    end
    eint0 = h0[5] .- 0.5f0 .* (h0[2].^2 .+ h0[3].^2 .+ h0[4].^2) ./ h0[1] .-
            0.5f0 .* (h0[6].^2 .+ h0[7].^2 .+ h0[8].^2)
    eint1 = h1[5] .- 0.5f0 .* (h1[2].^2 .+ h1[3].^2 .+ h1[4].^2) ./ h1[1] .-
            0.5f0 .* (h1[6].^2 .+ h1[7].^2 .+ h1[8].^2)
    @test maximum(abs, eint1 .- eint0) < 3f-7

    one_kick = allocate_state(backend(:cpu), Float32, (N, 1, 1); dx=1 / N)
    two_kicks = allocate_state(backend(:cpu), Float32, (N, 1, 1); dx=1 / N)
    for v in 1:9
        copyto!(one_kick.U[v], h0[v]); copyto!(two_kicks.U[v], h0[v])
    end
    apply_cell_center_gravity!(one_kick, phi, dt)
    apply_cell_center_gravity!(two_kicks, phi, dt / 2)
    apply_cell_center_gravity!(two_kicks, phi, dt / 2)
    ho = fields_to_host(one_kick); ht = fields_to_host(two_kicks)
    @test maximum(abs, ht[2] .- ho[2]) < 8f-8
    @test maximum(abs, ht[5] .- ho[5]) < 3f-7
end

@testset "lattice displacement particles preserve early-time drifts" begin
    N = 16
    np = N^3
    dxp = zeros(Float32, np); dyp = similar(dxp); dzp = similar(dxp)
    vx = ones(Float32, np); vy = zeros(Float32, np); vz = zeros(Float32, np)
    rho = zeros(Float32, np)
    initialize_lattice_displacements!(dxp, dyp, dzp, vx, vy, vz)
    fill!(vx, 1f0)

    increment = 5f-10
    for _ in 1:4096
        drift_lattice_displacements!(dxp, dyp, dzp, vx, vy, vz;
                                     coef=increment, wrap=true)
    end
    expected = Float32(4096) * increment
    @test dxp[1] ≈ expected rtol=5f-5 atol=1f-10
    @test dxp[1] > 10eps(Float32)

    fill!(dxp, 0f0)
    deposit_lattice_displacements!(rho, dxp, dyp, dzp, vx, vy, vz; N=N)
    @test rho == ones(Float32, np)

    amp_cells = 0.05f0
    for p in 1:np
        i = (p - 1) % N
        dxp[p] = amp_cells * sinpi(2f0 * (Float32(i) + 0.5f0) / Float32(N)) / Float32(N)
    end
    deposit_lattice_displacements!(rho, dxp, dyp, dzp, vx, vy, vz; N=N)
    @test sum(rho) ≈ Float32(np) rtol=2f-6
    @test sqrt(sum(abs2, rho .- 1f0) / np) > 0.005f0
end

function _terminal_ct_seed!(s; amp=0.08f0)
    N = s.dims[1]
    rho = Vector{Float32}(undef, N^3)
    Bx = similar(rho); By = similar(rho); Bz = similar(rho)
    E = similar(rho)
    eint = 1.5f0
    for k in 1:N, j in 1:N, i in 1:N
        c = linidx(N, N, i, j, k)
        x = Float32(2pi * (i - 1) / N)
        y = Float32(2pi * (j - 1) / N)
        z = Float32(2pi * (k - 1) / N)
        rho[c] = 1f0 + 0.02f0 * sin(x + y)
        Bx[c] = amp * sin(y)
        By[c] = amp * sin(z)
        Bz[c] = amp * sin(x)
        E[c] = eint + 0.5f0 * (Bx[c]^2 + By[c]^2 + Bz[c]^2)
    end
    copyto!(s.U[1], rho)
    copyto!(s.U[5], E)
    copyto!(s.U[6], Bx); copyto!(s.U[7], By); copyto!(s.U[8], Bz)
    KernelAbstractions.synchronize(s.be)
    return nothing
end

@testset "terminal CT Maxwell-stress force" begin
    N = 64
    s = allocate_state(backend(:cpu), Float32, (N, N, N); dx=1/N)
    B0 = 0.2f0
    by = zeros(Float32, N^3)
    for k in 1:N, j in 1:N, i in 1:N
        by[linidx(N, N, i, j, k)] = B0 * sinpi(2f0 * (i - 1) / N)
    end
    copyto!(s.U[7], by)
    fx, fy, fz = s.scratch[2:4]
    terminal_magnetic_force!(s, fx, fy, fz)
    hfx = Array(fx)
    exact = Vector{Float32}(undef, N)
    for i in 1:N
        x = Float32((i - 1) / N)
        exact[i] = -Float32(pi) * B0^2 * sinpi(4f0 * x)
    end
    rel_l2 = sqrt(sum(abs2, hfx[1:N] .- exact) / sum(abs2, exact))
    @test rel_l2 < 0.01
    @test maximum(abs, Array(fy)) < 2f-6
    @test maximum(abs, Array(fz)) < 2f-6
end

@testset "terminal CT preserves discrete divergence and thermal energy" begin
    N = 24
    s = allocate_state(backend(:cpu), Float32, (N, N, N); dx=1/N)
    _terminal_ct_seed!(s)
    h0 = fields_to_host(s)
    eint0 = h0[5] .- 0.5f0 .* (h0[6].^2 .+ h0[7].^2 .+ h0[8].^2)
    div0 = Float64(max_divb(s))
    b0 = copy(h0[6])
    for _ in 1:8
        nsub, vmax, eta = terminal_ct_induction!(s, 0.002;
            gamma_drag=4.0, pressure_coeff=0.2, cfl=0.4,
            dissipation=0.05, dissipation_order=4)
        @test nsub >= 1
        @test isfinite(vmax) && isfinite(eta)
    end
    h1 = fields_to_host(s)
    eint1 = h1[5] .- 0.5f0 .* (h1[6].^2 .+ h1[7].^2 .+ h1[8].^2)
    div1 = Float64(max_divb(s))
    @test maximum(abs, h1[6] .- b0) > 1f-7
    @test div1 * Float64(s.dx) < 2f-6
    @test div1 <= max(8 * div0, 2f-5)
    @test maximum(abs, eint1 .- eint0) < 2f-6
    @test all(isfinite, h1[5])
end

@testset "exact linear drag preserves thermal energy" begin
    N = 12
    s = allocate_state(backend(:cpu), Float32, (N, N, N); dx=1/N)
    fill!(s.U[1], 2f0)
    fill!(s.U[2], 3f0); fill!(s.U[3], -2f0); fill!(s.U[4], 1f0)
    fill!(s.U[6], 0.2f0); fill!(s.U[7], -0.1f0); fill!(s.U[8], 0.05f0)
    ek0 = 0.5f0 * (3f0^2 + 2f0^2 + 1f0^2) / 2f0
    eb = 0.5f0 * (0.2f0^2 + 0.1f0^2 + 0.05f0^2)
    fill!(s.U[5], 4f0 + ek0 + eb)
    apply_linear_drag!(s, exp(-3); target_velocity=(0.25, -0.5, 0.125))
    h = fields_to_host(s)
    decay = Float32(exp(-3))
    @test h[2][1] ≈ 0.5f0 + (3f0 - 0.5f0) * decay rtol=2f-6
    @test h[3][1] ≈ -1f0 + (-2f0 + 1f0) * decay rtol=2f-6
    @test h[4][1] ≈ 0.25f0 + (1f0 - 0.25f0) * decay rtol=2f-6
    ek1 = 0.5f0 * (h[2][1]^2 + h[3][1]^2 + h[4][1]^2) / h[1][1]
    eint1 = h[5][1] - ek1 - eb
    @test eint1 ≈ 4f0 atol=2f-6
end

@testset "exponential drag keeps the resolved force impulse" begin
    s = allocate_state(backend(:cpu), Float32, (4, 4, 4); dx=0.25)
    old_rho = 2f0
    old_v = (0.8f0, -0.3f0, 0.2f0)
    trial_rho = 2.5f0
    delta_v = (0.4f0, -0.2f0, 0.1f0)
    trial_v = ntuple(i -> old_v[i] + delta_v[i], 3)
    target = (0.1f0, -0.05f0, 0.02f0)
    B = (0.2f0, -0.1f0, 0.05f0)
    for c in eachindex(s.U[1])
        s.scratch[1][c] = old_rho
        for d in 1:3
            s.scratch[d + 1][c] = old_rho * old_v[d]
            s.U[d + 1][c] = trial_rho * trial_v[d]
            s.U[d + 5][c] = B[d]
        end
        s.U[1][c] = trial_rho
    end
    eb = 0.5f0 * sum(abs2, B)
    ek_trial = 0.5f0 * trial_rho * sum(abs2, trial_v)
    fill!(s.U[5], 4f0 + eb + ek_trial)

    q = 10f0
    apply_exponential_drag_increment!(s, s.scratch, q; target_velocity=target)
    h = fields_to_host(s)
    phi1 = Float32(-expm1(-q) / q)
    expected = ntuple(i -> target[i] + exp(-q) * (old_v[i] - target[i]) +
                           phi1 * delta_v[i], 3)
    for d in 1:3
        @test h[d + 1][1] / h[1][1] ≈ expected[d] rtol=3f-6 atol=2f-7
    end
    ek = 0.5f0 * sum(h[d + 1][1]^2 for d in 1:3) / h[1][1]
    @test h[5][1] - ek - eb ≈ 4f0 atol=2f-6

    for c in eachindex(s.U[1]), d in 1:3
        s.U[d + 1][c] = trial_rho * trial_v[d]
    end
    fill!(s.U[5], 4f0 + eb + ek_trial)
    apply_exponential_drag_increment!(s, s.scratch, 0; target_velocity=target)
    h0 = fields_to_host(s)
    for d in 1:3
        @test h0[d + 1][1] / h0[1][1] ≈ trial_v[d] rtol=2f-6
    end
end

@testset "drag regime handoff is hysteretic and one-way by default" begin
    controller = DragRegimeController(exit_drag_over_omega=8,
                                      enter_drag_over_omega=16,
                                      confirm_checks=2)
    @test update_drag_regime!(controller, 20, 0.1) == (terminal=true, transition=:none)
    @test update_drag_regime!(controller, 7.9, 0.1) == (terminal=true, transition=:none)
    @test update_drag_regime!(controller, 8.1, 0.1) == (terminal=true, transition=:none)
    @test update_drag_regime!(controller, 7.9, 0.1) == (terminal=true, transition=:none)
    @test update_drag_regime!(controller, 7.8, 0.1) ==
          (terminal=false, transition=:terminal_to_full)
    @test update_drag_regime!(controller, 32, 0.1) == (terminal=false, transition=:none)

    reentrant = DragRegimeController(exit_drag_over_omega=8,
                                     enter_drag_over_omega=16,
                                     confirm_checks=2,
                                     reenter_terminal=true)
    @test update_drag_regime!(reentrant, 4, 0.1).terminal == false
    @test update_drag_regime!(reentrant, 17, 0.1) == (terminal=false, transition=:none)
    @test update_drag_regime!(reentrant, 18, 0.1) ==
          (terminal=true, transition=:full_to_terminal)
end

@testset "terminal pressure CT uses non-aliased stencil inputs" begin
    s = allocate_state(backend(:cpu), Float32, (12, 12, 12); dx=1 / 12, gamma=5 / 3)
    _terminal_ct_seed!(s; amp=0.03f0)
    initial_arrays = Set(objectid.(s.U))
    function forward_euler_pressure!(dst, state, source, coeff_cells2, dt)
        @. dst = state + Float32(dt) * source
        return dst
    end
    result = terminal_pressure_ct_step!(forward_euler_pressure!, s, 2.0e-4;
                                        gamma_drag=6, pressure_coeff=0.2,
                                        induction_dissipation=0)
    fields = fields_to_host(s)
    @test result.nsub == 1
    @test all(isfinite, fields[1])
    @test minimum(fields[1]) > 0
    @test all(isfinite, fields[5])
    @test length(Set(objectid.(s.U))) == 9
    @test length(intersect(initial_arrays, Set(objectid.(s.U)))) == 5
    brms = sqrt((sum(abs2, fields[6]) + sum(abs2, fields[7]) +
                 sum(abs2, fields[8])) / length(fields[6]))
    @test max_divb(s) * s.dx / brms < 2e-5
end
