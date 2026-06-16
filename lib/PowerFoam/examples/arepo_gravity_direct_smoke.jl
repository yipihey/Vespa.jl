using Dates
using Printf
using PowerFoam

const OUTBASE = joinpath(@__DIR__, "out", "arepo_gravity_direct_smoke")
const RUN_TAG = Dates.format(now(), "yyyymmdd_HHMMSS")
const OUTDIR = joinpath(OUTBASE, RUN_TAG)

struct GravityCase
    name::String
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    m::Vector{Float64}
    expected_ax::Vector{Float64}
    expected_ay::Vector{Float64}
    expected_az::Vector{Float64}
    expected_pe::Float64
end

function analytic_acceleration(x, y, z, m; G = 1.0, softening = 0.0)
    ax = zeros(Float64, length(x))
    ay = zeros(Float64, length(y))
    az = zeros(Float64, length(z))
    eps2 = softening^2
    @inbounds for i in eachindex(x)
        for j in eachindex(x)
            i == j && continue
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dz = z[j] - z[i]
            r2 = dx * dx + dy * dy + dz * dz + eps2
            invr = inv(sqrt(r2))
            invr3 = invr * invr * invr
            pair = G * m[j] * invr3
            ax[i] += pair * dx
            ay[i] += pair * dy
            az[i] += pair * dz
        end
    end
    return ax, ay, az
end

function analytic_potential_energy(x, y, z, m; G = 1.0, softening = 0.0)
    pe = 0.0
    eps2 = softening^2
    @inbounds for i in 1:length(x)-1
        for j in i+1:length(x)
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dz = z[j] - z[i]
            r2 = dx * dx + dy * dy + dz * dz + eps2
            pe -= G * m[i] * m[j] / sqrt(r2)
        end
    end
    return pe
end

function build_cases()
    two_x = [0.0, 1.0]
    two_y = [0.0, 0.0]
    two_z = [0.0, 0.0]
    two_m = [2.0, 3.0]
    two_ax, two_ay, two_az = analytic_acceleration(two_x, two_y, two_z, two_m)
    two_pe = analytic_potential_energy(two_x, two_y, two_z, two_m)

    three_x = [0.0, 1.0, 0.0]
    three_y = [0.0, 0.0, 1.0]
    three_z = [0.0, 0.0, 0.0]
    three_m = [2.0, 3.0, 4.0]
    three_ax, three_ay, three_az = analytic_acceleration(three_x, three_y, three_z, three_m)
    three_pe = analytic_potential_energy(three_x, three_y, three_z, three_m)

    return GravityCase[
        GravityCase("two_body", two_x, two_y, two_z, two_m, two_ax, two_ay, two_az, two_pe),
        GravityCase("three_body", three_x, three_y, three_z, three_m, three_ax, three_ay, three_az, three_pe),
    ]
end

function max_component_error(actual, expected)
    return maximum(abs.(actual .- expected))
end

function momentum_residual(m, ax, ay, az)
    return (
        sum(m .* ax),
        sum(m .* ay),
        sum(m .* az),
    )
end

function evaluate_case(case::GravityCase; atol = 1e-12)
    oracle = arepo_direct_gravity_oracle(case.x, case.y, case.z, case.m)

    ax_err = max_component_error(oracle.ax, case.expected_ax)
    ay_err = max_component_error(oracle.ay, case.expected_ay)
    az_err = max_component_error(oracle.az, case.expected_az)
    pe_err = abs(oracle.potential_energy - case.expected_pe)
    residual = momentum_residual(case.m, oracle.ax, oracle.ay, oracle.az)
    residual_max = maximum(abs, residual)

    ax_err <= atol || error("$(case.name): ax mismatch exceeds tolerance: $(ax_err)")
    ay_err <= atol || error("$(case.name): ay mismatch exceeds tolerance: $(ay_err)")
    az_err <= atol || error("$(case.name): az mismatch exceeds tolerance: $(az_err)")
    pe_err <= atol || error("$(case.name): potential mismatch exceeds tolerance: $(pe_err)")
    residual_max <= atol || error("$(case.name): momentum residual exceeds tolerance: $(residual_max)")

    return (
        name = case.name,
        particle_count = length(case.m),
        potential_energy = oracle.potential_energy,
        expected_potential_energy = case.expected_pe,
        potential_error = pe_err,
        ax_error = ax_err,
        ay_error = ay_err,
        az_error = az_err,
        momentum_x = residual[1],
        momentum_y = residual[2],
        momentum_z = residual[3],
        momentum_residual_max = residual_max,
    )
end

function build_runtime_spec(case::GravityCase; dt = 0.125, advance_gravity = true)
    particles = (
        x = copy(case.x),
        y = copy(case.y),
        z = copy(case.z),
        m = copy(case.m),
        vx = zeros(Float64, length(case.x)),
        vy = zeros(Float64, length(case.y)),
        vz = zeros(Float64, length(case.z)),
    )
    return arepo_problem_spec(Symbol(case.name, :_runtime);
                              particle_count = length(case.m),
                              physics = (hydro = false,
                                         tessellation = false,
                                         gravity = true),
                              metadata = (particles = particles,
                                          softening = 0.0,
                                          G = 1.0,
                                          dt = dt,
                                          advance_gravity = advance_gravity))
end

function evaluate_runtime_case(case::GravityCase; dt = 0.125, atol = 1e-12)
    spec = build_runtime_spec(case; dt = dt)
    state = arepo_run_scaffold(spec)
    state.status == :gravity_advanced ||
        error("$(case.name): runtime dispatch did not advance gravity")
    result = state.payload.gravity
    oracle = arepo_direct_gravity_oracle(case.x, case.y, case.z, case.m)

    expected_vx = dt .* oracle.ax
    expected_vy = dt .* oracle.ay
    expected_vz = dt .* oracle.az
    expected_x = case.x .+ dt .* expected_vx
    expected_y = case.y .+ dt .* expected_vy
    expected_z = case.z .+ dt .* expected_vz

    result.accelerations.ax ≈ oracle.ax ||
        error("$(case.name): runtime ax mismatch")
    result.accelerations.ay ≈ oracle.ay ||
        error("$(case.name): runtime ay mismatch")
    result.accelerations.az ≈ oracle.az ||
        error("$(case.name): runtime az mismatch")
    result.potential_energy ≈ oracle.potential_energy ||
        error("$(case.name): runtime potential mismatch")
    result.after.vx ≈ expected_vx ||
        error("$(case.name): runtime vx mismatch")
    result.after.vy ≈ expected_vy ||
        error("$(case.name): runtime vy mismatch")
    result.after.vz ≈ expected_vz ||
        error("$(case.name): runtime vz mismatch")
    result.after.x ≈ expected_x ||
        error("$(case.name): runtime x mismatch")
    result.after.y ≈ expected_y ||
        error("$(case.name): runtime y mismatch")
    result.after.z ≈ expected_z ||
        error("$(case.name): runtime z mismatch")
    maximum(abs, collect(result.momentum_residual)) <= atol ||
        error("$(case.name): runtime momentum residual too large")

    return (
        name = case.name * "_runtime",
        particle_count = length(case.m),
        dt = dt,
        status = state.status,
        max_abs_accel = result.max_abs_accel,
        potential_energy = result.potential_energy,
        after_x = result.after.x,
        after_y = result.after.y,
        after_z = result.after.z,
    )
end

function csvquote(x)
    s = string(x)
    return "\"" * replace(s, "\"" => "\"\"") * "\""
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, "case,particle_count,potential_energy,expected_potential_energy,potential_error,ax_error,ay_error,az_error,momentum_x,momentum_y,momentum_z,momentum_residual_max")
        for row in rows
            vals = (
                row.name,
                row.particle_count,
                row.potential_energy,
                row.expected_potential_energy,
                row.potential_error,
                row.ax_error,
                row.ay_error,
                row.az_error,
                row.momentum_x,
                row.momentum_y,
                row.momentum_z,
                row.momentum_residual_max,
            )
            println(io, join((csvquote(v) for v in vals), ","))
        end
    end
end

function write_readme(path, rows; command)
    open(path, "w") do io
        println(io, "# AREPO Direct Gravity Smoke")
        println(io)
        println(io, "This smoke gate exercises the package-exported direct-gravity")
        println(io, "helpers through `using PowerFoam` on two tiny frozen particle")
        println(io, "systems: a two-body pair with closed-form integer accelerations")
        println(io, "and a small three-body triangle. It checks direct-force values,")
        println(io, "potential energy, the action-reaction momentum sum `sum(m .* a)`,")
        println(io, "and the metadata-driven runtime dispatch path on a frozen kick-drift")
        println(io, "step.")
        println(io)
        @printf(io, "- generated: %s\n", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        @printf(io, "- command: `%s`\n", command)
        println(io)
        println(io, "## Results")
        println(io)
        println(io, "| case | particles | potential | max accel err | max momentum residual |")
        println(io, "| --- | ---: | ---: | ---: | ---: |")
        for row in rows
            max_accel_err = max(row.ax_error, row.ay_error, row.az_error)
            @printf(io, "| %s | %d | %.16g | %.3e | %.3e |\n",
                    row.name, row.particle_count, row.potential_energy,
                    max_accel_err, row.momentum_residual_max)
        end
        println(io)
        println(io, "## Fixtures")
        println(io)
        println(io, "- `two_body`: `x=[0,1]`, `m=[2,3]`, expected potential `-6`.")
        println(io, "- `three_body`: points `(0,0,0)`, `(1,0,0)`, `(0,1,0)` with")
        println(io, "  masses `[2,3,4]`, expected potential `-(14 + 6sqrt(2))`.")
        println(io)
        println(io, "## Runtime Slice")
        println(io)
        println(io, "A frozen metadata payload also routes through `arepo_run_scaffold`")
        println(io, "with `gravity=true`, `particle_count>0`, and `advance_gravity=true`.")
    end
end

function main()
    cases = build_cases()
    rows = [evaluate_case(case) for case in cases]
    runtime_row = evaluate_runtime_case(first(cases))
    mkpath(OUTDIR)
    readme_path = joinpath(OUTDIR, "README.md")
    csv_path = joinpath(OUTDIR, "results.csv")
    command = "julia --project=lib/PowerFoam lib/PowerFoam/examples/arepo_gravity_direct_smoke.jl"
    write_readme(readme_path, rows; command)
    write_csv(csv_path, rows)
    @printf("wrote %s\n", readme_path)
    @printf("wrote %s\n", csv_path)
    for row in rows
        max_accel_err = max(row.ax_error, row.ay_error, row.az_error)
        @printf("%-10s particles=%d potential=%.16g max_accel_err=%.3e momentum_residual=%.3e\n",
                row.name, row.particle_count, row.potential_energy,
                max_accel_err, row.momentum_residual_max)
    end
    @printf("%-10s particles=%d status=%s dt=%.3f max_abs_accel=%.3e after_x=[%.6g, %.6g]\n",
            runtime_row.name, runtime_row.particle_count, runtime_row.status,
            runtime_row.dt, runtime_row.max_abs_accel,
            runtime_row.after_x[1], runtime_row.after_x[2])
end

main()
