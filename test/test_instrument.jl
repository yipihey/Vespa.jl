# Measurement at the seam (ADR-0001 P10): wrapping a backend in `Instrumented`
# satisfies the same interface and yields a per-operation cost report, with no
# change to the driver or kernels. The same wrapper will quantify exactly what a
# Rust or GPU backend buys, op-for-op, against this reference.

@testset "Instrumented{RefMesh} per-operation report" begin
    prob = sod_problem_defaults(n = 64)
    imesh = Instrumented(UniformMesh(prob.dims, prob.domain))
    sim = Simulation(imesh, prob)
    evolve!(sim)

    rep = span_report(imesh)
    @test !isempty(rep)
    ops = first.(rep)
    @test :for_each_cell in ops

    @info "Instrumented{RefMesh} cost report (name, calls, total_ms, mean_µs)"
    for row in rep
        @info "  span" name = row[1] calls = row[2] total_ms = round(row[3], digits = 3) mean_µs = round(row[4], digits = 3)
    end
end
