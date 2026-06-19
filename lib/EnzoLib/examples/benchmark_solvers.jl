# Benchmark solvers through the method-slot registry (ADR-0002): compare :enzo vs
# :julia implementations of a physics slot on ACCURACY and SPEED, with no observer
# effect — timing happens at the slot boundary (a whole hydro/gravity step), so the
# ~20 ns time_ns() pair is <1e-4 of the work. The SlotProbe is zero-overhead when
# absent, so production runs are unaffected; only this driver attaches one.
#
#   ENZOMODULES_GRID_LIB=.../libenzomodules_grid.dylib \
#   julia --project=lib/EnzoLib/test lib/EnzoLib/examples/benchmark_solvers.jl
#
# Methodology (so the numbers are trustworthy and comparable):
#   - WARM UP first (drop Julia JIT-compile time, which is meaningless).
#   - report MIN and MEDIAN per-call, not mean (min ≈ true cost; mean is GC/OS noise).
#   - both configs march the SAME dt sequence (Enzo's compute_dt) ⇒ identical work.
#   - normalize by cells (µs/cell-update) so resolutions are comparable + extrapolate.
#   - record provenance (CPU, threads, float precision of each side).

using EnzoLib, Vespa, MeshInterface, RefMesh, EnzoBackend
using Printf

const RUN = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "run"))
ms(ns) = ns / 1e6

# ── :julia hooks (Vespa kernels on the live grid, via EnzoBackend) ────────────
function julia_hydro_hook(; γ = 1.4, nghost = 3, domain = ((0.0, 1.0),), precision = Float64)
    cache = Ref{Any}(nothing); model = IdealHydro(γ)
    return function (h, level, dt)
        if cache[] === nothing || cache[][1].h != h     # run_amr frees+reinits the handle per rep
            mesh = EnzoGridMesh(h; grid = 0, nghost = nghost, domain = domain, precision = precision,
                                cons_density = density_index(model), cons_momentum = momentum_indices(model),
                                cons_energy = energy_index(model))
            N = MeshInterface.n_cells(mesh)
            prob = Problem(; name = "bench-hydro", dims = (N,), domain = domain, γ = γ, bcs = Outflow(),
                           tfinal = 1.0, cfl = 0.4, init = (x, y, z) -> (1.0, 0.0, 0.0, 0.0, 1.0))
            cache[] = (mesh, Simulation(mesh, prob; model = model))
        end
        mesh, sim = cache[]
        sync_from_enzo!(sim.sv, mesh); step!(sim, dt); sync_to_enzo!(mesh, sim.sv); nothing
    end
end

# `cold=true` zeros the potential before each solve, so the iterative CG is timed
# from scratch (no free warm-start from the previous solution). On a static
# density a warm CG would converge in 1 iteration — unrealistically favorable, and
# unfair vs Enzo's DIRECT (non-iterative) solve. Cold is the apples-to-apples cost.
function julia_gravity_hook(; G = 1.0, nghost = 3, domain = ((0.0, 1.0),), γ = 1.4, cold = true)
    cache = Ref{Any}(nothing); rec = Ref{Any}((iters = 0, relres = NaN))
    model = IdealHydro(γ)
    hook = function (h, level, dt)
        if cache[] === nothing || cache[][1].h != h
            mesh = EnzoGridMesh(h; grid = 0, nghost = nghost, domain = domain,
                                cons_density = density_index(model), cons_momentum = momentum_indices(model),
                                cons_energy = energy_index(model))
            N = MeshInterface.n_cells(mesh)
            prob = Problem(; name = "bench-grav", dims = (N,), domain = domain, γ = γ, bcs = Periodic(),
                           tfinal = 1.0, cfl = 0.4, init = (x, y, z) -> (1.0, 0.0, 0.0, 0.0, 1.0))
            sim = Simulation(mesh, prob; model = model)
            grav = enable_gravity!(sim; G = G, bcs = Periodic())
            cache[] = (mesh, sim, grav)
        end
        mesh, sim, grav = cache[]
        sync_from_enzo!(sim.sv, mesh)
        cold && for i in 1:MeshInterface.n_cells(mesh)
            grav.phiv[CartesianIndex(i)] = 0.0      # cold start
        end
        it, rr = solve_poisson!(sim, grav)
        rec[] = (iters = it, relres = rr); nothing
    end
    return hook, rec
end

# ── exact-Riemann accuracy oracle for the hydro problem ───────────────────────
function l1_vs_exact(density; WL, WR, disc, t, γ = 1.4)
    N = length(density); dx = 1.0 / N
    s(i) = ((i - 0.5) * dx - disc) / t
    sum(abs(density[i] - exact_riemann_sample(WL, WR, γ, s(i))[1]) for i in 1:N) / N
end

# UnigridTranspose=0 patch so Enzo's serial root-grid FFT gravity runs (matches the harness).
function patched(pf)
    tmp = tempname() * ".enzo"; cp(pf, tmp)
    open(tmp, "a") do io; println(io, "\nUnigridTranspose = 0"); end
    return tmp
end

# ── benchmarks ────────────────────────────────────────────────────────────────
# Full-run hydro: run_amr to `cycles` per config, timing the :hydro slot each step.
function bench_hydro(; cycles = 30, warmup = 1, reps = 3)
    pf = joinpath(RUN, "Hydro", "Hydro-1D", "Toro-1-ShockTube", "Toro-1-ShockTube.enzo")
    WL = (1.0, 0.75, 1.0); WR = (0.125, 0.0, 0.1); disc = 0.3; tend = 0.2
    hjulia(p) = EnzoLib.EngineConfig(; hydro = :julia, probe = EnzoLib.SlotProbe(),
                                     hooks = Dict{Symbol,Function}(:hydro => julia_hydro_hook(; precision = p)))
    rows = NamedTuple[]
    for (name, eng) in (("enzo f64", EnzoLib.EngineConfig(; hydro = :enzo, probe = EnzoLib.SlotProbe())),
                        ("julia f64", hjulia(Float64)),
                        ("julia f32", hjulia(Float32)))
        for _ in 1:warmup
            EnzoLib.run_amr(pf; engine = eng, reader = EnzoLib.read_density, regrid = false, maxcycle = cycles)
        end
        EnzoLib.reset!(eng.probe)
        d = nothing
        for _ in 1:reps
            d = EnzoLib.run_amr(pf; engine = eng, reader = EnzoLib.read_density, regrid = false, maxcycle = cycles)
        end
        s = EnzoLib.probe_summary(eng.probe)[:hydro]
        N = length(d)
        push!(rows, (name = name, N = N, acc = l1_vs_exact(d; WL = WL, WR = WR, disc = disc, t = tend),
                     acclabel = "L1 vs exact", s = s, iters = 0))
    end
    return ("Hydro slot — Toro-1 ($(cycles) cycles)", rows)
end

# Per-solve gravity microbench: time one :gravity solve on the live pancake density.
function bench_gravity(; reps = 40, warmup = 3)
    pf = patched(joinpath(RUN, "Cosmology", "ZeldovichPancake", "ZeldovichPancake.enzo"))
    rows = NamedTuple[]
    cd(EnzoLib._workdir(pf)) do
        h = EnzoLib.session_init(pf)
        try
            EnzoLib.session_set_boundary(h, 0)
            ghook, grec = julia_gravity_hook(; domain = ((0.0, 1.0),))
            configs = (("enzo", EnzoLib.EngineConfig(; gravity = :enzo, probe = EnzoLib.SlotProbe()), grec),
                       ("julia", EnzoLib.EngineConfig(; gravity = :julia, probe = EnzoLib.SlotProbe(),
                                                      hooks = Dict{Symbol,Function}(:gravity => ghook)), grec))
            for (name, eng, rec) in configs
                for _ in 1:warmup; EnzoLib.run_slot(:gravity, eng, h, 0, 1.0); end
                EnzoLib.reset!(eng.probe)
                for _ in 1:reps; EnzoLib.run_slot(:gravity, eng, h, 0, 1.0); end
                s = EnzoLib.probe_summary(eng.probe)[:gravity]
                it = name == "julia" ? rec[].iters : 0
                acc = name == "julia" ? rec[].relres : 0.0
                push!(rows, (name = name, N = EnzoLib.problem_grid_size(h, 0), acc = acc,
                             acclabel = "CG relres", s = s, iters = it))
            end
        finally
            EnzoLib.free_problem(h)
        end
    end
    return ("Gravity slot — ZeldovichPancake (per-solve)", rows)
end

# ── report ────────────────────────────────────────────────────────────────────
function provenance(; cycles, warmup, reps)
    bb, ib = try EnzoLib.check_precision() catch; (-1, -1) end
    @printf("Provenance: julia %s · %s · %d thread(s) · Enzo baryons %d-byte / ints %d-byte · Julia Float64\n",
            VERSION, Sys.CPU_NAME, Threads.nthreads(), bb, ib)
    @printf("            warmup=%d  reps=%d  (min ≈ true cost; both configs march identical dt)\n\n", warmup, reps)
end

function print_table(title, rows)
    println("### ", title)
    @printf("%-10s %6s %16s %6s %9s %9s %11s %9s %5s %6s\n",
            "impl", "cells", "accuracy", "calls", "min ms", "med ms", "µs/cell", "MB/call", "iters", "×base")
    base = rows[1].s.min_ns
    for r in rows
        s = r.s
        µs_cell = s.median_ns / max(r.N, 1) / 1e3
        mb_call = s.bytes / max(s.calls, 1) / 2^20
        @printf("%-10s %6d %16.7g %6d %9.3f %9.3f %11.4f %9.4f %5s %6.2f\n",
                r.name, r.N, r.acc, s.calls, ms(s.min_ns), ms(s.median_ns), µs_cell, mb_call,
                r.iters == 0 ? "—" : string(r.iters), s.min_ns / max(base, 1))
    end
    @printf("(accuracy column = %s)\n\n", rows[1].acclabel)
end

# ── precision study: PURE Vespa (no Enzo), state persists in T ───────────────
# The guest-on-Enzo runs above round-trip the state through Enzo's f64 BaryonField
# every step, so f32 barely shows. Here Vespa owns the state in T across the whole
# run, so the precision effect on BOTH accuracy and speed is real. Sweeping
# resolution shows the regime: at small N the cost is compute/overhead-bound (f32
# ≈ f64); f32's bandwidth win only appears once the field arrays stop fitting cache.
function l1_density(sim)
    γ, t = sim.problem.γ, sim.t
    WL = (1.0, 0.0, 1.0); WR = (0.125, 0.0, 0.1)
    num = 0.0; n = 0
    for (ctr, W) in cell_samples(sim)
        ρe = exact_riemann_sample(WL, WR, γ, (ctr[1] - 0.5) / t)[1]
        num += abs(W[1] - ρe); n += 1
    end
    return num / n
end

function bench_precision(; resolutions = (256, 1024, 4096), reps = 3)
    println("### Precision study — pure Vespa Sod to t=0.2 (state persists in T)")
    @printf("%-8s %7s %14s %8s %10s %9s %8s\n",
            "prec", "cells", "L1 vs exact", "cycles", "time ms", "µs/cell", "MB live")
    for N in resolutions
        prev = nothing
        for T in (Float64, Float32)
            prob = sod_problem_defaults(n = N)
            mk() = Simulation(UniformMesh(prob.dims, prob.domain; T = T), prob)
            evolve!(mk())                      # warm: compile for T + one run
            best = Inf; sim = nothing
            for _ in 1:reps
                s = mk(); e = @elapsed evolve!(s); best = min(best, e); sim = s
            end
            cyc = sim.step
            live = N * 5 * sizeof(T) / 2^20    # conserved state footprint
            ratio = prev === nothing ? "" : @sprintf(" (%.2f× f64 time)", best / prev)
            @printf("%-8s %7d %14.7g %8d %10.3f %9.5f %8.4f%s\n",
                    string(T), N, l1_density(sim), cyc, best * 1e3, best / (N * cyc) * 1e6, live, ratio)
            prev = T === Float64 ? best : prev
        end
    end
    println()
end

# Run the live :enzo-vs-:julia slot benchmarks FIRST (Enzo C++ on a clean process),
# then the pure-Julia precision study LAST — its heavy allocation/GC churn must not
# precede the Enzo calls in the same process (it can perturb the C++ heap → crash).
if !EnzoLib.grid_available()
    @info "Session bridge not built — skipping the live :enzo-vs-:julia slot benchmarks"
else
    cyc, wu, rp = 30, 1, 3
    provenance(; cycles = cyc, warmup = wu, reps = rp)
    for (title, rows) in (bench_hydro(; cycles = cyc, warmup = wu, reps = rp),
                          bench_gravity(; reps = 40, warmup = 3))
        print_table(title, rows)
    end
end

bench_precision(; resolutions = (256, 1024, 4096), reps = 2)   # pure Vespa; no Enzo bridge needed
