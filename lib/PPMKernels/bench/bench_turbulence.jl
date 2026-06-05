# Decaying-turbulence performance benchmark: the Metal PPM port (ppm_step_3d!)
# vs the same code on the CPU, plus the raw legacy Fortran 1-D kernel throughput.
#
# IC: uniform ρ and P (sound speed 1), a solenoidal (divergence-free) random-mode
# velocity field at a target RMS Mach number — representative decaying turbulence
# (not Enzo's exact HDF5 generator). The point is realistic cell counts + branch
# behaviour for a perf comparison.
#
# Run:  <juliaup-julia> --project=test bench/bench_turbulence.jl [sizes...]
#   e.g. ... bench/bench_turbulence.jl 32 64 128

using PPMKernels
using Printf
using Random
try
    @eval using Metal
catch err
    @info "Metal not loadable — GPU rows will be skipped" err
end
try
    @eval using EnzoLib
catch
end

const NG    = 4
const GAMMA = 1.4
const MACH  = 0.3

metal_ok() = PPMKernels.has_backend(:metal)

# ── representative decaying-turbulence IC on a padded (interior + 2·NG)³ grid ──
function turbulence_ic(n::Int; nmodes = 24, seed = 1234)
    Random.seed!(seed)
    N = n + 2NG
    tot = N^3
    lin(i, j, k) = i + N * (j - 1) + N * N * (k - 1)
    vx = zeros(tot); vy = zeros(tot); vz = zeros(tot)
    for _ in 1:nmodes
        kx, ky, kz = rand(-3:3), rand(-3:3), rand(-3:3)
        (kx == 0 && ky == 0 && kz == 0) && continue
        k2 = kx^2 + ky^2 + kz^2
        ax, ay, az = randn(), randn(), randn()
        ad = (ax * kx + ay * ky + az * kz) / k2          # project out the divergent part
        ax -= ad * kx; ay -= ad * ky; az -= ad * kz
        φ = 2π * rand(); amp = 1.0 / sqrt(k2)            # red-ish spectrum
        @inbounds for k in 1:N, j in 1:N, i in 1:N
            c = amp * cos(2π * (kx * (i - 1) + ky * (j - 1) + kz * (k - 1)) / N + φ)
            q = lin(i, j, k); vx[q] += ax * c; vy[q] += ay * c; vz[q] += az * c
        end
    end
    vrms = sqrt(sum(vx .^ 2 .+ vy .^ 2 .+ vz .^ 2) / tot)
    s = MACH / vrms; vx .*= s; vy .*= s; vz .*= s
    P0 = 1.0 / GAMMA                                      # c_s = sqrt(γP/ρ) = 1
    d  = ones(tot)
    ge = fill(P0 / ((GAMMA - 1)), tot)
    e  = ge .+ 0.5 .* (vx .^ 2 .+ vy .^ 2 .+ vz .^ 2)
    return (; d, e, ge, vx, vy, vz, gr = zeros(tot), N, dims = (N, N, N))
end

const FLAGS = (idual = 1, iflatten = 3, isteep = 0, idiff = 0, gravity = 0, eta2 = 0.1)

# time `nsteps` of ppm_step_3d! at element type T on a backend; returns sec/step
function time_steps(backend_name::Symbol, ::Type{T}, ic, nsteps::Int) where {T}
    be = PPMKernels.backend(backend_name)
    dev(a) = PPMKernels.to_device(be, a, T)
    d, e, ge = dev(ic.d), dev(ic.e), dev(ic.ge)
    vx, vy, vz = dev(ic.vx), dev(ic.vy), dev(ic.vz)
    grx = dev(ic.gr); gry = dev(ic.gr); grz = dev(ic.gr)
    dims = ic.dims; n = dims[1] - 2NG
    dx = 1.0 / n; dt = 0.2 * dx                           # ~CFL (c_s=1, Mach 0.3)
    step!(o) = PPMKernels.ppm_step_3d!(d, e, ge, vx, vy, vz, grx, gry, grz, dims, NG;
                                       dt = dt, gamma = GAMMA, order = o, FLAGS...)
    # the scratch pool recycles per-sweep buffers across steps (allocation is the
    # GPU bottleneck); the warm-up step also primes the pool + compiles.
    return PPMKernels.with_pool() do
        step!((1, 2, 3))
        (@elapsed for s in 1:nsteps
            step!(iseven(s) ? (3, 2, 1) : (1, 2, 3))
        end) / nsteps
    end
end

# raw legacy Fortran 1-D kernel throughput: one x-sweep worth of pencils (ccall'd)
function fortran_pencil_throughput(ic)
    (@isdefined EnzoLib) && EnzoLib.available() || return nothing
    N = ic.N; n = N - 2NG
    lin(i, j, k) = i + N * (j - 1) + N * N * (k - 1)
    p0 = [(GAMMA - 1) * ic.d[q] * (ic.e[q] - 0.5 * (ic.vx[q]^2 + ic.vy[q]^2 + ic.vz[q]^2)) for q in 1:N^3]
    pencils = [[lin(i, j, k) for i in 1:N] for k in 1:N for j in 1:N]
    sweep1(line) = begin
        d = ic.d[line]; e = ic.e[line]; u = ic.vx[line]; v = ic.vy[line]; w = ic.vz[line]
        ge = ic.ge[line]; p = p0[line]
        EnzoLib.ppm_sweep_1d_full!(d, e, u, v, w, p; i1 = NG + 1, i2 = N - NG, dx = 1.0 / n,
            dt = 0.2 / n, gamma = GAMMA, geslice = ge, idual = 1, iflatten = 3, eta2 = 0.1)
    end
    sweep1(pencils[1])                                    # warm up
    t = @elapsed for line in pencils
        sweep1(line)
    end
    return (N * n * n) / t                                # cells updated / sec
end

# ── driver ───────────────────────────────────────────────────────────────────
sizes = isempty(ARGS) ? [32, 64] : parse.(Int, ARGS)
nsteps_for(n) = n >= 128 ? 3 : n >= 64 ? 5 : 8

@printf("\n%-6s %-12s %-12s %-12s %-10s\n", "n^3", "backend/T", "sec/step", "Mcell/s", "speedup")
println("-"^58)

# each measurement is wrapped: an OOM/error on one backend/size reports and moves
# on instead of killing the whole run. The pool holds ~100 full-grid buffers, so
# CPU-f64 (8 B/elem) is the memory hog — skipped past `CPU_F64_MAX` cells.
const CPU_F64_MAX = 150^3

guard(f) = try f() catch err; @printf("    (skipped: %s)\n", sprint(showerror, err)); flush(stdout); nothing end

for n in sizes
    ic = turbulence_ic(n)
    ns = nsteps_for(n)
    ncell = n^3
    results = Dict{String,Float64}()

    # CPU baselines (same code as the GPU path) — f64 is the certified-original numerics
    Ts = ncell > CPU_F64_MAX ? (Float32,) : (Float64, Float32)
    for T in Ts
        guard() do
            sec = time_steps(:cpu, T, ic, ns)
            results["cpu/$T"] = sec
            @printf("%-8d %-12s %-12.4g %-12.2f %-10s\n", ncell, "cpu/$T", sec, ncell / sec / 1e6, "1.0×")
            flush(stdout)
        end
    end

    # Metal GPU (f32 only)
    if metal_ok()
        guard() do
            sec = time_steps(:metal, Float32, ic, ns)
            sp = haskey(results, "cpu/Float32") ? @sprintf("%.1f× vs cpu-f32", results["cpu/Float32"] / sec) : ""
            @printf("%-8d %-12s %-12.4g %-12.2f %-10s\n", ncell, "metal/F32", sec, ncell / sec / 1e6, sp)
            flush(stdout)
        end
    else
        @printf("%-8d %-12s %-12s\n", ncell, "metal/F32", "(no GPU)")
    end

    # raw legacy Fortran 1-D kernel throughput (single-thread, per-pencil ccall)
    guard() do
        fth = fortran_pencil_throughput(ic)
        fth === nothing || @printf("%-8d %-12s %-12s %-12.2f %-10s\n", ncell, "fortran-1d", "—",
                                   fth / 1e6, "(raw kernel)")
    end
    flush(stdout)
end
println()
