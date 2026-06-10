# Contact-preservation benchmark for PLM/PPM/PPML solver variants.
#
# Periodic, uniform-pressure advection isolates entropy/contact diffusion:
# the exact solution after an integer number of box crossings is the initial
# profile.  Metrics are intentionally contact-specific:
#   * L1 density error
#   * total 10-90% transition width across both contacts
#   * density overshoot/undershoot
#   * pressure wiggle amplitude relative to p0
#   * wall time
#
# Run from lib/PPMKernels:
#   julia --project=test bench/compare_contacts.jl [nx] [crossings] [case] [u0] [solver_substring]
# where case is one of: contact, top_hat, both.

using PPMKernels, KernelAbstractions, Printf, Statistics
try
    @eval using Metal
catch err
    @info "Metal not loadable - CPU fallback" err
end

const _P = PPMKernels
const NG = 4
const NY = 8
const GAMMA = 1.4
const RHO_LO = 1.0
const RHO_HI = 2.0
const P0 = 1.0

nx = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 256
crossings = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 4.0
case_arg = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :both
u0 = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0
solver_filter = length(ARGS) >= 5 ? ARGS[5] : ""

bkname = _P.has_backend(:metal) ? :metal : :cpu
be = _P.backend(bkname)
const T = Float32
dev(a) = _P.to_device(be, a, T)

const SOLVERS = [
    "Hancock-PLM",
    "Hancock-PPM-tr-2shk",
    "Local-PPM-tr-2shk",
    "Local-PPM-exact-label-THINC",
    "Local-PPM-carried-label-THINC",
    "PPM-DirectEuler",
    "PPML-trace",
]

@kernel function _fill_lagrangian_label_k!(A, nx_tot::Int, ny_tot::Int, nz_tot::Int,
                                           ng::Int, nx_active::Int, u0, t)
    g = @index(Global, Linear)
    i = (g - 1) % nx_tot + 1
    # Material coordinate for uniform advection: a(x,t)=x-u0*t mod 1.
    x = (i - ng - 0.5f0) / nx_active
    a = x - u0 * t
    A[g] = a - floor(a)
end

function fill_lagrangian_label!(A, ic, t)
    be = KernelAbstractions.get_backend(A)
    _fill_lagrangian_label_k!(be)(A, ic.dims[1], ic.dims[2], ic.dims[3],
                                  NG, nx, T(u0), T(t); ndrange = length(A))
    KernelAbstractions.synchronize(be)
    return A
end

@kernel function _init_label_moments_k!(A1, A2, @Const(D),
                                        nx_tot::Int, ny_tot::Int, nz_tot::Int,
                                        ng::Int, nx_active::Int)
    g = @index(Global, Linear)
    i = (g - 1) % nx_tot + 1
    a = (i - ng - 0.5f0) / nx_active
    a = a - floor(a)
    ρ = D[g]
    A1[g] = ρ * a
    A2[g] = ρ * a * a
end

function init_label_moments!(A1, A2, D, ic)
    be = KernelAbstractions.get_backend(A1)
    _init_label_moments_k!(be)(A1, A2, D, ic.dims[1], ic.dims[2], ic.dims[3],
                               NG, nx; ndrange = length(A1))
    KernelAbstractions.synchronize(be)
    return A1, A2
end

@kernel function _moments_to_labels_k!(L1, L2, @Const(A1), @Const(A2), @Const(D))
    g = @index(Global, Linear)
    ρ = max(D[g], eps(eltype(D)))
    a = A1[g] / ρ
    a2 = max(A2[g] / ρ, a*a)
    L1[g] = a
    L2[g] = a2
end

function moments_to_labels!(L1, L2, A1, A2, D, ic)
    be = KernelAbstractions.get_backend(A1)
    _moments_to_labels_k!(be)(L1, L2, A1, A2, D; ndrange = length(A1))
    _P.fill_periodic!(ic.dims, NG, L1, L2)
    KernelAbstractions.synchronize(be)
    return L1, L2
end

function density_profile(x, kind::Symbol)
    if kind === :contact
        return x < 0.5 ? RHO_HI : RHO_LO
    elseif kind === :top_hat
        return (0.25 <= x < 0.5) ? RHO_HI : RHO_LO
    else
        error("unknown contact case $kind")
    end
end

function contact_ic(kind::Symbol)
    nbx = nx + 2NG
    nby = NY + 2NG
    dims = (nbx, nby, nby)
    N = prod(dims)
    dx = 1.0 / nx
    d = Vector{Float64}(undef, N)
    u = fill(u0, N)
    pr = fill(P0, N)
    idx(i, j, k) = i + nbx * (j - 1) + nbx * nby * (k - 1)
    for k in 1:nby, j in 1:nby, i in 1:nbx
        x = mod((i - NG - 0.5) / nx, 1.0)
        d[idx(i, j, k)] = density_profile(x, kind)
    end
    eint = pr ./ ((GAMMA - 1) .* d)
    etot = eint .+ 0.5 .* u0^2
    z = zeros(N)
    return (; kind, d, u, vy = z, vz = z, eint, etot, dims, N, dx, nbx, nby)
end

function center_line(a, ic)
    j = NG + NY ÷ 2
    k = j
    return [Float64(a[i + ic.nbx * (j - 1) + ic.nbx * ic.nby * (k - 1)])
            for i in (NG + 1):(ic.nbx - NG)]
end

function pressure_line(D, S1, S2, S3, Tau, ic)
    d = center_line(D, ic)
    s1 = center_line(S1, ic)
    s2 = center_line(S2, ic)
    s3 = center_line(S3, ic)
    τ = center_line(Tau, ic)
    p = similar(d)
    @inbounds for i in eachindex(d)
        u = s1[i] / d[i]
        v = s2[i] / d[i]
        w = s3[i] / d[i]
        p[i] = (GAMMA - 1) * (τ[i] - 0.5 * d[i] * (u*u + v*v + w*w))
    end
    return p
end

function exact_line(ic)
    return [density_profile((i - 0.5) / nx, ic.kind) for i in 1:nx]
end

label_levels(kind::Symbol) =
    kind === :contact ? (0.5, -1.0) :
    kind === :top_hat ? (0.25, 0.5) :
    error("unknown contact case $kind")

function run_solver(name, ic, dt, nsteps)
    dims = ic.dims
    dx = ic.dx
    N = ic.N
    pbc5(a, b, c, d, e) = _P.fill_periodic!(dims, NG, a, b, c, d, e)
    pbc6(a, b, c, d, e, f) = _P.fill_periodic!(dims, NG, a, b, c, d, e, f)

    if name == "PPM-DirectEuler"
        d = dev(ic.d)
        e = dev(ic.etot)
        ge = dev(ic.eint)
        vx = dev(ic.u)
        vy = dev(ic.vy)
        vz = dev(ic.vz)
        z = dev(zeros(N))
        tw = _P.with_pool() do
            @elapsed for s in 1:nsteps
                _P.ppm_step_3d!(d, e, ge, vx, vy, vz, z, z, z, dims, NG;
                    dt, gamma = GAMMA, dx,
                    order = isodd(s) ? (1, 2, 3) : (3, 2, 1),
                    bc! = pbc6, idual = 0, iflatten = 3, isteep = 0,
                    idiff = 0, gravity = 0, eta2 = 0.1)
            end
        end
        hd = _P.to_host(d)
        hvx = _P.to_host(vx)
        he = _P.to_host(e)
        rho = center_line(hd, ic)
        p = similar(rho)
        @inbounds for i in eachindex(rho)
            q = NG + i + ic.nbx * (NG + NY ÷ 2 - 1) +
                ic.nbx * ic.nby * (NG + NY ÷ 2 - 1)
            p[i] = (GAMMA - 1) * hd[q] * (he[q] - 0.5 * hvx[q]^2)
        end
        return (; rho, pressure = p, wall = tw)
    end

    D = dev(ic.d)
    S1 = dev(ic.d .* ic.u)
    S2 = dev(zeros(N))
    S3 = dev(zeros(N))
    Tau = dev(ic.d .* ic.etot)
    st = name == "PPML-trace" ? _P.ppml_alloc_state(D, dims, NG) : nothing
    st === nothing || _P.ppml_init_state!(st, D, S1, S2, S3, Tau; gamma = GAMMA)
    exact_label = name == "Local-PPM-exact-label-THINC"
    carried_label = name == "Local-PPM-carried-label-THINC"
    Label = (exact_label || carried_label) ? dev(zeros(N)) : nothing
    Label2 = carried_label ? dev(zeros(N)) : nothing
    A1 = carried_label ? dev(zeros(N)) : nothing
    A2 = carried_label ? dev(zeros(N)) : nothing
    carried_label && init_label_moments!(A1, A2, D, ic)
    carried_label && moments_to_labels!(Label, Label2, A1, A2, D, ic)
    lev1, lev2 = label_levels(ic.kind)
    bc_local = carried_label ?
        ((a, b, c, d, e) -> begin
            pbc5(a, b, c, d, e)
            _P.fill_periodic!(dims, NG, A1, A2)
        end) : pbc5

    step! = if name == "Hancock-PLM"
        (o) -> _P.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, dims, NG;
            dt, gamma = GAMMA, dx, order = o, bc! = bc_local,
            recon = :plm, predictor = :hancock, riemann = :hll)
    elseif name == "Hancock-PPM-tr-2shk"
        (o) -> _P.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, dims, NG;
            dt, gamma = GAMMA, dx, order = o, bc! = bc_local,
            recon = :ppm, predictor = :trace, riemann = :twoshock)
    elseif name == "Local-PPM-tr-2shk"
        (o) -> _P.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, dims, NG;
            dt, gamma = GAMMA, dx, order = o, bc! = bc_local,
            face_periodic = true, recon = :ppm_local, predictor = :trace,
            riemann = :twoshock)
    elseif exact_label || carried_label
        (o) -> _P.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, dims, NG;
            dt, gamma = GAMMA, dx, order = o, bc! = bc_local,
            face_periodic = true, recon = :ppm_local, predictor = :trace,
            riemann = :twoshock, contact_label = Label,
            contact_label2 = Label2, contact_moment1 = A1, contact_moment2 = A2,
            contact_level1 = lev1, contact_level2 = lev2)
    elseif name == "PPML-trace"
        (o) -> _P.ppml_step_3d!(D, S1, S2, S3, Tau, dims, NG;
            state = st, dt, gamma = GAMMA, dx, order = o,
            face_periodic = true, predictor = :trace)
    else
        error("unknown solver $name")
    end

    tw = _P.with_pool() do
        @elapsed for s in 1:nsteps
            exact_label && fill_lagrangian_label!(Label, ic, (s - 1) * dt)
            carried_label && moments_to_labels!(Label, Label2, A1, A2, D, ic)
            step!(isodd(s) ? (1, 2, 3) : (3, 2, 1))
        end
    end
    hD = _P.to_host(D)
    hS1 = _P.to_host(S1)
    hS2 = _P.to_host(S2)
    hS3 = _P.to_host(S3)
    hTau = _P.to_host(Tau)
    return (; rho = center_line(hD, ic),
              pressure = pressure_line(hD, hS1, hS2, hS3, hTau, ic),
              wall = tw)
end

function contact_width(rho)
    lo = RHO_LO + 0.1 * (RHO_HI - RHO_LO)
    hi = RHO_LO + 0.9 * (RHO_HI - RHO_LO)
    return count(x -> lo < x < hi, rho)
end

function metrics(rho, pressure, exact)
    l1 = mean(abs.(rho .- exact))
    over = max(0.0, maximum(rho) - RHO_HI)
    under = max(0.0, RHO_LO - minimum(rho))
    pwig = maximum(abs.(pressure .- P0)) / P0
    return (; l1, width = contact_width(rho), over, under, pwig)
end

function write_profiles(path, x, exact, rows)
    open(path, "w") do io
        println(io, "x,exact," * join((replace(r.solver, '-' => '_') for r in rows), ","))
        for i in eachindex(x)
            print(io, x[i], ",", exact[i])
            for r in rows
                print(io, ",", r.rho[i])
            end
            println(io)
        end
    end
end

cases = case_arg === :both ? (:contact, :top_hat) : (case_arg,)
outdir = mkpath(joinpath(@__DIR__, "contact_out"))
metrics_path = joinpath(outdir, "contact_metrics.csv")
open(metrics_path, "w") do io
    println(io, "case,solver,nx,crossings,steps,dt,backend,l1,width_cells,overshoot,undershoot,pressure_wiggle,wall_s")
end

@printf("\nContact preservation — nx=%d, crossings=%.2f, backend=%s/%s\n", nx, crossings, bkname, T)
for kind in cases
    ic = contact_ic(kind)
    csmax = sqrt(GAMMA * P0 / RHO_LO)
    tfinal = crossings / abs(u0)
    dt0 = 0.3 * ic.dx / (abs(u0) + csmax)
    nsteps = ceil(Int, tfinal / dt0)
    dt = tfinal / nsteps
    exact = exact_line(ic)
    x = [(i - 0.5) / nx for i in 1:nx]
    rows = NamedTuple[]

    @printf("\ncase=%s u0=%.3f Mach_adv=%.2f t=%.4f dt=%.4e steps=%d\n",
            kind, u0, abs(u0) / csmax, tfinal, dt, nsteps)
    @printf("%-21s %-11s %-8s %-10s %-10s %-10s %-8s\n",
            "solver", "L1(rho)", "width", "overshoot", "undershoot", "pwiggle", "wall")
    println("-"^86)
    for solver in SOLVERS
        !isempty(solver_filter) && !occursin(solver_filter, solver) && continue
        try
            res = run_solver(solver, ic, dt, nsteps)
            m = metrics(res.rho, res.pressure, exact)
            @printf("%-21s %-11.4e %-8d %-10.3e %-10.3e %-10.3e %-8.2f\n",
                    solver, m.l1, m.width, m.over, m.under, m.pwig, res.wall)
            open(metrics_path, "a") do io
                @printf(io, "%s,%s,%d,%.6g,%d,%.12g,%s,%.12e,%d,%.12e,%.12e,%.12e,%.6f\n",
                        kind, solver, nx, crossings, nsteps, dt, bkname,
                        m.l1, m.width, m.over, m.under, m.pwig, res.wall)
            end
            push!(rows, (; solver, rho = res.rho))
        catch err
            @printf("%-21s failed: %s\n", solver, sprint(showerror, err))
        end
    end
    write_profiles(joinpath(outdir, "$(kind)_profiles.csv"), x, exact, rows)
end

println("\nwrote $metrics_path")
println("wrote profile CSVs under $outdir")
