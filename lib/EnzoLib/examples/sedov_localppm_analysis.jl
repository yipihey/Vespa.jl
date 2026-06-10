using EnzoLib
using Printf

const SRC = normpath(joinpath(@__DIR__, "..", "..", "..", "..",
    "run", "Hydro", "Hydro-2D", "SedovBlast", "SedovBlast.enzo"))

function replace_or_append(src::AbstractString, key::AbstractString, value::AbstractString)
    rx = Regex("^\\s*" * key * "\\s*=")
    lines = split(src, '\n'; keepempty = true)
    done = false
    for i in eachindex(lines)
        if occursin(rx, lines[i])
            lines[i] = value
            done = true
        end
    end
    done || push!(lines, value)
    return join(lines, '\n')
end

function patched_param(; method::Integer, n::Integer, stop::Real, nghost::Integer)
    txt = read(SRC, String)
    txt = replace_or_append(txt, "TopGridDimensions",
        @sprintf("TopGridDimensions         = %d %d", n, n))
    txt = replace_or_append(txt, "StopTime",
        @sprintf("StopTime                  = %.17g", Float64(stop)))
    txt = replace_or_append(txt, "dtDataDump", "dtDataDump                = 100.0")
    txt = replace_or_append(txt, "HydroMethod",
        @sprintf("HydroMethod               = %d", method))
    txt = replace_or_append(txt, "NumberOfGhostZones",
        @sprintf("NumberOfGhostZones        = %d", nghost))
    return txt
end

@inline function flat_index(i, j, k, dims)
    return i + dims[1] * (j - 1) + dims[1] * dims[2] * (k - 1)
end

function active_values(v, dims, rank, ng)
    nx = dims[1] - 2ng
    ny = rank >= 2 ? dims[2] - 2ng : 1
    out = Vector{Float64}(undef, nx * ny)
    p = 1
    @inbounds for j in 1:ny, i in 1:nx
        out[p] = v[flat_index(ng + i, rank >= 2 ? ng + j : 1, 1, dims)]
        p += 1
    end
    return out
end

function radial_profile(rho, dims, rank, ng, left, right; nbins = 160)
    nx = dims[1] - 2ng
    ny = rank >= 2 ? dims[2] - 2ng : 1
    dx = (right[1] - left[1]) / nx
    dy = rank >= 2 ? (right[2] - left[2]) / ny : dx
    rmax = hypot(max(abs(left[1] - 0.5), abs(right[1] - 0.5)),
                 max(abs(left[2] - 0.5), abs(right[2] - 0.5)))
    sumrho = zeros(Float64, nbins)
    count = zeros(Int, nbins)
    @inbounds for j in 1:ny, i in 1:nx
        x = left[1] + (i - 0.5) * dx
        y = left[2] + (j - 0.5) * dy
        r = hypot(x - 0.5, y - 0.5)
        b = clamp(fld(Int(floor(r / rmax * nbins)), 1) + 1, 1, nbins)
        sumrho[b] += rho[flat_index(ng + i, ng + j, 1, dims)]
        count[b] += 1
    end
    ravg = [(b - 0.5) / nbins * rmax for b in 1:nbins]
    prof = [count[b] == 0 ? NaN : sumrho[b] / count[b] for b in 1:nbins]
    return ravg, prof
end

function shock_radius(r, prof)
    best = 2
    bestgrad = -Inf
    for i in 2:length(prof)-1
        (isfinite(prof[i - 1]) && isfinite(prof[i + 1])) || continue
        g = prof[i + 1] - prof[i - 1]
        if g > bestgrad
            bestgrad = g
            best = i
        end
    end
    return r[best]
end

function analyze_handle(h; nghost)
    EnzoLib.session_rebuild(h, 0)
    g = EnzoLib.problem_grid_index_on_level(h, 0, 0)
    rank = EnzoLib.problem_grid_rank(h, g)
    dims = Tuple(EnzoLib.problem_grid_dims(h, g))
    left, right = EnzoLib.problem_grid_edge(h, g)
    iD = EnzoLib.field_index(h, 0; grid = g)
    iTE = EnzoLib.field_index(h, 1; grid = g)
    iV1 = EnzoLib.field_index(h, 4; grid = g)
    iV2 = EnzoLib.field_index(h, 5; grid = g)
    rho = EnzoLib.problem_get_field(h, iD, g)
    et = EnzoLib.problem_get_field(h, iTE, g)
    vx = EnzoLib.problem_get_field(h, iV1, g)
    vy = EnzoLib.problem_get_field(h, iV2, g)
    arho = active_values(rho, dims, rank, nghost)
    aet = active_values(et, dims, rank, nghost)
    nx = dims[1] - 2nghost
    ny = dims[2] - 2nghost
    cellvol = (right[1] - left[1]) / nx * (right[2] - left[2]) / ny
    mass = 0.0
    total_energy = 0.0
    radial_momentum = 0.0
    pmin = Inf
    pmax = -Inf
    q1 = q2 = q3 = q4 = 0.0
    @inbounds for j in 1:ny, i in 1:nx
        idx = flat_index(nghost + i, nghost + j, 1, dims)
        x = left[1] + (i - 0.5) * (right[1] - left[1]) / nx
        y = left[2] + (j - 0.5) * (right[2] - left[2]) / ny
        rr = hypot(x - 0.5, y - 0.5)
        mass += rho[idx] * cellvol
        total_energy += rho[idx] * et[idx] * cellvol
        kinetic = 0.5 * (vx[idx]^2 + vy[idx]^2)
        pressure = 0.4 * rho[idx] * (et[idx] - kinetic)
        pmin = min(pmin, pressure)
        pmax = max(pmax, pressure)
        rr > 0 && (radial_momentum += rho[idx] * (vx[idx] * (x - 0.5) + vy[idx] * (y - 0.5)) / rr * cellvol)
        if x >= 0.5 && y >= 0.5
            q1 += rho[idx] * cellvol
        elseif x < 0.5 && y >= 0.5
            q2 += rho[idx] * cellvol
        elseif x < 0.5 && y < 0.5
            q3 += rho[idx] * cellvol
        else
            q4 += rho[idx] * cellvol
        end
    end
    r, prof = radial_profile(rho, dims, rank, nghost, left, right)
    return (
        time = EnzoLib.session_time(h),
        cycle = EnzoLib.session_cycle(h),
        grids = EnzoLib.problem_num_grids(h),
        max_level = maximum((l for l in 0:8 if EnzoLib.session_num_grids_on_level(h, l) > 0); init = 0),
        dims = dims,
        active = (nx, ny),
        mass = mass,
        total_energy = total_energy,
        radial_momentum = radial_momentum,
        rho_min = minimum(arho),
        rho_max = maximum(arho),
        pressure_min = pmin,
        pressure_max = pmax,
        shock_radius = shock_radius(r, prof),
        expected_radius = sqrt(EnzoLib.session_time(h)),
        quadrant_mass_spread = maximum((q1, q2, q3, q4)) - minimum((q1, q2, q3, q4)),
        rho_active = arho,
        total_energy_active = aet,
    )
end

function run_case(; label, method, n, stop, nghost)
    mktempdir() do work
        pf = joinpath(work, "$label.enzo")
        write(pf, patched_param(; method, n, stop, nghost))
        cd(work) do
            h = EnzoLib.session_init(pf)
            h == C_NULL && error("session_init failed for $pf")
            try
                initial = analyze_handle(h; nghost)
                engine = method == EnzoLib.LOCAL_PPM_HYDROMETHOD ?
                    EnzoLib.local_ppm_engine(pf) : EnzoLib.EngineConfig()
                while EnzoLib.session_time(h) < EnzoLib.session_stop_time(h) &&
                      EnzoLib.session_cycle(h) < 100000
                    EnzoLib.evolve_level!(h, 0, 0.0; engine = engine, regrid = true)
                end
                final = analyze_handle(h; nghost)
                return (paramfile = pf, initial = initial, final = final)
            finally
                EnzoLib.free_problem(h)
            end
        end
    end
end

function print_case(name, r)
    i = r.initial
    f = r.final
    @printf("%s\n", name)
    @printf("  active=%dx%d dims=%s cycles=%d time=%.8f grids=%d max_level=%d\n",
        f.active[1], f.active[2], string(f.dims), f.cycle, f.time, f.grids, f.max_level)
    @printf("  mass:          %.16e  drift=%+.3e\n", f.mass, f.mass - i.mass)
    @printf("  total energy:  %.16e  drift=%+.3e\n", f.total_energy, f.total_energy - i.total_energy)
    @printf("  rho[min,max]:  %.8e  %.8e\n", f.rho_min, f.rho_max)
    @printf("  p[min,max]:    %.8e  %.8e\n", f.pressure_min, f.pressure_max)
    @printf("  shock radius:  %.8f  expected~%.8f  rel.err=%+.3e\n",
        f.shock_radius, f.expected_radius, (f.shock_radius - f.expected_radius) / f.expected_radius)
    @printf("  radial mom.:   %.8e\n", f.radial_momentum)
    @printf("  quadrant mass spread: %.3e\n", f.quadrant_mass_spread)
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 200
    stop = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.10
    localppm = run_case(; label = "sedov-localppm", method = 10, n, stop, nghost = 1)
    legacy = run_case(; label = "sedov-enzo-ppm", method = 0, n, stop, nghost = 3)
    print_case("Local PPM trace/two-shock, HydroMethod=10, NumberOfGhostZones=1", localppm)
    print_case("Legacy Enzo PPM, HydroMethod=0, NumberOfGhostZones=3", legacy)
    lf = localppm.final
    ef = legacy.final
    @printf("Comparison local - legacy:\n")
    @printf("  shock radius Δ = %+.8e\n", lf.shock_radius - ef.shock_radius)
    @printf("  rho max Δ      = %+.8e\n", lf.rho_max - ef.rho_max)
    @printf("  mass drift Δ   = %+.8e\n", (lf.mass - localppm.initial.mass) - (ef.mass - legacy.initial.mass))
    @printf("  energy drift Δ = %+.8e\n", (lf.total_energy - localppm.initial.total_energy) - (ef.total_energy - legacy.initial.total_energy))
    drho = lf.rho_active .- ef.rho_active
    dte = lf.total_energy_active .- ef.total_energy_active
    @printf("  rho rel L1/Linf = %.8e / %.8e\n",
        sum(abs, drho) / sum(abs, ef.rho_active),
        maximum(abs, drho) / maximum(abs, ef.rho_active))
    @printf("  specific Etot rel L1/Linf = %.8e / %.8e\n",
        sum(abs, dte) / sum(abs, ef.total_energy_active),
        maximum(abs, dte) / maximum(abs, ef.total_energy_active))
end

main()
