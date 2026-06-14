# Compare two Santa Barbara campaign diagnostic CSVs produced by sb_metal_amr.jl.
#
# Run:
#   <julia> lib/EnzoLib/examples/sb_compare_campaigns.jl cpu/diagnostics.csv metal/diagnostics.csv [out.md]

using Printf

function _parse_bool(s)
    lowercase(strip(s)) in ("true", "1", "yes")
end

function read_campaign_csv(path::AbstractString)
    lines = readlines(path)
    length(lines) >= 2 || error("empty campaign CSV: $path")
    rows = NamedTuple[]
    for ln in lines[2:end]
        isempty(strip(ln)) && continue
        f = split(ln, ',')
        length(f) == 9 || error("bad campaign CSV row in $path: $ln")
        push!(rows, (cycle = parse(Int, f[1]),
                     time = parse(Float64, f[2]),
                     grids = (parse(Int, f[3]), parse(Int, f[4]), parse(Int, f[5])),
                     rhomax = parse(Float64, f[6]),
                     mass_drift = parse(Float64, f[7]),
                     refined = _parse_bool(f[8]),
                     seconds = parse(Float64, f[9])))
    end
    return rows
end

function compare_campaigns(cpu_csv::AbstractString, other_csv::AbstractString;
                           out::Union{Nothing,String} = nothing,
                           label_a::AbstractString = "cpu-f32",
                           label_b::AbstractString = "candidate")
    a = read_campaign_csv(cpu_csv)
    b = read_campaign_csv(other_csv)
    n = min(length(a), length(b))
    n > 0 || error("no overlapping campaign rows")
    mismatched_cycles = Int[]
    mismatched_grids = Int[]
    max_dt = 0.0
    max_rel_rho = 0.0
    max_abs_mass = 0.0
    for i in 1:n
        a[i].cycle == b[i].cycle || push!(mismatched_cycles, i)
        a[i].grids == b[i].grids && a[i].refined == b[i].refined || push!(mismatched_grids, a[i].cycle)
        max_dt = max(max_dt, abs(a[i].time - b[i].time))
        scale = max(abs(a[i].rhomax), eps(Float64))
        max_rel_rho = max(max_rel_rho, abs(a[i].rhomax - b[i].rhomax) / scale)
        max_abs_mass = max(max_abs_mass, abs(a[i].mass_drift - b[i].mass_drift))
    end
    result = (rows = n,
              cycles_match = isempty(mismatched_cycles),
              grids_match = isempty(mismatched_grids),
              mismatched_grid_cycles = mismatched_grids,
              max_abs_time_diff = max_dt,
              max_rel_rhomax_diff = max_rel_rho,
              max_abs_mass_drift_diff = max_abs_mass,
              speedup = sum(r.seconds for r in a[1:n]) / sum(r.seconds for r in b[1:n]))
    if out !== nothing
        open(out, "w") do io
            println(io, "# Santa Barbara campaign comparison")
            println(io)
            println(io, "- reference: `$label_a` (`$cpu_csv`)")
            println(io, "- candidate: `$label_b` (`$other_csv`)")
            println(io, "- overlapping rows: `$(result.rows)`")
            println(io)
            println(io, "| metric | value |")
            println(io, "|---|---:|")
            println(io, "| cycles match | $(result.cycles_match) |")
            println(io, "| grids/refinement match | $(result.grids_match) |")
            @printf(io, "| max |Δt| | %.6e |\n", result.max_abs_time_diff)
            @printf(io, "| max relative Δρmax | %.6e |\n", result.max_rel_rhomax_diff)
            @printf(io, "| max |Δmass drift| | %.6e |\n", result.max_abs_mass_drift_diff)
            @printf(io, "| candidate speedup | %.3f |\n", result.speedup)
            if !isempty(result.mismatched_grid_cycles)
                println(io)
                println(io, "Mismatched grid/refinement cycles: `$(result.mismatched_grid_cycles)`")
            end
        end
    end
    return result
end

function main(args = ARGS)
    length(args) in (2, 3) ||
        error("usage: sb_compare_campaigns.jl cpu/diagnostics.csv candidate/diagnostics.csv [out.md]")
    out = length(args) == 3 ? args[3] : nothing
    r = compare_campaigns(args[1], args[2]; out = out)
    @printf("rows=%d cycles_match=%s grids_match=%s max_rel_rhomax=%.3e speedup=%.3f\n",
            r.rows, r.cycles_match, r.grids_match, r.max_rel_rhomax_diff, r.speedup)
    out === nothing || println("report: $out")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
