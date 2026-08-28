#!/usr/bin/env julia

# Reproducible short production gate for the spectral radiation/full-MHD path.
# It intentionally uses identical physical modes and P(k,mu) bins at all resolutions.

using Dates
using Printf

const DRIVER = joinpath(@__DIR__, "pmf_dm_lattice_bench.jl")
const DEFAULT_NS = (64, 128, 256)
const K_CUT = 4pi
const K_MAX = 8pi

function _parse_ns(s::AbstractString)
    ns = Tuple(parse.(Int, filter(!isempty, strip.(split(s, ',')))))
    isempty(ns) && error("MHD_GATE_NS must contain at least one resolution")
    all(n -> n >= 64 && ispow2(n), ns) ||
        error("MHD_GATE_NS entries must be powers of two >= 64")
    return ns
end

function _last_tsv_row(path::AbstractString)
    lines = filter(!isempty, readlines(path))
    length(lines) >= 2 || error("no data rows in $path")
    names = split(lines[1], '\t')
    vals = split(lines[end], '\t')
    length(names) == length(vals) || error("malformed TSV row in $path")
    return Dict(names .=> vals)
end

_number(row, key) = parse(Float64, row[key])

function _initial_b_spectrum(path::AbstractString)
    lines = readlines(path)
    header = split(lines[1], '\t')
    col = Dict(name => i for (i, name) in pairs(header))
    power = Dict{Tuple{Int,Int},Tuple{Float64,Int}}()
    for line in @view lines[2:end]
        v = split(line, '\t')
        v[col["field"]] == "B_vector" || continue
        parse(Int, v[col["cycle"]]) == 0 || continue
        key = (parse(Int, v[col["kbin"]]), parse(Int, v[col["mubin"]]))
        p = parse(Float64, v[col["P"]])
        n = parse(Int, v[col["Nmodes"]])
        power[key] = (p, n)
    end
    isempty(power) && error("no initial B_vector spectrum in $path")
    return power
end

function _run_case(outdir::AbstractString, n::Int; max_dlna=2.5e-4,
                   steps=20, warmup=2, suffix="",zend=20)
    tag = "N$(n)$(suffix)"
    history = joinpath(outdir, "$(tag)_history.tsv")
    pk = joinpath(outdir, "$(tag)_pk.tsv")
    summary = joinpath(outdir, "$(tag)_summary.tsv")
    log = joinpath(outdir, "$(tag).log")
    for path in (history, pk, summary, log)
        ispath(path) && error("refusing to overwrite $path; choose a new MHD_GATE_OUT")
    end

    env = copy(ENV)
    merge!(env, Dict(
        "MHD_BACKEND" => "metal",
        "MHD_PMF_N" => string(n),
        "MHD_STEPS" => string(steps),
        "MHD_WARMUP" => string(warmup),
        "MHD_MAX_CYCLES" => string(steps + warmup),
        "MHD_ZSTART" => "1000000",
        "MHD_ZEND" => string(zend),
        "MHD_MAXEXP" => "1e-3",
        "MHD_RADIATION_MAX_DLNA" => string(max_dlna),
        "MHD_RADIATION_RESPONSE_MODEL" =>
            get(ENV,"MHD_GATE_RESPONSE_MODEL","moments"),
        "MHD_RADIATION_ENERGY_LEDGER_EVERY" => "5",
        "MHD_COMPTON_DRAG" => "1",
        "MHD_DRAG_DT_MODE" => "radiation",
        "MHD_GLM_PROJECT_EVERY" => "1",
        "MHD_TERMINAL_LBOX_CKPC" => "2",
        # Code coordinates span one periodic domain; its physical length is
        # supplied independently by MHD_TERMINAL_LBOX_CKPC.
        "MHD_BOXSIZE" => "1",
        "MHD_CHEM_VUNIT" => "1e8",
        "MHD_P0" => "auto",
        "MHD_PMF_B0_NG" => "0.001",
        "MHD_PMF_SEED" => "42",
        "MHD_PMF_KCUT" => string(K_CUT),
        "MHD_PMF_KMAX" => string(K_MAX),
        "MHD_PMF_MIN_CELLS_PER_WAVELENGTH" => "16",
        "MHD_PMF_MODE_LOCK_N" => "64",
        "MHD_PMF_CURL_SYMBOL" => "continuum",
        "MHD_GRAVITY" => "false",
        "MHD_CHEM" => "off",
        "MHD_INITIAL_HISTORY" => "1",
        "MHD_INITIAL_PK" => "1",
        "MHD_PK_EVERY" => string(steps),
        "MHD_PK_NBINS" => "10",
        "MHD_PK_NMU" => "4",
        "MHD_PK_KMAX" => string(K_MAX),
        "MHD_HISTORY_TSV" => history,
        "MHD_PK_TSV" => pk,
        "MHD_OUT_TSV" => summary,
        "MHD_PRINT_EVERY" => "1000",
        "MHD_CYCLE_PRINT_EVERY" => "1000",
        "MHD_CHECK_EVERY" => "0",
    ))
    project = @__DIR__
    cmd = `$(Base.julia_cmd()) --project=$project $DRIVER`
    @printf("running %s: max_dlna=%.3g steps=%d\n", tag, max_dlna, steps)
    flush(stdout)
    open(log, "w") do io
        run(pipeline(setenv(cmd, env), stdout=io, stderr=io))
    end
    return (; tag, history, pk, summary, log, row=_last_tsv_row(summary))
end

function _spatial_gate(cases)
    sort!(cases; by=c -> parse(Int, c.row["N"]))
    br0 = [_number(c.row, "brms0") for c in cases]
    br = [_number(c.row, "brms") for c in cases]
    delta = [_number(c.row, "delta_b_rms") for c in cases]
    divb = [_number(c.row, "divBdx_over_brms") for c in cases]
    throughput = [_number(c.row, "throughput_macro_mcells_s") for c in cases]
    ns = [parse(Int, c.row["N"]) for c in cases]

    maximum(abs.(br0 ./ br0[end] .- 1)) < 1e-5 ||
        error("matched-mode initial Brms differs across resolutions: $br0")
    maximum(divb) < 1e-5 || error("divB gate failed: $divb")
    all(delta[i+1] <= max(1.1*delta[i],1e-9) for i in 1:length(delta)-1) ||
        error("density response is not spatially convergent above the f32 diagnostic floor: $delta")

    i128 = findfirst(==(128), ns)
    i256 = findfirst(==(256), ns)
    if i128 !== nothing && i256 !== nothing
        abs(br[i128] / br[i256] - 1) < 5e-4 ||
            error("N128/N256 final Brms mismatch: $(br[i128]) vs $(br[i256])")
        delta[i128] < 2e-8 || error("N128 density residual exceeds gate: $(delta[i128])")
        delta[i256] < 5e-9 || error("N256 density residual exceeds gate: $(delta[i256])")
    end

    spectra = [_initial_b_spectrum(c.pk) for c in cases]
    keys0 = Set(keys(spectra[end]))
    all(Set(keys(s)) == keys0 for s in spectra) ||
        error("fixed physical P(k,mu) bins/mode counts differ across resolutions")
    for key in keys0
        counts = [s[key][2] for s in spectra]
        all(==(counts[end]), counts) || error("mode count mismatch in bin $key: $counts")
    end
    peak = maximum(v[1] for v in values(spectra[end]) if v[2] > 0)
    for key in keys0
        pref, count = spectra[end][key]
        count > 0 && pref > peak * 1e-10 || continue
        rel = [abs(s[key][1] / pref - 1) for s in spectra]
        maximum(rel) < 2e-4 ||
            error("matched initial B power differs in bin $key: relative errors $rel")
    end

    println("\nSpatial production gate")
    println("N\tBrms0\tBrms_final\tdelta_b_rms\tdivBdx/Brms\tMcell/s")
    for i in eachindex(ns)
        @printf("%d\t%.8e\t%.8e\t%.8e\t%.3e\t%.1f\n",
                ns[i], br0[i], br[i], delta[i], divb[i], throughput[i])
    end
end

function main()
    ns = _parse_ns(get(ENV, "MHD_GATE_NS", join(DEFAULT_NS, ',')))
    stamp = Dates.format(now(), dateformat"yyyymmdd-HHMMSS")
    outdir = abspath(get(ENV, "MHD_GATE_OUT", joinpath(tempdir(), "radiation-gate-$stamp")))
    mkpath(outdir)
    cases = [_run_case(outdir, n) for n in ns]
    _spatial_gate(cases)

    if get(ENV, "MHD_GATE_TEMPORAL_REFERENCE", "0") == "1"
        128 in ns || error("temporal reference requires N=128 in MHD_GATE_NS")
        base = only(filter(c -> parse(Int, c.row["N"]) == 128, cases))
        target_z = _number(base.row,"final_z")
        # The refined run is redshift-limited, not cycle-count matched. The
        # generous step ceiling lets the scale-factor integrator clamp its last
        # step to exactly the coarse run's final epoch.
        ref = _run_case(outdir, 128; max_dlna=1e-5, steps=2000, warmup=2,
                        suffix="_dtref",zend=target_z)
        abs(_number(ref.row,"final_z")-target_z) <=
            1e-9*max(target_z,1.0) ||
            error("temporal reference stopped at the wrong redshift")
        ebr = abs(_number(base.row, "brms") / _number(ref.row, "brms") - 1)
        ebr < 0.01 || error("production max_dlna Brms error $ebr exceeds 1%")
        @printf("Temporal gate: N128 Brms relative error %.4g (limit 0.01)\n", ebr)
    end
    println("PASS: radiation/full-MHD production gate; artifacts in $outdir")
end

main()
