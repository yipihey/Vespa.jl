#!/usr/bin/env julia

using Dates
using Printf
using PowerFoam

const OUTBASE = joinpath(@__DIR__, "out", "arepo_runtime_param_spec_smoke")
const RUN_TAG = Dates.format(now(), "yyyymmdd_HHMMSS")
const OUTDIR = joinpath(OUTBASE, RUN_TAG)

const PARAM_TEXT = """
% Representative KH2D-style AREPO runtime slice for parser/spec smoke
InitCondFile ics/kh2d.hdf5
ICFormat 3
OutputDir output/kh2d
SnapshotFileBase snap
SnapFormat 3
NumFilesPerSnapshot 1
NumFilesWrittenInParallel 1
OutputListOn 1
OutputListFilename output_list.txt
TimeBegin 0.0
TimeMax 1.0
TimeBetSnapshot 0.1
TimeOfFirstSnapshot 0.1
TimeBetStatistics 0.05
CpuTimeBetRestartFile 30.0
TimeLimitCPU 120.0
BoxSize 1.0
PeriodicBoundariesOn 1
ComovingIntegrationOn 0
CourantFac 0.4
MaxSizeTimestep 0.02
MinSizeTimestep 1.0e-6
TypeOfTimestepCriterion courant
InitGasTemp 1.0
MinGasTemp 0.1
MinEgySpec 1.0e-8
MinimumDensityOnStartUp 1.0e-12
DesNumNgb 32
MaxNumNgbDeviation 2
MultipleDomains 1
TopNodeFactor 2.0
ActivePartFracForNewDomainDecomp 0.05
CellShapingSpeed 0.5
CellMaxAngleFactor 1.4
ErrTolIntAccuracy 0.025
ErrTolTheta 0.7
ErrTolForceAcc 0.0025
GasSoftFactor 1.5
GravityConstantInternal 1.0
SofteningComovingType0 0.001
SofteningMaxPhysType0 0.001
SofteningTypeOfPartType0 0
NumGasCells 4096
ProblemName kh2d_runtime_param_spec
MeshMotion moving
HydroCase kelvin_helmholtz
"""

const CONFIG_TEXT = """
# Representative KH2D-style Config.sh slice for parser/spec smoke
TWODIMS
DOUBLEPRECISION=1
INPUT_IN_DOUBLEPRECISION
OUTPUT_IN_DOUBLEPRECISION
HAVE_HDF5=1
OUTPUT_PRESSURE
OUTPUT_VOLUME
REGULARIZE_MESH_CM_DRIFT
REGULARIZE_MESH_FACE_ANGLE
LOCAL_PPM
RIEMANN_HLL
"""

function csvquote(x)
    s = string(x)
    return "\"" * replace(s, "\"" => "\"\"") * "\""
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, "section,key,value")
        for row in rows
            println(io, join((csvquote(v) for v in row), ","))
        end
    end
end

function maybe_get(nt::NamedTuple, key::Symbol, default)
    return key in keys(nt) ? getfield(nt, key) : default
end

function infer_problem_spec(params::ArepoParameterSet)
    normalized = params.normalized
    extras = normalized.extras
    features = normalized.features

    dim = features.twodims ? 2 : 3
    box_size = something(normalized.domain.box_size, 1.0)
    domain = dim == 2 ?
        ((0.0, box_size), (0.0, box_size)) :
        ((0.0, box_size), (0.0, box_size), (0.0, box_size))
    periodic = ntuple(_ -> something(normalized.domain.periodic_boundaries_on, false), dim)
    gas_cell_count = parse(Int, maybe_get(extras, :NumGasCells, "0"))
    problem_name = maybe_get(extras, :ProblemName, "arepo_runtime_param_spec_smoke")
    hydro_case = Symbol(maybe_get(extras, :HydroCase, "unknown"))
    mesh_motion = Symbol(maybe_get(extras, :MeshMotion, "unknown"))
    solver = normalized.features.riemann_hll ? :hll : :unspecified
    reconstruction = normalized.features.local_ppm ? :local_ppm : :unspecified

    return arepo_problem_spec(
        problem_name;
        dimensionality = dim,
        domain = domain,
        periodic = periodic,
        gas_cell_count = gas_cell_count,
        particle_count = 0,
        physics = (
            hydro = true,
            tessellation = true,
            gravity = false,
        ),
        initial_conditions = (
            family = hydro_case,
            solver = solver,
            reconstruction = reconstruction,
            mesh_motion = mesh_motion,
            init_cond_file = normalized.io.init_cond_file,
        ),
        metadata = (
            source = :arepo_param_config_parser_smoke,
            output_dir = normalized.io.output_dir,
            snapshot_file_base = normalized.io.snapshot_file_base,
            output_list_on = normalized.io.output_list_on,
            validator_warning_count = length(validate_arepo_parameters(params).warnings),
        ),
    )
end

function summarize_rows(params, validation, spec, smoke, state)
    normalized = params.normalized
    return [
        ("validation", "valid", validation.valid),
        ("validation", "warning_count", length(validation.warnings)),
        ("runtime", "problem_name", spec.name),
        ("runtime", "dimensionality", spec.dimensionality),
        ("runtime", "box_size", normalized.domain.box_size),
        ("runtime", "periodic_boundaries_on", normalized.domain.periodic_boundaries_on),
        ("runtime", "gas_cell_count", spec.gas_cell_count),
        ("runtime", "init_cond_file", normalized.io.init_cond_file),
        ("runtime", "output_dir", normalized.io.output_dir),
        ("runtime", "courant_fac", normalized.hydro.courant_fac),
        ("runtime", "des_num_ngb", normalized.mesh.des_num_ngb),
        ("features", "twodims", normalized.features.twodims),
        ("features", "double_precision", normalized.features.double_precision),
        ("features", "local_ppm", normalized.features.local_ppm),
        ("features", "riemann_hll", normalized.features.riemann_hll),
        ("spec", "periodic", spec.periodic),
        ("spec", "physics", spec.physics),
        ("smoke", "status", smoke.status),
        ("smoke", "eligible", smoke.eligible),
        ("smoke", "requirements", join(string.(smoke.requirements), ";")),
        ("scaffold", "status", state.status),
        ("scaffold", "unsupported", join(string.(state.unsupported), ";")),
        ("extras", "count", length(keys(normalized.extras))),
    ]
end

function write_readme(path, rows, validation, spec, smoke, state; command)
    open(path, "w") do io
        println(io, "# AREPO Runtime Param-Spec Smoke")
        println(io)
        println(io, "This example stays on the exported `PowerFoam` parser/runtime surface.")
        println(io, "It parses representative `param.txt` and `Config.sh` text, normalizes")
        println(io, "the fields, builds an `ArepoProblemSpec` from the normalized runtime")
        println(io, "view where possible, classifies the pure-KA hydro smoke fit, and")
        println(io, "records the scaffold/runtime summary.")
        println(io)
        @printf(io, "- generated: %s\n", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        @printf(io, "- command: `%s`\n", command)
        @printf(io, "- spec: `%s`\n", string(spec.name))
        @printf(io, "- validation: `%s`\n", validation.valid ? "valid" : "invalid")
        @printf(io, "- smoke status: `%s`\n", string(smoke.status))
        @printf(io, "- scaffold status: `%s`\n", string(state.status))
        println(io)
        println(io, "## Summary")
        println(io)
        println(io, "| section | key | value |")
        println(io, "| --- | --- | --- |")
        for row in rows
            @printf(io, "| %s | %s | %s |\n", row[1], row[2], row[3])
        end
        println(io)
        println(io, "## Smoke Reasons")
        println(io)
        for reason in smoke.reasons
            println(io, "- ", reason)
        end
        println(io)
        println(io, "## Scaffold Diagnostics")
        println(io)
        for diag in state.diagnostics
            println(io, "- ", diag)
        end
        if !isempty(validation.warnings)
            println(io)
            println(io, "## Validation Warnings")
            println(io)
            for warning in validation.warnings
                println(io, "- ", warning)
            end
        end
    end
end

function print_summary(rows, smoke, outdir)
    println("AREPO runtime param-spec smoke")
    println("==============================")
    @printf("artifact_dir=%s\n", outdir)
    @printf("smoke_status=%s eligible=%s\n", string(smoke.status), string(smoke.eligible))
    for (section, key, value) in rows
        @printf("%-10s %-24s %s\n", string(section), string(key), string(value))
    end
end

function main()
    raw = parse_arepo_param_text(PARAM_TEXT)
    flags = parse_arepo_config_text(CONFIG_TEXT)
    params = normalize_arepo_parameters(raw, flags)
    validation = validate_arepo_parameters(params)
    validation.valid || error("validation failed: $(join(validation.errors, "; "))")

    spec = infer_problem_spec(params)
    smoke = classify_ka_hydro_smoke(spec)
    state = arepo_run_scaffold(
        spec;
        options = ArepoRunOptions(
            start_time = something(params.normalized.time.time_begin, 0.0),
            final_time = something(params.normalized.time.time_max, 0.0),
            max_steps = 1,
            cfl = something(params.normalized.hydro.courant_fac, 0.4),
        ),
    )

    rows = summarize_rows(params, validation, spec, smoke, state)
    mkpath(OUTDIR)
    readme_path = joinpath(OUTDIR, "README.md")
    csv_path = joinpath(OUTDIR, "summary.csv")
    command = "julia --project=lib/PowerFoam lib/PowerFoam/examples/arepo_runtime_param_spec_smoke.jl"
    write_readme(readme_path, rows, validation, spec, smoke, state; command)
    write_csv(csv_path, rows)

    @printf("wrote %s\n", readme_path)
    @printf("wrote %s\n", csv_path)
    print_summary(rows, smoke, OUTDIR)
end

main()
