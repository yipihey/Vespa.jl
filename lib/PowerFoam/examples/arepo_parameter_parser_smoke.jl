#!/usr/bin/env julia

using Dates
using Printf
using PowerFoam

const OUTBASE = joinpath(@__DIR__, "out", "arepo_parameter_parser_smoke")
const RUN_TAG = Dates.format(now(), "yyyymmdd_HHMMSS")
const OUTDIR = joinpath(OUTBASE, RUN_TAG)

const PARAM_TEXT = """
% Representative AREPO param.txt slice for parser smoke
InitCondFile ics/sedov_2d.hdf5
ICFormat 3
OutputDir output/sedov2d
SnapshotFileBase snap
SnapFormat 3
NumFilesPerSnapshot 1
NumFilesWrittenInParallel 1
OutputListOn 1
OutputListFilename output_list.txt
TimeBegin 0.0
TimeMax 0.2
TimeBetSnapshot 0.05
TimeOfFirstSnapshot 0.05
TimeBetStatistics 0.01
CpuTimeBetRestartFile 30.0
TimeLimitCPU 120.0
BoxSize 1.0
PeriodicBoundariesOn 0
ComovingIntegrationOn 0
CourantFac 0.4
MaxSizeTimestep 0.02
MinSizeTimestep 1.0e-6
TypeOfTimestepCriterion courant
InitGasTemp 100.0
MinGasTemp 10.0
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
SofteningComovingType1 0.002
SofteningMaxPhysType0 0.001
SofteningMaxPhysType1 0.002
SofteningTypeOfPartType0 0
SofteningTypeOfPartType1 1
UnitVelocity_in_cm_per_s 1.0e5
"""

const CONFIG_TEXT = """
# Representative Config.sh slice for parser smoke
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
SHOCK_FOLLOWING_MESH=0
SHOCK_FOLLOWING_MESH_GAIN=0.25
SHOCK_FOLLOWING_MESH_WIDTH=0.05
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

function write_readme(path, rows, validation; command)
    open(path, "w") do io
        println(io, "# AREPO Parameter Parser Smoke")
        println(io)
        println(io, "This example exercises the package-exported AREPO parameter/config")
        println(io, "parser through `using PowerFoam`, normalizes a representative")
        println(io, "runtime slice, validates it, and records a tiny summary artifact.")
        println(io)
        @printf(io, "- generated: %s\n", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        @printf(io, "- command: `%s`\n", command)
        @printf(io, "- validation: `%s`\n", validation.valid ? "valid" : "invalid")
        @printf(io, "- warnings: %d\n", length(validation.warnings))
        println(io)
        println(io, "## Key fields")
        println(io)
        println(io, "| section | key | value |")
        println(io, "| --- | --- | --- |")
        for row in rows
            @printf(io, "| %s | %s | %s |\n", row[1], row[2], row[3])
        end
        if !isempty(validation.warnings)
            println(io)
            println(io, "## Warnings")
            println(io)
            for warning in validation.warnings
                println(io, "- ", warning)
            end
        end
        if !isempty(validation.errors)
            println(io)
            println(io, "## Errors")
            println(io)
            for err in validation.errors
                println(io, "- ", err)
            end
        end
    end
end

function summarize_rows(params, validation)
    io = params.normalized.io
    time = params.normalized.time
    domain = params.normalized.domain
    hydro = params.normalized.hydro
    mesh = params.normalized.mesh
    gravity = params.normalized.gravity
    features = params.normalized.features

    return [
        ("validation", "valid", validation.valid),
        ("validation", "warning_count", length(validation.warnings)),
        ("io", "init_cond_file", io.init_cond_file),
        ("io", "output_dir", io.output_dir),
        ("io", "snapshot_file_base", io.snapshot_file_base),
        ("time", "time_begin", time.time_begin),
        ("time", "time_max", time.time_max),
        ("domain", "box_size", domain.box_size),
        ("domain", "periodic_boundaries_on", domain.periodic_boundaries_on),
        ("hydro", "courant_fac", hydro.courant_fac),
        ("mesh", "des_num_ngb", mesh.des_num_ngb),
        ("gravity", "softening_type0", gravity.softening_type_of_part === nothing ? nothing : gravity.softening_type_of_part[1]),
        ("features", "twodims", features.twodims),
        ("features", "double_precision", features.double_precision),
        ("features", "have_hdf5", features.have_hdf5),
        ("features", "local_ppm", features.local_ppm),
        ("features", "riemann_hll", features.riemann_hll),
        ("extras", "extra_key_count", length(keys(params.normalized.extras))),
    ]
end

function print_summary(rows, outdir)
    println("AREPO parameter parser smoke")
    println("============================")
    @printf("artifact_dir=%s\n", outdir)
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

    rows = summarize_rows(params, validation)
    mkpath(OUTDIR)
    readme_path = joinpath(OUTDIR, "README.md")
    csv_path = joinpath(OUTDIR, "summary.csv")
    command = "julia --project=lib/PowerFoam lib/PowerFoam/examples/arepo_parameter_parser_smoke.jl"
    write_readme(readme_path, rows, validation; command)
    write_csv(csv_path, rows)

    @printf("wrote %s\n", readme_path)
    @printf("wrote %s\n", csv_path)
    print_summary(rows, OUTDIR)
end

main()
