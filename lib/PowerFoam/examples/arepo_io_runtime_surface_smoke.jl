#!/usr/bin/env julia

const POWERFOAM_ROOT = normpath(joinpath(@__DIR__, ".."))
const SRC_DIR = joinpath(POWERFOAM_ROOT, "src")

using PowerFoam

const PLANNED_TYPES = [
    :ArepoConfigFlags,
    :ArepoParameterSet,
    :ArepoRuntimeFeatureSet,
    :ArepoICHeader,
    :ArepoGasICBlock,
    :ArepoICData,
    :ArepoSnapshotLocator,
    :ArepoSnapshotHeader,
    :ArepoGasSnapshotBlock,
    :ArepoSnapshotData,
    :ArepoRestartStamp,
    :ArepoRestartAssessment,
    :ArepoDiagnosticPolicy,
    :ArepoRunArtifactLayout,
]

const PLANNED_FUNCTIONS = [
    :read_arepo_param_file,
    :parse_arepo_param_text,
    :read_arepo_config_flags,
    :parse_arepo_config_text,
    :normalize_arepo_parameters,
    :arepo_runtime_features,
    :validate_arepo_parameters,
    :read_arepo_ic,
    :write_arepo_ic,
    :validate_arepo_ic,
    :read_arepo_snapshot,
    :write_arepo_snapshot,
    :locate_arepo_snapshot,
    :validate_arepo_snapshot,
    :build_arepo_restart_stamp,
    :assess_arepo_restart_compatibility,
    :default_arepo_diagnostic_policy,
    :build_arepo_run_tag,
    :materialize_arepo_artifact_layout,
    :write_arepo_run_summary,
]

const PLANNED_SOURCE_FILES = [
    "arepo_io_runtime.jl",
    "arepo_io_parameters.jl",
    "arepo_io_ic.jl",
    "arepo_io_snapshots.jl",
    "arepo_io_restart.jl",
    "arepo_io_diagnostics.jl",
]

function main()
    println("# PowerFoam IO runtime surface")
    println()
    println("## Types")
    present_types = Symbol[]
    missing_types = Symbol[]
    for name in PLANNED_TYPES
        present = isdefined(PowerFoam, name)
        println("- ", name, ": ", present ? "present" : "missing")
        push!(present ? present_types : missing_types, name)
    end

    println()
    println("## Functions")
    present_functions = Symbol[]
    missing_functions = Symbol[]
    for name in PLANNED_FUNCTIONS
        present = isdefined(PowerFoam, name)
        println("- ", name, ": ", present ? "present" : "missing")
        push!(present ? present_functions : missing_functions, name)
    end

    println()
    println("## Planned source files")
    missing = String[]
    for file in PLANNED_SOURCE_FILES
        path = joinpath(SRC_DIR, file)
        exists = isfile(path)
        println("- ", file, ": ", exists ? "present" : "missing")
        exists || push!(missing, file)
    end

    println()
    println("Present type count: ", length(present_types), " / ", length(PLANNED_TYPES))
    println("Present function count: ", length(present_functions), " / ", length(PLANNED_FUNCTIONS))
    println("Missing source file count: ", length(missing))
end

main()
