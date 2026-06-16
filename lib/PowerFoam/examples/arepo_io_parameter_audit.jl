#!/usr/bin/env julia

using Printf

const POWERFOAM_ROOT = normpath(joinpath(@__DIR__, ".."))
const EXAMPLES_DIR = @__DIR__
const SRC_DIR = joinpath(POWERFOAM_ROOT, "src")
const PLANNING_DIR = joinpath(POWERFOAM_ROOT, "planning")

function relpath_from_powerfoam(path)
    return relpath(path, POWERFOAM_ROOT)
end

function recursive_files(root)
    files = String[]
    for (dir, _, names) in walkdir(root)
        for name in names
            push!(files, joinpath(dir, name))
        end
    end
    sort!(files)
    return files
end

function basename_matches(path, needles)
    name = lowercase(basename(path))
    any(needle -> occursin(needle, name), needles)
end

function find_matches(root, needles)
    return [path for path in recursive_files(root) if basename_matches(path, needles)]
end

function find_exactish_matches(root, names)
    lower_names = Set(lowercase.(names))
    return [
        path for path in recursive_files(root)
        if lowercase(splitext(basename(path))[1]) in lower_names ||
           lowercase(basename(path)) in lower_names
    ]
end

function find_case_dirs()
    dirs = String[]
    for path in sort(readdir(EXAMPLES_DIR; join = true))
        !isdir(path) && continue
        startswith(basename(path), "arepo_") || continue
        files = Set(readdir(path))
        has_case_surface =
            "write_arepo_cases.py" in files ||
            "profile_snapshots.py" in files ||
            "generate_tables.jl" in files ||
            "README.md" in files
        has_case_surface || continue
        push!(dirs, path)
    end
    return dirs
end

function print_list(title, items)
    println(title)
    if isempty(items)
        println("- none")
        return
    end
    for item in items
        println("- ", item)
    end
end

function main()
    case_dirs = find_case_dirs()
    top_level_arepo_examples = [
        relpath_from_powerfoam(path) for path in sort(readdir(EXAMPLES_DIR; join = true))
        if isfile(path) && startswith(basename(path), "arepo_") && endswith(path, ".jl")
    ]

    println("# PowerFoam AREPO IO/parameter audit")
    println()
    println("## Candidate case directories")
    if isempty(case_dirs)
        println("- none found")
    else
        for dir in case_dirs
            println("- ", relpath_from_powerfoam(dir))
            for name in (
                "README.md",
                "generate_tables.jl",
                "write_arepo_cases.py",
                "profile_snapshots.py",
                "csv_to_arepo_ic.c",
                "profile_arepo_snapshot.c",
                "profile_noh_snapshot.c",
                "apply_shock_following_patch.sh",
                "arepo_shock_following_mesh.patch",
            )
                path = joinpath(dir, name)
                isfile(path) || continue
                println("  - ", relpath_from_powerfoam(path))
            end
        end
    end

    println()
    print_list("## Top-level AREPO bridge/gate examples", top_level_arepo_examples)

    println()
    print_list("## Runtime/planning anchors", [
        relpath_from_powerfoam(joinpath(SRC_DIR, "arepo_runtime_scaffold.jl")),
        relpath_from_powerfoam(joinpath(PLANNING_DIR, "arepo_jl_full_rewrite_master_plan.md")),
        relpath_from_powerfoam(joinpath(POWERFOAM_ROOT, "arepo_physics_parity_plan.md")),
        relpath_from_powerfoam(joinpath(POWERFOAM_ROOT, "arepo_physics_parity_audit.md")),
    ])

    println()
    surface_specs = [
        ("parameter/config parser in src",
         find_matches(SRC_DIR, ["param", "config"])),
        ("snapshot/io module in src",
         find_exactish_matches(SRC_DIR, ["snapshot_io", "snapshot_reader",
                                         "snapshot_writer", "arepo_io", "hdf5_io"])),
        ("initial-condition module in src",
         find_exactish_matches(SRC_DIR, ["initial_conditions", "ic_io", "ic_reader",
                                         "ic_writer"])),
        ("restart compatibility module in src",
         find_matches(SRC_DIR, ["restart"])),
        ("diagnostic/output policy module in src",
         find_exactish_matches(SRC_DIR, ["diagnostics", "diagnostic_io",
                                         "output_policy", "output_writer",
                                         "reporting"])),
    ]
    println("## Missing runtime surfaces by file existence")
    for (label, raw_matches) in surface_specs
        matches = [relpath_from_powerfoam(path) for path in raw_matches]
        if isempty(matches)
            println("- missing: ", label)
        else
            println("- present: ", label)
            for match in matches
                println("  - ", match)
            end
        end
    end
end

main()
