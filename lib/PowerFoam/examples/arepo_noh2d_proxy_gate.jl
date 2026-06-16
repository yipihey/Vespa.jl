using Dates
using Printf

const EXAMPLE_DIR = normpath(joinpath(@__DIR__, "arepo_noh_proxy"))
const OUTBASE = joinpath(@__DIR__, "out", "arepo_noh2d_proxy_gate")
const RUN_TAG = Dates.format(now(), "yyyymmdd_HHMMSS")
const OUTDIR = joinpath(OUTBASE, RUN_TAG)

struct CheckRow
    group::String
    item::String
    path::String
    required::Bool
    present::Bool
    nonempty::Bool
    status::String
    detail::String
end

csvquote(x) = "\"" * replace(string(x), "\"" => "\"\"") * "\""
rel(path::AbstractString) = relpath(path, @__DIR__)

function file_row(group, path; item = basename(path), required = true)
    present = isfile(path)
    nonempty = present && filesize(path) > 0
    status = required ? (nonempty ? "PASS" : "FAIL") : (nonempty ? "PASS" : "INFO")
    detail = !present ? "missing" : nonempty ? @sprintf("%d bytes", filesize(path)) : "empty file"
    return CheckRow(group, item, rel(path), required, present, nonempty, status, detail)
end

function collect_rows()
    rows = CheckRow[]

    required_sources = [
        joinpath(EXAMPLE_DIR, "README.md"),
        joinpath(EXAMPLE_DIR, "generate_tables.jl"),
        joinpath(EXAMPLE_DIR, "write_arepo_cases.py"),
        joinpath(EXAMPLE_DIR, "profile_snapshots.py"),
        joinpath(EXAMPLE_DIR, "profile_noh_snapshot.c"),
        joinpath(EXAMPLE_DIR, "results", "README.md"),
    ]
    for path in required_sources
        push!(rows, file_row("required-source", path))
    end

    generated_files = [
        joinpath(EXAMPLE_DIR, "out", "metadata.txt"),
        joinpath(EXAMPLE_DIR, "out", "standard_ic.csv"),
        joinpath(EXAMPLE_DIR, "out", "powerfoam_ic.csv"),
    ]
    for path in generated_files
        push!(rows, file_row("generated-artifact", path; required = false))
    end

    results_data_dir = joinpath(EXAMPLE_DIR, "results", "data")
    results_fig_dir = joinpath(EXAMPLE_DIR, "results", "figures")
    for dir in (results_data_dir, results_fig_dir)
        if isdir(dir)
            for name in sort(readdir(dir))
                path = joinpath(dir, name)
                isfile(path) || continue
                push!(rows, file_row("result-artifact", path; required = false))
            end
        end
    end

    return rows
end

function required_ok(rows)
    all(row -> !row.required || row.nonempty, rows)
end

function count_group(rows, group, pred)
    count(row -> row.group == group && pred(row), rows)
end

function overall_status(rows)
    source_ok = required_ok(rows)
    generated_present = count_group(rows, "generated-artifact", r -> r.nonempty)
    result_present = count_group(rows, "result-artifact", r -> r.nonempty)
    pass = source_ok && (generated_present > 0 || result_present > 0)
    return (; pass, source_ok, generated_present, result_present)
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, "group,item,path,required,present,nonempty,status,detail")
        for row in rows
            vals = (
                row.group,
                row.item,
                row.path,
                string(row.required),
                string(row.present),
                string(row.nonempty),
                row.status,
                row.detail,
            )
            println(io, join((csvquote(v) for v in vals), ","))
        end
    end
end

function write_report(path, rows, summary)
    open(path, "w") do io
        println(io, "# AREPO Noh2D Proxy Readiness Gate")
        println(io)
        println(io, "This is an executable lightweight gate for the local")
        println(io, "`lib/PowerFoam/examples/arepo_noh_proxy/` material. It does not")
        println(io, "run original AREPO or claim physics parity. It only checks that")
        println(io, "the proxy workflow files exist and records whether generated and")
        println(io, "archived result artifacts are present in the repo.")
        println(io)
        @printf(io, "- generated: %s\n", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        @printf(io, "- proxy root: `%s`\n", rel(EXAMPLE_DIR))
        @printf(io, "- overall status: `%s`\n", summary.pass ? "PASS" : "FAIL")
        @printf(io, "- required source files ready: `%s`\n", summary.source_ok ? "PASS" : "FAIL")
        @printf(io, "- generated artifacts found: `%d`\n", summary.generated_present)
        @printf(io, "- archived result artifacts found: `%d`\n", summary.result_present)
        println(io)
        println(io, "## Readiness Rule")
        println(io)
        println(io, "The gate passes when all required proxy workflow files are present")
        println(io, "and at least one generated or archived result artifact is available")
        println(io, "for inspection. Missing generated or result artifacts are reported,")
        println(io, "but they only become gate-failing if the proxy surface has no")
        println(io, "evidence beyond the source files themselves.")
        println(io)
        println(io, "## Checks")
        println(io)
        println(io, "| group | item | required | status | detail |")
        println(io, "| --- | --- | --- | --- | --- |")
        for row in rows
            required = row.required ? "yes" : "no"
            @printf(io, "| %s | `%s` | %s | %s | %s |\n",
                    row.group, row.path, required, row.status, row.detail)
        end
    end
end

function main()
    isdir(EXAMPLE_DIR) || error("proxy directory not found at $(EXAMPLE_DIR)")
    rows = collect_rows()
    summary = overall_status(rows)
    mkpath(OUTDIR)
    csv_path = joinpath(OUTDIR, "readiness.csv")
    report_path = joinpath(OUTDIR, "README.md")
    write_csv(csv_path, rows)
    write_report(report_path, rows, summary)
    @printf("overall=%s\n", summary.pass ? "PASS" : "FAIL")
    @printf("required_sources=%s\n", summary.source_ok ? "PASS" : "FAIL")
    @printf("generated_artifacts=%d\n", summary.generated_present)
    @printf("result_artifacts=%d\n", summary.result_present)
    @printf("wrote %s\n", report_path)
    @printf("wrote %s\n", csv_path)
    summary.pass || exit(1)
end

main()
