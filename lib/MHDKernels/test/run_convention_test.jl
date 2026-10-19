# The PMF driver as a conforming run (koine docs/plan-run-convention.md, rollout step 4).
#
#   julia --project=lib/MHDKernels/test lib/MHDKernels/test/run_convention_test.jl
#
# (a) `--schema` names every MHD_* environment variable the driver reads;
# (b) a tiny CPU run writes a run directory that `run-check` accepts, a resume from its
#     checkpoint records lineage, and a `never` parameter cannot change on resume;
# (c) the old `env MHD_…=…` invocation and the new `--group.name=value` one produce
#     byte-identical history tables.
# Every driver run is a foreground subprocess with a timeout (CLAUDE.md, background-run
# discipline); set RUN_CHECK to the checker binary when it is not at the default path.

using Test
using RunConvention
using RunConvention: parse_json

const DRIVER = joinpath(@__DIR__, "pmf_dm_lattice_bench.jl")
const PROJECT = @__DIR__
const RUN_CHECK = get(ENV, "RUN_CHECK", joinpath(homedir(), "Projects", "koine", "target", "debug", "run-check"))
const TIMEOUT_S = parse(Int, get(ENV, "RUN_CONVENTION_TEST_TIMEOUT", "900"))

# get(ENV, "MHD_…") sites in the driver before it adopted the convention: 231 literal
# sites (MHD_MEMPROBE was read twice) plus MHD_CHEM_FALPHA_ZS and MHD_CHEM_FALPHA_VALS,
# which went through _parse_optional_list; 232 distinct names. MHD_RUN_DIR is new.
const REPLACED_ENV_NAMES = 232
const NEW_ENV_NAMES = ["MHD_RUN_DIR"]

# The driver's own environment is scrubbed of MHD_* so a developer's shell cannot leak in.
function clean_env(extra::Dict{String,String})
    env = Dict{String,String}(k => v for (k, v) in ENV if !startswith(k, "MHD_") && k != "VESPA_RUN_DIR")
    merge!(env, extra)
    env
end

"Run the driver to completion in the foreground; returns (exit code, log text)."
function run_driver(args::Vector{String}, env::Dict{String,String}, log::String)
    cmd = setenv(`$(Base.julia_cmd()) --project=$PROJECT $DRIVER $(args)`, clean_env(env))
    proc = run(pipeline(cmd; stdout=log, stderr=log); wait=false)
    t0 = time()
    while process_running(proc)
        time() - t0 > TIMEOUT_S && (kill(proc); error("driver exceeded $(TIMEOUT_S)s: $log"))
        sleep(0.5)
    end
    return proc.exitcode, read(log, String)
end

function run_check(dir::String)
    io = IOBuffer()
    proc = run(pipeline(ignorestatus(`$RUN_CHECK $dir`); stdout=io, stderr=io))
    return proc.exitcode, String(take!(io))
end

read_json(path) = parse_json(read(path, String))
read_events(dir) = [parse_json(l) for l in eachline(joinpath(dir, "events.jsonl")) if !isempty(strip(l))]

const TINY = Dict("MHD_BACKEND" => "cpu", "MHD_PMF_N" => "16", "MHD_STEPS" => "4",
                  "MHD_PK_EVERY" => "1", "MHD_CYCLE_PRINT_EVERY" => "1", "MHD_CHECK_EVERY" => "1",
                  "MHD_PRINT_EVERY" => "1")

@testset "run convention" begin
    @testset "schema names every MHD_* variable" begin
        d = mktempdir()
        code, out = run_driver(["--schema"], Dict("MHD_BACKEND" => "cpu"), joinpath(d, "schema.log"))
        @test code == 0
        start = findfirst('{', out)
        @test start !== nothing
        schema = parse_json(out[start:end])
        props = schema["properties"]
        envs = String[p["x-run"]["env"] for (_, p) in props if haskey(p["x-run"], "env")]
        @test length(envs) == length(props)                    # every parameter names its variable
        @test length(unique(envs)) == length(envs)
        @test all(startswith("MHD_"), envs)
        @test length(envs) == REPLACED_ENV_NAMES + length(NEW_ENV_NAMES)
        @test all(in(envs), NEW_ENV_NAMES)
        @test all(occursin('.', k) for k in keys(props))       # component.name, never a flat global
        @test Set(p["x-run"]["mutable"] for (_, p) in props) == Set(["never", "restart", "always"])
        @test props["io.print_every"]["x-run"]["mutable"] == "always"
        @test props["pmf.seed"]["x-run"]["mutable"] == "never"
        @test props["mhd.cfl"]["x-run"]["mutable"] == "restart"
        @test props["pmf.b0_ng"]["x-run"]["unit"] == "nG"
        metrics = [m["name"] for m in schema["x-run"]["metrics"]]
        @test "brms" in metrics && "xHII_mean" in metrics
        # No raw environment read survives in the driver.
        @test !occursin("get(ENV, \"MHD_", read(DRIVER, String))
        @test !occursin("get(ENV,\"MHD_", read(DRIVER, String))
    end

    @testset "tiny CPU run, checker, resume with lineage" begin
        if !isfile(RUN_CHECK)
            @warn "run-check not found; set RUN_CHECK" RUN_CHECK
            @test_skip isfile(RUN_CHECK)
        else
            d = mktempdir()
            ckpt = joinpath(d, "ckpt.jls")
            env = merge(TINY, Dict("MHD_CHECKPOINT_EVERY" => "2", "MHD_CHECKPOINT_PATH" => ckpt,
                                   "MHD_HISTORY_TSV" => joinpath(d, "history.tsv"), "MHD_RUN_DIR" => d))
            code, log = run_driver(String[], env, joinpath(d, "run.log"))
            @test code == 0
            @test isfile(ckpt)
            for f in ("run.json", "params.json", "events.jsonl", "outputs.json")
                @test isfile(joinpath(d, f))
            end
            @test isdir(joinpath(d, "control"))
            ccode, cout = run_check(d)
            @test ccode == 0
            ccode == 0 || println(cout)
            run = read_json(joinpath(d, "run.json"))
            @test run["convention"] == "run@1"
            @test run["status"]["state"] == "done" && run["status"]["exit"]["code"] == 0
            @test run["problem"]["name"] == "pmf-dm-lattice"
            params = read_json(joinpath(d, "params.json"))
            @test params["pmf.n"]["value"] == 16 && params["pmf.n"]["source"] == "env:MHD_PMF_N"
            @test params["cosmo.h"]["source"] == "default"
            @test params["terminal.vcap"]["value"] == "Infinity"     # non-finite values are spelled
            events = read_events(d)
            kinds = [e["kind"] for e in events]
            @test kinds[1] == "started" && kinds[end] == "finished"
            @test count(==("progress"), kinds) >= 4
            @test count(==("metric"), kinds) >= 4
            @test any(e -> e["kind"] == "checkpoint" && e["path"] == "ckpt.jls", events)
            @test any(e -> e["kind"] == "output" && e["path"] == "history.tsv", events)
            @test [e["seq"] for e in events] == 1:length(events)
            outputs = read_json(joinpath(d, "outputs.json"))["outputs"]
            @test any(o -> o["kind"] == "checkpoint" && o["path"] == "ckpt.jls", outputs)

            # Resume for two more cycles: a restart-class change is applied and recorded.
            env2 = merge(env, Dict("MHD_STEPS" => "6"))
            code2, log2 = run_driver(String[], env2, joinpath(d, "resume.log"))
            @test code2 == 0
            @test occursin("restore_checkpoint", log2)
            children = filter(x -> startswith(x, "run-") && isdir(joinpath(d, x)), readdir(d))
            @test length(children) == 1
            child = joinpath(d, children[1])
            ccode2, cout2 = run_check(child)
            @test ccode2 == 0
            ccode2 == 0 || println(cout2)
            crun = read_json(joinpath(child, "run.json"))
            @test crun["lineage"]["parent"]["run"] == run["id"]
            @test crun["lineage"]["parent"]["relation"] == "resume"
            @test crun["lineage"]["parent"]["checkpoint"]["path"] == "ckpt.jls"
            @test crun["problem"] == run["problem"]
            cevents = read_events(child)
            @test any(e -> e["kind"] == "changed" && e["param"] == "run.steps" && e["by"] == "restart" &&
                           e["from"] == 4 && e["to"] == 6, cevents)
            @test any(e -> e["kind"] == "checkpoint", cevents)
            @test cevents[end]["kind"] == "finished" && cevents[end]["exit"]["code"] == 0
            @test read_json(joinpath(child, "params.json"))["run.steps"]["source"] == "env:MHD_STEPS"

            # Non-finite values survive the resume: the fresh run wrote the spelled strings,
            # the library read them back as Inf / NaN (the driver asserts they equal its own
            # values), the child records the same spelling, and run-check hashed it (exit 0 above).
            cparams = read_json(joinpath(child, "params.json"))
            @test cparams["terminal.vcap"]["value"] == "Infinity"
            @test cparams["drag.rate_code"]["value"] == "NaN"
            @test RunConvention.coerce_value(Float64, cparams["terminal.vcap"]["value"], "terminal.vcap") == Inf
            @test run["params"]["key"] != crun["params"]["key"]     # steps changed, so the key did

            # Resume a resume: the child rewrote ckpt.jls in place, so the file now belongs to
            # the child, not to the run above it. Run directories are named to the second.
            sleep(1.2)
            env4 = merge(env, Dict("MHD_STEPS" => "8"))
            code4, log4 = run_driver(String[], env4, joinpath(d, "grandchild.log"))
            @test code4 == 0
            grandchildren = filter(x -> startswith(x, "run-") && isdir(joinpath(d, x)) && x != children[1], readdir(d))
            @test length(grandchildren) == 1
            grand = joinpath(d, grandchildren[1])
            gcode, gout = run_check(grand)
            @test gcode == 0
            gcode == 0 || println(gout)
            grun = read_json(joinpath(grand, "run.json"))
            @test grun["lineage"]["parent"]["run"] == crun["id"]
            @test grun["lineage"]["parent"]["run"] != run["id"]
            @test grun["lineage"]["parent"]["checkpoint"]["path"] == "../ckpt.jls"
            @test any(e -> e["kind"] == "checkpoint" && e["sha256"] == grun["lineage"]["parent"]["checkpoint"]["sha256"], cevents)
            @test any(e -> e["kind"] == "changed" && e["param"] == "run.steps" && e["from"] == 6 && e["to"] == 8 &&
                           e["by"] == "restart", read_events(grand))
            @test read_json(joinpath(grand, "params.json"))["terminal.vcap"]["value"] == "Infinity"

            # A never-class parameter cannot change on resume: the library refuses before
            # the checkpoint fingerprint even gets to look.
            env3 = merge(env, Dict("MHD_STEPS" => "6", "MHD_PMF_SEED" => "43"))
            code3, log3 = run_driver(String[], env3, joinpath(d, "refused.log"))
            @test code3 != 0
            @test occursin("pmf.seed", log3) && occursin("never", log3)
        end
    end

    @testset "old env invocation == new --name=value invocation" begin
        a = mktempdir(); b = mktempdir()
        enva = merge(TINY, Dict("MHD_HISTORY_TSV" => joinpath(a, "history.tsv"), "MHD_PMF_SEED" => "7",
                                "MHD_CFL" => "0.3"))
        codea, loga = run_driver(String[], enva, joinpath(a, "run.log"))
        @test codea == 0
        @test isfile(joinpath(a, "run.json"))        # the run directory is the table's directory
        argsb = ["--run.backend=cpu", "--pmf.n=16", "--run.steps=4", "--diag.pk_every=1",
                 "--io.cycle_print_every=1", "--io.check_every=1", "--io.print_every=1",
                 "--io.history_tsv=" * joinpath(b, "history.tsv"), "--pmf.seed=7", "--mhd.cfl=0.3"]
        codeb, logb = run_driver(argsb, Dict{String,String}(), joinpath(b, "run.log"))
        @test codeb == 0
        ha = read(joinpath(a, "history.tsv"))
        hb = read(joinpath(b, "history.tsv"))
        @test length(ha) > 200
        @test ha == hb
        pa = read_json(joinpath(a, "params.json")); pb = read_json(joinpath(b, "params.json"))
        @test pa["pmf.seed"]["source"] == "env:MHD_PMF_SEED" && pb["pmf.seed"]["source"] == "cli"
        # Every value agrees except the table path, which names each run's own directory.
        @test Dict(k => v["value"] for (k, v) in pa if k != "io.history_tsv") ==
              Dict(k => v["value"] for (k, v) in pb if k != "io.history_tsv")
        # An unknown name is refused (R-D2.5).
        c = mktempdir()
        codec, logc = run_driver(["--pmf.sead=7"], Dict("MHD_BACKEND" => "cpu", "MHD_PMF_N" => "16"),
                                 joinpath(c, "run.log"))
        @test codec != 0
        @test occursin("unknown parameter", logc)
    end
end
