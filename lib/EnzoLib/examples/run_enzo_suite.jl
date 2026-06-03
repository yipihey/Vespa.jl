# Run Enzo's own test problems through the NEW Julia driver and confirm identical
# results — the generalization of E2/E4 to Enzo's whole quicksuite. For each
# `.enzotest` problem it runs Enzo's EvolveHierarchy and the Julia EvolveLevel
# (full replication, physics inferred from the .enzo) in an ISOLATED subprocess
# (so an Enzo C++ abort on one problem can't sink the run) and reports the max
# normalized field error, categorized.
#
#   ENZOMODULES_GRID_LIB=.../libenzomodules_grid.dylib \
#   julia --project=lib/EnzoLib/test lib/EnzoLib/examples/run_enzo_suite.jl [category...]
#
# e.g. `... run_enzo_suite.jl Hydro MHD`  (default: all quicksuite categories).

const HERE = @__DIR__
const TESTDIR = normpath(joinpath(HERE, "..", "test"))
const RUN = normpath(joinpath(HERE, "..", "..", "..", "..", "run"))
const WORKER = joinpath(TESTDIR, "cmp_one.jl")
const PASS_TOL = 1e-2          # normalized L∞; bit-for-bit single-grid, ~5e-5 AMR
# Per-attempt wall-clock cap (s) and how many extra attempts a CRASHED run gets.
# Enzo's serial build can intermittently abort during init under batch load (a
# fresh subprocess that succeeds standalone) — so crashes are retried before
# being believed; a genuine failure crashes every attempt. Override via env.
const SUITE_TIMEOUT = parse(Int, get(ENV, "ENZO_SUITE_TIMEOUT", "300"))
const RETRIES       = parse(Int, get(ENV, "ENZO_SUITE_RETRIES", "2"))

# Discover quicksuite problems: (.enzotest with quicksuite=True) → its .enzo file.
function quicksuite_problems(categories)
    probs = Tuple{String,String,String}[]   # (category, name, enzo_path)
    for (root, _, files) in walkdir(RUN)
        for f in files
            endswith(f, ".enzotest") || continue
            txt = read(joinpath(root, f), String)
            occursin(r"quicksuite\s*=\s*True", txt) || continue
            cat = split(relpath(root, RUN), "/")[1]
            isempty(categories) || cat in categories || continue
            enzo = joinpath(root, basename(root) * ".enzo")
            isfile(enzo) || continue
            push!(probs, (cat, basename(root), enzo))
        end
    end
    return sort(probs)
end

# One subprocess attempt → (status, note, err); status ∈ {ok, particle_only, crash}.
# A crash note distinguishes a kill at the timeout (timeout_enzo/timeout_julia)
# from a process that died on its own (enzo_abort/julia_abort, e.g. an uncatchable
# EnzoFatalException) — keyed on whether the reference printed REFOK before stopping.
function run_once(enzo_path; timeout = SUITE_TIMEOUT)
    cmd = setenv(`$(Base.julia_cmd()) --project=$TESTDIR $WORKER $enzo_path`, ENV)
    # Redirect to a TEMP FILE, not an in-memory IOBuffer: a chatty problem (an AMR
    # run prints ~160 KB) overflows the OS pipe buffer, and without a concurrent
    # drainer the subprocess stalls on write before it ever prints RESULT. A file
    # sink has no such limit, so the result line is never lost.
    logf = tempname()
    p = run(pipeline(cmd; stdout = logf, stderr = devnull); wait = false)
    t0 = time()
    while process_running(p) && time() - t0 < timeout; sleep(0.5); end
    timedout = process_running(p)
    if timedout
        kill(p)
        while process_running(p); sleep(0.1); end
    end
    lines = isfile(logf) ? split(read(logf, String), '\n') : String[]
    try; rm(logf; force = true); catch; end
    refok = any(l -> startswith(l, "REFOK|"), lines)
    timedout && return ("crash", refok ? "timeout_julia" : "timeout_enzo", NaN)
    ri = findfirst(l -> startswith(l, "RESULT|"), lines)
    ri === nothing && return ("crash", refok ? "julia_abort" : "enzo_abort", NaN)
    parts = split(lines[ri], '|')
    si = findfirst(p -> startswith(p, "status="), parts)
    status = si === nothing ? "?" : replace(parts[si], "status=" => "")
    ei = findfirst(p -> startswith(p, "err="), parts)
    err = ei === nothing ? NaN : parse(Float64, replace(parts[ei], "err=" => ""))
    ni = findfirst(p -> startswith(p, "nfields="), parts)
    nf = ni === nothing ? -1 : parse(Int, replace(parts[ni], "nfields=" => ""))
    pei = findfirst(p -> startswith(p, "perr="), parts)
    perr = pei === nothing ? NaN : parse(Float64, replace(parts[pei], "perr=" => ""))
    npi = findfirst(p -> startswith(p, "nparticles="), parts)
    npart = npi === nothing ? 0 : parse(Int, replace(parts[npi], "nparticles=" => ""))
    # No baryon fields → it's a particle problem; judge by particle-position error.
    if nf == 0
        npart == 0 && return ("particle_only", "no fields/particles", NaN)
        return (status, "particles=$npart", perr)   # GravityTest / TestOrbit
    end
    return (status, "", err)
end

# Retry a CRASHED attempt (transient Enzo init abort / contention) before believing
# it; a definitive result (ok / particle_only) is returned immediately. `note`
# records the retries so a flaky-but-eventually-ok run is visible.
function run_one(enzo_path)
    r = run_once(enzo_path)
    retried = 0
    while r[1] == "crash" && retried < RETRIES
        retried += 1
        r = run_once(enzo_path)
    end
    return retried > 0 && r[1] != "crash" ? (r[1], "$(r[2]) (retry $retried)", r[3]) : r
end

categories = ARGS
probs = quicksuite_problems(categories)
println("Enzo quicksuite via the Julia driver — ", length(probs), " problems",
        isempty(categories) ? "" : " (categories: $(join(categories, ", ")))")
println(rpad("category", 14), rpad("problem", 36), rpad("outcome", 12), "max field error")
tally = Dict{String,Int}()
for (cat, nm, pf) in probs
    status, note, err = run_one(pf)
    outcome = status == "ok" ? (isfinite(err) && err < PASS_TOL ? "PASS" : "MISMATCH") :
              status == "crash" ? uppercase(note) : uppercase(status)
    tally[outcome] = get(tally, outcome, 0) + 1
    println(rpad(cat, 14), rpad(nm, 36), rpad(outcome, 12),
            status == "ok" ? string(round(err, sigdigits = 4)) : note)
    flush(stdout)
end
println("\nsummary: ", join(["$k=$v" for (k, v) in sort(collect(tally))], "  "))
