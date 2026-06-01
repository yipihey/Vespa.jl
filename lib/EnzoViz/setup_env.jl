#!/usr/bin/env julia
# Provision the Python environment EnzoViz drives, the painless `uv` way.
#
# EnzoViz renders through the yipihey/veusz fork (GPU Vello painter + the
# <veusz-figure> WASM web component). The fork already ships a working `.venv`
# with the compiled `_paint_ext` (Vello) extension; the default and fastest path
# is simply to **reuse that env**. This script:
#
#   1. If a usable veusz is already importable from a chosen interpreter, record
#      its path to `lib/EnzoViz/.python-path` and stop (zero work).
#   2. Otherwise, use `uv` to create `lib/EnzoViz/.venv` and
#      `uv pip install -e <veusz fork>` into it, then record that interpreter.
#
# EnzoViz reads `.python-path` at load time and points PythonCall at it
# (CondaPkg backend = Null), so no Conda is involved. Run once:
#
#   julia lib/EnzoViz/setup_env.jl [path-to-veusz-fork]
#
const HERE = @__DIR__
const DEFAULT_FORK = get(ENV, "ENZOVIZ_VEUSZ", "/Users/tabel/Projects/veusz")
fork = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_FORK

"Does `python` import veusz with a usable paint backend?"
function veusz_ok(python::AbstractString)
    isfile(python) || return false
    code = "import veusz; from veusz.paint import _paint_ext as p; " *
           "assert p.available_backends(); print('OK')"
    try
        out = read(setenv(`$python -c $code`, "QT_QPA_PLATFORM" => "offscreen"), String)
        return occursin("OK", out)
    catch
        return false
    end
end

record(python) = (write(joinpath(HERE, ".python-path"), python);
                  @info "EnzoViz Python interpreter recorded" python;
                  python)

# 1. Reuse the fork's bundled venv if it works.
fork_py = joinpath(fork, ".venv", "bin", "python")
if veusz_ok(fork_py)
    @info "Reusing the veusz fork's existing .venv (Vello already built)"
    record(fork_py); exit(0)
end

# 2. Build a dedicated env with uv.
uv = Sys.which("uv")
uv === nothing && error("`uv` not found on PATH. Install uv (https://docs.astral.sh/uv/) " *
                        "or pass an interpreter that already has veusz importable.")
venv = joinpath(HERE, ".venv")
@info "Creating EnzoViz venv with uv" venv
run(`$uv venv $venv`)
py = joinpath(venv, "bin", "python")
@info "Installing the veusz fork (editable) into the venv" fork
run(setenv(`$uv pip install --python $py -e $fork`))
if !veusz_ok(py)
    error("veusz still not importable from $py after install. " *
          "Check the fork builds its Vello/_paint_ext extension, or use tiny-skia.")
end
record(py)
@info "EnzoViz environment ready."
