# Reproducibly build the EnzoModules pilot shared library that EnzoLib binds.
# Run:  julia --project=lib/EnzoLib/test lib/EnzoLib/deps/build.jl
# (or `Pkg.build("EnzoLib")`). Pure pilot scope — needs only gfortran + g++,
# no HDF5/MPI/full-Enzo build.
#
# Why this wrapper: on macOS the system linker (Apple clang) doesn't know
# Homebrew gfortran's library directory, so `-lgfortran` fails to resolve at the
# final link of build_pilot.sh. We locate it via `gfortran -print-file-name` and
# put it on LIBRARY_PATH (honored by clang for -l search). Harmless on Linux.

# EnzoModules lives in the sibling enzo-dev repo (Vespa.jl was extracted from it).
# Default to the sibling layout ~/Projects/{Vespa.jl,enzo-dev}; override with ENV["ENZO_DEV_REPO"].
const REPO = get(ENV, "ENZO_DEV_REPO",
                 normpath(joinpath(@__DIR__, "..", "..", "..", "..", "enzo-dev")))
const SCRIPT = joinpath(REPO, "EnzoModules", "deps", "build_pilot.sh")
const SO = joinpath(REPO, "EnzoModules", "deps", "libenzomodules_pilot.so")

isfile(SCRIPT) || error("build_pilot.sh not found at $SCRIPT (set ENV[\"ENZO_DEV_REPO\"] to the enzo-dev checkout)")

fc = get(ENV, "FC", "gfortran")
gflib = try
    dirname(readchomp(`$fc -print-file-name=libgfortran.dylib`))
catch
    ""
end
env = copy(ENV)
if !isempty(gflib) && isdir(gflib)
    env["LIBRARY_PATH"] = string(gflib, Sys.iswindows() ? ';' : ':', get(env, "LIBRARY_PATH", ""))
end

@info "building EnzoModules pilot library" script=SCRIPT gfortran_lib=gflib
run(setenv(`bash $SCRIPT`, env))
isfile(SO) || error("build reported success but $SO is missing")
@info "EnzoModules pilot library ready" lib=SO
