# ── LazyLib: one lazily-dlopen'd shared library ──────────────────────────────
#
# The loading pattern every wrapper hand-rolled: a runtime-determined library
# path (env override else in-repo default) that cannot appear in a
# `ccall((:sym, "lib"), …)` literal inside a precompiled module, so the library
# is dlopen'd once (lazily) and symbols resolved through the handle.  `path`,
# `flags` and `preopen` are zero-arg closures evaluated AT OPEN TIME, because
# they may depend on runtime state (e.g. EnzoLib's MPI flavor selects both the
# library file and RTLD_GLOBAL, and must promote MPItrampoline first).

mutable struct LazyLib
    path::Function                      # () -> String (resolved at open time)
    flags::Function                     # () -> Union{Nothing,Integer} dlopen flags
    preopen::Function                   # () -> nothing; runs once, before dlopen
    hint::String                        # build instruction, appended to errors
    handle::Base.RefValue{Ptr{Cvoid}}
end

LazyLib(path::Function; flags::Function = () -> nothing,
        preopen::Function = () -> nothing, hint::AbstractString = "") =
    LazyLib(path, flags, preopen, String(hint), Ref(Ptr{Cvoid}(C_NULL)))

"""
    LazyLib(; env, default, flags=…, preopen=…, hint=…)

Env-override convenience: the library path is `ENV[env]` when set (made
absolute), else `default`.  The env var is consulted at every `libpath` call,
matching the wrappers' original behaviour.
"""
LazyLib(; env::AbstractString, default::AbstractString, kwargs...) =
    LazyLib(let e = String(env), d = String(default)
                () -> (p = get(ENV, e, ""); isempty(p) ? d : abspath(p))
            end; kwargs...)

"Absolute path the library will be (or was) loaded from."
libpath(l::LazyLib) = l.path()

"True when the shared library exists on disk (callers can skip live calls without it)."
available(l::LazyLib) = isfile(libpath(l))

"The dlopen handle, opening the library on first use."
function handle(l::LazyLib)
    if l.handle[] == C_NULL
        p = libpath(l)
        isfile(p) || error("shared library not found at $p." *
                           (isempty(l.hint) ? "" : " " * l.hint))
        l.preopen()
        f = l.flags()
        l.handle[] = f === nothing ? Libdl.dlopen(p) : Libdl.dlopen(p, f)
    end
    return l.handle[]
end

"Resolve a C symbol in the library (dlsym through the lazy handle)."
sym(l::LazyLib, name::Symbol) = Libdl.dlsym(handle(l), name)
