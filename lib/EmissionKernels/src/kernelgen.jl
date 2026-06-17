# Codegen for the per-formula kernels.  A reaction rate / cooling coefficient is a
# pure scalar function `f(T)::T`; `@scalarkernel f` generates the device-agnostic
# launcher `f_grid(backend_name, T, Ts)` that evaluates `f` over a temperature
# array on any backend at any precision — the exact shape the verification ladder
# (`check_scalar_kernel`) expects.  Keeps each Wave-1 file to the PHYSICS (the
# `@inline f(T)=…` line) plus a one-line `@scalarkernel f`.

export @scalarkernel

"""
    @scalarkernel f

Generate, for a scalar formula `f(T)::T`, a KernelAbstractions kernel `_f_k!` and a
host launcher `f_grid(name::Symbol, ::Type{T}, Ts) -> Vector{T}` that runs `f` over
`Ts` on backend `name` in precision `T`. Mutation lives only here; `f` stays pure
(and thus AD-friendly).
"""
macro scalarkernel(f)
    kname = Symbol("_", f, "_k!")
    gname = Symbol(f, "_grid")
    esc(quote
        @kernel function $(kname)(out, @Const(xs))
            i = @index(Global)
            @inbounds out[i] = $(f)(xs[i])
        end
        function $(gname)(name::Symbol, ::Type{Tprec}, xs) where {Tprec}
            be = EmissionKernels.backend(name)
            d  = EmissionKernels.to_device(be, collect(xs), Tprec)
            o  = EmissionKernels.device_zeros(be, Tprec, (length(xs),))
            $(kname)(be)(o, d; ndrange = length(xs))
            EmissionKernels.to_host(o)
        end
    end)
end
