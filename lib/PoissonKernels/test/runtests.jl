using Test
using PoissonKernels
using KernelAbstractions

# Bring up the Metal backend if this machine has one (Apple Silicon). The
# extension self-skips when no functional device is present, so this is safe on
# any host — `metal_ready()` then reports false and the f32 GPU layers skip.
try
    @eval using Metal
catch err
    @info "Metal not loadable; GPU parity layers will skip" err
end

using EnzoLib

include("harness.jl")

@testset "PoissonKernels" begin
    @testset "scaffolding smoke" begin
        @test PoissonKernels.has_backend(:cpu)
        @test PoissonKernels.backend(:cpu) isa KernelAbstractions.CPU
        @info "Metal backend present?" metal = metal_ready()
        @info "Enzo grid dylib available?" grid = EnzoLib.grid_available()

        # A trivial KA kernel on every device proves the one-source launch path +
        # the device-array helpers + the f32 CPU↔Metal parity machinery end-to-end.
        @kernel function _scale!(y, @Const(x), a)
            i = @index(Global, Linear)
            @inbounds y[i] = a * x[i] + one(eltype(y))
        end
        run_scale(name, ::Type{T}) where {T} = begin
            be = PoissonKernels.backend(name)
            x = PoissonKernels.to_device(be, T.(1:16), T)
            y = PoissonKernels.device_zeros(be, T, size(x))
            _scale!(be)(y, x, T(2); ndrange = length(x))
            KernelAbstractions.synchronize(be)
            PoissonKernels.to_host(y)
        end
        @test run_scale(:cpu, Float64) == (2.0 .* (1.0:16.0) .+ 1.0)
        layerB!("smoke_scale", run_scale)

        # Pure-Julia check of the V-cycle dim schedule (no dylib needed).
        @test PoissonKernels.mg_dims_schedule((33, 33, 33)) ==
              [(33, 33, 33), (17, 17, 17), (9, 9, 9), (5, 5, 5), (3, 3, 3)]
    end

    include("test_relax.jl")
    include("test_defect.jl")
    include("test_restrict.jl")
    include("test_prolong.jl")
    include("test_comp_accel.jl")
    include("test_vcycle.jl")
    include("test_fft_poisson.jl")
    include("test_deposit.jl")
    include("test_particle_push.jl")
    include("test_field_ops.jl")
end
