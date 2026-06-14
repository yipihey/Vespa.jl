using Test
using PPMKernels
using KernelAbstractions

# Bring up the Metal backend if this machine has one (Apple Silicon). The
# extension self-skips when no functional device is present, so this is safe on
# any host — `metal_ready()` then reports false and the f32 GPU layers skip.
try
    @eval using Metal
catch err
    @info "Metal not loadable; GPU parity layers will skip" err
end

include("harness.jl")

@testset "PPMKernels" begin
    @testset "Phase 0 — scaffolding smoke" begin
        @test PPMKernels.has_backend(:cpu)
        @test PPMKernels.backend(:cpu) isa KernelAbstractions.CPU
        @info "Metal backend present?" metal = metal_ready()

        # A trivial KA kernel exercised on every available device proves the
        # one-source launch path + the device-array helpers + the f32 CPU↔Metal
        # parity machinery end-to-end, before any physics lands.
        @kernel function _scale!(y, @Const(x), a)
            i = @index(Global, Linear)
            @inbounds y[i] = a * x[i] + one(eltype(y))
        end

        run_scale(name, ::Type{T}) where {T} = begin
            be = PPMKernels.backend(name)
            x = PPMKernels.to_device(be, T.(1:16), T)
            y = PPMKernels.device_zeros(be, T, size(x))
            _scale!(be)(y, x, T(2); ndrange = length(x))
            KernelAbstractions.synchronize(be)
            PPMKernels.to_host(y)
        end

        # Layer A analogue: CPU f64 matches the closed-form answer exactly.
        want = 2.0 .* (1.0:16.0) .+ 1.0
        @test run_scale(:cpu, Float64) == want
        # Layer B analogue: CPU f32 ≡ Metal f32 (skips if no GPU).
        layerB!("smoke_scale", run_scale)
    end

    # Component test files are included here as each is ported + certified:
    include("test_eos.jl")              # Phase 2.1 ✓
    include("test_calcdiss.jl")         # Phase 2.2 ✓
    include("test_inteuler.jl")         # Phase 2.3 ✓
    include("test_twoshock.jl")         # Phase 2.4 ✓
    include("test_flux_twoshock.jl")    # Phase 2.5 ✓
    include("test_euler.jl")            # Phase 2.6 ✓
    include("test_sweep.jl")            # Phase 3 ✓
    include("test_ppm_grid.jl")         # Phase 4 ✓
    include("test_muscl.jl")            # MUSCL PLM+HLL vs live Enzo hydro_rk ✓
    include("test_muscl_grid.jl")       # 3-D unsplit RK2 MUSCL driver (HydroMethod=3)
    include("test_ppml.jl")             # PPML stateful char-traced solver (Ustyugov+ 2009)
    include("test_colour.jl")           # passive species (colour) advection on the PPM mass flux
    # include("test_sweep.jl")          # Phase 3
end
