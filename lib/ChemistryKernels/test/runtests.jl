# ChemistryKernels test suite. Run directly against the test project (NOT via
# Pkg.test, whose sandbox does not inherit [sources]):
#   <julia> --project=lib/ChemistryKernels/test lib/ChemistryKernels/test/runtests.jl
using ChemistryKernels
using Test

# Metal lights up the :metal backend where available (Apple Silicon); harmless
# elsewhere — the B/C layers then skip cleanly.
try
    @eval using Metal
catch
end

# CUDA lights up the :cuda backend where available (NVIDIA); harmless elsewhere.
try
    @eval using CUDA
catch
end

include("oracle.jl");  using .ChemOracle
include("harness.jl")

@testset "ChemistryKernels" begin
    include("test_smoke.jl")
    include("test_rates_atomic.jl")
    include("test_rates_h2.jl")
    include("test_rates_deuterium.jl")
    include("test_rates_cmb.jl")
    include("test_cooling_atomic.jl")
    include("test_cooling_h2hd.jl")
    include("test_temperature.jl")
    include("test_equilibrium.jl")
    include("test_network_step.jl")
    include("test_edot.jl")
    include("test_driver.jl")
    include("test_onezone.jl")
    include("test_recombination_mixing.jl")
    include("test_recombination_field.jl")
end
