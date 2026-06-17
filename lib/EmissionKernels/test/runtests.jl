# EmissionKernels test suite. Run standalone (no macOS C-grackle oracle needed):
#   <julia> --project=lib/EmissionKernels/test lib/EmissionKernels/test/runtests.jl
using EmissionKernels
using Test

try
    @eval using Metal
catch
end

@testset "EmissionKernels" begin
    include("test_emission.jl")
end
