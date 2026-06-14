using ChemistryKernels, KernelAbstractions, Test
try; @eval using Metal; catch; end
include(joinpath(@__DIR__, "oracle.jl")); using .ChemOracle
include(joinpath(@__DIR__, "harness.jl"))
module UnitDeut
  using ChemistryKernels, KernelAbstractions
  using ChemistryKernels: MH, TINY, TEV_PER_K
  include(joinpath(@__DIR__, "..", "src", "rates_deuterium.jl"))
end
ChemOracle.set_flags!(); Ts = ChemOracle.tgrid()

@testset "rates_deuterium vs grackle" begin
  for rn in ("k50","k51","k52","k53","k54","k55","k56")
    ref = [ChemOracle.rate(rn, t) for t in Ts]
    check_scalar_kernel(rn, getfield(UnitDeut, Symbol(rn, "_grid")), ref, Ts)
  end
end
