using ChemistryKernels, KernelAbstractions, Test
try; @eval using Metal; catch; end
include(joinpath(@__DIR__, "oracle.jl")); using .ChemOracle
include(joinpath(@__DIR__, "harness.jl"))
module UnitH2
  using ChemistryKernels, KernelAbstractions
  using ChemistryKernels: MH, TINY, TEV_PER_K
  include(joinpath(@__DIR__, "..", "src", "rates_h2.jl"))
end
ChemOracle.set_flags!(); Ts = ChemOracle.tgrid()

@testset "rates_h2 vs grackle" begin
  for rn in ("k7","k8","k9","k10","k11","k12","k13","k14","k15","k16","k17","k18","k19","k22")
    ref = [ChemOracle.rate(rn, t) for t in Ts]
    check_scalar_kernel(rn, getfield(UnitH2, Symbol(rn, "_grid")), ref, Ts; f32rtol = f32rtol_for(rn))
  end
end
