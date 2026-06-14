using ChemistryKernels, KernelAbstractions, Test
try; @eval using Metal; catch; end
include(joinpath(@__DIR__, "oracle.jl")); using .ChemOracle
include(joinpath(@__DIR__, "harness.jl"))

module UnitAtomic
  using ChemistryKernels, KernelAbstractions
  using ChemistryKernels: MH, TINY, TEV_PER_K
  include(joinpath(@__DIR__, "..", "src", "rates_atomic.jl"))
end

ChemOracle.set_flags!()             # CaseB on
Ts = ChemOracle.tgrid()

@testset "rates_atomic vs grackle" begin
  for rn in ("k1","k2","k3","k4","k5","k6","k57","k58")
    ref = [ChemOracle.rate(rn, t) for t in Ts]
    check_scalar_kernel(rn, getfield(UnitAtomic, Symbol(rn, "_grid")), ref, Ts; f32rtol = f32rtol_for(rn))
  end
end
