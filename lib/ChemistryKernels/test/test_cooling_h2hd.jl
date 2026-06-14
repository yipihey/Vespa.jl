using ChemistryKernels, KernelAbstractions, Test
try; @eval using Metal; catch; end
include(joinpath(@__DIR__, "oracle.jl")); using .ChemOracle
include(joinpath(@__DIR__, "harness.jl"))

module UnitCoolMol
  using ChemistryKernels, KernelAbstractions
  using ChemistryKernels: MH, TINY, TEV_PER_K
  include(joinpath(@__DIR__, "..", "src", "cooling_h2.jl"))
  include(joinpath(@__DIR__, "..", "src", "cooling_hd.jl"))
end

ChemOracle.set_flags!()
Ts = ChemOracle.tgrid()
@testset "cooling_h2hd vs grackle" begin
  for nm in ("GAHI","GAH2","GAHe","GAHp","GAel","H2LTE","HDlte","HDlow")
    ref = [ChemOracle.cool(nm, t) for t in Ts]
    check_scalar_kernel(nm, getfield(UnitCoolMol, Symbol(nm, "_grid")), ref, Ts; f32rtol = f32rtol_for(nm))
  end
end
