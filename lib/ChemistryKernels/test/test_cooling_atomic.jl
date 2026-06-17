using ChemistryKernels, KernelAbstractions, Test
try; @eval using Metal; catch; end
include(joinpath(@__DIR__, "oracle.jl")); using .ChemOracle
include(joinpath(@__DIR__, "harness.jl"))

module UnitCoolAtom
  # cooling coefficients now live in EmissionKernels; re-include them here (re-runs
  # @scalarkernel against EmissionKernels' backend) for the grackle-oracle parity check.
  using EmissionKernels, KernelAbstractions
  include(joinpath(@__DIR__, "..", "..", "EmissionKernels", "src", "cooling_atomic.jl"))
end

ChemOracle.set_flags!()            # CaseB on
Ts = ChemOracle.tgrid()
@testset "cooling_atomic vs grackle" begin
  for nm in ("ceHI","ceHeI","ceHeII","ciHI","ciHeI","ciHeII","ciHeIS","reHII","reHeII1","reHeII2","reHeIII","brem")
    ref = [ChemOracle.cool(nm, t) for t in Ts]
    check_scalar_kernel(nm, getfield(UnitCoolAtom, Symbol(nm, "_grid")), ref, Ts; f32rtol = f32rtol_for(nm))
  end
end
