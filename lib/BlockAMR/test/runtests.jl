# BlockAMR test suite.  Run directly against the test project (NOT via Pkg.test):
#   <julia> --project=lib/BlockAMR/test lib/BlockAMR/test/runtests.jl
#
# CPU always runs; the CUDA layers light up when `using CUDA` succeeds AND a
# functional device is present (BACKENDS below then includes :cuda).
using BlockAMR
using Test

try; @eval using CUDA; catch; end

const BACKENDS = has_backend(:cuda) ? (:cpu, :cuda) : (:cpu,)
@info "BlockAMR tests" backends = BACKENDS

@testset "BlockAMR" begin
    include("test_topology.jl")
    include("test_transfer.jl")
    include("test_hydro.jl")
    include("test_regrid.jl")
    include("test_subcycle.jl")
    include("test_ctu.jl")
    include("test_checkpoint.jl")
    include("test_f16.jl")
    include("test_gravity.jl")
    include("test_chem.jl")
    include("test_gravsolve.jl")
    include("test_collapse.jl")
    include("test_particles.jl")
end
