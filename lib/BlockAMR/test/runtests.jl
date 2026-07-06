# BlockAMR test suite.  Run directly against the test project (NOT via Pkg.test):
#   <julia> --project=lib/BlockAMR/test lib/BlockAMR/test/runtests.jl
#
# CPU always runs; a GPU layer lights up when its package loads AND a functional
# device is present — `using CUDA` adds :cuda (NVIDIA), `using Metal` adds :metal
# (Apple).  BACKENDS below picks up whichever registered.
using BlockAMR
using Test

try; @eval using CUDA; catch; end
try; @eval using Metal; catch; end

const BACKENDS = (:cpu,
                  (has_backend(:cuda) ? (:cuda,) : ())...,
                  (has_backend(:metal) ? (:metal,) : ())...)
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
