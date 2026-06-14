# Wave-0 smoke: the package loads, a trivial precision-generic @kernel runs on
# every available backend, and the grackle oracle is reachable + sane.
using Test, KernelAbstractions
const KA = KernelAbstractions

@kernel function _square!(out, @Const(x))
    i = @index(Global)
    T = eltype(out)
    @inbounds out[i] = x[i] * x[i]
end

function run_square(name, ::Type{T}) where {T}
    be = ChemistryKernels.backend(name)
    x  = ChemistryKernels.to_device(be, Float64[1, 2, 3, 4], T)
    o  = ChemistryKernels.device_zeros(be, T, (4,))
    _square!(be)(o, x; ndrange = 4)
    ChemistryKernels.to_host(o)
end

@testset "smoke: backends" begin
    @test ChemistryKernels.has_backend(:cpu)
    @test run_square(:cpu, Float64) == [1.0, 4.0, 9.0, 16.0]
    @test run_square(:cpu, Float32) == Float32[1, 4, 9, 16]
    if metal_ready()
        @test run_square(:metal, Float32) == Float32[1, 4, 9, 16]
        @info "Metal backend present"
    else
        @info "Metal not available — CPU-only run"
    end
end

@testset "smoke: oracle reachable" begin
    @test ChemOracle.available()
    ChemOracle.set_flags!()
    @test ChemOracle.rate("k1", 1e4) > 0          # k1(1e4) ≈ 7.24e-16
    @test ChemOracle.rate("k2", 1e4) > 0          # caseB recomb
    @test ChemOracle.cool("ceHI", 1e4) > 0
    @test ChemOracle.cool("GAHI", 1e3) > 0
    @test ChemOracle.rate("nonsense", 1e4) == -1.0
    @test length(ChemOracle.tgrid()) > 200
end
