using Test
using CodeBridge

# ── compile the fixture C library ─────────────────────────────────────────────
const CSRC = """
double cb_add_d(double a, double b) { return a + b; }
int    cb_add_i(int a, int b)       { return a + b; }
void   cb_scale(double* v, int n, double s) { for (int i = 0; i < n; i++) v[i] *= s; }
void   cb_sum(const double* v, int n, double* out) {
    double t = 0; for (int i = 0; i < n; i++) t += v[i]; *out = t;
}
int    cb_strlen(const char* s) { int n = 0; while (s[n]) n++; return n; }
void   cb_iota(int* v, int n)   { for (int i = 0; i < n; i++) v[i] = i + 1; }
int    cb_answer(void)          { return 42; }
"""

cc = Sys.which("cc")
if cc === nothing
    @warn "no C compiler on PATH; skipping CodeBridge live tests"
    @testset "CodeBridge (no cc)" begin
        @test_skip false
    end
else
    libdir = mktempdir()
    csrc = joinpath(libdir, "cbfix.c")
    libfile = joinpath(libdir, "libcbfix." * (Sys.isapple() ? "dylib" : "so"))
    write(csrc, CSRC)
    run(`$cc -shared -O1 -o $libfile $csrc`)
    ENV["CB_FIXTURE_LIB"] = libfile          # inherited by the worker subprocess

    include("fixture.jl")
    using .CBFixture

    @testset "CodeBridge" begin
        @testset "LazyLib + Bridge basics" begin
            B = CBFixture.BRIDGE
            @test CodeBridge.libpath(B) == libfile
            @test CodeBridge.available(B)
            @test CodeBridge.backend(B) === :local
            @test CodeBridge.flavor(B) === :main          # single-lib default
            @test_throws ErrorException CodeBridge.lib(B, :nope)
            @test_throws ErrorException CodeBridge.set_backend!(B, :bogus)
            # env override is consulted at call time
            old = ENV["CB_FIXTURE_LIB"]
            ENV["CB_FIXTURE_LIB"] = "/tmp/elsewhere.dylib"
            @test CodeBridge.libpath(B) == "/tmp/elsewhere.dylib"
            ENV["CB_FIXTURE_LIB"] = old
        end

        @testset "local @xcall" begin
            @test CBFixture.add_d(1.5, 2.25) === 3.75
            @test CBFixture.add_i(2, 3) === Int32(5)
            @test CBFixture.scale!([1.0, 2.0, 3.0], 2.0) == [2.0, 4.0, 6.0]
            @test CBFixture.sumto([1.0, 2.0, 3.5]) === 6.5
            @test CBFixture.strlen("hello") === Int32(5)
            @test CBFixture.iota!(zeros(Int32, 4)) == Int32[1, 2, 3, 4]
            @test CBFixture.answer() === Int32(42)
        end

        @testset "manifest + contract" begin
            B = CBFixture.BRIDGE
            m = CodeBridge.manifest(B)
            @test length(m) == 7
            @test haskey(m, :cb_add_d) && haskey(m, :cb_iota)
            h1 = CodeBridge.contract_hash(B)
            @test h1 isa UInt64
            @test h1 == CodeBridge.contract_hash(B)              # stable
            @test startswith(CodeBridge.contract_canonical(B), "cbfixture-contract-v1")
            # the hash is over the canonical string — a seed change changes it
            @test CodeBridge.fnv1a64("a") != CodeBridge.fnv1a64("b")
        end

        @testset "worker RPC: local ≡ remote (the parity oracle)" begin
            B = CBFixture.BRIDGE
            # reference results through the local backend
            ref_add = CBFixture.add_d(Float64(π), Float64(ℯ))
            ref_vec = CBFixture.scale!([1.1, 2.2, 3.3], Float64(π))
            ref_sum = CBFixture.sumto([0.1, 0.2, 0.3])
            ref_len = CBFixture.strlen("ünïcode✓")
            ref_iot = CBFixture.iota!(zeros(Int32, 5))

            shm = tempname()
            wcmd = `$(Base.julia_cmd()) --startup-file=no --project=$(@__DIR__) $(joinpath(@__DIR__, "worker.jl")) $shm`
            CodeBridge.connect_worker!(B, wcmd; shm = shm)
            try
                @test CodeBridge.backend(B) === :remote
                @test CBFixture.add_d(Float64(π), Float64(ℯ)) === ref_add        # bit-identical
                @test CBFixture.scale!([1.1, 2.2, 3.3], Float64(π)) == ref_vec   # buffer round-trip
                @test CBFixture.sumto([0.1, 0.2, 0.3]) === ref_sum               # Ref round-trip
                @test CBFixture.strlen("ünïcode✓") === ref_len                   # Cstring
                @test CBFixture.iota!(zeros(Int32, 5)) == ref_iot                # Int32 buffer
                @test CBFixture.answer() === Int32(42)                           # zero-arg call
            finally
                CodeBridge.disconnect_worker!(B)
            end
            @test CodeBridge.backend(B) === :local
            @test CBFixture.add_d(1.0, 1.0) === 2.0          # local path restored
        end
    end
end
