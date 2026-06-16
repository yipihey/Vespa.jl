# field_ops — the baryon time-centering copy and the root-grid periodic ghost fill.
#   * copy_field! is an exact device-agnostic copy (any rank/eltype);
#   * fill_periodic_ghosts! ≡ wrapping each ghost to its active periodic image
#     (faces, edges AND corners), leaving the active interior untouched;
#   * Metal f32 ≡ CPU f32 for the ghost fill.

@testset "field_ops — baryon copy + periodic ghosts" begin
    # (1) copy_field! exact (3-D and 1-D)
    a = [sin(0.3i + 0.7j + 0.1k) for i in 1:9, j in 1:8, k in 1:7]
    b = zeros(size(a))
    PoissonKernels.copy_field!(b, a)
    @test b == a
    v = collect(1.0:50.0); w = zeros(50); PoissonKernels.copy_field!(w, v); @test w == v
    @test_throws DimensionMismatch PoissonKernels.copy_field!(zeros(3, 3), zeros(3, 4))

    # (2) periodic ghosts ≡ active-image reference (one pass fills corners too)
    N = 10; NG = 3; M = N + 2NG
    f0 = [cos(0.11i - 0.07j + 0.05k) for i in 1:M, j in 1:M, k in 1:M]
    wa(i) = i <= NG ? i + N : (i > NG + N ? i - N : i)
    ref = [f0[wa(i), wa(j), wa(k)] for i in 1:M, j in 1:M, k in 1:M]

    g = copy(f0); PoissonKernels.fill_periodic_ghosts!(g; ng = NG)
    @test maximum(abs.(g .- ref)) == 0
    R = (NG+1):(NG+N)
    @test g[R, R, R] == f0[R, R, R]                     # active interior preserved
    # opposite-face equality (a defining property of periodic ghosts)
    @test g[1:NG, R, R] == g[(N+1):(N+NG), R, R]

    @test_throws ErrorException PoissonKernels.fill_periodic_ghosts!(zeros(5, 5, 5); ng = 3)  # ng>N

    # (3) Metal f32 ≡ CPU f32 ghost fill
    if metal_ready()
        runghost(name) = begin
            be = PoissonKernels.backend(name)
            d = PoissonKernels.to_device(be, f0, Float32)
            PoissonKernels.fill_periodic_ghosts!(d; ng = NG)
            PoissonKernels.to_host(d)
        end
        gm = runghost(:metal); gc = runghost(:cpu)
        @info "fill_periodic_ghosts Metal≡CPU f32" maxabs = maximum(abs.(gm .- gc))
        @test maximum(abs.(gm .- gc)) == 0
    else
        @test_skip "Metal not available — GPU ghost-fill parity skipped"
    end
end
