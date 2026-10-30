# test_movie.jl — movie output (src/movie.jl): derived fields rasterized from a
# 2-level 2-D hierarchy, 16-bit codes in RGB PNGs, manifest, share bundles.
#   * PNG writer ↔ reader round-trip is exact (all four row filters exercised);
#   * every pixel's ρ is the covering cell's ρ (fine where refined, else coarse),
#     to the 16-bit quantum; the :level image is the refinement map;
#   * ω_z, ∇·v of a shear flow match the analytic values (2nd-order differences;
#     bounded seams where stencils read prolonged C/F ghosts); CPU ≡ GPU codes;
#   * movie_bundle sheets (several frames per PNG, downsampling) decode back to
#     the per-frame codes; the manifest parses back.
using Printf: @sprintf
using BlockAMR: _write_png_codes, _read_png_codes

@testset "movie: PNG code round-trip" begin
    dir = mktempdir()
    for (W, H) in ((1, 1), (37, 11), (256, 3))
        codes = rand(UInt16, W * H); codes[1:min(end, 40)] .= 0x1234   # flat run → Sub/Up filters
        p = joinpath(dir, "c.png"); _write_png_codes(p, codes, W, H)
        c2, w2, h2 = _read_png_codes(p)
        @test (w2, h2) == (W, H) && c2 == codes
    end
end

# decoded stored quantity (log10 v or v) of field f, frame k: NaN = no data
function movie_values(dir, f, k)
    m = read_movie_manifest(dir); fi = only(x for x in m["fields"] if x["key"] == String(f))
    codes, W, H = _read_png_codes(joinpath(dir, fi["dir"], @sprintf("%05d.png", k)))
    lo, hi = fi["lohi"][k + 1]
    v = [c == 0 ? NaN : lo + (c - 1) / 65534 * (hi - lo) for c in codes]
    return reshape(fi["enc"] == "log" ? exp10.(v) : v, W, H), codes, m
end

const MOVIE_CODES = Dict{Symbol,Vector{UInt16}}()
for BE in BACKENDS
@testset "movie: derived fields on a 2-level 2-D hierarchy [$BE]" begin
    for physics in (:hydro, :mhd)
        h = static2d(; nbase = (64, 32), kids = KIDS_CENTER, box = 2.0, backend = BE, physics,
                     divb = physics === :mhd ? :ct : :glm)
        setall!(h, x -> (1 + 0.5sin(π * x[1]), sin(2π * x[2]), 0.0, 0.0, 1.0))
        dir = mktempdir()
        mw = MovieWriter(dir, h; level = 1, title = "test")
        @test mw.nx == 128 && mw.ny == 64
        @test (:B2 in mw.fields) == (physics === :mhd) && !(:tcool in mw.fields)
        for t in 0:2; write_frame!(mw, h, 0.1t); end
        @test isfile(joinpath(dir, "index.html"))
        ρ, cρ, m = movie_values(dir, :rho, 1)
        @test m["nframes"] == 3 && m["nx"] == 128 && m["times"][3][1] ≈ 0.2
        c0 = cellmap2(h, 0, :D); c1 = cellmap2(h, 1, :D)
        err = maximum(abs(ρ[a + 1, b + 1] / get(c1, (a, b), c0[(a ÷ 2, b ÷ 2)]) - 1) for a in 0:127, b in 0:63)
        @test err < 2e-4
        lv, _, _ = movie_values(dir, :level, 0)
        @test all(lv[a + 1, b + 1] == (haskey(c1, (a, b)) ? 1 : 0) for a in 0:127, b in 0:63)
        # shear flow vx = sin 2πy: ω_z = −2π cos 2πy, ∇·v = 0 (pixel centre y at level 1)
        wz, _, _ = movie_values(dir, :wz, 0); dv, _, _ = movie_values(dir, :divv, 0)
        ex = [-2π * cos(2π * (b + 0.5) / 64) for a in 0:127, b in 0:63]
        # pixels whose 5-point stencil (in level-1 pixels, ±2) stays on one level vs
        # C/F seams, where the centred difference reads prolonged ghosts: there the
        # coarse curvature error / dx_fine shows (a few % of the ω scale; the solver
        # sees the same ghosts)
        seam = [any(lv[clamp(a + d, 1, 128), clamp(b + e, 1, 64)] != lv[a, b] for d in -2:2, e in -2:2)
                for a in 1:128, b in 1:64]
        @test count(seam) > 0
        @test maximum(abs.(wz .- ex)[.!seam]) < 0.06 * 2π   # coarse cells: 1 − sinc + cell offset
        @test maximum(abs.(dv)[.!seam]) < 1e-4
        @test maximum(abs.(dv)[seam]) < 0.05 * 2π
        c1st = get!(MOVIE_CODES, physics, cρ)                # CPU ≡ GPU (first backend sets it)
        @test maximum(abs.(Int.(c1st) .- Int.(cρ))) <= 1      # log10 may differ by an ulp

        # share bundle: 2 frames per sheet, then 2× downsampled
        b1 = movie_bundle(dir, joinpath(dir, "b1"); maxfiles = 2 * length(mw.fields), fields = nothing)
        @test b1.per == 2 && b1.nfiles == 2 * length(mw.fields)
        mb = read_movie_manifest(joinpath(dir, "b1"))
        @test mb["pack"]["per"] == 2 && mb["nframes"] == 3
        for k in 0:2
            s, W, H = _read_png_codes(joinpath(dir, "b1", "rho", @sprintf("%05d.png", k ÷ 2)))
            t = k % 2; cols = mb["pack"]["cols"]; tc = t % cols; tr = t ÷ cols
            _, ck, _ = movie_values(dir, :rho, k)
            tile = [s[(H - 1 - (tr * 64 + 63 - y)) * W + tc * 128 + x + 1] for x in 0:127, y in 0:63]
            @test vec(tile) == ck
        end
        b2 = movie_bundle(dir, joinpath(dir, "b2"); downsample = 2, fields = (:rho,), every = 2)
        mb2 = read_movie_manifest(joinpath(dir, "b2"))
        @test mb2["nx"] == 64 && mb2["nframes"] == 2 && length(mb2["fields"]) == 1 && b2.nfiles == 2
        s, W, H = _read_png_codes(joinpath(dir, "b2", "rho", "00000.png"))
        @test (W, H) == (64, 32)
        @test abs(Int(s[5 * 64 + 7 + 1]) - round(Int, sum(Int(cρ[(2 * 5 + j) * 128 + 2 * 7 + i + 1])
                                                         for i in 0:1, j in 0:1) / 4)) <= 1
    end
end
end # for BE
