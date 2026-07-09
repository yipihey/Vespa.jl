# test_particles.jl — DM particle infrastructure on blocks:
#   * CIC deposit at level resolution: exact weights vs a host reference,
#     total mass conserved, sibling ghost-accumulate at block faces;
#   * accel gather from the FINEST covering level (fine ≠ coarse verified);
#   * KDK kick/drift smoke with periodic wrap.
const BA = BlockAMR

for BE in BACKENDS
@testset "particle deposit: CIC exactness + ghost accumulate [$BE]" begin
    hier = AMRHierarchy(; nbase = (32, 32, 32), B = 16, backend = BE)
    lev0 = init_base_level!(hier)
    p0 = lev0.byorigin[(UInt128(0), UInt128(0), UInt128(0))]
    c1 = add_block!(hier, 1, p0, (0, 0, 0))     # fine active [0,16)³ (level-1 cells)
    c2 = add_block!(hier, 1, p0, (8, 0, 0))     # fine active [16,32)×[0,16)²
    build_level_tables!(hier, 0); build_level_tables!(hier, 1)
    be = hier.be
    # particle 1: interior of c1; particle 2: cloud straddles the c1|c2 face
    xs = Float32[0.11, 0.24993896]              # x=0.25 face is at fine cell 16
    ys = Float32[0.13, 0.11]
    zs = Float32[0.07, 0.09]
    parts = (px = to_device(be, xs, Float32), py = to_device(be, ys, Float32),
             pz = to_device(be, zs, Float32),
             vx = to_device(be, zeros(Float32, 2), Float32),
             vy = to_device(be, zeros(Float32, 2), Float32),
             vz = to_device(be, zeros(Float32, 2), Float32))
    mass = 2.5
    deposit_particles_level!(hier, 1, parts; mass_code = mass)
    lev1 = hier.levels[2]
    hdm = Array(lev1.dm)
    dx1 = BA.level_dx(hier, 1)
    # host reference on the fine lattice (active cells of the two blocks)
    ref = Dict{NTuple{3,Int},Float64}()
    for (x, y, z) in zip(xs, ys, zs)
        g = (x, y, z) .* 64 .- 0.5
        i0 = floor.(Int, g); f = g .- i0
        for c3 in 0:1, c2_ in 0:1, c1_ in 0:1
            w = (c1_ == 0 ? 1 - f[1] : f[1]) * (c2_ == 0 ? 1 - f[2] : f[2]) *
                (c3 == 0 ? 1 - f[3] : f[3])
            key = (i0[1] + c1_, i0[2] + c2_, i0[3] + c3)
            ref[key] = get(ref, key, 0.0) + mass / dx1^3 * w
        end
    end
    tot = 0.0; worst = 0.0
    for s in lev1.live
        m = lev1.meta[s]; base = (Int(s) - 1) * lev1.stride
        for k in 1:16, j in 1:16, i in 1:16
            gcell = (Int(m.origin[1]) + i - 1, Int(m.origin[2]) + j - 1,
                     Int(m.origin[3]) + k - 1)
            v = Float64(hdm[base + ((lev1.ng+k-1)*lev1.nd + (lev1.ng+j-1))*lev1.nd + (lev1.ng+i-1) + 1])
            tot += v * dx1^3
            worst = max(worst, abs(v - get(ref, gcell, 0.0)))
        end
    end
    @test isapprox(tot, 2mass; rtol = 1e-5)              # both particles fully captured
    @test worst < 1e-3 * mass / dx1^3                    # per-cell CIC exact (f32)
end

@testset "accel gather: finest covering level wins [$BE]" begin
    hier = center_refined(; backend = BE, T = Float32)
    # manufactured φ on both levels' pools
    for l in 0:1
        lev = hier.levels[l + 1]
        hphi = zeros(Float32, lev.cap * lev.stride)
        Nl = 32 * 2^l
        for s in lev.live
            m = lev.meta[s]; base = (Int(s)-1)*lev.stride
            for k in 1:lev.nd, j in 1:lev.nd, i in 1:lev.nd
                x = ntuple(d -> (Float64(m.origin[d]) + (i,j,k)[d] - 1 - lev.ng + 0.5) / Nl, 3)
                hphi[base + ((k-1)*lev.nd + (j-1))*lev.nd + (i-1) + 1] =
                    Float32(cos(2π*x[1]) * cos(2π*x[2]) * cos(2π*x[3]))
            end
        end
        copyto!(lev.phi, hphi)
    end
    be = hier.be
    xs = Float32[0.52, 0.03]     # inside refined center; outside (level 0 only)
    ys = Float32[0.47, 0.05]
    zs = Float32[0.50, 0.06]
    parts = (px = to_device(be, xs, Float32), py = to_device(be, ys, Float32),
             pz = to_device(be, zs, Float32),
             vx = to_device(be, zeros(Float32, 2), Float32),
             vy = to_device(be, zeros(Float32, 2), Float32),
             vz = to_device(be, zeros(Float32, 2), Float32))
    ax = device_zeros(be, Float32, (2,)); ay = device_zeros(be, Float32, (2,))
    az = device_zeros(be, Float32, (2,))
    gather_accel_particles!(hier, parts, ax, ay, az)
    hax = Array(ax)
    aex(x, y, z) = 2π * sin(2π*x) * cos(2π*y) * cos(2π*z)
    for (p, tol) in ((1, 0.05), (2, 0.12))               # fine h² ≪ coarse h²
        @test isapprox(Float64(hax[p]), aex(xs[p], ys[p], zs[p]); atol = tol * 2π)
    end
    # the interior particle really used level 1: its residual vs analytic must be
    # far smaller than the coarse-grid truncation at that point
    @test abs(Float64(hax[1]) - aex(xs[1], ys[1], zs[1])) < 0.02 * 2π
    # kick/drift smoke with periodic wrap
    particles_kick!(hier, parts, ax, ay, az, 0.1)
    particles_drift!(hier, parts, 0.5)
    @test all(0 .<= Array(parts.px) .< 1)
    @test all(isfinite, Array(parts.vx))
end
end # for BE

# ── out-of-core particle streaming: tiled launches must match the whole-array
# path — bit-identical for the per-particle-independent kernels (gather/kick/
# drift), and identical to atomic round-off for the CIC deposit.  A non-divisor
# tile (7 over Np=1000) exercises the remainder tile.
for BE in BACKENDS
@testset "particle streaming tiling parity [$BE]" begin
    N = 32; Np = 1000
    function build()
        hier = AMRHierarchy(; nbase = (N, N, N), B = 16, backend = BE, T = Float16,
                            nsp = 0, gamma = 5/3, cfl = 0.3, Lcap = 1, scheme = :ctu)
        init_base_level!(hier); build_level_tables!(hier, 0)
        plev = hier.levels[1]; Bh = hier.B ÷ 2
        for ps in plev.live                                # refine one octant → 2-level gather
            m = plev.meta[ps]
            (m.origin[1] < UInt128(N÷2) && m.origin[2] < UInt128(N÷2)) || continue
            for oz in 0:1, oy in 0:1, ox in 0:1
                add_block!(hier, 1, ps, (Int16(ox*Bh), Int16(oy*Bh), Int16(oz*Bh)); sync = false)
            end
        end
        BA.sync_live!(hier.levels[2]); build_level_tables!(hier, 1)
        for lev in hier.levels                             # deterministic φ so gather ≠ 0
            h = Array(lev.phi); for i in eachindex(h); h[i] = Float32(sin(0.017*i)); end
            copyto!(lev.phi, h)
        end
        hier
    end
    dev(v) = BA.to_device(BA.backend(BE), v, Float32)
    mkparts() = ((px = dev(Float32[mod(0.5+0.3*sin(3.1*i),1f0) for i in 1:Np]),
                  py = dev(Float32[mod(0.5+0.3*cos(2.7*i),1f0) for i in 1:Np]),
                  pz = dev(Float32[mod(0.5+0.3*sin(1.3*i+1),1f0) for i in 1:Np]),
                  vx = dev(Float32[0.01*sin(i) for i in 1:Np]),
                  vy = dev(Float32[0.01*cos(i) for i in 1:Np]),
                  vz = dev(Float32[0.01*sin(2i) for i in 1:Np])),
                 dev(Float32[1.0/Np*(1+0.5*(i%3)) for i in 1:Np]))
    function runit(tile)
        set_particle_tile!(tile)
        hier = build(); parts, pm = mkparts()
        deposit_particles_level!(hier, 0, parts; mass_code = pm)
        deposit_particles_level!(hier, 1, parts; mass_code = pm)
        dm0 = copy(Array(hier.levels[1].dm)); dm1 = copy(Array(hier.levels[2].dm))
        ax = BA.device_zeros(hier.be, Float32, (Np,)); ay = similar(ax); az = similar(ax)
        gather_accel_particles!(hier, parts, ax, ay, az)
        A = (Array(ax), Array(ay), Array(az))
        particles_kick!(hier, parts, ax, ay, az, 0.123)
        particles_drift!(hier, parts, 0.077)
        (dm0, dm1, A, Array(parts.vx), Array(parts.px))
    end
    ref = runit(0); tl = runit(7)
    @test maximum(abs.(ref[1] .- tl[1])) <= 1e-5 * maximum(abs.(ref[1])) + 1f-8
    @test maximum(abs.(ref[2] .- tl[2])) <= 1e-5 * maximum(abs.(ref[2])) + 1f-8
    @test ref[3][1] == tl[3][1] && ref[3][2] == tl[3][2] && ref[3][3] == tl[3][3]
    @test ref[4] == tl[4]                                  # velocities after kick — exact
    @test ref[5] == tl[5]                                  # positions after drift — exact
    set_particle_tile!(0)
end
end # for BE
