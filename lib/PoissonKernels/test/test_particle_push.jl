# particle_push — gravity interpolation + leapfrog drift/kick.
#   * interp_accel_to_particles! ≡ a scalar port of cic_interp.F (exact, f64);
#   * particle_kick! ≡ the semi-implicit comoving velocity update (f64 round-off);
#   * particle_drift! ≡ x += coef·v, with optional periodic wrap;
#   * Metal f32 ≡ CPU f32 for the full interp→kick→drift→kick push.
# The Fortran formulas ARE the bit-for-bit spec; the live-Enzo differential gate
# (one session_update_particles vs this push) lives in the MultiCode suite, which
# can boot a grid hierarchy.

@testset "particle_push — interp + leapfrog drift/kick" begin
    N = 16; NG = 3; M = N + 2NG; np = 4000
    cellsize = 1 / N
    le = (-NG * cellsize, -NG * cellsize, -NG * cellsize)
    dcoef = 0.013

    # deterministic, reproducible inputs (no global RNG to keep CPU/Metal in sync)
    px = [mod(sin(0.7p) * 0.5 + 0.5, 1.0) for p in 1:np]
    py = [mod(cos(1.3p) * 0.5 + 0.5, 1.0) for p in 1:np]
    pz = [mod(sin(2.1p + 1) * 0.5 + 0.5, 1.0) for p in 1:np]
    vx = [cos(0.9p) for p in 1:np]; vy = [sin(1.7p) for p in 1:np]; vz = [cos(0.3p + 2) for p in 1:np]
    gx = [sin(0.1i + 0.2j + 0.3k) for i in 1:M, j in 1:M, k in 1:M]
    gy = [cos(0.2i - 0.1j + 0.15k) for i in 1:M, j in 1:M, k in 1:M]
    gz = [sin(0.05i + 0.25j - 0.2k) for i in 1:M, j in 1:M, k in 1:M]

    # ── reference CIC interp (mirror cic_interp.F:133-179) ──
    function ref_interp(g, dcoef, cellsize, le)
        d1, d2, d3 = size(g); half = 0.5001; fact = 1 / cellsize
        e1 = d1 - half; e2 = d2 - half; e3 = d3 - half
        a = zeros(np)
        @inbounds for n in 1:np
            xq = px[n] + dcoef * vx[n]; yq = py[n] + dcoef * vy[n]; zq = pz[n] + dcoef * vz[n]
            xpos = min(max((xq - le[1]) * fact, half), e1)
            ypos = min(max((yq - le[2]) * fact, half), e2)
            zpos = min(max((zq - le[3]) * fact, half), e3)
            i1 = trunc(Int, xpos + 0.5); j1 = trunc(Int, ypos + 0.5); k1 = trunc(Int, zpos + 0.5)
            dx = i1 + 0.5 - xpos; dy = j1 + 0.5 - ypos; dz = k1 + 0.5 - zpos
            a[n] = g[i1, j1, k1] * dx * dy * dz + g[i1+1, j1, k1] * (1-dx) * dy * dz +
                   g[i1, j1+1, k1] * dx * (1-dy) * dz + g[i1+1, j1+1, k1] * (1-dx) * (1-dy) * dz +
                   g[i1, j1, k1+1] * dx * dy * (1-dz) + g[i1+1, j1, k1+1] * (1-dx) * dy * (1-dz) +
                   g[i1, j1+1, k1+1] * dx * (1-dy) * (1-dz) + g[i1+1, j1+1, k1+1] * (1-dx) * (1-dy) * (1-dz)
        end
        a
    end

    run_interp(name, ::Type{T}) where {T} = begin
        be = PoissonKernels.backend(name); d(x) = PoissonKernels.to_device(be, x, T)
        axp = PoissonKernels.device_zeros(be, T, (np,)); ayp = similar(axp); azp = similar(axp)
        PoissonKernels.interp_accel_to_particles!(axp, ayp, azp,
            d(px), d(py), d(pz), d(vx), d(vy), d(vz), d(gx), d(gy), d(gz);
            dcoef = dcoef, cellsize = cellsize, leftedge = le)
        map(PoissonKernels.to_host, (axp, ayp, azp))
    end

    # (1) CPU f64 interp ≡ scalar cic_interp.F reference (exact)
    axp, ayp, azp = run_interp(:cpu, Float64)
    @test maximum(abs.(axp .- ref_interp(gx, dcoef, cellsize, le))) < 1e-12
    @test maximum(abs.(ayp .- ref_interp(gy, dcoef, cellsize, le))) < 1e-12
    @test maximum(abs.(azp .- ref_interp(gz, dcoef, cellsize, le))) < 1e-12

    # (2) kick ≡ semi-implicit comoving update v ← ((1-coef)v + g·ts)/(1+coef)
    ts, coef = 0.02, 0.0011
    bek = PoissonKernels.backend(:cpu)
    vxk = PoissonKernels.to_device(bek, vx, Float64); vyk = PoissonKernels.to_device(bek, vy, Float64)
    vzk = PoissonKernels.to_device(bek, vz, Float64)
    adx = PoissonKernels.to_device(bek, axp, Float64); ady = PoissonKernels.to_device(bek, ayp, Float64)
    adz = PoissonKernels.to_device(bek, azp, Float64)
    PoissonKernels.particle_kick!(vxk, vyk, vzk, adx, ady, adz; ts = ts, coef = coef)
    vref = ((1 - coef) .* vx .+ axp .* ts) ./ (1 + coef)
    @test maximum(abs.(PoissonKernels.to_host(vxk) .- vref)) < 1e-14

    # (2b) f16-stored velocities still kick in f32 and only round on store.
    vxh = PoissonKernels.to_device(bek, Float16.(vx), Float16)
    vyh = PoissonKernels.to_device(bek, Float16.(vy), Float16)
    vzh = PoissonKernels.to_device(bek, Float16.(vz), Float16)
    adx32 = PoissonKernels.to_device(bek, axp, Float32)
    ady32 = PoissonKernels.to_device(bek, ayp, Float32)
    adz32 = PoissonKernels.to_device(bek, azp, Float32)
    PoissonKernels.particle_kick!(vxh, vyh, vzh, adx32, ady32, adz32; ts = ts, coef = coef)
    vhref = Float16.(((1f0 - Float32(coef)) .* Float32.(Float16.(vx)) .+ Float32.(axp) .* Float32(ts)) ./
                     (1f0 + Float32(coef)))
    @test PoissonKernels.to_host(vxh) == vhref

    # (3) drift x += coef·v (no wrap exact; wrap maps into [0,1))
    cd = 0.031
    pxd = PoissonKernels.to_device(bek, px, Float64); pyd = PoissonKernels.to_device(bek, py, Float64)
    pzd = PoissonKernels.to_device(bek, pz, Float64)
    vd = PoissonKernels.to_device(bek, vx, Float64)
    PoissonKernels.particle_drift!(pxd, pyd, pzd, vd, vd, vd; coef = cd, wrap = 0)
    @test maximum(abs.(PoissonKernels.to_host(pxd) .- (px .+ cd .* vx))) < 1e-14
    pxw = PoissonKernels.to_device(bek, px, Float64)
    PoissonKernels.particle_drift!(pxw, PoissonKernels.to_device(bek, py, Float64),
        PoissonKernels.to_device(bek, pz, Float64), vd, vd, vd; coef = 5.0, wrap = 1.0)
    h = PoissonKernels.to_host(pxw)
    @test all(0 .<= h .< 1)
    @test maximum(abs.(h .- mod.(px .+ 5.0 .* vx, 1.0))) < 1e-14

    # (3b) global nc³ potential + periodic wrap matches the padded-potential path.
    φg = [sin(0.17i + 0.11j - 0.09k) + 0.2cos(0.07i * j) for i in 1:N, j in 1:N, k in 1:N]
    φpad = zeros(Float64, M, M, M)
    @views φpad[NG+1:NG+N, NG+1:NG+N, NG+1:NG+N] .= φg
    PoissonKernels.fill_periodic_ghosts!(φpad; ng = NG)
    force_pair(name, ::Type{T}) where {T} = begin
        be = PoissonKernels.backend(name); d(x) = PoissonKernels.to_device(be, x, T)
        apx = PoissonKernels.device_zeros(be, T, (np,)); apy = similar(apx); apz = similar(apx)
        agx = similar(apx); agy = similar(apx); agz = similar(apx)
        PoissonKernels.interp_force_from_potential!(apx, apy, apz,
            d(px), d(py), d(pz), d(vx), d(vy), d(vz), d(φpad);
            dcoef = dcoef, cellsize = cellsize, leftedge = le)
        PoissonKernels.interp_force_from_global_potential!(agx, agy, agz,
            d(px), d(py), d(pz), d(vx), d(vy), d(vz), d(φg);
            dcoef = dcoef, nc = (N, N, N))
        map(PoissonKernels.to_host, (apx, apy, apz, agx, agy, agz))
    end
    fp = force_pair(:cpu, Float64)
    @test maximum(abs.(fp[1] .- fp[4])) < 1e-12
    @test maximum(abs.(fp[2] .- fp[5])) < 1e-12
    @test maximum(abs.(fp[3] .- fp[6])) < 1e-12

    # (3c) fused global-potential KDK matches the old four-kernel path and keeps
    # the same Float16 velocity rounding after the first kick.
    function run_global_kdk(name, ::Type{Tp}, ::Type{Tv}; fused::Bool) where {Tp,Tv}
        be = PoissonKernels.backend(name)
        qx = PoissonKernels.to_device(be, px, Tp); qy = PoissonKernels.to_device(be, py, Tp)
        qz = PoissonKernels.to_device(be, pz, Tp)
        wx = PoissonKernels.to_device(be, vx, Tv); wy = PoissonKernels.to_device(be, vy, Tv)
        wz = PoissonKernels.to_device(be, vz, Tv)
        ph = PoissonKernels.to_device(be, φg, Tp)
        if fused
            PoissonKernels.particle_kdk_from_global_potential!(qx, qy, qz, wx, wy, wz, ph;
                dcoef = dcoef, ts = ts, driftcoef = cd, nc = (N, N, N), wrap = 1.0)
        else
            ax = PoissonKernels.device_zeros(be, Tp, (np,)); ay = similar(ax); az = similar(ax)
            PoissonKernels.interp_force_from_global_potential!(ax, ay, az, qx, qy, qz, wx, wy, wz, ph;
                dcoef = dcoef, nc = (N, N, N))
            PoissonKernels.particle_kick!(wx, wy, wz, ax, ay, az; ts = ts, coef = 0.0)
            PoissonKernels.particle_drift!(qx, qy, qz, wx, wy, wz; coef = cd, wrap = 1.0)
            PoissonKernels.particle_kick!(wx, wy, wz, ax, ay, az; ts = ts, coef = 0.0)
        end
        map(PoissonKernels.to_host, (qx, qy, qz, wx, wy, wz))
    end
    guf = run_global_kdk(:cpu, Float32, Float32; fused=false)
    gf = run_global_kdk(:cpu, Float32, Float32; fused=true)
    @test maximum(abs.(guf[1] .- gf[1])) == 0
    @test maximum(abs.(guf[4] .- gf[4])) == 0
    guf16 = run_global_kdk(:cpu, Float32, Float16; fused=false)
    gf16 = run_global_kdk(:cpu, Float32, Float16; fused=true)
    @test guf16[1] == gf16[1]
    @test guf16[4] == gf16[4]

    # (4) Metal f32 ≡ CPU f32 for the full interp→kick→drift→kick push
    if metal_ready()
        run_push(name, ::Type{T}) where {T} = begin
            be = PoissonKernels.backend(name); d(x) = PoissonKernels.to_device(be, x, T)
            qx = d(px); qy = d(py); qz = d(pz); wx = d(vx); wy = d(vy); wz = d(vz)
            ax = PoissonKernels.device_zeros(be, T, (np,)); ay = similar(ax); az = similar(ax)
            PoissonKernels.interp_accel_to_particles!(ax, ay, az, qx, qy, qz, wx, wy, wz,
                d(gx), d(gy), d(gz); dcoef = dcoef, cellsize = cellsize, leftedge = le)
            PoissonKernels.particle_kick!(wx, wy, wz, ax, ay, az; ts = ts, coef = coef)
            PoissonKernels.particle_drift!(qx, qy, qz, wx, wy, wz; coef = cd, wrap = 1.0)
            PoissonKernels.particle_kick!(wx, wy, wz, ax, ay, az; ts = ts, coef = coef)
            map(PoissonKernels.to_host, (ax, qx, wx))
        end
        am = run_push(:metal, Float32); ac = run_push(:cpu, Float32)
        rel(a, b) = maximum(abs.(a .- b)) / (maximum(abs.(b)) + 1f-30)
        @info "particle_push Metal≡CPU f32" interp = rel(am[1], ac[1]) pos = rel(am[2], ac[2]) vel = rel(am[3], ac[3])
        @test rel(am[1], ac[1]) < 1f-4
        @test rel(am[2], ac[2]) < 1f-4
        @test rel(am[3], ac[3]) < 1f-4

        fm = force_pair(:metal, Float32); fc = force_pair(:cpu, Float32)
        @info "global potential force Metal≡CPU f32" ax = rel(fm[4], fc[4]) ay = rel(fm[5], fc[5]) az = rel(fm[6], fc[6])
        @test rel(fm[4], fc[4]) < 1f-4
        @test rel(fm[5], fc[5]) < 1f-4
        @test rel(fm[6], fc[6]) < 1f-4

        mf = run_global_kdk(:metal, Float32, Float32; fused=true)
        cu = run_global_kdk(:cpu, Float32, Float32; fused=false)
        @info "fused global KDK Metal≡CPU f32" pos = rel(mf[1], cu[1]) vel = rel(mf[4], cu[4])
        @test rel(mf[1], cu[1]) < 1f-4
        @test rel(mf[4], cu[4]) < 1f-4

        mf16 = run_global_kdk(:metal, Float32, Float16; fused=true)
        cu16 = run_global_kdk(:cpu, Float32, Float16; fused=false)
        @info "fused global KDK Metal≡CPU f16v" pos = rel(Float32.(mf16[1]), Float32.(cu16[1])) vel = rel(Float32.(mf16[4]), Float32.(cu16[4]))
        @test rel(Float32.(mf16[1]), Float32.(cu16[1])) < 1f-4
        @test rel(Float32.(mf16[4]), Float32.(cu16[4])) < 1f-4
    else
        @test_skip "Metal not available — GPU particle-push parity skipped"
    end
end
