# PPML (Piecewise-Parabolic Method on a Local stencil, Ustyugov+ 2009) — the
# stateful characteristic-traced solver. Tests: (1) the ported per-cell primitives
# (mirroring the Rust unit tests), (2) rotation invariance of the 3-D dim-split step
# (the same 1-D problem along x/y/z gives bit-identical profiles ⇒ the transpose +
# momentum-rotation machinery is correct), (3) conservation to round-off, (4)
# with_pool ≡ no-pool, (5) metal-f32 ≡ cpu-f32 parity, (6) fluxrec reproduces the
# conservative update (the AMR-reflux enabler).

const _P = PPMKernels

@testset "PPML solver (Ustyugov+ 2009 — stateful char-traced)" begin
    g = 1.4

    # ── A. ported per-cell primitives (mirror morton_code ppml.rs #[cfg(test)]) ──
    @testset "primitives (trace / RGK / char projections)" begin
        w = (1.0, 0.0, 0.0, 0.0, 1.0)                # constant ⇒ constant traced face
        fr = _P._ppml_face_right(w..., w..., w..., 0.1, g)
        fl = _P._ppml_face_left(w..., w..., w..., 0.1, g)
        @test maximum(abs.(fr .- w)) < 1e-14
        @test maximum(abs.(fl .- w)) < 1e-14
        # dt=0 ⇒ geometric endpoint (face_right→wR, face_left→wL)
        wl = (0.9, 0.1, 0.05, 0.0, 0.95); wa = (1.0, 0.2, 0.0, 0.0, 1.0); wr = (1.1, 0.3, -0.05, 0.0, 1.05)
        @test maximum(abs.(_P._ppml_face_right(wl..., wa..., wr..., 0.0, g) .- wr)) < 1e-13
        @test maximum(abs.(_P._ppml_face_left(wl..., wa..., wr..., 0.0, g) .- wl)) < 1e-13
        # characteristic projection is an isomorphism
        ρ, p = 1.2, 0.8; cs = sqrt(g * p / ρ); δ = (0.05, -0.02, 0.03, -0.04, 0.1)
        rt = _P._ppml_char2prim(_P._ppml_prim2char(δ..., ρ, cs)..., ρ, cs)
        @test maximum(abs.(rt .- δ)) < 1e-13
        # RGK on a constant triple ⇒ face pair = cell average
        rg = _P._ppml_rgk(w..., w..., w..., w..., w..., g)
        @test maximum(abs.(rg .- (w..., w...))) < 1e-14
        # RGK clamps a gross overshoot toward the (zero) neighbour gradient
        ov = _P._ppml_rgk(w..., w..., w..., (2.0, 0.0, 0.0, 0.0, 1.0)..., w..., g)
        @test abs(ov[1] - 1.0) < 1e-12               # overshooting wL density clamps to avg
        # WENO5 reconstructs a LINEAR field exactly at the two interfaces
        b = 0.3; lin(i) = 1.0 + b * i
        (qL, qR) = _P._ppml_weno5(lin(-2), lin(-1), lin(0), lin(1), lin(2))
        @test abs(qL - (1.0 - 0.5b)) < 1e-12 && abs(qR - (1.0 + 0.5b)) < 1e-12
        # smooth-extremum fallback FIRES at a smooth peak (limiter had clamped to (1,1))…
        (mL, mR) = _P._ppml_extremum_fix(1.0, 1.0, 0.0, 0.8, 1.0, 0.8, 0.0)
        @test !(mL == 1.0 && mR == 1.0) && mL > 1.0 - 1e-9 - 0.5 && mR < 1.0 + 1e-9
        # …and leaves a monotone ramp untouched (no extremum ⇒ keep the limited values)
        (rL, rR) = _P._ppml_extremum_fix(0.95, 1.05, 0.8, 0.9, 1.0, 1.1, 1.2)
        @test rL == 0.95 && rR == 1.05
    end

    # ── B. rotation invariance: same 1-D problem on x/y/z ⇒ identical profiles ──
    # A state varying only along axis `a` (uniform transverse) evolves only through
    # the `a`-sweep (transverse sweeps are no-ops); embedding the SAME 1-D problem
    # along each axis must give bit-identical 1-D profiles — certifying the per-axis
    # transpose + momentum-role rotation.
    @testset "rotation invariance (x ≡ y ≡ z)" begin
        ng = 3; na = 24; nb = na + 2ng; dims = (nb, nb, nb); N = nb^3
        dx = 1.0 / na; dt = 0.012; nsteps = 3
        # 1-D periodic profile (active cells 1..na): ρ, normal velocity, pressure
        prof(s) = (1.0 + 0.3sinpi(2s), 0.25sinpi(2s + 0.4), 0.08cospi(2s), -0.05,
                   0.7 + 0.2cospi(2s))               # (ρ, vn, vt1, vt2, p)
        nx, ny, nz = dims
        idx(i, j, k) = i + nx * (j - 1) + nx * ny * (k - 1)
        line(a) = begin
            D = zeros(N); S = (zeros(N), zeros(N), zeros(N)); Tau = zeros(N)
            for k in 1:nz, j in 1:ny, i in 1:nx
                t = (a == 1 ? i : a == 2 ? j : k)
                s = (t - ng - 0.5) / na
                (ρ, vn, vt1, vt2, p) = prof(s)
                vc = (0.0, 0.0, 0.0)
                vc = Base.setindex(vc, vn, a)                 # normal vel along axis a
                vc = Base.setindex(vc, vt1, a % 3 + 1)        # transverse roles (cyclic)
                vc = Base.setindex(vc, vt2, (a % 3 + 1) % 3 + 1)
                q = idx(i, j, k); D[q] = ρ
                S[1][q] = ρ * vc[1]; S[2][q] = ρ * vc[2]; S[3][q] = ρ * vc[3]
                Tau[q] = ρ * (p / ((g - 1) * ρ) + 0.5 * (vc[1]^2 + vc[2]^2 + vc[3]^2))
            end
            st = _P.ppml_alloc_state(D, dims, ng); _P.ppml_init_state!(st, D, S..., Tau; gamma = g)
            for _ in 1:nsteps
                _P.ppml_step_3d!(D, S[1], S[2], S[3], Tau, dims, ng; state = st,
                                 dt = dt, gamma = g, dx = dx, face_periodic = true)
            end
            # extract the 1-D profile along axis a (conserved D, normal momentum, τ)
            pick(t) = a == 1 ? idx(t, ng + 1, ng + 1) : a == 2 ? idx(ng + 1, t, ng + 1) : idx(ng + 1, ng + 1, t)
            Sn = S[a]
            [(D[pick(t)], Sn[pick(t)], Tau[pick(t)]) for t in (ng + 1):(nb - ng)]
        end
        lx = line(1); ly = line(2); lz = line(3)
        dmax(p, q) = maximum(maximum(abs.(values(a) .- values(b))) for (a, b) in zip(p, q))
        @test dmax(lx, ly) < 1e-12                   # x ≡ y (bit-tight up to FP reassoc)
        @test dmax(lx, lz) < 1e-12                   # x ≡ z
    end

    # ── C. conservation (mass, momentum, energy) to round-off, multi-step ───────
    @testset "conservation (periodic, round-off)" begin
        ng = 3; n = 20; nb = n + 2ng; dims = (nb, nb, nb); N = nb^3; dx = 1.0 / n; dt = 0.01
        nx, ny, nz = dims; idx(i, j, k) = i + nx * (j - 1) + nx * ny * (k - 1)
        D = zeros(N); S1 = zeros(N); S2 = zeros(N); S3 = zeros(N); Tau = zeros(N)
        for k in 1:nz, j in 1:ny, i in 1:nx
            x = (i - ng - 0.5) / n; y = (j - ng - 0.5) / n; z = (k - ng - 0.5) / n; q = idx(i, j, k)
            ρ = 1.0 + 0.3sinpi(2x) * cospi(2y); pr = 0.6 + 0.2cospi(2z)
            u = 0.2sinpi(2y); v = 0.15cospi(2z); w = -0.1sinpi(2x)
            D[q] = ρ; S1[q] = ρ * u; S2[q] = ρ * v; S3[q] = ρ * w
            Tau[q] = ρ * (pr / ((g - 1) * ρ) + 0.5 * (u^2 + v^2 + w^2))
        end
        st = _P.ppml_alloc_state(D, dims, ng); _P.ppml_init_state!(st, D, S1, S2, S3, Tau; gamma = g)
        tot(f) = _P.total_field(f, dims, ng, dx)
        m0 = tot(D); px0 = tot(S1); py0 = tot(S2); pz0 = tot(S3); e0 = tot(Tau)
        _P.with_pool() do
            for s in 1:6
                _P.ppml_step_3d!(D, S1, S2, S3, Tau, dims, ng; state = st, dt = dt, gamma = g, dx = dx,
                                 order = isodd(s) ? (1, 2, 3) : (3, 2, 1), face_periodic = true)
            end
        end
        @test abs(tot(D) - m0) / abs(m0) < 1e-12
        @test abs(tot(S1) - px0) < 1e-11
        @test abs(tot(S2) - py0) < 1e-11
        @test abs(tot(S3) - pz0) < 1e-11
        @test abs(tot(Tau) - e0) / abs(e0) < 1e-12
        @test !any(isnan, D)
    end

    # ── C2. WENO5 smooth-extremum fallback preserves a smooth advected wave ─────
    # The full-PPML (Ustyugov §6) WENO5 fallback recovers 5th order at SMOOTH extrema
    # that the median+CW84 limiter would otherwise slowly clip. A smooth entropy wave
    # (ρ=1+A·sin, u=const, p=const ⇒ pure advection of ρ) advected ~1 period must keep
    # MORE amplitude with WENO5 on than off — the limiter never re-adds amplitude, so
    # the direction is deterministic.
    @testset "WENO5 preserves smooth extrema (entropy-wave advection)" begin
        ng = 3; n = 32; nb = n + 2ng; dims = (nb, nb, nb); N = nb^3; dx = 1.0 / n
        A = 0.4; u0 = 1.0; p0 = 1.0; nx, ny, nz = dims
        idx(i, j, k) = i + nx * (j - 1) + nx * ny * (k - 1)
        amp(weno) = begin
            D = zeros(N); S1 = zeros(N); S2 = zeros(N); S3 = zeros(N); Tau = zeros(N)
            for k in 1:nz, j in 1:ny, i in 1:nx
                x = (i - ng - 0.5) / n; q = idx(i, j, k); ρ = 1.0 + A * sinpi(2x)
                D[q] = ρ; S1[q] = ρ * u0; Tau[q] = ρ * (p0 / ((g - 1) * ρ) + 0.5 * u0^2)
            end
            st = _P.ppml_alloc_state(D, dims, ng); _P.ppml_init_state!(st, D, S1, S2, S3, Tau; gamma = g)
            dt = 0.2 * dx / (u0 + sqrt(g * p0))
            _P.with_pool() do
                for s in 1:120
                    _P.ppml_step_3d!(D, S1, S2, S3, Tau, dims, ng; state = st, dt = dt, gamma = g, dx = dx,
                                     order = isodd(s) ? (1, 2, 3) : (3, 2, 1), face_periodic = true, weno5 = weno)
                end
            end
            ρl = [D[idx(i, ng + 1, ng + 1)] for i in (ng + 1):(nb - ng)]
            maximum(ρl) - minimum(ρl)
        end
        a_off = amp(false); a_on = amp(true)
        @test a_on > a_off                               # WENO5 retains more of the smooth wave
        @test a_on <= 2A + 1e-9                           # but does not amplify (no overshoot)
        @test a_on > 0.97 * 2A                            # both preserve the wave well
    end

    # ── D. with_pool ≡ no pool (bitwise) ────────────────────────────────────────
    @testset "with_pool ≡ no pool (bitwise)" begin
        ng = 3; n = 16; nb = n + 2ng; dims = (nb, nb, nb); N = nb^3; dx = 1.0 / n; dt = 0.01
        nx, ny, nz = dims; idx(i, j, k) = i + nx * (j - 1) + nx * ny * (k - 1)
        D0 = zeros(N); v1 = zeros(N); v2 = zeros(N); v3 = zeros(N); et = zeros(N)
        for k in 1:nz, j in 1:ny, i in 1:nx
            x = (i - ng - 0.5) / n; q = idx(i, j, k)
            D0[q] = 1.0 + 0.2sinpi(2x); v1[q] = 0.1cospi(2x); et[q] = 0.7 / ((g - 1) * D0[q]) + 0.5 * v1[q]^2
        end
        run() = begin
            D = copy(D0); S1 = D0 .* v1; S2 = D0 .* v2; S3 = D0 .* v3; Tau = D0 .* et
            st = _P.ppml_alloc_state(D, dims, ng); _P.ppml_init_state!(st, D, S1, S2, S3, Tau; gamma = g)
            f = () -> for s in 1:3
                _P.ppml_step_3d!(D, S1, S2, S3, Tau, dims, ng; state = st, dt = dt, gamma = g, dx = dx,
                                 order = isodd(s) ? (1, 2, 3) : (3, 2, 1), face_periodic = true)
            end
            f(); D
        end
        a = run(); b = _P.with_pool() do; run(); end
        @test a == b
    end

    # ── E. metal-f32 ≡ cpu-f32 parity (layer B) ─────────────────────────────────
    @testset "metal-f32 ≡ cpu-f32" begin
        ng = 3; n = 20; nb = n + 2ng; dims = (nb, nb, nb); N = nb^3; dx = 1.0 / n; dt = 0.01
        nx, ny, nz = dims; idx(i, j, k) = i + nx * (j - 1) + nx * ny * (k - 1)
        rho = zeros(N); vx = zeros(N); vy = zeros(N); vz = zeros(N); etot = zeros(N)
        for k in 1:nz, j in 1:ny, i in 1:nx
            x = (i - ng - 0.5) / n; y = (j - ng - 0.5) / n; z = (k - ng - 0.5) / n; q = idx(i, j, k)
            rho[q] = 1.0 + 0.3sinpi(2x) * cospi(2y); pr = 0.6 + 0.2cospi(2z)
            vx[q] = 0.2sinpi(2y); vy[q] = 0.15cospi(2z); vz[q] = -0.1sinpi(2x)
            etot[q] = rho[q] * (pr / ((g - 1) * rho[q]) + 0.5 * (vx[q]^2 + vy[q]^2 + vz[q]^2))
        end
        run(bk, ::Type{T}) where {T} = begin
            be = _P.backend(bk); dev(a) = _P.to_device(be, a, T)
            D = dev(rho); S1 = dev(rho .* vx); S2 = dev(rho .* vy); S3 = dev(rho .* vz); Tau = dev(etot)
            st = _P.ppml_alloc_state(D, dims, ng); _P.ppml_init_state!(st, D, S1, S2, S3, Tau; gamma = g)
            _P.with_pool() do
                for s in 1:4
                    _P.ppml_step_3d!(D, S1, S2, S3, Tau, dims, ng; state = st, dt = dt, gamma = g, dx = dx,
                                     order = isodd(s) ? (1, 2, 3) : (3, 2, 1), face_periodic = true)
                end
            end
            _P.to_host(D)
        end
        layerB!("ppml.density", run)
    end

    # ── E2. reduced ghost zones (:cw3 flatten) + HLL/HLLC both conserve ─────────
    # PPML's reconstruction is 1-ghost-local; the 3-point flattener drops the ng
    # requirement to 2 (periodic). Both Riemann solvers conserve to round-off, and the
    # reduced-ghost f32 path stays CPU≡Metal.
    @testset "reduced ghosts (:cw3, ng=2) + riemann ∈ {hll,hllc}" begin
        ng = 2; n = 18; nb = n + 2ng; dims = (nb, nb, nb); N = nb^3; dx = 1.0 / n; dt = 0.01
        nx, ny, nz = dims; idx(i, j, k) = i + nx * (j - 1) + nx * ny * (k - 1)
        rho = zeros(N); vx = zeros(N); vy = zeros(N); vz = zeros(N); etot = zeros(N)
        for k in 1:nz, j in 1:ny, i in 1:nx
            x = (i - ng - 0.5) / n; y = (j - ng - 0.5) / n; z = (k - ng - 0.5) / n; q = idx(i, j, k)
            rho[q] = 1.0 + 0.3sinpi(2x) * cospi(2y); pr = 0.6 + 0.2cospi(2z)
            vx[q] = 0.2sinpi(2y); vy[q] = 0.15cospi(2z); vz[q] = -0.1sinpi(2x)
            etot[q] = rho[q] * (pr / ((g - 1) * rho[q]) + 0.5 * (vx[q]^2 + vy[q]^2 + vz[q]^2))
        end
        run(bk, ::Type{T}, rie) where {T} = begin
            be = _P.backend(bk); dev(a) = _P.to_device(be, a, T)
            D = dev(rho); S1 = dev(rho .* vx); S2 = dev(rho .* vy); S3 = dev(rho .* vz); Tau = dev(etot)
            st = _P.ppml_alloc_state(D, dims, ng); _P.ppml_init_state!(st, D, S1, S2, S3, Tau; gamma = g)
            tot(f) = _P.total_field(f, dims, ng, dx); m0 = tot(D)
            _P.with_pool() do
                for s in 1:5
                    _P.ppml_step_3d!(D, S1, S2, S3, Tau, dims, ng; state = st, dt = dt, gamma = g, dx = dx,
                                     order = isodd(s) ? (1, 2, 3) : (3, 2, 1), face_periodic = true,
                                     flatten = :cw3, riemann = rie)
                end
                (abs(tot(D) - m0) / abs(m0), _P.to_host(D))
            end
        end
        for rie in (:hllc, :hll)
            drift, d = run(:cpu, Float64, rie)
            @test drift < 1e-12                              # cw3 ng=2 conserves to round-off
            @test !any(isnan, d)
        end
        # ng below the requirement is rejected (cw3 + periodic needs ng ≥ 2)
        @test_throws ErrorException _P.ppml_step_3d!(
            _P.to_device(_P.backend(:cpu), rho, Float64), _P.to_device(_P.backend(:cpu), rho .* vx, Float64),
            _P.to_device(_P.backend(:cpu), rho .* vy, Float64), _P.to_device(_P.backend(:cpu), rho .* vz, Float64),
            _P.to_device(_P.backend(:cpu), etot, Float64), dims, 1;
            state = _P.ppml_alloc_state(_P.to_device(_P.backend(:cpu), rho, Float64), dims, 1),
            dt = dt, gamma = g, dx = dx, face_periodic = true, flatten = :cw3)
        # metal-f32 ≡ cpu-f32 on the reduced-ghost HLLC path
        if metal_ready()
            (_, dc) = run(:cpu, Float32, :hllc); (_, dm) = run(:metal, Float32, :hllc)
            @check("ppml.cw3.density", dm, dc, RTOL_B)
        end
    end

    # ── F. fluxrec reproduces the conservative update (AMR reflux enabler) ──────
    @testset "flux recording reproduces the update" begin
        ng = 3; n = 16; nb = n + 2ng; dims = (nb, nb, nb); N = nb^3; dx = 1.0 / n; dt = 0.01
        nx, ny, nz = dims; idx(i, j, k) = i + nx * (j - 1) + nx * ny * (k - 1)
        D = zeros(N); S1 = zeros(N); S2 = zeros(N); S3 = zeros(N); Tau = zeros(N)
        for k in 1:nz, j in 1:ny, i in 1:nx
            x = (i - ng - 0.5) / n; y = (j - ng - 0.5) / n; z = (k - ng - 0.5) / n; q = idx(i, j, k)
            ρ = 1.0 + 0.3sinpi(2x) * cospi(2y); pr = 0.6 + 0.2cospi(2z)
            u = 0.2sinpi(2y); v = 0.15cospi(2z); w = -0.1sinpi(2x)
            D[q] = ρ; S1[q] = ρ * u; S2[q] = ρ * v; S3[q] = ρ * w
            Tau[q] = ρ * (pr / ((g - 1) * ρ) + 0.5 * (u^2 + v^2 + w^2))
        end
        st = _P.ppml_alloc_state(D, dims, ng); _P.ppml_init_state!(st, D, S1, S2, S3, Tau; gamma = g)
        U0 = (copy(D), copy(S1), copy(S2), copy(S3), copy(Tau))
        frec = ntuple(_ -> ntuple(_ -> zeros(N), 6), 3)
        _P.ppml_step_3d!(D, S1, S2, S3, Tau, dims, ng; state = st, dt = dt, gamma = g, dx = dx,
                         face_periodic = true, fluxrec = frec)
        U1 = (D, S1, S2, S3, Tau); dtdx = dt / dx; ea = (1, nx, nx * ny)
        for fld in 1:5
            maxres = 0.0
            for k in (ng + 1):(nz - ng), j in (ng + 1):(ny - ng), i in (ng + 1):(nx - ng)
                c = idx(i, j, k); div = 0.0
                for a in 1:3
                    div += (frec[a][fld][c + ea[a]] - frec[a][fld][c]) * dtdx
                end
                maxres = max(maxres, abs((U1[fld][c] - U0[fld][c]) + div))
            end
            @test maxres < 1e-13
        end
    end
end
