# test_cooling.jl — tabulated Grackle (Cloudy-mode) cooling, src/cooling.jl.
# Reference numbers from libgrackle 3.2 (tools/grackle_ref/grackle_tab_ref.c;
# primordial_chemistry = 0, metal cooling, UVB, Z = 0.3 Z☉, FG2011 tables at z = 0).
# Skipped when the exported tables are not on this host
# (tools/cloudy_table_export.py → ~/data/grackle_tables/FG2011[_shielded]_z0).
const BA = BlockAMR
const TABDIR = expanduser(get(ENV, "BAM_COOLING_TABLES", "~/data/grackle_tables"))

if !isdir(joinpath(TABDIR, "FG2011_z0"))
    @info "test_cooling.jl skipped: no exported FG2011 table under $TABDIR"
else
const UCOOL = CoolingUnits(density = BA._MH_CGS, length = 3.0857e18, velocity = 1e5)
# (n_H, T0, Δt [Myr], e0 [erg/g], T1 [K]) — monotone-cooling cells (libgrackle)
const GREF = Dict(
    "FG2011_z0" => [(4.67e-4, 1e6, 10.0, 1.9949889665e14, 9.8440641009e5),
                    (0.1, 1e5, 0.01, 1.9864210713e13, 8.5366204542e4),
                    (9.1e-4, 1e5, 1.0, 1.9868307625e13, 8.6890526170e4)],
    "FG2011_shielded_z0" => [(4.67e-4, 1e6, 10.0, 1.9949889665e14, 9.8423213811e5),
                             (0.1, 1e5, 0.01, 1.9882624534e13, 8.7413425835e4),
                             (9.1e-4, 1e5, 1.0, 1.9871988102e13, 8.7520781485e4)])

@testset "cooling: T(e) and integration vs libgrackle (tabulated mode)" begin
    for (tab, ref) in GREF
        c = load_cloudy_cooling(joinpath(TABDIR, tab); be = BA.backend(:cpu), units = UCOOL,
                                metallicity = 0.3)
        h = Float64.(Array(c.tab)); g = (5/3 - 1) * BA._MH_CGS / BA._KB_CGS
        for (nH, T0, dtM, e0, T1) in ref
            ρ = nH * BA._MH_CGS / c.XH
            @test isapprox(cloudy_temperature(c, ρ, e0)[1], T0; rtol = 1e-6)   # Grackle's T(e)
            e1 = BA._cloudy_integrate(e0, ρ, dtM * 3.15576e13, h, c.nn, c.nt, c.n0, c.dn, c.t0,
                                      c.dT, c.XH, c.Zrel * c.Zsun, c.Zrel, g, c.Tfloor, c.eps, c.maxsub)
            @test isapprox(cloudy_temperature(c, ρ, e1)[1], T1; rtol = 2e-3)
        end
    end
    # the thin table's warm equilibrium on P/k = 1000 (the plan's "near 8000 K")
    c = load_cloudy_cooling(joinpath(TABDIR, "FG2011_z0"); be = BA.backend(:cpu), units = UCOOL,
                            metallicity = 0.3)
    @test 8000 < maximum(cloudy_isobar(c, 1000.0).Teq) < 9500
end

for BE in BACKENDS
@testset "cooling: block kernel ≡ scalar integrator; ρ, ρv untouched (hydro + MHD) [$BE]" begin
    for physics in (:hydro, :mhd)
        h = AMRHierarchy(; nbase = (32, 32), B = 16, backend = BE, physics, T = Float64,
                         divb = physics === :mhd ? :ct : :glm)
        init_base_level!(h); build_level_tables!(h, 0)
        # code units: length 1 pc, velocity 1 km/s, density m_H cm⁻³
        c = load_cloudy_cooling(joinpath(TABDIR, "FG2011_z0"); be = BA.backend(BE), units = UCOOL,
                                metallicity = 0.3)
        lev = h.levels[1]; n = lev.cap * lev.stride
        a = [zeros(n) for _ in 1:10]; γ = h.gamma
        for s in lev.live, j in 0:15, i in 0:15
            o = lev.meta[s].origin; idx = (Int(s) - 1) * lev.stride + (j + 2) * lev.nd + (i + 2) + 1
            X = (Int(o[1]) + i) / 32; Y = (Int(o[2]) + j) / 32
            ρ = 10.0^(-3 + 3X); Tg = 10.0^(2 + 4.5Y); v = 30.0
            p = ρ * BA._KB_CGS * Tg / (0.62 * BA._MH_CGS) / (BA._MH_CGS * 1e10)   # code pressure
            ge = p / (γ - 1); bx = physics === :mhd ? 0.0 : 0.0
            vals = (ρ, ρ * v, 0.0, 0.0, ge + 0.5ρ * v^2, ge, bx, 0.0, 0.0, 0.0)
            for f in 1:10; a[f][idx] = vals[f]; end
        end
        physics === :mhd ? BA.encode_mhd_from_host!(lev, a...) : BA.encode_from_host!(lev, a[1:6]...)
        D0 = Array(lev.D); S0 = Array(lev.S1); G0 = Array(lev.Ge)
        dt = 0.02                                     # code time = 0.02 pc/(km/s) ≈ 0.02 Myr
        cool_level!(h, 0, dt, c)
        D1 = Array(lev.D); S1 = Array(lev.S1); G1 = Array(lev.Ge); E1 = Array(lev.Tau)
        @test D1 == D0 && S1 == S0
        hc = Float64.(Array(c.tab)); g = (γ - 1) * BA._MH_CGS / BA._KB_CGS
        worst = 0.0; wE = 0.0
        for s in lev.live, j in 0:15, i in 0:15
            idx = (Int(s) - 1) * lev.stride + (j + 2) * lev.nd + (i + 2) + 1
            ρc = D0[idx] * BA._MH_CGS; e0 = G0[idx] / D0[idx] * 1e10
            e1 = BA._cloudy_integrate(e0, ρc, dt * BA.time_unit(UCOOL), hc, c.nn, c.nt, c.n0,
                                      c.dn, c.t0, c.dT, c.XH, c.Zrel * c.Zsun, c.Zrel, g,
                                      c.Tfloor, c.eps, c.maxsub)
            worst = max(worst, abs(G1[idx] / D0[idx] * 1e10 / e1 - 1))
            # E changes by exactly ρΔe (kinetic energy untouched)
            wE = max(wE, abs((E1[idx] - G1[idx]) - 0.5 * S0[idx]^2 / D0[idx]) / E1[idx])
        end
        @test worst < 1e-9
        @test wE < 1e-12
    end
end
end # for BE
end # tables present
