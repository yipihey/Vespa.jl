# E3 — the :julia slot swap: a Julia physics method running on LIVE Enzo grid
# memory. In the Julia-driven EvolveLevel, the hydro slot is replaced by EnzoNG's
# certified HLLC+PLM+SSP-RK2 kernels, which read the live grid's BaryonField
# (Density, Velocity1, TotalEnergy via problem_get_field), advance one step, and
# write it back (problem_set_field) — mutating the same Enzo memory Enzo's own
# kernels would. This is mix-and-match: Enzo owns init/BC/timestep/AMR, a Julia
# method owns hydro. Validated against the exact Riemann solution.
#
# Guarded on grid_available() (needs the heavy Session bridge library).

const SWAP_PROB = normpath(joinpath(@__DIR__, "..", "..", "..", "..",
                                    "run", "Hydro", "Hydro-1D", "Toro-1-ShockTube",
                                    "Toro-1-ShockTube.enzo"))
const SWAP_NGHOST = 3   # PPM ghost zones (Enzo default for this problem)

# One SSP-RK2 HLLC+PLM step on the live Enzo grid arrays, reusing EnzoNG kernels.
function julia_hydro_step!(h, dt; γ = 1.4)
    T  = EnzoLib.problem_grid_size(h)
    di = EnzoLib.field_index(h, 0)            # Density
    vi = EnzoLib.field_index(h, 4)            # Velocity1 (x)
    ei = EnzoLib.field_index(h, 1)            # TotalEnergy (specific)
    d = EnzoLib.problem_get_field(h, di)
    v = EnzoLib.problem_get_field(h, vi)
    e = EnzoLib.problem_get_field(h, ei)
    N = T - 2 * SWAP_NGHOST
    dx = 1.0 / N                              # domain [0,1]
    i1, i2 = SWAP_NGHOST + 1, SWAP_NGHOST + N
    prim(k) = (d[k], v[k], 0.0, 0.0, (γ - 1) * d[k] * (e[k] - 0.5 * v[k]^2))
    U  = [EnzoNG.prim2cons(prim(k), γ) for k in 1:T]
    U0 = copy(U)
    # net flux-divergence update of the active cells, src → dst (ghosts held)
    function fluxdiv!(Us, Ud, hdt)
        Wp = [EnzoNG.cons2prim(Us[k], γ) for k in 1:T]
        slope(k) = ntuple(c -> EnzoNG.limited_slope(Wp[k-1][c], Wp[k][c], Wp[k+1][c]), 5)
        for k in i1:i2
            sm, s0, sp = slope(k - 1), slope(k), slope(k + 1)
            FL = EnzoNG.hllc_flux(ntuple(c -> Wp[k-1][c] + 0.5sm[c], 5),
                                  ntuple(c -> Wp[k][c]   - 0.5s0[c], 5), γ, 1)
            FR = EnzoNG.hllc_flux(ntuple(c -> Wp[k][c]   + 0.5s0[c], 5),
                                  ntuple(c -> Wp[k+1][c] - 0.5sp[c], 5), γ, 1)
            Ud[k] = ntuple(c -> Us[k][c] - hdt / dx * (FR[c] - FL[c]), 5)
        end
    end
    U1 = copy(U); fluxdiv!(U, U1, dt)         # stage 1 (Euler)
    U2 = copy(U1); fluxdiv!(U1, U2, dt)       # stage 2
    for k in i1:i2
        U[k] = ntuple(c -> 0.5 * U0[k][c] + 0.5 * U2[k][c], 5)
        W = EnzoNG.cons2prim(U[k], γ)
        d[k] = W[1]; v[k] = W[2]
        e[k] = W[5] / ((γ - 1) * W[1]) + 0.5 * W[2]^2   # back to specific total energy
    end
    EnzoLib.problem_set_field(h, di, d)
    EnzoLib.problem_set_field(h, vi, v)
    EnzoLib.problem_set_field(h, ei, e)
    return nothing
end

if !EnzoLib.grid_available()
    @info "Session bridge not built — skipping :julia slot-swap test"
else
    @testset "E3: Julia HLLC slot on live Enzo grid (mix-and-match)" begin
        dj = EnzoLib.session_evolve_density(SWAP_PROB, julia_hydro_step!)  # Julia hydro
        de = EnzoLib.reference_density(SWAP_PROB)                           # Enzo PPM
        N = length(dj); dx = 1.0 / N
        x = [(k - 0.5) * dx for k in 1:N]
        WL = (1.0, 0.75, 1.0); WR = (0.125, 0.0, 0.1)                       # Toro-1, disc at 0.3
        ρexact(xi) = exact_riemann_sample(WL, WR, 1.4, (xi - 0.3) / 0.2)[1]
        l1j = sum(abs(dj[i] - ρexact(x[i])) for i in 1:N) / N
        l1e = sum(abs(de[i] - ρexact(x[i])) for i in 1:N) / N
        @info "slot swap vs exact Riemann" L1_julia_HLLC = l1j L1_enzo_PPM = l1e
        @test all(isfinite, dj) && all(>(0), dj)        # ran on live Enzo memory, stayed physical
        @test l1j < 0.03                                # Julia HLLC matches truth
        @test maximum(abs.(dj .- de)) < 0.1             # agrees with Enzo PPM (scheme-level)
    end
end
