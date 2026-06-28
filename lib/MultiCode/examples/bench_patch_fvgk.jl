# bench_patch_fvgk.jl — throughput (Mcell/s) of the in-process PatchGrid hydro + chemistry at the
# CICASS config (128³, 3 species, super-comoving cosmo units at z), isolating each GPU phase.
#
#   FVGK hydro : patch_step!(solver=:fvgk, do_hydro=true,  do_chem=false)   — EulerColors species advection
#   CK   chem  : patch_step!(             do_hydro=false, do_chem=true)     — solve_chem_device! (+rate tables)
#   (BENCH_SOLVER=ppm compares the PPMKernels split-sweep hydro.)
#
# Mcell/s = N³ · nsteps / wall  (active cells; the chem kernel additionally touches the ghost shell).
# Run: BENCH_N=128 BENCH_NP=2 BENCH_STEPS=30 <julia> --project=lib/MultiCode/test lib/MultiCode/examples/bench_patch_fvgk.jl

using MultiCode, ChemistryKernels, EmissionKernels
import PPMKernels
using Printf, Statistics

const N      = parse(Int, get(ENV, "BENCH_N", "128"))
const NP     = parse(Int, get(ENV, "BENCH_NP", "2"))
const STEPS  = parse(Int, get(ENV, "BENCH_STEPS", "30"))
const WARMUP = parse(Int, get(ENV, "BENCH_WARMUP", "3"))
const SOLVER = Symbol(get(ENV, "BENCH_SOLVER", "fvgk"))
const ZBENCH = parse(Float64, get(ENV, "BENCH_Z", "100.0"))
const PACKED = get(ENV, "BENCH_PACKED", "0") == "1"     # UInt16 packed species storage (vs f32 ρxᵢ)
const GAMMA  = 5/3; const NG = 4

have_gpu = try; using CUDA; CUDA.functional(); catch; false; end
SOLVER === :fvgk && (@eval using FiniteVolumeGodunovKA)

function make_ic(nc, u)
    D  = Array{Float32}(undef, nc); S1=zero(D); S2=zero(D); S3=zero(D); Tau=zero(D); Ge=zero(D)
    sp = [zeros(Float32, nc) for _ in 1:3]
    # T ≈ 250 K IGM at z~100: eint = T / ((γ-1)·μ·temperature_units)
    tuf  = ChemistryKernels.temperature_units(u.l, u.t)
    eint = Float32(250.0 / ((GAMMA - 1) * 1.22 * tuf))
    @inbounds for k in 1:nc[3], j in 1:nc[2], i in 1:nc[1]
        x=2π*(i-1)/nc[1]; y=2π*(j-1)/nc[2]; z=2π*(k-1)/nc[3]
        ρ = 1f0 + 0.05f0*(sin(x)*cos(y) + sin(z))             # few-% baryon perturbation
        u1=0.02f0*sin(y); u2=0.015f0*cos(z); u3=0.01f0*sin(x)
        D[i,j,k]=ρ; S1[i,j,k]=ρ*u1; S2[i,j,k]=ρ*u2; S3[i,j,k]=ρ*u3
        Ge[i,j,k]=ρ*eint; Tau[i,j,k]=ρ*(eint+0.5f0*(u1*u1+u2*u2+u3*u3))
        sp[1][i,j,k]=ρ*2f-4; sp[2][i,j,k]=ρ*2f-6; sp[3][i,j,k]=ρ*1f-9   # x_HII, x_H2I, x_HDI
    end
    return (D=D, S1=S1, S2=S2, S3=S3, Tau=Tau, Ge=Ge, species=sp)
end

function timed(f, n)            # n calls of f(), synchronized; returns seconds/total
    have_gpu && CUDA.synchronize()
    t0 = time()
    for _ in 1:n; f(); end
    have_gpu && CUDA.synchronize()
    return time() - t0
end

function main()
    if !have_gpu
        @info "no functional CUDA — benchmark needs the GPU"; return
    end
    c = Cosmo(; Om=0.3111, OL=0.6889, h0=67.66, box=0.2, Ob=0.0490)
    a = z_to_a(ZBENCH); u = cosmo_units(c, a)
    ncell=(N,N,N); np=(NP,NP,NP); dx=1.0/N
    spbytes = PACKED ? 2 : 4
    @printf("PatchGrid bench: %d³ → %d patches of %d³  z=%.0f  solver=%s  species=%s (%d B/cell)  (CUDA Float32)\n",
            N, prod(np), N÷NP, ZBENCH, SOLVER, PACKED ? "UInt16-packed" : "f32 ρxᵢ", spbytes); flush(stdout)
    pg = build_patchgrid(; ng=NG, ncell=ncell, np=np, dx=dx, gamma=GAMMA, nspecies=3,
                         besym=:cuda, T=Float32, du=u.d, lu=u.l, tu=u.t, deut=true,
                         packed_species=PACKED)
    ic = make_ic(ncell, u)
    scatter_global!(pg, ic)
    ratetab = ChemistryKernels.build_rate_tables(; precision=Float64, backend=:cuda)
    cooltab = EmissionKernels.build_cooling_tables(; precision=Float64, backend=:cuda)

    # a representative sub-CFL dt from the IC (sound speed + bulk velocity), so nsub=1
    csmax = sqrt(GAMMA*(GAMMA-1)*maximum(ic.Ge ./ ic.D))
    vmax  = maximum(sqrt.(ic.S1.^2 .+ ic.S2.^2 .+ ic.S3.^2) ./ ic.D)
    dτ = 0.4 * dx / (csmax + vmax)
    cells = N^3

    # ── FVGK (or PPM) hydro only ──
    hyd() = patch_step!(pg, dτ; a_value=a, order=(1,2,3), accel=nothing, chem=false,
                        solver=SOLVER, du=u.d, lu=u.l, tu=u.t, do_hydro=true, do_chem=false)
    timed(hyd, WARMUP)
    th = timed(hyd, STEPS)
    mc_h = cells * STEPS / th / 1e6

    # ── ChemistryKernels chemistry only (GPU, with rate/cooling tables, like CICASS) ──
    chm() = patch_step!(pg, dτ; a_value=a, order=(1,2,3), chem=true, du=u.d, lu=u.l, tu=u.t,
                        do_hydro=false, do_chem=true, rate_tables=ratetab, cool_tables=cooltab)
    timed(chm, WARMUP)
    tc = timed(chm, STEPS)
    mc_c = cells * STEPS / tc / 1e6

    @printf("\n  hydro (%-4s): %8.1f Mcell/s   (%.3f ms/step, %d steps)\n", SOLVER, mc_h, 1e3*th/STEPS, STEPS)
    @printf("  chem  (CK)  : %8.1f Mcell/s   (%.3f ms/step, %d steps)\n", mc_c, 1e3*tc/STEPS, STEPS)
    @printf("  combined    : %8.1f Mcell/s   (hydro+chem per cycle: %.3f ms)\n",
            cells / (th/STEPS + tc/STEPS) / 1e6, 1e3*(th/STEPS + tc/STEPS))
    # sanity
    g = gather_global(pg)
    @printf("  state: ρ∈[%.3f,%.3f]  finite=%s\n", minimum(g.D), maximum(g.D),
            all(isfinite, g.D) && all(isfinite, g.Tau) && all(isfinite, g.species[1]))
end

main()
