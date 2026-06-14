using ChemistryKernels, Test
using ChemistryKernels: solve_chem!

# Wave-5 end-to-end gate: a single cosmological parcel from z=1000→20, comparing
# the ChemistryKernels operator (solve_chem!) to the grackle reduced lib
# (GrackleChem.grackle_reduced_step!) at EVERY step on identical input — the
# decisive "do they compute the same chemistry+cooling" check.  Skips cleanly if
# the reduced lib / Cloudy data file is absent.

# GrackleChem is a self-contained ccall module (no MultiCode deps) — include it.
const _GRSRC = abspath(joinpath(@__DIR__, "..", "..", "MultiCode", "src", "grackle_service.jl"))
const _GRDATA = let
    cands = [joinpath(homedir(),"Research","codes","grackle","input","CloudyData_noUVB.h5"),
             joinpath(homedir(),"Research","codes","grackle","grackle_data_files","input","CloudyData_noUVB.h5")]
    idx = findfirst(isfile, cands); idx === nothing ? "" : cands[idx]
end

if !isfile(_GRSRC)
    @info "Wave-5 one-zone: grackle_service.jl not found, skipping"
elseif (include(_GRSRC); !GrackleChem.available()) || isempty(_GRDATA)
    @info "Wave-5 one-zone: reduced lib or Cloudy data absent, skipping"
else
@testset "one-zone z=1000→20 vs grackle reduced lib" begin
    # cosmology
    h, Om, OL = 0.71, 0.27, 0.73
    H0 = h*100*1e5/3.0856775807e24             # s⁻¹
    Ob = 0.045
    rhocrit = 1.8788e-29*h^2                    # g/cm³
    γ = 5/3; fh = 0.76; kB = 1.3806504e-16; mh = 1.67262171e-24
    rhob(z) = Ob*rhocrit*(1+z)^3
    # flat ΛCDM cosmic time [s]
    tcos(z) = (2/(3*H0*sqrt(OL)))*asinh(sqrt(OL/Om)*(1+z)^-1.5)

    # code units: cosmological density_units (keeps grackle's absolute scratch
    # floors negligible), length=time=1 ⇒ e_code≡e_cgs, dt_code≡dt_s.
    DU = rhob(1000.0)
    GrackleChem.grackle_reduced_init!(; hubble=h*100, Om=Om, OL=OL, a_value=1/1001,
        fh=fh, density_units=DU, length_units=1.0, time_units=1.0,
        data_file=_GRDATA, deuterium=false)

    # initial parcel at z=1000: T = T_cmb (still Compton-coupled), residual x_e.
    z = 1000.0
    T = 2.73*(1+z)
    μ = 1.22                                    # neutral primordial mean weight
    e = kB*T/((γ-1)*μ*mh)                       # erg/g
    xHII = 0.05; xH2 = 2e-6

    zs = exp.(range(log(1000.0), log(20.0); length=40))
    worst_e = 0.0; worst_x = 0.0
    for i in 2:length(zs)
        zn = zs[i]; a = 1/(1+zn)
        ρ  = rhob(zn)
        # adiabatic expansion cooling between calls (T ∝ ρ^(γ-1))
        e *= (rhob(zn)/rhob(zs[i-1]))^(γ-1)
        dt = tcos(zn) - tcos(zs[i-1])           # s

        # identical input to both operators — ALL in code units (ρ_code=ρ/DU,
        # species = x·fh·ρ_code; e_code≡e_cgs since length=time=1).
        rc = ρ/DU
        rho_c = [rc]
        HIIm  = xHII*fh*rc; H2Im = xH2*fh*rc
        # grackle (f32 lib)
        eg=[e]; Hg=[HIIm]; H2g=[H2Im]
        GrackleChem.grackle_reduced_step!(rho_c, eg, Hg, H2g; a_value=a, dt=dt)
        # kernels (f64)
        ek=[e]; Hk=[HIIm]; H2k=[H2Im]
        solve_chem!(rho_c, ek, Hk, H2k; a_value=a, dt=dt, density_units=DU,
                    length_units=1.0, time_units=1.0, hubble=h*100, Om=Om, OL=OL,
                    fh=fh, backend=:cpu, precision=Float64)

        worst_e = max(worst_e, abs(ek[1]-eg[1])/abs(eg[1]))
        worst_x = max(worst_x, abs(Hk[1]-Hg[1])/abs(Hg[1]))
        # advance the shared trajectory using grackle as the reference
        e = eg[1]; xHII = Hg[1]/(fh*rc); xH2 = H2g[1]/(fh*rc)
    end
    @info "one-zone operator agreement" worst_rel_e=worst_e worst_rel_xHII=worst_x
    # per-step operator agreement along the full recombination+H2-formation
    # history: sub-percent on both energy and the HII channel (grackle here is
    # f32, kernels f64; the residual is f32 roundoff + dtit-heuristic).
    @test worst_e < 1e-2
    @test worst_x < 1e-2
end
end
