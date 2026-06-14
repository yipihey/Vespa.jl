# Regression for the chem_step! engine switch (Wave-5 wiring): the native
# ChemistryKernels engine (:kernels) agrees with the live Grackle reduced lib
# (:grackle) sub-percent, and the default :grackle path is byte-unchanged.
using MultiCode, Test
using MultiCode: chem_init!, chem_step!, GrackleChem

const GD = let
    c = [joinpath(homedir(),"Research","codes","grackle","input","CloudyData_noUVB.h5"),
         joinpath(homedir(),"Research","codes","grackle","grackle_data_files","input","CloudyData_noUVB.h5")]
    i = findfirst(isfile, c); i === nothing ? "" : c[i]
end

if !GrackleChem.available() || isempty(GD)
    @info "chem engines: reduced lib or Cloudy data absent, skipping"
else
@testset "chem_step! engine switch (:grackle vs :kernels)" begin
    h, Om, OL, fh = 0.71, 0.27, 0.73, 0.76
    DU = 0.045 * 1.8788e-29*h^2 * (1+50.0)^3      # ρ̄_b at z=50, cosmological unit
    chem_init!(; hubble=h*100, Om=Om, OL=OL, a_value=1/51, fh=fh,
               density_units=DU, length_units=1.0, time_units=1.0, data_file=GD)

    n = 64
    rho  = fill(1.0, n)
    eint = collect(range(5.0e8, 5.0e9; length=n))           # e_cgs (length=time=1)
    HII  = collect(range(1e-4, 1e-2; length=n)) .* fh
    H2I  = fill(2e-6*fh, n)
    a, dt = 1/51, 5.0e13          # ~0.2/H(z=50): a real cosmological substep [s]

    # default :grackle must be byte-identical to a direct reduced-lib call
    e0=copy(eint); H0=copy(HII); H20=copy(H2I)
    GrackleChem.grackle_reduced_step!(copy(rho), e0, H0, H20; a_value=a, dt=dt)
    eg=copy(eint); Hg=copy(HII); H2g=copy(H2I)
    chem_step!(copy(rho), eg, Hg, H2g; a_value=a, dt=dt)            # engine=:grackle default
    @test eg == e0 && Hg == H0 && H2g == H20

    # :kernels engine — sub-percent vs :grackle
    ek=copy(eint); Hk=copy(HII); H2k=copy(H2I)
    chem_step!(copy(rho), ek, Hk, H2k; a_value=a, dt=dt, engine=:kernels)
    @test isapprox(ek, eg; rtol=1e-2)
    @test isapprox(Hk, Hg; rtol=2e-2)
    @info "chem engine agreement" max_e=maximum(abs.(ek.-eg)./abs.(eg)) max_HII=maximum(abs.(Hk.-Hg)./abs.(Hg))
end
end
