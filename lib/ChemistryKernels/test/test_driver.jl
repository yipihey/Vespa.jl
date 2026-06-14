using ChemistryKernels, Test
using ChemistryKernels: peebles_k2, hubble_z_of, evolve_cell, solve_chem!

@testset "Peebles k2 (transcription)" begin
    Hz = hubble_z_of(1000.0; hubble=71.0, Om=0.27, OL=0.73)
    @test Hz > 0
    for (T, nHI) in ((3000.0, 1.0), (300.0, 1.0e2), (8000.0, 1.0e-2))
        tt = T/1e4
        aB = 1e-19*4.309*tt^-0.6166/(1+0.6703*tt^0.53)
        n1s = nHI*1e6
        bet = aB*(1.799920e14*T)^1.5*exp(-3.945150e4/T)
        K = 1.215668e-7^3/(8*pi*Hz)
        C = (1+K*8.2245809*n1s)/(1+K*8.2245809*n1s+K*bet*n1s)
        @test isapprox(peebles_k2(T, nHI, Hz), aB*1e6*C; rtol=1e-13)
    end
    # the C-factor suppresses recombination ⇒ Peebles k2 < the bare α_B·1e6
    T = 5000.0; nHI = 1.0
    aB = 1e-19*4.309*(T/1e4)^-0.6166/(1+0.6703*(T/1e4)^0.53)
    @test peebles_k2(T, nHI, Hz) < aB*1e6
end

@testset "evolve_cell physical sanity" begin
    fh = 0.76; rho = 1.0e-25
    HII = 1e-4*fh*rho; H2I = 1e-6*fh*rho
    # high z: strong Compton cooling toward the CMB ⇒ energy drops
    e1,_,_,_ = evolve_cell(rho, 5.0e11, HII, H2I, 0.0, 1.0e13, 1000.0)
    @test e1 < 5.0e11
    # ionization stays sane (no runaway): xHII bounded
    _,hii,_,_ = evolve_cell(rho, 1.0e11, HII, H2I, 0.0, 1.0e13, 100.0)
    @test 0 < hii/(fh*rho) < 1.0
end

@testset "solve_chem! CPU≡Metal parity" begin
    # batch of cells in code units (density_units chosen cosmological)
    n = 32; fh = 0.76
    DU, LU, TU = 1.0e-24, 3.0e24, 3.0e15
    rho = fill(1.0, n)
    eint = fill(1.0e11 / (LU/TU)^2, n)            # ~1e11 erg/g in code units
    xHII = range(1e-4, 1e-2; length=n) |> collect
    HII = xHII .* fh .* rho
    H2I = fill(1e-6*fh, n)
    a = 1.0/(1+50.0)
    args = (; a_value=a, dt=0.05, density_units=DU, length_units=LU, time_units=TU,
            hubble=71.0, Om=0.27, OL=0.73, fh=fh)

    e_ref = copy(eint); HII_ref = copy(HII); H2I_ref = copy(H2I)
    solve_chem!(rho, e_ref, HII_ref, H2I_ref; backend=:cpu, precision=Float64, args...)
    @test all(isfinite, e_ref) && all(>(0), e_ref)

    if ChemistryKernels.has_backend(:metal)
        e_c = copy(eint); HII_c = copy(HII); H2I_c = copy(H2I)
        solve_chem!(rho, e_c, HII_c, H2I_c; backend=:cpu, precision=Float32, args...)
        e_m = copy(eint); HII_m = copy(HII); H2I_m = copy(H2I)
        solve_chem!(rho, e_m, HII_m, H2I_m; backend=:metal, precision=Float32, args...)
        @test isapprox(e_m, e_c; rtol=1e-3)
        @test isapprox(HII_m, HII_c; rtol=1e-3)
    end
end
