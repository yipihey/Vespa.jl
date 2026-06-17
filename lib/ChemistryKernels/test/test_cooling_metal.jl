# test_cooling_metal.jl — INTEGRATION of metal cooling into the chemistry network.
# (The metal/emission PHYSICS is tested in EmissionKernels/test/test_emission.jl; here
# we check the ChemistryKernels-side wiring: cooling_edot delegates correctly, metals
# add cooling through solve_chem!, and metals=nothing is bit-identical.)

using ChemistryKernels
using Test
const CK = ChemistryKernels

@testset "cooling_metal_integration" begin

    @testset "GOLDEN bit-identity: cooling_edot unchanged by the EmissionKernels move" begin
        # captured from ChemistryKernels BEFORE the radiative physics was relocated.
        ab = metal_abund(solar=1.0)
        let s = (nHI=0.7,nHII=0.3,nHeI=0.06,nde=0.3,nH2=1e-3,nHD=1e-6,T=3000.0,z=30.0)
            @test cooling_edot(s.nHI,s.nHII,s.nHeI,s.nde,s.nH2,s.nHD,s.T,s.z) == -2.396333826229013e-25
            @test cooling_edot(s.nHI,s.nHII,s.nHeI,s.nde,s.nH2,s.nHD,s.T,s.z; nH=1.0, metals=ab) ==
                  -5.593173118817311e-25
        end
        let s = (nHI=0.99,nHII=0.01,nHeI=0.08,nde=0.01,nH2=1e-5,nHD=1e-8,T=150.0,z=0.0)
            @test cooling_edot(s.nHI,s.nHII,s.nHeI,s.nde,s.nH2,s.nHD,s.T,s.z) == -4.074876222048002e-29
            @test cooling_edot(s.nHI,s.nHII,s.nHeI,s.nde,s.nH2,s.nHD,s.T,s.z; nH=1.0, metals=ab) ==
                  -3.9535955487627824e-27
        end
    end

    @testset "cooling_edot: metals add cooling; off-path bit-identical" begin
        nHI=0.7; nHII=0.3; nHeI=0.06; nde=0.3; nH2=1e-3; nHD=0.0; T=3000.0; z=0.0
        e_no = cooling_edot(nHI,nHII,nHeI,nde,nH2,nHD,T,z)
        ab   = metal_abund(solar=1.0)
        e_yes= cooling_edot(nHI,nHII,nHeI,nde,nH2,nHD,T,z; nH=1.0, metals=ab)
        @test e_yes < e_no
        @test cooling_edot(nHI,nHII,nHeI,nde,nH2,nHD,T,z; nH=1.0, metals=nothing) == e_no
        @test cooling_edot(nHI,nHII,nHeI,nde,nH2,nHD,T,z; nH=1.0,
                           metals=MetalAbundances(0.0,0.0,0.0,0.0)) == e_no
    end

    @testset "solve_chem!: a metals field cools the cell more; metals=nothing identical" begin
        n=4
        rho=fill(1.0*CK.MH/0.76, n); HII=fill(0.3*CK.MH, n); H2I=fill(1e-3*CK.MH, n)
        e0 = [3000.0 * 1.5 * CK.KBOLTZ * (0.76*(1+0.3/0.76) + 0.06) / CK.MH for _ in 1:n]
        mk()=(copy(e0), copy(HII), copy(H2I))
        dt = 3.0e11
        e1,h1,m1 = mk(); solve_chem!(rho, e1, h1, m1; a_value=1.0, dt=dt,
                                     density_units=1.0, length_units=1.0, time_units=1.0)
        met = (C=fill(2.69e-4,n), O=fill(4.9e-4,n), Si=fill(3.24e-5,n), Fe=fill(3.16e-5,n))
        e2,h2,m2 = mk(); solve_chem!(rho, e2, h2, m2; a_value=1.0, dt=dt,
                                     density_units=1.0, length_units=1.0, time_units=1.0, metals=met)
        @test e2[1] < e1[1]
        e3,h3,m3 = mk(); solve_chem!(rho, e3, h3, m3; a_value=1.0, dt=dt,
                                     density_units=1.0, length_units=1.0, time_units=1.0, metals=nothing)
        @test e3 == e1
    end
end

println("cooling_metal_integration tests complete.")
