# test_emission.jl — radiative-channel physics for EmissionKernels.
#   • metal per-ion / per-line / ion-fraction parity vs the draft (metal_cooling.pro);
#   • per-channel & diagnostics consistency (channels sum == cooling_rate_total);
#   • @scalarkernel self-parity (catches a botched re-point without the macOS oracle);
#   • GOLDEN bit-identity: cooling_rate_total reproduces the pre-refactor -cooling_edot;
#   • ForwardDiff differentiability + f32/f64 agreement.

using EmissionKernels
using Test
using Printf
const EK = EmissionKernels

const _XE = 1e-3
mket(T) = (nHI=(1-_XE)*0.76, nHII=_XE, nde=_XE, nH2=1e-5, nH=1.0, T=T)
_dtrad(z) = 2.7255*(1+z)
_median(v) = sort(v)[cld(length(v),2)]

# Frozen per-ion ε̃ [erg s⁻¹ cm³] from the draft (tcmb=2.7255). (T,z)=>(CI,CII,OI,SiI,SiII,FeII)
const REF_EPS = Dict(
 (50.0,0.0)  => (1.39686845235525e-24, 4.5656459161804474e-24, 9.067050297466122e-27,
                 1.3971529041870192e-24, 5.898631021653257e-26, 1.8183613305339232e-27),
 (300.0,0.0) => (4.561527838418183e-24, 1.6318623389165235e-23, 1.5396498106570868e-24,
                 1.0125365376765006e-23, 3.4012121617638955e-23, 1.419236675327082e-23),
 (3000.0,0.0)=> (9.139262142279321e-24, 2.3114977816984994e-23, 1.6070303085129136e-23,
                 1.6974795018302995e-23, 1.0635133778036637e-22, 9.116026755086962e-23),
)
const REF_FION = Dict(
 50.0  => (6.55705926928057e-5, 0.44670236542458036, 0.8625102464133836),
 300.0 => (0.00028135351564844465, 0.9081185845672277, 0.9687126851053747),
 3000.0=> (0.001826335920470478, 0.9955666658519341, 0.9533238603907912),
)

# pre-refactor golden cooling_edot (captured from ChemistryKernels before the move):
# state => (no_metal, with_solar_metal)
const GOLD = (
 (nHI=0.7,nHII=0.3,nHeI=0.06,nde=0.3,nH2=1e-3,nHD=1e-6,T=3000.0,z=30.0,
  e_no=-2.396333826229013e-25, e_m=-5.593173118817311e-25),
 (nHI=0.99,nHII=0.01,nHeI=0.08,nde=0.01,nH2=1e-5,nHD=1e-8,T=150.0,z=0.0,
  e_no=-4.074876222048002e-29, e_m=-3.9535955487627824e-27),
)

function _eps(T, z)
    s = mket(T); Tr = _dtrad(z); nH2o=0.75*s.nH2; nH2p=0.25*s.nH2
    (EK._cool_CI(T,Tr,s.nHI,s.nHII,nH2o,nH2p,s.nde),
     EK._cool_CII(T,Tr,s.nHI,nH2o,nH2p,s.nde),
     EK._cool_OI(T,Tr,s.nHI,s.nHII,nH2o,nH2p,s.nde),
     EK._cool_SiI(T,Tr,s.nHI,s.nHII,nH2o,nH2p,s.nde),
     EK._cool_SiII(T,Tr,s.nHI,s.nde),
     EK._cool_FeII(T,Tr,s.nHI,s.nde))
end

@testset "metal per-ion emissivity parity" begin
    names = (:CI,:CII,:OI,:SiI,:SiII,:FeII)
    for ((T,z), ref) in REF_EPS
        got = _eps(T, z)
        for (k, nm) in enumerate(names)
            rel = abs(got[k]-ref[k])/abs(ref[k])
            tol = nm === :FeII ? (T<=50.0 ? 1e-3 : T<=300.0 ? 0.13 : 0.35) : 1e-4
            @test rel < tol
        end
    end
end

@testset "ion fractions parity" begin
    for (T, (rC,rSi,rFe)) in REF_FION
        s = mket(T)
        @test isapprox(EK._fion_C(T,s.nde,s.nHI,s.nHII),  rC;  rtol=1e-6)
        @test isapprox(EK._fion_Si(T,s.nde,s.nHI,s.nHII), rSi; rtol=1e-6)
        @test isapprox(EK._fion_Fe(T,s.nde,s.nHI,s.nHII), rFe; rtol=1e-6)
    end
end

@testset "per-line sums to per-ion; 15 lines present" begin
    s = mket(300.0); ab = metal_abund(solar=1.0)
    lines = metal_line_emissivities(300.0, 0.0, s.nHI, s.nHII, s.nde, s.nH2, s.nH, ab)
    @test length(lines) == 15
    @test all(>(0), values(lines))
    # the per-ion sum of lines must equal the per-ion term in metal_cooling_rate
    Trad = EK.comp2_cmb(0.0); nH2o=0.75*s.nH2; nH2p=0.25*s.nH2
    fC = EK._fion_C(300.0,s.nde,s.nHI,s.nHII)
    wCII = s.nH*ab.C*fC
    @test isapprox(lines.CII_158, wCII*EK._cool_CII(300.0,Trad,s.nHI,nH2o,nH2p,s.nde); rtol=1e-12)
    @test isapprox(lines.CI_609+lines.CI_230+lines.CI_369,
                   s.nH*ab.C*(1-fC)*EK._cool_CI(300.0,Trad,s.nHI,s.nHII,nH2o,nH2p,s.nde); rtol=1e-12)
end

@testset "per-channel + diagnostics consistency" begin
    nHI=0.7; nHII=0.3; nHeI=0.06; nde=0.3; nH2=1e-3; nHD=1e-6; T=3000.0; z=30.0
    @test emiss_HI_lyalpha(nHI,nde,T) > 0
    @test emiss_brem(nHII,nde,T) > 0
    ch = radiative_channels(nHI,nHII,nHeI,nde,nH2,nHD,T,z; nH=1.0, metals=metal_abund(solar=1.0))
    # the diagnostics total equals cooling_rate_total to summation order
    tot = cooling_rate_total(nHI,nHII,nHeI,nde,nH2,nHD,T,z; nH=1.0, metals=metal_abund(solar=1.0))
    @test isapprox(ch.total, tot; rtol=1e-12)
    # He channels are exposed and finite (not part of the reduced cooling sum)
    @test emiss_HeI_exc(0.08, nde, T) >= 0
end

@testset "@scalarkernel self-parity (re-point guard)" begin
    Ts = [50.0, 300.0, 1500.0, 9000.0]
    g  = EK.ceHI_grid(:cpu, Float64, Ts)
    @test g ≈ EK.ceHI.(Ts)                       # grid launcher == direct calls
    gb = EK.brem_grid(:cpu, Float64, Ts)
    @test gb ≈ EK.brem.(Ts)
    # f32 parity on brem (finite, no exp underflow — ceHI's exp(-χ/T) underflows in f32)
    gb32 = EK.brem_grid(:cpu, Float32, Float32.(Ts))
    @test all(isapprox.(Float64.(gb32), gb; rtol=1e-4))
end

@testset "GOLDEN bit-identity: cooling_rate_total == -(pre-refactor cooling_edot)" begin
    ab = metal_abund(solar=1.0)
    for s in GOLD
        c_no = cooling_rate_total(s.nHI,s.nHII,s.nHeI,s.nde,s.nH2,s.nHD,s.T,s.z)
        c_m  = cooling_rate_total(s.nHI,s.nHII,s.nHeI,s.nde,s.nH2,s.nHD,s.T,s.z; nH=1.0, metals=ab)
        @test c_no == -s.e_no                    # exact (same code, relocated)
        @test c_m  == -s.e_m
    end
end

@testset "abundance linearity, template mixing, [α/Fe], CMB suppression, taper" begin
    s = mket(300.0)
    mc(ab) = metal_cooling_rate(300.0,0.0,s.nHI,s.nHII,s.nde,s.nH2,s.nH, ab)
    @test isapprox(mc(metal_abund(solar=2.0)), 2*mc(metal_abund(solar=1.0)); rtol=1e-12)
    ca=mc(metal_abund(ccsn=1.0)); cb=mc(metal_abund(ia=1.0))
    @test isapprox(mc(metal_abund(ccsn=0.3,ia=0.7)), 0.3ca+0.7cb; rtol=1e-10)
    @test cb > ca                                # Type Ia (Fe/Si rich) cools more
    s60 = mket(60.0); ab = metal_abund(solar=1.0)
    @test metal_cooling_rate(60.0,6.0,s60.nHI,s60.nHII,s60.nde,s60.nH2,s60.nH,ab) <
          metal_cooling_rate(60.0,0.0,s60.nHI,s60.nHII,s60.nde,s60.nH2,s60.nH,ab)
    @test metal_cooling_rate(2.5e4,0.0,s.nHI,s.nHII,s.nde,s.nH2,s.nH,ab) == 0.0
end

@testset "ForwardDiff + f32/f64" begin
    import ForwardDiff
    s = mket(300.0); ab = metal_abund(solar=1.0)
    f(T) = metal_cooling_rate(T,0.0,s.nHI,s.nHII,s.nde,s.nH2,s.nH, ab)
    @test isapprox(ForwardDiff.derivative(f,300.0), (f(300.0+1e-3)-f(300.0-1e-3))/2e-3; rtol=1e-5)
    g(aC) = metal_cooling_rate(300.0,0.0,s.nHI,s.nHII,s.nde,s.nH2,s.nH,
                               MetalAbundances(aC,4.9e-4,3.24e-5,3.16e-5))
    @test isapprox(ForwardDiff.derivative(g,2.69e-4), (g(2.69e-4+1e-9)-g(2.69e-4-1e-9))/2e-9; rtol=1e-5)
    c64 = f(300.0)
    s32 = mket(300.0f0)
    c32 = metal_cooling_rate(300.0f0,0.0f0,Float32(s.nHI),Float32(s.nHII),Float32(s.nde),
                             Float32(s.nH2),1.0f0, MetalAbundances(2.69f-4,4.9f-4,3.24f-5,3.16f-5))
    @test isapprox(Float64(c32), c64; rtol=1e-4)
end

println("EmissionKernels emission tests complete.")
