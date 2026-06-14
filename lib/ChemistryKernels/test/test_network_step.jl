using ChemistryKernels, Test
using ChemistryKernels: network_step, equilibrium_HM, equilibrium_H2II, equilibrium_DII

# network_step has no closed grackle oracle (step_rate_g is internal); the
# transcription risk is structural (which rate in which scoef/acoef, the /2,/3,2×
# molecular factors, the Gauss-Seidel ordering).  We pin all rates + a state and
# check each provisional against an INDEPENDENT literal re-evaluation of
# solve_rate_cool_g.F:2336-2581.  End-to-end accuracy vs grackle is the Wave-5
# one-zone gate.
@testset "network_step (pinned-rate transcription)" begin
    d, fh = 1.0, 0.76
    # state (grackle mass-equiv convention: yH2I=2·n(H2), yH2II=2·n(H2⁺), yHDI=3·n(HD))
    yHI, yHII, yde = 0.70, 0.03, 0.03
    yH2I, yHM, yH2II = 0.02, 1.0e-9, 1.0e-11
    yDI, yDII, yHDI = 6.8e-5*0.70, 6.8e-5*0.03, 1.0e-7
    dt = 1.0e10
    yHeI = (1-fh)*d

    K = (; k1=1.1e-10, k2=2.2e-12, k3=1.7e-13, k4=1.0e-12, k5=1.0e-13, k6=1.0e-12,
         k7=3.0e-16, k8=1.3e-9, k9=2.0e-10, k10=6.0e-10, k11=5.0e-11, k12=4.0e-9,
         k13=1.0e-9, k14=1.0e-9, k15=2.0e-9, k16=4.0e-9, k17=8.0e-10, k18=1.3e-6,
         k19=5.0e-7, k22=1.3e-32, k57=1.0e-15, k58=1.0e-16, k27=7.0e-12, k28=4.0e-13,
         k50=9.0e-11, k51=9.5e-11, k52=1.0e-9, k53=1.5e-9, k54=2.0e-9, k55=2.5e-9,
         k56=3.0e-9)
    k = K  # shorthand

    out = network_step(d, fh, yHI,yHII,yde,yH2I,yHM,yH2II,yDI,yDII,yHDI, K, dt;
                       deuterium=true)

    # --- independent re-evaluation (literal Fortran) ---
    HIp = (k.k2*yHII*yde + 2*k.k13*yHI*yH2I/2 + k.k11*yHII*yH2I/2 + 2*k.k12*yde*yH2I/2
           + k.k14*yHM*yde + k.k15*yHM*yHI + 2*k.k16*yHM*yHII
           + 2*k.k18*yH2II*yde/2 + k.k19*yH2II*yHM/2) * dt + yHI
    HIp /= 1 + (k.k1*yde + k.k7*yde + k.k8*yHM + k.k9*yHII + k.k10*yH2II/2
                + 2*k.k22*yHI^2 + k.k57*yHI + k.k58*yHeI/4) * dt
    @test isapprox(out.yHI, max(HIp,1e-20); rtol=1e-12)

    HIIp = (k.k1*yHI*yde + k.k10*yH2II*yHI/2 + k.k57*yHI*yHI + k.k58*yHI*yHeI/4)*dt + yHII
    HIIp /= 1 + (k.k2*yde + k.k9*yHI + k.k11*yH2I/2 + k.k16*yHM + k.k17*yHM)*dt
    @test isapprox(out.yHII, max(HIIp,1e-20); rtol=1e-12)

    H2Ip = 2*(k.k8*yHM*yHI + k.k10*yH2II*yHI/2 + k.k19*yH2II*yHM/2 + k.k22*yHI*yHI^2)*dt + yH2I
    H2Ip /= 1 + (k.k13*yHI + k.k11*yHII + k.k12*yde)*dt
    @test isapprox(out.yH2I, max(H2Ip,1e-20); rtol=1e-12)

    # HM equilibrium (OLD), H2II equilibrium (NEW provisionals + dep)
    HMp = equilibrium_HM(yHI,yHII,yde,yH2II, k.k7,k.k8,k.k14,k.k15,k.k16,k.k17,k.k19,k.k27)
    @test isapprox(out.yHM, max(HMp,1e-20); rtol=1e-12)

    sc_de = k.k8*yHM*yHI + k.k15*yHM*yHI + k.k17*yHM*yHII + k.k57*yHI*yHI + k.k58*yHI*yHeI/4
    ac_de = -(k.k1*yHI - k.k2*yHII + k.k3*yHeI/4 - k.k6*1e-20/4 + k.k5*1e-20/4
              - k.k4*1e-20/4 + k.k14*yHM - k.k7*yHI - k.k18*yH2II/2)
    dep = (sc_de*dt + yde)/(1 + ac_de*dt)
    H2IIp = equilibrium_H2II(HIp,HIIp,H2Ip,dep,HMp, k.k9,k.k10,k.k11,k.k17,k.k18,k.k19,k.k28)
    @test isapprox(out.yH2II, max(H2IIp,1e-20); rtol=1e-12)

    # deuterium
    DIp = (k.k2*yDII*yde + k.k51*yDII*yHI + 2*k.k55*yHDI*yHI/3)*dt + yDI
    DIp /= 1 + (k.k1*yde + k.k50*yHII + k.k54*yH2I/2 + k.k56*yHM)*dt
    @test isapprox(out.yDI, max(DIp,1e-20); rtol=1e-12)

    DIIp = equilibrium_DII(yDI,yde,yHI,yHII,yH2I,yHDI, k.k1,k.k2,k.k50,k.k51,k.k52,k.k53)
    @test isapprox(out.yDII, max(DIIp,1e-20); rtol=1e-12)

    HDIp = 3*(k.k52*yDII*yH2I/2/2 + k.k54*yDI*yH2I/2/2 + 2*k.k56*yDI*yHM/2)*dt + yHDI
    HDIp /= 1 + (k.k53*yHII + k.k55*yHI)*dt
    @test isapprox(out.yHDI, max(HDIp,1e-20); rtol=1e-12)

    # charge conservation: nₑ = HII_new + HeII/4 + HeIII/2 − HM_old + H2II_old/2
    de_n = max(HIIp,1e-20) + 1e-20/4 + 1e-20/2 - yHM + yH2II/2
    @test isapprox(out.yde, de_n; rtol=1e-12)

    # f32 path runs and tracks
    o32 = network_step(Float32(d), Float32(fh),
                       Float32.((yHI,yHII,yde,yH2I,yHM,yH2II,yDI,yDII,yHDI))...,
                       map(Float32, K), Float32(dt); deuterium=true)
    @test isapprox(Float64(o32.yHI),  out.yHI;  rtol=1e-4)
    @test isapprox(Float64(o32.yHII), out.yHII; rtol=1e-4)
end
