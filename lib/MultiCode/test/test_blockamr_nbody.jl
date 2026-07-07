# #4 — N-body cross-check for the BlockAMR DM dynamics.
#
# The same zero-velocity Zel'dovich plane wave (MultiCode.ZeldovichSpec — an n³
# lattice with x = q + A·sin(2π q_x), v = 0, EdS) is evolved through TWO
# independent particle-mesh engines and compared to the exact closed-form
# mixed-mode growth b(x) = 0.6·x + 0.4·x^(-3/2):
#   * BlockAMR's DM path  (topgrid CIC → FFT Poisson 1.5Ωm·a → trilinear accel →
#     KDK → super-comoving dτ), and
#   * RAMSES's PM  (bin64sc, super-comoving units, the certified capi driver).
# Zero initial velocity excites BOTH the growing (∝a) and decaying (∝a^-3/2)
# modes, so matching b(x) tests the full leapfrog — not just the growing mode.
# The measurement (MultiCode.zeldovich_measure) is identity-free: y,z never move
# (x-only forces), so sorting each lattice line pairs x with q with no ID map.
#
# Run:  RAMSES_LIB_COSMO=<.../bin64sc/libramses3d.so> \
#         julia --project=lib/MultiCode/test lib/MultiCode/test/test_blockamr_nbody.jl
# (the .so path is auto-detected on this host if the env var is unset; the RAMSES
#  arm is skipped cleanly if the cosmo flavor is unavailable.)
using MultiCode, BlockAMR, RamsesLib, CodeBridge, Printf
import PoissonKernels
try; @eval using CUDA; catch; end
const BEbam = BlockAMR.has_backend(:cuda) ? :cuda : :cpu

# auto-point the RAMSES cosmo flavor at this host's Linux .so if unset (the
# bridge default path assumes the macOS mini-ramses/bin64sc/*.dylib layout)
if !haskey(ENV, "RAMSES_LIB_COSMO")
    for p in ("/home/tabel/Projects/mini-ramses-metal/bin64sc/libramses3d.so",)
        isfile(p) && (ENV["RAMSES_LIB_COSMO"] = p; break)
    end
end

"""
    run_blockamr_zeldovich(spec; nstep, be) -> (; xp, a, steps, seconds)

Evolve the shared zero-velocity Zel'dovich lattice through BlockAMR's DM path
from a_i = 1/(1+z_init) to a_ratio·a_i (EdS: D=a, dτ = a·dlna/a^{3/2}).  Returns
the final positions (N×3, box units) and a = a_final/a_i.
"""
function run_blockamr_zeldovich(spec::MultiCode.ZeldovichSpec; nstep::Int = 400, be = BEbam)
    ics = MultiCode.zeldovich_particles(spec)
    N = spec.n; Np = N^3
    ai = 1/(1+spec.z_init); af = spec.a_ratio*ai; Om = 1.0
    px = Float32.(ics.x[:,1]); py = Float32.(ics.x[:,2]); pz = Float32.(ics.x[:,3])
    z0 = zeros(Float32, Np)
    beobj = BlockAMR.backend(be); dev(v) = BlockAMR.to_device(beobj, v, Float32)
    parts = (px=dev(px), py=dev(py), pz=dev(pz),
             vx=dev(copy(z0)), vy=dev(copy(z0)), vz=dev(copy(z0)))
    mass_code = 1.0/Np
    B = 16; while N % B != 0; B ÷= 2; end
    hier = AMRHierarchy(; nbase=(N,N,N), B=B, backend=be, T=Float16, nsp=0,
                        gamma=5/3, cfl=0.3, Lcap=0, scheme=:ctu)
    BlockAMR.init_base_level!(hier); BlockAMR.build_level_tables!(hier, 0)
    ρg = BlockAMR.device_zeros(hier.be, Float32, (N^3,))
    φg = BlockAMR.device_zeros(hier.be, Float32, (N^3,))
    pax = BlockAMR.device_zeros(hier.be, Float32, (Np,)); pay = similar(pax); paz = similar(pax)
    a = ai; dlna = (log(af)-log(ai))/nstep; t0 = time()
    for _ in 1:nstep
        dτ = a*dlna/sqrt(a^3)                               # EdS dtau_for_dlna
        fill!(ρg, 0.0f0)
        PoissonKernels.cic_deposit!(ρg, parts.px, parts.py, parts.pz,
            parts.vx, parts.vy, parts.vz, Float32(mass_code*N^3); N=N, disp=0.0f0, shift=-0.5f0)
        ρg .-= 1.0f0
        PoissonKernels.fft_poisson_rfft!(reshape(φg,N,N,N), reshape(ρg,N,N,N);
            G=1.5*Om*a, a=1.0, boxsize=1.0)
        BlockAMR.phi_from_global!(hier, φg)
        BlockAMR.gather_accel_particles!(hier, parts, pax, pay, paz)
        BlockAMR.particles_kick!(hier, parts, pax, pay, paz, 0.5*dτ)
        BlockAMR.particles_drift!(hier, parts, dτ)
        BlockAMR.particles_kick!(hier, parts, pax, pay, paz, 0.5*dτ)
        a = exp(log(a)+dlna)
    end
    xp = hcat(Float64.(Array(parts.px)), Float64.(Array(parts.py)), Float64.(Array(parts.pz)))
    return (xp=xp, a=a/ai, steps=nstep, seconds=time()-t0)
end

function main()
    spec = MultiCode.ZeldovichSpec()
    @printf("N-body cross-check — ZeldovichSpec: n=%d A=%.4f z_init=%.0f a_i→%.1f·a_i box=%.0f Mpc/h (EdS, v=0)\n",
            spec.n, spec.A, spec.z_init, spec.a_ratio, spec.box_mpch)
    println("engine       a/a_i   bA meas   bA exact   ratio    shape resid/A")
    rows = NamedTuple[]

    rb = run_blockamr_zeldovich(spec)
    mb = MultiCode.zeldovich_measure(rb.xp, spec)
    bAx = MultiCode.zeldovich_growth(rb.a)*spec.A
    @printf("blockamr-pm  %.3f   %.5f  %.5f   %.4f   %.4f\n",
            rb.a, mb.bA, bAx, mb.bA/bAx, mb.rms_resid/spec.A)
    push!(rows, (label="blockamr", psi=mb.bA/MultiCode.zeldovich_growth(rb.a),
                 ratio=mb.bA/bAx, resid=mb.rms_resid/spec.A, a=rb.a))

    ramses_ok = CodeBridge.available(RamsesLib.BRIDGE, :cosmo)
    if ramses_ok
        rr = MultiCode.run_ramses_zeldovich(spec)
        try
            mr = MultiCode.zeldovich_measure(rr.xp, spec)
            bAx2 = MultiCode.zeldovich_growth(rr.a)*spec.A
            @printf("ramses-pm    %.3f   %.5f  %.5f   %.4f   %.4f\n",
                    rr.a, mr.bA, bAx2, mr.bA/bAx2, mr.rms_resid/spec.A)
            push!(rows, (label="ramses", psi=mr.bA/MultiCode.zeldovich_growth(rr.a),
                         ratio=mr.bA/bAx2, resid=mr.rms_resid/spec.A, a=rr.a))
        finally
            rr.free()
        end
    else
        @warn "RAMSES cosmo flavor unavailable — set RAMSES_LIB_COSMO to bin64sc/libramses3d.so; running BlockAMR-vs-exact only"
    end

    println("─"^62)
    bam = rows[1]
    ok_bam = abs(bam.ratio-1) < 0.03 && bam.resid < 0.06
    @printf("BlockAMR vs exact:  growth %+.2f%%  shape resid %.2f%%   %s\n",
            100*(bam.ratio-1), 100*bam.resid, ok_bam ? "PASS" : "FAIL")
    if length(rows) == 2
        ram = rows[2]
        dpsi = abs(bam.psi - ram.psi)/spec.A
        @printf("RAMSES   vs exact:  growth %+.2f%%  shape resid %.2f%%\n",
                100*(ram.ratio-1), 100*ram.resid)
        @printf("BlockAMR vs RAMSES: Δψ̂/A = %.4f   %s\n",
                dpsi, dpsi < 0.02 ? "PASS (independent N-body agrees <2%)" : "FAIL")
        println(ok_bam && dpsi < 0.02 ?
                "N-BODY CROSS-CHECK: BlockAMR DM ≡ RAMSES PM ≡ exact Zel'dovich ✓" :
                "N-BODY CROSS-CHECK: FAILED")
    else
        println(ok_bam ? "BlockAMR DM matches exact Zel'dovich ✓ (RAMSES arm skipped)" :
                         "BlockAMR DM FAILED vs exact")
    end
end

main()
