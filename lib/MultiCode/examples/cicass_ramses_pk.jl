# RAMSES on the SAME CICASS streaming ICs as the Enzo run (cicass_highz_pk.jl):
# boot mini-ramses (cosmo) on the CICASS grafic realization, evolve with the real
# amr_step time loop, and measure the baryon (gas) and DM (particle CIC) density
# power spectra at the same output redshifts — so Enzo and RAMSES can be compared
# directly for consistency in the densities (and, next, the potential), both
# seeded by one CICASS realization.  P(k) FFT runs on the GPU (power_spectrum_gpu).
#
# Run:  BACKEND=metal RAMSES_LIB=<bin64sc>/libramses3d.dylib \
#   <julia> --project=lib/MultiCode/test lib/MultiCode/examples/cicass_ramses_pk.jl

using EnzoLib, MultiCode, CICASSLib, RamsesLib, PoissonKernels, Printf, Statistics
import CodeBridge
if Symbol(get(ENV, "BACKEND", "metal")) === :metal
    using Metal                       # activates PoissonKernels' Metal backend (GPU P(k) FFT)
end

const BOX    = parse(Float64, get(ENV, "CIC_BOX",   "0.128"))
const ZSTART = parse(Float64, get(ENV, "CIC_ZSTART","1000.0"))
const ZEND   = parse(Float64, get(ENV, "CIC_ZEND",  "20.0"))
const NGRID  = parse(Int,     get(ENV, "CIC_NGRID", "128"))
const NOUT   = parse(Int,     get(ENV, "CIC_NOUT",  "7"))
const OMEGA_M = parse(Float64, get(ENV, "CIC_OMEGAM","0.27"))
const VBC    = parse(Float64, get(ENV, "CIC_VBC",   "30.0"))
const BE     = Symbol(get(ENV, "BACKEND", "metal"))
const T      = Float32
const REPORTS = joinpath(@__DIR__, "..", "..", "..", "reports", "multicode")

dev(be, a) = PoissonKernels.to_device(be, a, T)

# overdensity δ = ρ/ρ̄ − 1 of an N³ grid → P(k) via the GPU FFT
function pk_of(grid::Array{Float64,3})
    δ = copy(grid); m = sum(δ)/length(δ); δ ./= m; δ .-= 1.0
    bep = PoissonKernels.backend(BE)
    r = PoissonKernels.power_spectrum_gpu(dev(bep, δ); boxsize=BOX, nbins=size(δ,1)÷2)
    return (k = collect(r.k), P = collect(r.P))
end

# raw-field power (mean-subtracted, no /mean) — for the potential φ
function pk_field(grid::Array{Float64,3})
    f = copy(grid); f .-= sum(f)/length(f)
    bep = PoissonKernels.backend(BE)
    r = PoissonKernels.power_spectrum_gpu(dev(bep, f); boxsize=BOX, nbins=size(f,1)÷2)
    return (k = collect(r.k), P = collect(r.P))
end

# CIC-deposit particle positions (box fraction) onto an N³ density grid
function cic_density(xp::AbstractMatrix, N::Integer)
    ρ = zeros(N, N, N)
    @inbounds for p in 1:size(xp,1)
        gx=mod(xp[p,1],1.0)*N; gy=mod(xp[p,2],1.0)*N; gz=mod(xp[p,3],1.0)*N
        i=floor(Int,gx);fx=gx-i;j=floor(Int,gy);fy=gy-j;k=floor(Int,gz);fz=gz-k
        i0=mod(i,N)+1;i1=mod(i+1,N)+1;j0=mod(j,N)+1;j1=mod(j+1,N)+1;k0=mod(k,N)+1;k1=mod(k+1,N)+1
        ρ[i0,j0,k0]+=(1-fx)*(1-fy)*(1-fz);ρ[i1,j0,k0]+=fx*(1-fy)*(1-fz)
        ρ[i0,j1,k0]+=(1-fx)*fy*(1-fz);ρ[i1,j1,k0]+=fx*fy*(1-fz)
        ρ[i0,j0,k1]+=(1-fx)*(1-fy)*fz;ρ[i1,j0,k1]+=fx*(1-fy)*fz
        ρ[i0,j1,k1]+=(1-fx)*fy*fz;ρ[i1,j1,k1]+=fx*fy*fz
    end
    ρ
end

# RAMSES gas density on the uniform level grid → N³ (NGP: cells tile the grid)
function ramses_gas_grid(h, lev, N)
    cs = MultiCode.ramses_extract(h; lev=lev, boxlen=1.0, lib=:cosmo)
    dep = MultiCode.deposit_to_grid(cs, N; method=:ngp, periodic=true)
    return Array{Float64,3}(reshape(dep.rho, N, N, N))
end

# RAMSES mesh field (:phi, :rho, …) on the uniform level grid → N³ via the
# ckey→cell-position convention (x_d = (2·ck_d + bit_d(c) + 0.5)·dx).
function ramses_field_grid(h, lev, N, which::Symbol)
    ck, val = RamsesLib.get_field(h, which, lev; lib=:cosmo)
    noct = size(ck, 1); dx = 1.0 / 2^lev
    out = zeros(Float64, N, N, N)
    @inbounds for i in 1:noct, c in 1:8
        ix = floor(Int, (2*ck[i,1] + ((c-1)>>0 & 1) + 0.5)*dx * N)
        iy = floor(Int, (2*ck[i,2] + ((c-1)>>1 & 1) + 0.5)*dx * N)
        iz = floor(Int, (2*ck[i,3] + ((c-1)>>2 & 1) + 0.5)*dx * N)
        out[mod(ix,N)+1, mod(iy,N)+1, mod(iz,N)+1] = val[i, c]
    end
    out
end

# Override mini-ramses's gas velocity (which it sets from ic_velc = CDM) with the
# ACTUAL CICASS baryon velocity field snap.gas_vel — so RAMSES baryons feel the
# pressure-suppressed gas velocities (like Enzo) instead of tracing CDM.  Maps each
# RAMSES cell → CICASS grid index (C-order, i fastest); ρu = ρ·(v_kms/unit_v).
function inject_gas_velocity!(h, lev, snap, unit_v, N)
    ck, rho = RamsesLib.get_hydro(h, :uold, 1, lev; lib=:cosmo)
    noct = size(ck,1); dx = 1.0/2^lev; gv = snap.gas_vel
    for d in 1:3
        _, mom = RamsesLib.get_hydro(h, :uold, 1+d, lev; lib=:cosmo)
        @inbounds for i in 1:noct, c in 1:8
            ix = mod(floor(Int,(2*ck[i,1]+((c-1)>>0&1)+0.5)*dx*N), N)
            iy = mod(floor(Int,(2*ck[i,2]+((c-1)>>1&1)+0.5)*dx*N), N)
            iz = mod(floor(Int,(2*ck[i,3]+((c-1)>>2&1)+0.5)*dx*N), N)
            idx = ix + N*(iy + N*iz) + 1
            mom[i,c] = rho[i,c] * (gv[idx,d]/unit_v)
        end
        RamsesLib.set_hydro!(h, :uold, 1+d, lev, ck, mom; lib=:cosmo)
    end
end

# ── Grackle chemistry SUBPROCESS worker (isolates Grackle from the live RAMSES,
#    which co-resident segfaults inside solve_rate_cool_g) ──
function spawn_grackle_worker(; hubble, Om, OL, a0, fh, du, lu, tu, deut, data_file)
    script = joinpath(@__DIR__, "..", "deps", "grackle_worker.jl")
    logf = joinpath(tempdir(), "grackle_worker.log")
    proj = Base.active_project()
    p = open(pipeline(`$(Base.julia_cmd()) --project=$proj $script`; stderr=logf), "r+")
    write(p, Float64(hubble), Float64(Om), Float64(OL), Float64(a0), Float64(fh),
          Float64(du), Float64(lu), Float64(tu))
    write(p, Int64(deut ? 1 : 0), Int64(ncodeunits(data_file))); write(p, data_file); flush(p)
    read(p, Int64) == 1 || error("grackle worker init failed (see $logf)")
    @printf("grackle worker ready (PID isolated from RAMSES)\n"); flush(stdout)
    return p
end

# extract RAMSES state → clamp → send to worker → write back cooled E + species
function worker_chem_step!(p, h, lev, N; a_value, dt, du, lu, tu, deut)
    ck, U = RamsesLib.get_hydro_all(h, :uold, lev; lib=:cosmo)
    noct = size(U,1)
    rho  = Float64.(vec(@view U[:,:,1])); Etot = Float64.(vec(@view U[:,:,5]))
    mx   = Float64.(vec(@view U[:,:,2])); my = Float64.(vec(@view U[:,:,3])); mz = Float64.(vec(@view U[:,:,4]))
    HII  = Float64.(vec(@view U[:,:,6])); H2I = Float64.(vec(@view U[:,:,7]))
    HDI  = deut ? Float64.(vec(@view U[:,:,8])) : nothing
    r = max.(rho, eps()); kin = 0.5 .* (mx.^2 .+ my.^2 .+ mz.^2) ./ r
    eint = (Etot .- kin) ./ r
    mh=1.6726e-24; kB=1.380649e-16; velu=lu/tu; Tunits=mh*velu^2/kB; ec=1.0/(Tunits*(5/3-1)*1.22)
    rfloor = maximum(rho)*1e-20+eps(); emin=1.0*ec; emax=1e8*ec
    @inbounds for i in eachindex(rho)
        rho[i]  = (isfinite(rho[i]) && rho[i]>rfloor) ? rho[i] : rfloor
        eint[i] = (isfinite(eint[i]) && eint[i]>emin) ? min(eint[i],emax) : emin
        HII[i]  = isfinite(HII[i]) ? clamp(HII[i],0.0,rho[i]) : 0.0
        H2I[i]  = isfinite(H2I[i]) ? clamp(H2I[i],0.0,rho[i]) : 0.0
        deut && (HDI[i] = isfinite(HDI[i]) ? clamp(HDI[i],0.0,rho[i]) : 0.0)
    end
    n = length(rho)
    # subcycle the chemistry: one giant Grackle step over the full RAMSES dt
    # overruns the high-z stiff-chemistry subcycle limit → cap each substep.
    dtmax = 0.05; nsub = max(1, ceil(Int, dt/dtmax)); dts = dt/nsub
    for _ in 1:nsub
        write(p, Int64(1), Int64(n), Float64(a_value), Float64(dts))
        write(p, rho); write(p, eint); write(p, HII); write(p, H2I)
        deut && write(p, HDI); flush(p)
        read!(p, eint); read!(p, HII); read!(p, H2I); deut && read!(p, HDI)
    end
    Etot_new = eint .* rho .+ kin
    r8(v) = reshape(v, noct, 8)
    RamsesLib.set_hydro!(h, :uold, 5, lev, ck, r8(Etot_new); lib=:cosmo)
    RamsesLib.set_hydro!(h, :uold, 6, lev, ck, r8(HII); lib=:cosmo)
    RamsesLib.set_hydro!(h, :uold, 7, lev, ck, r8(H2I); lib=:cosmo)
    deut && RamsesLib.set_hydro!(h, :uold, 8, lev, ck, r8(HDI); lib=:cosmo)
    return (xHII1 = HII[1]/rho[1],)
end

function main_ramses()
    CodeBridge_ok = CodeBridge.available(RamsesLib.BRIDGE, :cosmo)
    CodeBridge_ok || error("RAMSES cosmo lib not available (set RAMSES_LIB to bin64sc/libramses3d.dylib)")
    @printf("RAMSES on CICASS ICs: box=%.4f Mpc/h, %d³, z=%.0f→%.0f\n", BOX, NGRID, ZSTART, ZEND)
    COURANT = parse(Float64, get(ENV, "CIC_COURANT", "0.8"))   # = Enzo CourantSafetyNumber (CFL parity)
    res = MultiCode.run_cicass_ramses(; vbc=VBC, boxlength=BOX, zstart=ZSTART, omega_m=OMEGA_M, courant=COURANT)
    h = res.handle; lev = res.lev; N = res.n
    if get(ENV, "CIC_GASVEL", "1") == "1"      # inject the proper baryon velocity field
        inject_gas_velocity!(h, lev, res.snap, res.unit_v_kms, N)
        @printf("injected CICASS gas_vel into RAMSES baryons (overriding ic_velc/CDM)\n")
    end
    # ── v2026 reduced H+D chemistry on RAMSES (NPSCAL=3: HII@6,H2I@7,HDI@8) ──
    chem = get(ENV, "CIC_CHEM", "0") == "1"
    # chemistry engine: :grackle = C reduced lib (worker or in-proc); :kernels =
    # the native ChemistryKernels Julia port (always in-proc, GPU-capable).
    CHEM_ENG = Symbol(get(ENV, "CIC_CHEM_ENGINE", "grackle"))
    CHEM_BK  = Symbol(get(ENV, "CIC_CHEM_BACKEND", "metal"))
    CHEM_PREC = CHEM_BK === :metal ? Float32 : Float64
    INPROC = get(ENV, "CIC_CHEM_INPROC", "0") == "1" || CHEM_ENG === :kernels
    local dens_u, len_u, time_u, gw, gwp
    chem_nstep = 0; chem_accum_dt = 0.0
    # chemistry every hydro step (species advect every step → react every step for a
    # consistent operator split). The fast kernels chem makes per-step affordable.
    CHEM_EVERY = parse(Int, get(ENV, "CIC_CHEM_EVERY", "1"))
    if chem
        a0 = 1.0/(1.0+ZSTART)
        u = RamsesLib.get_units(h; lib=:cosmo)              # EXACT cgs unit factors from RAMSES
        dens_u = u.scale_d; len_u = u.scale_l; time_u = u.scale_t
        @printf("RAMSES units: scale_d=%.4e g/cc  scale_l=%.4e cm  scale_t=%.4e s  scale_nH=%.4e\n",
                dens_u, len_u, time_u, u.scale_nH); flush(stdout)
        GD = get(ENV, "GRACKLE_DATA_FILE",
                 joinpath(homedir(),"Research","codes","grackle","input","CloudyData_noUVB.h5"))
        s = nothing
        try; s = CICASSLib.thermal_state(ZSTART); catch; end
        xHII0 = s===nothing ? 0.047 : s.x_e
        Tgas0 = s===nothing ? 2728.0 : s.T_gas
        ck, rr = RamsesLib.get_hydro(h, :uold, 1, lev; lib=:cosmo)   # density (noct×8)
        RamsesLib.set_hydro!(h, :uold, 6, lev, ck, rr .* xHII0; lib=:cosmo)      # HII = ρ·x_e
        RamsesLib.set_hydro!(h, :uold, 7, lev, ck, rr .* 1e-6;  lib=:cosmo)      # H2I
        RamsesLib.set_hydro!(h, :uold, 8, lev, ck, rr .* (6.8e-5*xHII0); lib=:cosmo) # HDI
        # initialize the gas ENERGY from the CICASS thermal T (grafic had no ic_tempb,
        # so RAMSES left e~0 → Grackle saw T≈0 → out-of-bounds cooling-table segfault)
        mh = 1.6726e-24; kB = 1.380649e-16; vel_u = len_u/time_u
        Tunits = mh*vel_u^2/kB; eint0 = Tgas0/Tunits/(5/3-1)/1.22       # specific eint (code)
        _, mx = RamsesLib.get_hydro(h, :uold, 2, lev; lib=:cosmo)
        _, my = RamsesLib.get_hydro(h, :uold, 3, lev; lib=:cosmo)
        _, mz = RamsesLib.get_hydro(h, :uold, 4, lev; lib=:cosmo)
        kin = 0.5 .* (mx.^2 .+ my.^2 .+ mz.^2) ./ max.(rr, eps())
        RamsesLib.set_hydro!(h, :uold, 5, lev, ck, eint0 .* rr .+ kin; lib=:cosmo)
        @printf("init gas T=%.1f K (eint=%.3e code), Tunits=%.3e, x_HII=%.3e\n", Tgas0, eint0, Tunits, xHII0); flush(stdout)
        # sanity: what density/temperature does Grackle SEE for cell 1?
        nH1 = Float64(rr[1,1])*dens_u*0.76/mh
        @printf("  cell1: rho_code=%.3e → n_H=%.3e cm^-3,  T_init≈%.1f K\n", rr[1,1], nH1, eint0*Tunits*(5/3-1)*1.22); flush(stdout)
        gwp = (; hubble=71.0, Om=OMEGA_M, OL=1-OMEGA_M, a0=a0, fh=0.76,
               du=dens_u, lu=len_u, tu=time_u, deut=true, data_file=GD)
        if INPROC
            MultiCode.chem_init!(; hubble=71.0, Om=OMEGA_M, OL=1-OMEGA_M, a_value=a0, fh=0.76,
                density_units=dens_u, length_units=len_u, time_units=time_u, data_file=GD,
                deuterium=true, engine=CHEM_ENG)
            @printf("H+D chemistry IN-PROCESS engine=%s backend=%s (cosmo NPSCAL=3): x_HII0=%.3e\n",
                    CHEM_ENG, CHEM_BK, xHII0)
        else
            gw = spawn_grackle_worker(; gwp...)
            @printf("H+D chemistry via worker (cosmo NPSCAL=3): x_HII0=%.3e\n", xHII0)
        end
    end
    pk_results = NamedTuple[]
    mkpath(REPORTS); datafile = joinpath(REPORTS, "cicass_ramses_pk.dat")
    function write_tables()
        open(datafile, "w") do io
            println(io, "# RAMSES on CICASS ICs z=$ZSTART→$ZEND box=$BOX Mpc/h N=$NGRID")
            println(io, "# block: '@ z=<z> <baryon|dm>' then k[h/Mpc] P[(Mpc/h)^3]")
            for r in pk_results, tag in (:baryon, :dm, :phi)
                c = getfield(r, tag)
                println(io, "@ z=$(round(r.z,digits=3)) $tag")
                for i in eachindex(c.k); @printf(io, "%.6e %.6e\n", c.k[i], c.P[i]); end
            end
        end
    end
    try
        znow() = 1.0/RamsesLib.get_dt(h, lev; lib=:cosmo).aexp - 1.0
        a_start = 1.0/(1.0+ZSTART); a_end = 1.0/(1.0+ZEND)
        a_outs = exp.(range(log(a_start), log(a_end), length=NOUT)); ai = 1
        function record!(z)
            gas = ramses_gas_grid(h, lev, N)
            p = RamsesLib.get_particles(h, N^3; lib=:cosmo)
            dm = cic_density(p.xp, N)
            phi = ramses_field_grid(h, lev, N, :phi)         # RAMSES gravitational potential
            push!(pk_results, (z=z, baryon=pk_of(gas), dm=pk_of(dm), phi=pk_field(phi)))
            # save the φ and density grids for cross-correlation vs Enzo (same realization)
            open(joinpath(REPORTS, "ramses_fields_z$(round(Int,z)).bin"), "w") do io
                write(io, Float64.(vec(phi))); write(io, Float64.(vec(dm)))
            end
            # ── per-cell physical phase dump (ρ/ρ̄, n_H[cm⁻³], T[K], f_H2, x_HII) ──
            # matched μ=1.22 + γ=5/3 + X_H=0.76 with the Enzo run for a direct comparison.
            let γ=5/3, XH=0.76
                # CURRENT-redshift unit factors (scale_d/_T2/_nH are a-dependent — the
                # init-time dens_u/len_u/time_u would be ~(1+z_i)³/(1+z)³ off at z=20).
                u = RamsesLib.get_units(h; lib=:cosmo)
                ck, U = RamsesLib.get_hydro_all(h, :uold, lev; lib=:cosmo)
                rho=Float64.(vec(@view U[:,:,1])); Et=Float64.(vec(@view U[:,:,5]))
                mx=Float64.(vec(@view U[:,:,2])); my=Float64.(vec(@view U[:,:,3])); mz=Float64.(vec(@view U[:,:,4]))
                HII=Float64.(vec(@view U[:,:,6])); H2I=Float64.(vec(@view U[:,:,7]))
                r=max.(rho,eps()); eint=(Et .- 0.5.*(mx.^2 .+ my.^2 .+ mz.^2)./r)./r
                rrel = rho./(sum(rho)/length(rho))
                nH   = rho .* u.scale_nH                       # cm⁻³ at current a
                Tcell= eint .* ((γ-1)*1.22*u.scale_T2)         # K (T2=T/μ; ×μ=1.22)
                fH2  = (H2I./r)./XH
                xHIIc= (HII./r)./XH
                open(joinpath(REPORTS, "ramses_phase_z$(round(Int,z)).bin"), "w") do io
                    write(io, Int64(length(rrel)))
                    write(io, rrel); write(io, nH); write(io, Tcell); write(io, fH2); write(io, xHIIc)
                end
            end
            write_tables()
            @printf("  ● RAMSES output z=%.2f  [%d written]\n", z, length(pk_results)); flush(stdout)
        end
        @printf("%-5s %-9s %-10s %-8s\n", "step", "z", "ρmax", "sec")
        chemt = 0.0; steptot = 0.0       # wall-time breakdown accumulators
        for step in 0:100000
            z = znow(); a = 1.0/(1.0+z)
            while ai <= NOUT && a >= a_outs[ai] - 1e-12; record!(z); ai += 1; end
            z <= ZEND && break
            t0 = time(); RamsesLib.amr_step!(h, lev, 1; lib=:cosmo)
            chem_accum_dt += chem ? RamsesLib.get_dt(h, lev; lib=:cosmo).dtnew : 0.0
            # run the (expensive) chemistry every CHEM_EVERY hydro steps over the
            # ACCUMULATED dt — valid operator-split since the recombination / H2-HD
            # formation timescales are ≫ the CFL hydro step at high z.
            if chem && (step % CHEM_EVERY == 0)
                znew = znow(); dtc = chem_accum_dt; chem_accum_dt = 0.0
                # The Grackle worker can segfault from a cumulative per-call leak;
                # it is STATELESS (species live in RAMSES), so recycle it every 10
                # steps and, on failure, respawn + retry with several fresh workers.
                # If a step still won't process, skip its chemistry (advection-only
                # for one step is negligible) so the run never dies.
                if !INPROC && chem_nstep > 0 && chem_nstep % 10 == 0
                    try; write(gw, Int64(0)); flush(gw); close(gw); catch; end
                    gw = spawn_grackle_worker(; gwp...)
                end
                if INPROC
                    # in-process (fast: no pipe transfer). engine=:kernels uses the
                    # native ChemistryKernels port (no Grackle dylib); :grackle uses
                    # the in-proc C reduced lib.
                    _ct = time()
                    MultiCode.ramses_chem_step!(h, lev; dt=dtc, a_value=1.0/(1.0+znew),
                        density_units=dens_u, length_units=len_u, time_units=time_u,
                        iHII=6, iH2I=7, iHDI=8, lib=:cosmo,
                        engine=CHEM_ENG, backend=CHEM_BK, precision=CHEM_PREC)
                    chemt += time() - _ct
                else
                    ok = false
                    for attempt in 1:6
                        try
                            worker_chem_step!(gw, h, lev, N; a_value=1.0/(1.0+znew), dt=dtc,
                                du=dens_u, lu=len_u, tu=time_u, deut=true)
                            ok = true; break
                        catch e
                            e isa EOFError || rethrow()
                            try; close(gw); catch; end
                            gw = spawn_grackle_worker(; gwp...)   # fresh worker, retry
                        end
                    end
                    ok || (@printf("  [chem skipped at z=%.1f after 6 worker deaths]\n", znew); flush(stdout))
                end
                chem_nstep += 1
            end
            sec = time()-t0; steptot += sec
            if step % 20 == 0
                gas = ramses_gas_grid(h, lev, N)
                @printf("%-5d %-9.3f %-10.3f %-8.2f\n", step, z, maximum(gas), sec); flush(stdout)
            end
        end
        ai <= NOUT && record!(znow())
        write_tables()
        @printf("\nwrote %d RAMSES outputs → %s\n", length(pk_results), datafile)
        @printf("TIMING RAMSES: step_loop=%.1fs  chem=%.1fs (%.1f%%)  host(hydro+grav)=%.1fs\n",
                steptot, chemt, 100*chemt/max(steptot,eps()), steptot-chemt); flush(stdout)
    finally
        if chem && @isdefined(gw)
            try; write(gw, Int64(0)); flush(gw); close(gw); catch; end   # tell worker to quit
        end
        res.free()
    end
end

main_ramses()
