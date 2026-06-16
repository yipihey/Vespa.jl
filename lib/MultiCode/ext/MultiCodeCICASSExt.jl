# ── the CICASS streaming-velocity injector ───────────────────────────────────
#
# CICASS uniquely realizes the baryon–dark-matter STREAMING VELOCITY (Tseliakhovich
# & Hirata 2010; McQuinn & O'Leary 2012): a 2D (k⊥,k∥) transfer function that gives
# the gas a coherent BULK VELOCITY OFFSET relative to the dark matter — physics a
# single-phase generator (MUSIC's amplitude-only two-component path, DISCO-DJ's 1LPT)
# cannot express.
#
# This extension turns a CICASS `.cicass` realization into RAMSES's native grafic
# IC format, splitting the gas (`ic_velb*`, `ic_deltab`) from the dark matter
# (`ic_velc*`) so the streaming offset lives where RAMSES reads it:
# mean(ic_velb) − mean(ic_velc).  The gate confirms the offset survives the
# realization → grafic round-trip (≈ vbc·(1+z)/1001 km/s along one axis).
#
# A package extension: `using CICASSLib` activates it.

module MultiCodeCICASSExt

using MultiCode
using MultiCode: EnzoLib, RamsesLib, CodeBridge
using CICASSLib
using CICASSLib: CICASSSnapshot

_mean(x) = sum(x) / length(x)
_rms(x) = sqrt(sum(abs2, x) / length(x))

# Enzo cosmological velocity unit in km/s (= CICASS enzo_out.c setEnzoUnits OV/1e5);
# km/s ↔ Enzo code velocity for injecting/reading CICASS physical-peculiar km/s.
_enzo_vunit_kms(box, omega_m, z) = 1.22475e7 * box * sqrt(omega_m) * sqrt(1 + z) / 1e5

# Write a grafic field file: 44-byte header record (3×Int32 dims + 8×Float32
# cosmology) then n planes, each a Fortran record of n² Float32 (i fastest, then j;
# planes index k). Matches RAMSES init_grafic.f90 / the zeldovich.jl zero-writer.
function _write_grafic_field(path::AbstractString, field::AbstractVector, n::Integer,
                             dx_mpc::Real, astart::Real, omega_m::Real, omega_l::Real, h0::Real)
    A = reshape(field, n, n, n)
    open(path, "w") do io
        write(io, Int32(44))
        write(io, Int32(n), Int32(n), Int32(n))
        write(io, Float32(dx_mpc), Float32(0), Float32(0), Float32(0))
        write(io, Float32(astart), Float32(omega_m), Float32(omega_l), Float32(h0))
        write(io, Int32(44))
        for k in 1:n
            write(io, Int32(4 * n * n))
            write(io, Float32.(vec(@view A[:, :, k])))
            write(io, Int32(4 * n * n))
        end
    end
    return path
end

# Read a grafic field back (for the round-trip gate).
function _read_grafic_field(path::AbstractString)
    open(path, "r") do io
        read(io, Int32)                                   # 44
        n = Int(read(io, Int32)); read(io, Int32); read(io, Int32)
        for _ in 1:8; read(io, Float32); end
        read(io, Int32)
        A = Array{Float32}(undef, n, n, n)
        plane = Vector{Float32}(undef, n * n)
        for k in 1:n
            read(io, Int32)
            read!(io, plane)
            A[:, :, k] = reshape(plane, n, n)
            read(io, Int32)
        end
        return A
    end
end

# Map a per-particle CICASS array a[l] (lattice index l = x·N²+y·N+z, x slowest)
# onto a grafic-order grid vector (x fastest: m = x+y·N+z·N²) — the transpose
# RAMSES's plane reader (i1 fastest) consumes.  CICASS particle l sits on the
# corner lattice (x/N,y/N,z/N); RAMSES particle (x,y,z) on the cell-center lattice
# ((x+0.5)/N,…).  Used for both ic_velc (exact dm velocity) and ic_posc (exact
# displacement); pass `subtract_center=true` to emit dm_pos − center (the
# displacement that lands the RAMSES particle exactly on the CICASS position).
function _particle_to_grafic(a::AbstractVector, n::Integer;
                             subtract_center::Bool = false, dim::Integer = 1)
    g = Vector{Float64}(undef, n^3)
    mini(d) = d - round(d)
    @inbounds for z in 0:n-1, y in 0:n-1, x in 0:n-1
        m = x + y * n + z * n * n + 1            # grafic linear (x fastest)
        l = x * n * n + y * n + z + 1            # CICASS particle (x slowest)
        if subtract_center
            c = ((x, y, z)[dim] + 0.5) / n
            g[m] = mini(a[l] - c)
        else
            g[m] = a[l]
        end
    end
    return g
end

function MultiCode.write_grafic_streaming(dir::AbstractString, snap::CICASSSnapshot;
                                          h0::Real = 71.0, vboost = (0.0, 0.0, 0.0),
                                          baryon_delta = nothing)
    mkpath(dir)
    n = snap.n
    a_i = 1 / (1 + snap.zinit)
    dx_mpc = (snap.box / snap.hconst) / n          # box [Mpc/h] → [Mpc] per cell
    om, ol = snap.omega_m, snap.omega_l
    # `vboost` (km/s, per component) is subtracted from BOTH species — a uniform
    # Galilean shift of the IC velocity field (e.g. into the baryon rest frame),
    # leaving the relative gas−DM streaming offset unchanged.
    # gas (baryon) velocity grids — already gridded by CICASS
    for (d, name) in zip(1:3, ("ic_velbx", "ic_velby", "ic_velbz"))
        _write_grafic_field(joinpath(dir, name), snap.gas_vel[:, d] .- vboost[d], n, dx_mpc, a_i, om, ol, h0)
    end
    # dark-matter velocity grids — EXACT per-particle CICASS velocities (km/s),
    # transposed into grafic order.  RAMSES rescales ic_velc→code velocity; with
    # ic_posc present below it is consumed ONLY as the kinematic velocity (the
    # Zeldovich xp+=vp step is skipped), so this is the genuine peculiar velocity.
    for (d, name) in zip(1:3, ("ic_velcx", "ic_velcy", "ic_velcz"))
        vc = _particle_to_grafic(snap.dm_vel[:, d] .- vboost[d], n)
        _write_grafic_field(joinpath(dir, name), vc, n, dx_mpc, a_i, om, ol, h0)
    end
    # dark-matter displacement grids (ic_posc*) — the EXACT CICASS displacement
    # dm_pos − cell-center lattice, in Mpc/h (RAMSES divides by boxlen_ini = box
    # [Mpc/h]).  RAMSES reads these directly and lands each particle exactly on the
    # CICASS position (identical to Enzo), bypassing its radiation-free Zeldovich
    # reconstruction from ic_velc that previously under-displaced DM by ~0.71×.
    for (d, name) in zip(1:3, ("ic_poscx", "ic_poscy", "ic_poscz"))
        ψ = _particle_to_grafic(snap.dm_pos[:, d], n; subtract_center = true, dim = d) .* snap.box
        _write_grafic_field(joinpath(dir, name), ψ, n, dx_mpc, a_i, om, ol, h0)
    end
    # baryon overdensity field; `baryon_delta` overrides snap.gas_delta (e.g. zeros for
    # a uniform-density start, or the smooth Fourier δb) — the physical recombination IC.
    δb = baryon_delta === nothing ? snap.gas_delta : baryon_delta
    _write_grafic_field(joinpath(dir, "ic_deltab"), δb, n, dx_mpc, a_i, om, ol, h0)
    return dir
end

function MultiCode.run_cicass_streaming(; vbc::Real = 30.0, boxlength::Real = 0.2,
                                        zstart::Real = 100.0, ngrid::Integer = 128,
                                        workdir::AbstractString = mktempdir())
    CICASSLib.available() || error("libcicass_capi not found — build cicass/deps/build_cicass_darwin.sh")
    spec = CICASSSpec(boxlength = boxlength, zstart = zstart, ngrid = ngrid,
                      vbc = vbc, filename = "cic_stream")
    res = CICASSLib.generate(spec; workdir = workdir)
    snap = CICASSLib.read_snapshot(res.output)

    # streaming offset straight from the realization (physical peculiar km/s)
    dv_snap = CICASSLib.streaming_velocity(snap)

    # write the RAMSES-native grafic streaming set and read it back
    gdir = MultiCode.write_grafic_streaming(joinpath(workdir, "ics"), snap)
    offb = ntuple(d -> _mean(_read_grafic_field(joinpath(gdir, ("ic_velbx", "ic_velby", "ic_velbz")[d]))), 3)
    offc = ntuple(d -> _mean(_read_grafic_field(joinpath(gdir, ("ic_velcx", "ic_velcy", "ic_velcz")[d]))), 3)
    off_grafic = ntuple(d -> offb[d] - offc[d], 3)

    vexp = vbc * (1 + zstart) / 1001.0
    return (; n = ngrid, vbc = float(vbc), zstart = float(zstart),
            offset_snapshot = dv_snap, offset_grafic = off_grafic,
            expected = vexp, grafic_dir = gdir, output = res.output)
end

function MultiCode.run_cicass_enzo(; vbc::Real = 30.0, boxlength::Real = 0.2,
                                   zstart::Real = 100.0, omega_m::Real = 0.27,
                                   ngrid::Integer = 128, param_extra::AbstractString = "",
                                   zero_baryon_bulk::Bool = false,
                                   uniform_baryons::Bool = false,
                                   baryon_ic::Symbol = :particle,
                                   init_temperature::Real = 0.0, mu_init::Real = 1.22,
                                   workdir::AbstractString = mktempdir())
    CICASSLib.available() || error("libcicass_capi not found")
    EnzoLib.grid_available() || error("Enzo grid bridge not built")

    # baryon density IC source:
    #   :particle (default) — CICASS's glass-CIC-interpolated δb (shot-noise prone at high z)
    #   :smooth             — the RAW Fourier-realized δb grid from the CAMB/CLASS baryon
    #                         transfer function (same phases as the DM), no glass/CIC → no
    #                         particle shot noise.  Requires the rebuilt libcicass_capi that
    #                         honors CICASS_SMOOTH_BARYON.  Gas starts at rest (rest frame).
    #   :uniform            — δb=0 (set via uniform_baryons).
    mode = uniform_baryons ? :uniform : baryon_ic
    if mode == :smooth
        ENV["CICASS_SMOOTH_BARYON"] = "1"          # honored by makeCosICs/main.c OUTPUT_CAPI
    else
        delete!(ENV, "CICASS_SMOOTH_BARYON")
    end

    # 1) CICASS streaming realization (ngrid³)
    spec = CICASSSpec(boxlength = boxlength, zstart = zstart, ngrid = ngrid,
                      vbc = vbc, Omega_m = omega_m, filename = "cic_enzo")
    res = CICASSLib.generate(spec; workdir = workdir)
    snap = CICASSLib.read_snapshot(res.output)
    N = snap.n

    ob   = snap.omega_b
    ocdm = max(omega_m - ob, 0.0)
    hh   = snap.hconst
    # Radiation EXACTLY as CICASS (vbc_transfer/main.cc:170): OmegaR = 4.15e-5/h²
    # (T_cmb=2.726 K, 3 relativistic neutrino species).  CICASS evolves a FLAT
    # background with radiation, H²/H0² = (1−Ωm−Ωr) + Ωm(1+z)³ + Ωr(1+z)⁴
    # (main.cc:918); Ω_r/Ω_m ≈ 0.27 at z=900.  Keep flat: ΩΛ = 1−Ωm−Ωr.
    orad = 4.15e-5 / hh^2
    ol   = 1.0 - omega_m - orad
    conv = _enzo_vunit_kms(boxlength, omega_m, zstart)       # km/s per Enzo velocity unit

    # 2) Translate the CICASS realization into NATIVE Enzo HDF5 IC files (a fully
    #    self-contained CosmologySimulation; no external host template).  Everything
    #    in Enzo code units: baryon density = the
    #    cosmological baryon fraction Ωb/Ωm × (1+δb) (so the GravitatingMassField sums
    #    to 1); gas velocity = gas_vel/conv; DM particle pos = box-fraction, vel/conv.
    #    An HDF5.jl SUBPROCESS writes the HDF5 (libenzo + HDF5.jl cannot co-reside);
    #    Enzo then initializes directly from these — no post-init bridge injection.
    # Optionally boost into the BARYON REST FRAME: subtract the coherent gas bulk
    # velocity (mean of gas_vel, = the streaming flow) from BOTH the gas field and the
    # DM particles.  This zeroes mean(gas_vel) while PRESERVING the relative streaming
    # offset mean(gas_vel−dm_vel) — a Galilean boost.  Growth depends only on the
    # relative velocity, so a correct code must give identical baryon+DM growth.
    # `uniform_baryons` = the PHYSICAL post-recombination start: the baryons are
    # Silk-damped to a UNIFORM density (δb=0) and at rest in the CMB frame (v_b=0),
    # while the DM carries its structure and the coherent streaming (−v_bc).  Combined
    # with the per-cycle Compton-drag operator in the driver, this lets the baryon
    # fluctuations GROW physically from DM gravity (instead of seeding the shot-noise-
    # dominated z~990 CICASS baryon field).  It forces the baryon rest frame (boosts DM).
    # The :smooth/:uniform "physical recombination start" runs in EITHER Galilean frame
    # (zero_baryon_bulk picks it):
    #   BOOSTED (zero_baryon_bulk=true) — CMB/baryon REST frame: gas at rest (v=0), DM
    #     streams at −v_bc.  The Compton drag damps v_b → 0 (v_cmb = 0).
    #   UNBOOSTED (false) — standard CICASS frame: gas streams at +v_bc (= the CMB bulk,
    #     baryons were locked to the CMB), DM at rest.  The drag damps v_b → +v_bc.
    # Both share the SAME relative gas−DM streaming, so a Galilean-correct code must agree.
    # `v_cmb` (Enzo code velocity) is the drag target returned for the driver.
    physical = mode == :smooth || mode == :uniform
    boosted  = zero_baryon_bulk
    gas_vel = snap.gas_vel; dm_vel = snap.dm_vel
    gbulk = (0.0, 0.0, 0.0); v_cmb = (0.0, 0.0, 0.0)
    if physical
        gbulk = ntuple(d -> _mean(@view snap.gas_vel[:, d]), 3)       # the coherent v_bc (km/s)
        if boosted
            gas_vel = zero(snap.gas_vel)                              # gas at rest (CMB frame)
            dm_vel  = snap.dm_vel .- reshape(collect(gbulk), 1, 3)    # DM streams at −v_bc
            v_cmb   = (0.0, 0.0, 0.0)
            @info "run_cicass_enzo: physical :$mode BOOSTED (gas rest, DM stream −$(round.(gbulk,digits=3)) km/s)"
        else
            gas_vel = repeat(reshape(collect(gbulk), 1, 3), size(snap.gas_vel, 1), 1)  # gas streams at CMB vel
            dm_vel  = snap.dm_vel                                     # DM at rest
            v_cmb   = gbulk ./ conv                                   # drag target (code units)
            @info "run_cicass_enzo: physical :$mode UNBOOSTED (gas stream +$(round.(gbulk,digits=3)) km/s, DM rest)"
        end
    elseif zero_baryon_bulk                                           # legacy :particle Galilean boost
        gbulk = ntuple(d -> _mean(@view snap.gas_vel[:, d]), 3)
        gas_vel = snap.gas_vel .- reshape(collect(gbulk), 1, 3)
        dm_vel  = snap.dm_vel  .- reshape(collect(gbulk), 1, 3)
        @info "run_cicass_enzo: baryon rest frame; subtracted gas bulk = $(round.(gbulk, digits=4)) km/s"
    end

    # baryon density grid (Enzo code units, mean = Ωb/Ωm):
    #   :uniform → flat;  :smooth → (Ωb/Ωm)(1+δb_Fourier);  :particle → (Ωb/Ωm)(1+δb_CIC)
    edir = joinpath(workdir, "enzo"); mkpath(edir)
    dens0 = mode == :uniform ? fill(ob / omega_m, length(snap.gas_delta)) :
                               (ob / omega_m) .* (1.0 .+ snap.gas_delta)
    @info "run_cicass_enzo: baryon IC = :$mode (δb rms = $(round(_rms(snap.gas_delta), sigdigits=3)))" *
          (mode == :uniform ? "  [uniform]" : "")
    write(joinpath(edir, "density.f32"), Float32.(dens0))
    write(joinpath(edir, "gridvel.f32"), Float32.(vec(gas_vel ./ conv)))
    write(joinpath(edir, "partpos.f32"), Float32.(vec(snap.dm_pos)))
    write(joinpath(edir, "partvel.f32"), Float32.(vec(dm_vel ./ conv)))
    Np = size(snap.dm_pos, 1)
    writer = normpath(joinpath(@__DIR__, "..", "deps", "write_enzo_cicass_ic.jl"))
    run(`$(Base.julia_cmd()) --project=$(Base.active_project()) $writer $edir $N $Np`)

    # Gas temperature [K].  Caller passes init_temperature (e.g. the RECFAST T_gas at
    # zstart); else default to the CICASS snapshot mean (or T_cmb).  Enzo's CosmologySim
    # InitialTemperature path converts T→internal-energy with μ≈0.6 (IONIZED), but the
    # post-recombination gas is NEUTRAL (μ≈1.22), so it would start ~2× too hot.  We
    # therefore OWN the gas energy explicitly below (after boot) using the SPECIFIED
    # μ=`mu_init`, and the param value here is only a placeholder Enzo overwrites.
    Tinit = init_temperature > 0 ? float(init_temperature) :
            (isdefined(snap, :tavg) ? float(snap.tavg) : 2.726 * (1 + zstart))
    par = """
    ProblemType                = 30
    TopGridRank                = 3
    TopGridDimensions          = $N $N $N
    SelfGravity                = 1
    TopGridGravityBoundary     = 0
    LeftFaceBoundaryCondition  = 3 3 3
    RightFaceBoundaryCondition = 3 3 3
    ComovingCoordinates        = 1
    GravitationalConstant      = 1
    CosmologySimulationOmegaBaryonNow       = $ob
    CosmologySimulationOmegaCDMNow          = $ocdm
    CosmologySimulationDensityName          = GridDensity
    CosmologySimulationVelocity1Name        = GridVelocities
    CosmologySimulationVelocity2Name        = GridVelocities
    CosmologySimulationVelocity3Name        = GridVelocities
    CosmologySimulationParticlePositionName = ParticlePositions
    CosmologySimulationParticleVelocityName = ParticleVelocities
    CosmologySimulationInitialTemperature   = $Tinit
    CosmologyOmegaMatterNow    = $omega_m
    CosmologyOmegaLambdaNow    = $ol
    CosmologyOmegaRadiationNow = $orad
    CosmologyHubbleConstantNow = $hh
    CosmologyComovingBoxSize   = $boxlength
    CosmologyInitialRedshift   = $zstart
    CosmologyFinalRedshift     = 0
    CosmologyMaxExpansionRate  = 0.015
    HydroMethod                = 0
    Gamma                      = 1.6667
    DualEnergyFormalism        = 1
    PPMDiffusionParameter      = 0
    InterpolationMethod        = 1
    CourantSafetyNumber        = 0.5
    ParticleCourantSafetyNumber = 0.8
    RadiativeCooling           = 0
    MultiSpecies               = 0
    StaticHierarchy            = 1
    MaximumRefinementLevel     = 0
    """
    isempty(param_extra) || (par *= "\n" * param_extra * "\n")   # driver appends chem/dt overrides
    pf = joinpath(edir, "cic.enzo"); write(pf, par)

    return cd(edir) do
        h = EnzoLib.session_init(pf)
        h == C_NULL && error("session_init failed on the native CICASS ICs")
        try
            dims = EnzoLib.problem_grid_dims(h, 0)
            ghost = (dims[1] - N) ÷ 2
            act = ntuple(d -> (ghost + 1):(ghost + N), 3)
            np = EnzoLib.problem_num_particles(h, 0)
            np == N^3 || error("Enzo native ICs gave $np particles, expected $(N^3)")

            # ── EXPLICIT gas internal energy for T = `Tinit` at μ = `mu_init` (NEUTRAL).
            # Overrides Enzo's μ≈0.6 InitialTemperature conversion (which would leave the
            # gas ~2× too hot → wrong pressure/Jeans scale AND chemistry).  Specific
            # internal energy e = kB·T / ((γ−1)·μ·m_H); GasEnergy[code] = e / VelocityUnits²
            # (VU = conv·1e5 cm/s, constant at z_init).  TotalEnergy = GasEnergy + ½v².
            let γ = 5/3, kB = 1.380649e-16, mh = 1.6726e-24, VUcgs = conv * 1e5
                eint_code = kB * Tinit / ((γ - 1) * mu_init * mh) / VUcgs^2
                iGE = EnzoLib.field_index(h, 2; grid = 0)        # GasEnergy (DualEnergyFormalism)
                iTE = EnzoLib.field_index(h, 1; grid = 0)        # TotalEnergy
                nf  = length(EnzoLib.problem_get_field(h, iGE, 0))
                v1 = EnzoLib.problem_get_field(h, EnzoLib.field_index(h, 4; grid = 0), 0)
                v2 = EnzoLib.problem_get_field(h, EnzoLib.field_index(h, 5; grid = 0), 0)
                v3 = EnzoLib.problem_get_field(h, EnzoLib.field_index(h, 6; grid = 0), 0)
                ge = fill(eint_code, nf)
                EnzoLib.problem_set_field(h, iGE, ge; grid = 0)
                EnzoLib.problem_set_field(h, iTE, ge .+ 0.5 .* (v1.^2 .+ v2.^2 .+ v3.^2); grid = 0)
                @info "run_cicass_enzo: gas internal energy set for T=$(round(Tinit,digits=1)) K, μ=$mu_init (neutral; avoids the 2× μ=0.6 over-heat)"
            end

            # read the bulk offset back out of Enzo's live structures (native ICs;
            # no injection — the HDF5 files already carry the CICASS data)
            gasmean = ntuple(d -> begin
                fi = EnzoLib.field_index(h, 4 + (d - 1); grid = 0)
                f = reshape(EnzoLib.problem_get_field(h, fi, 0), dims...)
                sum(@view f[act...]) / N^3 * conv
            end, 3)
            dmmean = ntuple(d -> sum(EnzoLib.problem_get_particle_vel(h, d - 1, 0)) / np * conv, 3)
            offset = ntuple(d -> gasmean[d] - dmmean[d], 3)
            return (; n = N, vbc = float(vbc), conv_kms = conv,
                    gas_mean = gasmean, dm_mean = dmmean, offset_enzo = offset,
                    offset_snapshot = CICASSLib.streaming_velocity(snap),
                    expected = vbc * (1 + zstart) / 1001.0,
                    baryon_mode = mode, physical = physical, boosted = boosted,
                    v_cmb = v_cmb,                       # drag target (Enzo code velocity)
                    handle = h, edir = edir, dims = dims, act = act, ghost = ghost,
                    boxlength = float(boxlength), zstart = float(zstart),
                    omega_m = float(omega_m), snap = snap, workdir = workdir,
                    pk_file = joinpath(workdir, "cic_enzo.pk"),
                    free = () -> EnzoLib.free_problem(h))
        catch
            EnzoLib.free_problem(h)
            rethrow()
        end
    end
end

function _cicass_ramses_namelist(n::Integer, level::Integer; courant::Real = 0.8,
                                 omega_b::Real = 0.046)
    return """
    CICASS streaming ICs, hydro+cosmo (generated by MultiCodeCICASSExt)

    &RUN_PARAMS
    cosmo=.true.
    pic=.true.
    poisson=.true.
    hydro=.true.
    nrestart=0
    nremap=0
    nsubcycle=10*1
    nstepmax=100000
    ncontrol=1
    verbose=.false.
    /

    &AMR_PARAMS
    levelmin=$(level)
    levelmax=$(level)
    ngridtot=5000000
    ncachemax=400000
    npartmax=$(2 * n^3)
    nexpand=1
    boxlen=1.0
    /

    &INIT_PARAMS
    filetype='grafic'
    initfile(1)='ics'
    omega_b=$(omega_b)
    /

    &HYDRO_PARAMS
    gamma=1.6667
    courant_factor=$(courant)
    slope_type=1
    $(get(ENV, "CIC_DUAL_ENERGY", "1") == "1" ? "entropy=.true.\n    dual_energy=1d-3\n    T2_fix=1d9" : "")
    /

    &OUTPUT_PARAMS
    foutput=0
    aend=1.0
    /

    &POISSON_PARAMS
    epsilon=1.d-5
    /

    &REFINE_PARAMS
    m_refine=10*1.d30
    /
    """
end

function MultiCode.run_cicass_ramses(; vbc::Real = 30.0, boxlength::Real = 0.2,
                                     zstart::Real = 100.0, omega_m::Real = 0.27,
                                     courant::Real = 0.8, zero_baryon_bulk::Bool = false,
                                     uniform_baryons::Bool = false, baryon_ic::Symbol = :particle,
                                     workdir::AbstractString = mktempdir())
    CICASSLib.available() || error("libcicass_capi not found")
    CodeBridge.available(RamsesLib.BRIDGE, :cosmo) ||
        error("RAMSES cosmo library not found (bin64sc)")

    # SAME baryon-IC modes as run_cicass_enzo: :particle (default), :smooth (raw CAMB/CLASS
    # Fourier δb grid), :uniform (δb=0).  The :smooth/:uniform "physical recombination
    # start" modes put the baryons at rest (gas v=0) with the DM carrying structure + the
    # −v_bc stream, to pair with the driver's Compton-drag operator.
    mode = uniform_baryons ? :uniform : baryon_ic
    if mode == :smooth
        ENV["CICASS_SMOOTH_BARYON"] = "1"
    else
        delete!(ENV, "CICASS_SMOOTH_BARYON")
    end
    physical = mode == :smooth || mode == :uniform

    spec = CICASSSpec(boxlength = boxlength, zstart = zstart, ngrid = 128,
                      vbc = vbc, Omega_m = omega_m, filename = "cic_ramses")
    res = CICASSLib.generate(spec; workdir = workdir)
    snap = CICASSLib.read_snapshot(res.output)
    n = snap.n
    level = round(Int, log2(n))

    # Optionally boost into the BARYON REST FRAME: subtract the coherent gas bulk
    # velocity (mean of gas_vel, = the streaming flow) from BOTH species, leaving the
    # relative gas−DM streaming offset unchanged (a Galilean shift).  The gas is
    # boosted by the driver's inject_gas_velocity! using `gas_bulk_boost` (returned).
    # The DM is boosted POST-init via RamsesLib.boost_particles! — NOT through grafic:
    # RAMSES's grafic reader strips the mean (DC) velocity, so a bulk DM stream set in
    # ic_velc is silently discarded (DM ends at rest → relative offset collapses to 0
    # → no streaming).  boost_particles! imposes −gbulk directly on the live particles.
    # Frame choice (Galilean): BOOSTED (zero_baryon_bulk) = CMB/baryon rest frame, gas at
    # rest + DM streams −v_bc, drag→0.  UNBOOSTED = standard CICASS frame, gas streams +v_bc
    # (the CMB bulk) + DM at rest, drag→+v_bc.  Both have the SAME relative streaming.
    boosted = zero_baryon_bulk
    boost = physical ? boosted : zero_baryon_bulk
    gbulk = (physical || zero_baryon_bulk) ? ntuple(d -> _mean(@view snap.gas_vel[:, d]), 3) : (0.0, 0.0, 0.0)
    @info "run_cicass_ramses: baryon IC = :$mode, frame = $(boosted ? "BOOSTED (gas rest, DM stream)" : "UNBOOSTED (gas stream, DM rest)"); v_bc = $(round.(gbulk, digits=4)) km/s"

    # boot RAMSES on the grafic set: gas from ic_deltab/ic_velb, DM positions from
    # ic_posc (the EXACT CICASS displacement — identical to Enzo, bypassing RAMSES's
    # radiation-free Zeldovich reconstruction) + DM velocities from ic_velc.  For
    # :uniform we write a flat ic_deltab (δb=0); :smooth gets the smooth CICASS δb above.
    bδ = mode == :uniform ? zeros(eltype(snap.gas_delta), length(snap.gas_delta)) : nothing
    MultiCode.write_grafic_streaming(joinpath(workdir, "ics"), snap; baryon_delta = bδ)
    # Pass Ω_b explicitly — the grafic header carries Ω_m/Ω_Λ/h but NOT Ω_b, so RAMSES
    # would otherwise default to 0.045 (≠ CICASS/Enzo's 0.046), giving a ~2% baryon-density
    # (n_H) mismatch that shows up as a systematic x_HII/T offset vs Enzo.
    write(joinpath(workdir, "cic.nml"),
          _cicass_ramses_namelist(n, level; courant = courant, omega_b = snap.omega_b))

    dv_kms = collect(CICASSLib.streaming_velocity(snap))   # [≈3.027, 0, 0] km/s

    return cd(workdir) do
        h = RamsesLib.init("cic.nml"; lib = :cosmo)
        lev = RamsesLib.info(h; lib = :cosmo).levelmin
        try
            # mini-ramses sets the gas velocity from ic_velc (CDM), so the
            # streaming offset is absent after init. It is COHERENT across the box
            # (Tseliakhovich–Hirata), so we inject it as a uniform gas-velocity
            # boost via set_hydro! — exactly the right physics for streaming.
            #
            # RAMSES supercomoving velocity unit, derived empirically: the DM
            # particles were built from ic_velc (km/s) → code-unit vp, so
            # unit_v[km/s] = rms(dm_vel[km/s]) / rms(vp[code]).
            p = RamsesLib.get_particles(h, n^3; lib = :cosmo)
            unit_v = _rms(snap.dm_vel) / _rms(p.vp)

            # In the BOOSTED frame the DM must stream at −v_bc — but grafic strips the
            # mean velocity, so impose it directly on the live particles post-init.
            if boost
                RamsesLib.boost_particles!(h, -gbulk[1] / unit_v, -gbulk[2] / unit_v,
                                           -gbulk[3] / unit_v; lib = :cosmo)
                pb = RamsesLib.get_particles(h, n^3; lib = :cosmo)
                dmb = ntuple(d -> sum(@view pb.vp[:, d]) / size(pb.vp, 1) * unit_v, 3)
                @info "run_cicass_ramses: DM bulk after boost_particles! = $(round.(dmb, digits=3)) km/s (want $(round.(.-gbulk, digits=3)))"
            end

            ck, dens = RamsesLib.get_hydro(h, :uold, 1, lev; lib = :cosmo)
            v_cmb = (0.0, 0.0, 0.0)
            if physical
                # gas target velocity (code units): 0 in the boosted frame, +v_bc/unit_v
                # in the unboosted frame (the CMB bulk).  Set the momentum AND fix uold(5)
                # so the total energy keeps the (unchanged) internal energy + the new KE.
                gtarget = boosted ? (0.0, 0.0, 0.0) : ntuple(d -> gbulk[d] / unit_v, 3)
                v_cmb = gtarget
                _, E = RamsesLib.get_hydro(h, :uold, 5, lev; lib = :cosmo)
                for d in 1:3
                    _, mom = RamsesLib.get_hydro(h, :uold, 1 + d, lev; lib = :cosmo)
                    newmom = dens .* gtarget[d]
                    # E += ½(new_mom² − old_mom²)/ρ  (internal energy unchanged)
                    E .+= 0.5 .* (newmom .^ 2 .- mom .^ 2) ./ max.(dens, eps())
                    RamsesLib.set_hydro!(h, :uold, 1 + d, lev, ck, newmom; lib = :cosmo)
                end
                RamsesLib.set_hydro!(h, :uold, 5, lev, ck, E; lib = :cosmo)
            else
                # Coherent streaming boost (gas streams at v_bc in the unboosted frame).
                # The bulk velocity is NOT tiny (~30 km/s) — its kinetic energy is
                # comparable to / exceeds the thermal energy at z~1000 — so uold(5)=E_tot
                # MUST be updated with the added KE (else e_int goes negative → floored).
                _, E = RamsesLib.get_hydro(h, :uold, 5, lev; lib = :cosmo)
                for d in 1:3
                    Δ = dv_kms[d] - gbulk[d]                 # streaming, less the frame boost
                    abs(Δ) < 1e-9 && continue
                    _, mom = RamsesLib.get_hydro(h, :uold, 1 + d, lev; lib = :cosmo)
                    newmom = mom .+ dens .* (Δ / unit_v)     # ρu += ρ·Δv (coherent boost)
                    E .+= 0.5 .* (newmom .^ 2 .- mom .^ 2) ./ max.(dens, eps())  # add bulk KE
                    RamsesLib.set_hydro!(h, :uold, 1 + d, lev, ck, newmom; lib = :cosmo)
                end
                RamsesLib.set_hydro!(h, :uold, 5, lev, ck, E; lib = :cosmo)
            end

            # read the bulk offset back out of RAMSES (mass-weighted), → km/s
            sden = sum(dens)
            gas = ntuple(d -> begin
                _, mom = RamsesLib.get_hydro(h, :uold, 1 + d, lev; lib = :cosmo)
                sum(mom) / sden * unit_v
            end, 3)
            p2 = RamsesLib.get_particles(h, n^3; lib = :cosmo)
            dm = ntuple(d -> sum(@view p2.vp[:, d]) / size(p2.vp, 1) * unit_v, 3)
            offset = ntuple(d -> gas[d] - dm[d], 3)
            return (; n, vbc = float(vbc), unit_v_kms = unit_v,
                    gas_bulk = gas, dm_bulk = dm, offset_ramses = offset,
                    offset_snapshot = Tuple(dv_kms), gas_bulk_boost = Tuple(gbulk),
                    baryon_mode = mode, physical = physical, boosted = boosted, v_cmb = v_cmb,
                    handle = h, lev = lev, workdir = workdir, snap = snap,
                    boxlength = float(boxlength), zstart = float(zstart),
                    free = () -> RamsesLib.finalize(h; lib = :cosmo))
        catch
            RamsesLib.finalize(h; lib = :cosmo)
            rethrow()
        end
    end
end

end # module
