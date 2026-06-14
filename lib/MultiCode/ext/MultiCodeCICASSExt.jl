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

# the 128³ hydro cosmology Enzo host (gas BaryonFields + DM particles)
const _SB_HOST = normpath(joinpath(@__DIR__, "..", "..", "..", "..",
                                   "run", "CosmologySimulation", "SantaBarbaraCluster"))

# Enzo cosmological velocity unit in km/s (= CICASS enzo_out.c setEnzoUnits OV/1e5);
# km/s ↔ Enzo code velocity for injecting/reading CICASS physical-peculiar km/s.
_enzo_vunit_kms(box, omega_m, z) = 1.22475e7 * box * sqrt(omega_m) * sqrt(1 + z) / 1e5

# CIC-deposit one per-particle velocity component onto an N³ grid (mass-weighted
# cell mean), in CICASS c-order idx = i + j*N + k*N² (i fastest). The bulk (k=0)
# mode — the streaming offset — is preserved exactly by the conservative deposit.
function _cic_deposit_velocity(pos::AbstractMatrix, vel::AbstractVector, n::Integer)
    N3 = n * n * n
    num = zeros(Float64, N3)        # Σ w·v
    den = zeros(Float64, N3)        # Σ w
    np = size(pos, 1)
    @inbounds for p in 1:np
        gx = pos[p, 1] * n; gy = pos[p, 2] * n; gz = pos[p, 3] * n
        i0 = floor(Int, gx); j0 = floor(Int, gy); k0 = floor(Int, gz)
        fx = gx - i0; fy = gy - j0; fz = gz - k0
        vp = vel[p]
        for (di, wx) in ((0, 1 - fx), (1, fx))
            ii = mod(i0 + di, n)
            for (dj, wy) in ((0, 1 - fy), (1, fy))
                jj = mod(j0 + dj, n)
                for (dk, wz) in ((0, 1 - fz), (1, fz))
                    kk = mod(k0 + dk, n)
                    idx = ii + jj * n + kk * n * n + 1
                    w = wx * wy * wz
                    num[idx] += w * vp
                    den[idx] += w
                end
            end
        end
    end
    @inbounds for i in 1:N3
        num[i] = den[i] > 0 ? num[i] / den[i] : 0.0
    end
    return num
end

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

function MultiCode.write_grafic_streaming(dir::AbstractString, snap::CICASSSnapshot; h0::Real = 71.0)
    mkpath(dir)
    n = snap.n
    a_i = 1 / (1 + snap.zinit)
    dx_mpc = (snap.box / snap.hconst) / n          # box [Mpc/h] → [Mpc] per cell
    om, ol = snap.omega_m, snap.omega_l
    # gas (baryon) velocity grids — already gridded by CICASS
    for (d, name) in zip(1:3, ("ic_velbx", "ic_velby", "ic_velbz"))
        _write_grafic_field(joinpath(dir, name), snap.gas_vel[:, d], n, dx_mpc, a_i, om, ol, h0)
    end
    # dark-matter velocity grids — CIC-deposit the DM particle velocities
    for (d, name) in zip(1:3, ("ic_velcx", "ic_velcy", "ic_velcz"))
        vc = _cic_deposit_velocity(snap.dm_pos, snap.dm_vel[:, d], n)
        _write_grafic_field(joinpath(dir, name), vc, n, dx_mpc, a_i, om, ol, h0)
    end
    _write_grafic_field(joinpath(dir, "ic_deltab"), snap.gas_delta, n, dx_mpc, a_i, om, ol, h0)
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
                                   workdir::AbstractString = mktempdir())
    CICASSLib.available() || error("libcicass_capi not found")
    EnzoLib.grid_available() || error("Enzo grid bridge not built")
    isdir(_SB_HOST) || error("SantaBarbaraCluster (128³ hydro) host not found at $_SB_HOST")

    # 1) CICASS streaming realization (ngrid³)
    spec = CICASSSpec(boxlength = boxlength, zstart = zstart, ngrid = ngrid,
                      vbc = vbc, Omega_m = omega_m, filename = "cic_enzo")
    res = CICASSLib.generate(spec; workdir = workdir)
    snap = CICASSLib.read_snapshot(res.output)
    N = snap.n

    # 2) boot Enzo on the SB 128³ hydro host, patched to the CICASS cosmology so
    #    Enzo's velocity units match what CICASS realized (the IC field files are
    #    overwritten below — only the grid structure + cosmology matter here).
    edir = joinpath(workdir, "enzo"); mkpath(edir)
    for f in readdir(_SB_HOST)
        p = joinpath(_SB_HOST, f)
        isfile(p) && cp(p, joinpath(edir, f); force = true)
    end
    par = read(joinpath(_SB_HOST, "SantaBarbaraCluster.enzo"), String)
    # patch the FULL cosmology self-consistently from the CICASS realization, so
    # Enzo derives correct length/density/velocity units AND particle masses
    # (∝ ΩCDM/ΩMatter).  The SB host is EdS Ωm=1, ΩCDM=0.9, h=0.5 — leaving those
    # stale while only setting ΩMatter gives ΩCDM>ΩMatter → DM masses (hence gravity)
    # several× too large.  CICASS cosmology: Ωm, Ωb, h from the snapshot header.
    ol  = 1.0 - omega_m
    ob  = snap.omega_b
    ocdm = max(omega_m - ob, 0.0)
    hh  = snap.hconst
    for (pat, rep) in (
            r"CosmologyOmegaMatterNow\s*=\s*\S+" => "CosmologyOmegaMatterNow    = $(omega_m)",
            r"CosmologyOmegaLambdaNow\s*=\s*\S+" => "CosmologyOmegaLambdaNow    = $(ol)",
            r"CosmologyHubbleConstantNow\s*=\s*\S+" => "CosmologyHubbleConstantNow = $(hh)",
            r"CosmologySimulationOmegaBaryonNow\s*=\s*\S+" => "CosmologySimulationOmegaBaryonNow = $(ob)",
            r"CosmologySimulationOmegaCDMNow\s*=\s*\S+" => "CosmologySimulationOmegaCDMNow = $(ocdm)",
            r"CosmologyComovingBoxSize\s*=\s*\S+" => "CosmologyComovingBoxSize   = $(boxlength)",
            r"CosmologyInitialRedshift\s*=\s*\S+" => "CosmologyInitialRedshift   = $(zstart)")
        par = replace(par, pat => rep)
    end
    par *= "\nStaticHierarchy = 1\nMaximumRefinementLevel = 0\n"
    isempty(param_extra) || (par *= "\n" * param_extra * "\n")
    pf = joinpath(edir, "cic.enzo"); write(pf, par)

    conv = _enzo_vunit_kms(boxlength, omega_m, zstart)       # km/s per Enzo velocity unit

    return cd(edir) do
        h = EnzoLib.session_init(pf)
        h == C_NULL && error("session_init failed on the CICASS-patched SB host")
        try
            dims = EnzoLib.problem_grid_dims(h, 0)
            ghost = (dims[1] - N) ÷ 2
            act = ntuple(d -> (ghost + 1):(ghost + N), 3)
            np = EnzoLib.problem_num_particles(h, 0)
            np == N^3 || error("Enzo host has $np particles, expected $(N^3)")

            # gas velocity field (FieldType 4/5/6): read full (with ghosts),
            # overwrite the active region (CICASS c-order ≡ Julia col-major,
            # i fastest), write back flat.
            for d in 0:2
                fi = EnzoLib.field_index(h, 4 + d; grid = 0)
                full = reshape(EnzoLib.problem_get_field(h, fi, 0), dims...)
                full[act...] = reshape(snap.gas_vel[:, d + 1] ./ conv, N, N, N)
                EnzoLib.problem_set_field(h, fi, vec(full); grid = 0)
            end
            # DM particles (positions box-fraction, velocities → Enzo units)
            for d in 0:2
                EnzoLib.problem_set_particle_pos(h, d, snap.dm_pos[:, d + 1])
                EnzoLib.problem_set_particle_vel(h, d, snap.dm_vel[:, d + 1] ./ conv)
            end

            # read the bulk offset back out of Enzo's live structures
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

function _cicass_ramses_namelist(n::Integer, level::Integer; courant::Real = 0.8)
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
    /

    &HYDRO_PARAMS
    gamma=1.6667
    courant_factor=$(courant)
    slope_type=1
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
                                     courant::Real = 0.8,
                                     workdir::AbstractString = mktempdir())
    CICASSLib.available() || error("libcicass_capi not found")
    CodeBridge.available(RamsesLib.BRIDGE, :cosmo) ||
        error("RAMSES cosmo library not found (bin64sc)")

    spec = CICASSSpec(boxlength = boxlength, zstart = zstart, ngrid = 128,
                      vbc = vbc, Omega_m = omega_m, filename = "cic_ramses")
    res = CICASSLib.generate(spec; workdir = workdir)
    snap = CICASSLib.read_snapshot(res.output)
    n = snap.n
    level = round(Int, log2(n))

    # boot RAMSES purely on the grafic streaming set: gas from ic_deltab/ic_velb,
    # DM particles from ic_velc — both on the same (grafic) unit convention.
    MultiCode.write_grafic_streaming(joinpath(workdir, "ics"), snap)
    write(joinpath(workdir, "cic.nml"), _cicass_ramses_namelist(n, level; courant = courant))

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

            ck, dens = RamsesLib.get_hydro(h, :uold, 1, lev; lib = :cosmo)
            for d in 1:3
                abs(dv_kms[d]) < 1e-9 && continue
                _, mom = RamsesLib.get_hydro(h, :uold, 1 + d, lev; lib = :cosmo)
                mom .+= dens .* (dv_kms[d] / unit_v)        # ρu += ρ·Δv (coherent boost)
                RamsesLib.set_hydro!(h, :uold, 1 + d, lev, ck, mom; lib = :cosmo)
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
                    offset_snapshot = Tuple(dv_kms),
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
