# ── reduced-chemistry guest slot for RAMSES / Arepo ───────────────────────────
#
# Wires the code-neutral GrackleChem service (grackle_service.jl) onto a host
# code's gas state so it can run early-universe primordial chemistry advecting
# only TWO species, HII and H2I.  The host carries those two as density-weighted
# passive scalars (mass density rho*x), exactly the GrackleChem convention.
#
#   chem_init!(...)                 -- initialise the service for the host units
#   chem_step!(rho,eint,HII,H2I;..) -- generic core (the testable seam)
#   ramses_chem_step!(h,lev;..)     -- extract/inject for a RAMSES host
#
# Arepo uses the same chem_step! core once ArepoLib exposes the passive scalars
# (see the note in arepo_chem_step! below).

using .GrackleChem

const _CHEM_DATA_FILE = Ref{String}(
    joinpath(homedir(), "Research", "codes", "grackle", "input", "CloudyData_noUVB.h5"))

# Cosmology+units captured at chem_init! so the stateless ChemistryKernels engine
# (engine=:kernels) can be called per step with the same configuration grackle
# was initialised with.  The default :grackle engine ignores this.
const _CHEM_CFG = Ref{NamedTuple}((; hubble=71.0, Om=0.27, OL=0.73, fh=0.76,
    density_units=1.0, length_units=1.0, time_units=1.0, deuterium=false,
    hubble_expansion=false))

"""
    chem_init!(; hubble, Om, OL, a_value, fh, density_units, length_units,
                 time_units, data_file=<CloudyData_noUVB.h5>)

Initialise the reduced primordial-chemistry service for a host's code units
(`*_units` convert host code units → CGS; `hubble` = H0 in km/s/Mpc).
"""
function chem_init!(; hubble::Real, Om::Real, OL::Real, a_value::Real, fh::Real=0.76,
        density_units::Real, length_units::Real, time_units::Real,
        data_file::AbstractString=_CHEM_DATA_FILE[], deuterium::Bool=false,
        engine::Symbol=:grackle, hubble_expansion::Bool=false)
    _CHEM_CFG[] = (; hubble=Float64(hubble), Om=Float64(Om), OL=Float64(OL),
        fh=Float64(fh), density_units=Float64(density_units),
        length_units=Float64(length_units), time_units=Float64(time_units),
        deuterium=deuterium, hubble_expansion=hubble_expansion)
    # The native ChemistryKernels engine is pure Julia — no Grackle dylib / data
    # file / subprocess worker needed.  Only init the C reduced lib for :grackle.
    engine === :kernels && return nothing
    GrackleChem.grackle_reduced_init!(; hubble=hubble, Om=Om, OL=OL, a_value=a_value,
        fh=fh, density_units=density_units, length_units=length_units,
        time_units=time_units, data_file=data_file, deuterium=deuterium)
end

"""
    chem_step!(rho, eint, HII, H2I, [HDI]; a_value, dt, engine=:grackle,
               backend=:cpu, precision=Float64)

Advance the chemistry+cooling one step.  `eint` (specific internal energy),
`HII`, `H2I` (mass densities rho*x) are updated in place.  This is the code-
neutral core both RAMSES and Arepo call after extracting their gas state.

`engine=:grackle` (default, byte-unchanged) calls the live Grackle reduced lib;
`engine=:kernels` calls the native, device-agnostic `ChemistryKernels.solve_chem!`
(the table-free KA port — same v2026 reduced model, sub-percent agreement) using
the cosmology+units captured by `chem_init!`, on `backend` at `precision`.
"""
function chem_step!(rho, eint, HII, H2I, HDI=nothing; a_value, dt,
                    engine::Symbol=:grackle, backend::Symbol=:cpu,
                    precision::Type=Float64, adot_over_a::Real=NaN)
    if engine === :grackle
        GrackleChem.grackle_reduced_step!(rho, eint, HII, H2I, HDI; a_value=a_value, dt=dt)
    elseif engine === :kernels
        cfg = _CHEM_CFG[]
        ChemistryKernels.solve_chem!(rho, eint, HII, H2I, HDI;
            a_value=a_value, dt=dt, density_units=cfg.density_units,
            length_units=cfg.length_units, time_units=cfg.time_units,
            hubble=cfg.hubble, Om=cfg.Om, OL=cfg.OL, fh=cfg.fh,
            deuterium=cfg.deuterium && HDI !== nothing,
            hubble_expansion=cfg.hubble_expansion, adot_over_a=adot_over_a,
            backend=backend, precision=precision)
    else
        error("unknown chem engine :$engine (use :grackle or :kernels)")
    end
end

# ── RAMSES wiring ─────────────────────────────────────────────────────────────
# RAMSES stores uold = (rho, rho*u, E_total, [passive scalars rho*x ...]).  The
# two chemistry species live at hydro var indices `iHII`, `iH2I` (density-
# weighted).  Requires a RAMSES built with nvar >= max(iHII,iH2I) (e.g. -DNVAR=7
# with iHII=6, iH2I=7); RamsesLib.get_hydro/set_hydro already handle any ivar.

"""
    ramses_chem_step!(h, lev; dt, a_value, density_units, length_units,
                      time_units, iHII=6, iH2I=7)

Run one reduced-chemistry step on a RAMSES level: pull (rho, momentum, E_total)
and the two species via `RamsesLib.get_hydro_all`, form the specific internal
energy, call `chem_step!`, and write back E_total and the two species via
`RamsesLib.set_hydro!`.  `chem_init!` must have been called first.
"""
function ramses_chem_step!(h, lev::Integer; dt::Real, a_value::Real,
        density_units::Real, length_units::Real, time_units::Real,
        iHII::Integer=6, iH2I::Integer=7, iHDI::Union{Nothing,Integer}=nothing,
        ientropy::Union{Nothing,Integer}=nothing,
        lib::Symbol=:cpu, engine::Symbol=:grackle, backend::Symbol=:cpu,
        precision::Type=Float64)
    _prof = get(ENV, "CIC_CHEMPROF", "0") == "1"
    _t0 = _prof ? time() : 0.0
    ck, U = RamsesLib.get_hydro_all(h, :uold, lev; lib=lib)   # U :: noct × 8 × nvar
    _prof && (println("    [chemprof] get_hydro_all = ", round(time()-_t0,digits=3), "s"); flush(stdout))
    nv = size(U, 3)
    need = iHDI === nothing ? max(iHII, iH2I) : max(iHII, iH2I, iHDI)
    nv >= need ||
        error("RAMSES nvar=$nv < $need; rebuild with -DNPSCAL>=$(need-5) to carry the species")

    rho  = Float64.(vec(@view U[:, :, 1]))
    mx   = Float64.(vec(@view U[:, :, 2]))
    my   = Float64.(vec(@view U[:, :, 3]))
    mz   = Float64.(vec(@view U[:, :, 4]))
    Etot = Float64.(vec(@view U[:, :, 5]))
    HII  = Float64.(vec(@view U[:, :, iHII]))
    H2I  = Float64.(vec(@view U[:, :, iH2I]))
    HDI  = iHDI === nothing ? nothing : Float64.(vec(@view U[:, :, iHDI]))

    r    = max.(rho, eps())
    kin  = 0.5 .* (mx.^2 .+ my.^2 .+ mz.^2) ./ r          # kinetic energy density
    eint = (Etot .- kin) ./ r                             # specific internal energy

    # Robustness: boundary/empty octs (ρ≈0) make eint blow up to Inf/NaN, which
    # drives Grackle's temperature-table index out of bounds → segfault.  Clamp
    # ρ, eint and the species to physical ranges so such cells pass through inert.
    # eint is clamped to a PHYSICAL temperature window [1, 1e8] K (Grackle's table
    # spans ~1–1e9 K; absurd T from numerics indexes past the table → crash).
    mh = 1.6726e-24; kB = 1.380649e-16; velu = length_units/time_units
    Tunits = mh*velu^2/kB; ec = 1.0/(Tunits*(5/3-1)*1.22)   # code eint per Kelvin (μ≈1.22)
    rfloor = maximum(rho) * 1e-20 + eps()
    emin, emax = 1.0*ec, 1e8*ec                           # T ∈ [1, 1e8] K
    @inbounds for i in eachindex(rho)
        rho[i] = (isfinite(rho[i]) && rho[i] > rfloor) ? rho[i] : rfloor
        eint[i] = (isfinite(eint[i]) && eint[i] > emin) ? min(eint[i], emax) : emin
        HII[i]  = isfinite(HII[i]) ? clamp(HII[i], 0.0, rho[i]) : 0.0
        H2I[i]  = isfinite(H2I[i]) ? clamp(H2I[i], 0.0, rho[i]) : 0.0
        HDI === nothing || (HDI[i] = isfinite(HDI[i]) ? clamp(HDI[i], 0.0, rho[i]) : 0.0)
    end

    if get(ENV, "CHEM_DEBUG", "0") == "1"
        @info "chem_step extrema" rho=extrema(rho) eint=extrema(eint) HII=extrema(HII) H2I=extrema(H2I) HDI=(HDI===nothing ? nothing : extrema(HDI)) dt=dt a_value=a_value
        flush(stderr)
    end
    _t1 = _prof ? time() : 0.0
    chem_step!(rho, eint, HII, H2I, HDI; a_value=a_value, dt=dt,
               engine=engine, backend=backend, precision=precision)
    _prof && (println("    [chemprof] solve_chem!  = ", round(time()-_t1,digits=3), "s"); flush(stdout))

    _t2 = _prof ? time() : 0.0
    Etot_new = eint .* rho .+ kin                         # cooled internal + same kinetic
    noct = size(U, 1)
    reshape8(v) = reshape(v, noct, 8)
    RamsesLib.set_hydro!(h, :uold, 5,    lev, ck, reshape8(Etot_new); lib=lib)
    if ientropy !== nothing
        # Keep entropy consistent: s = eint*(γ-1)/ρ^(γ-1), matching RAMSES convention
        # e_prim = uold[ientropy]*ρ^(γ-1)/(γ-1)  →  uold[ientropy] = eint*(γ-1)/ρ^(γ-1)
        γm1 = 5/3 - 1
        entropy_new = eint .* γm1 ./ (r .^ γm1)
        RamsesLib.set_hydro!(h, :uold, ientropy, lev, ck, reshape8(entropy_new); lib=lib)
    end
    RamsesLib.set_hydro!(h, :uold, iHII, lev, ck, reshape8(HII); lib=lib)
    RamsesLib.set_hydro!(h, :uold, iH2I, lev, ck, reshape8(H2I); lib=lib)
    iHDI === nothing || RamsesLib.set_hydro!(h, :uold, iHDI, lev, ck, reshape8(HDI); lib=lib)
    _prof && (println("    [chemprof] set_hydro×N  = ", round(time()-_t2,digits=3), "s"); flush(stdout))
    # Push the chem-updated uold to the device (no-op on CPU/Metal; required on CUDA,
    # where the mesh is device-resident, so the GPU hydro sees the new energy+species).
    _t3 = _prof ? time() : 0.0
    RamsesLib.set_uold_device!(h; lib=lib)
    _prof && (println("    [chemprof] set_uold_dev = ", round(time()-_t3,digits=3), "s"); flush(stdout))
    return (; ncells = length(rho))
end

# ── Enzo wiring (cooling=:julia hook) ─────────────────────────────────────────
# Enzo runs COMOVING (ComovingCoordinates=1) and stores its reduced-chem fields
# as code-unit mass densities; GasEnergy (FieldType 2, DualEnergyFormalism=1) is
# the specific internal energy.  We convert to physical CGS using Enzo's exact
# CosmologyGetUnits.C scalings (DensityUnits, VelocityUnits, TimeUnits — the first
# z-dependent, the latter two constant at z_init), run the native ChemistryKernels
# engine, and convert back.  FieldType ints: Density=0, TotalEnergy=1, GasEnergy=2,
# HII=9, H2I=14, HDI=18.  Used as the `hooks[:cooling]` of an EngineConfig.

# Enzo CosmologyGetUnits.C: physical CGS scalings at redshift z (box in Mpc/h,
# hub = HubbleConstantNow = H0/100, zri = initial redshift).
function _enzo_units(z; Om, hub, box, zri)
    zp1 = 1 + z
    DensityUnits  = 1.8788e-29 * Om * hub^2 * zp1^3                 # ∝(1+z)³
    VelocityUnits = 1.22475e7 * box * sqrt(Om) * sqrt(1 + zri)      # constant
    TimeUnits     = 2.519445e17 / sqrt(Om) / hub / (1 + zri)^1.5    # constant
    return DensityUnits, VelocityUnits, TimeUnits
end

"""
    enzo_chem_step!(h, level, dt; Om, OL, hub, box, zri, fh=0.76, deuterium=true,
                    engine=:kernels, backend=:cpu, precision=Float64)

Cooling-slot hook: run one reduced chemistry+cooling step on Enzo's LIVE level-0
baryon fields with `ChemistryKernels.solve_chem!`.  Extracts Density/GasEnergy/
TotalEnergy/HII/H2I/HDI (flat, incl. ghosts), converts to physical CGS via Enzo's
cosmology units, evolves, and writes back GasEnergy+TotalEnergy and the species.
Pass as `hooks[:cooling] = (h,lev,dt)->enzo_chem_step!(h,lev,dt; …)` with
`cooling=:julia`.
"""
function enzo_chem_step!(h, level, dt; Om::Real, OL::Real, hub::Real, box::Real,
        zri::Real, fh::Real=0.76, deuterium::Bool=true, ng::Integer=3,
        engine::Symbol=:kernels, backend::Symbol=:cpu, precision::Type=Float64)
    z = EnzoLib.session_cosmology(h)[2]                       # z at step BEGIN
    DU, VU, TU = _enzo_units(z; Om=Om, hub=hub, box=box, zri=zri)
    VU2 = VU^2

    # Adiabatic-cooling rate for the coupled adiabatic+Compton kernel, sourced from
    # ENZO's OWN CosmologyComputeExpansionFactor (not an analytic H(z)).  The cooling
    # slot runs BEFORE session_advance_time, so session_time = t_begin; with a0=a(t0),
    # a1=a(t0+dt), ȧ/a_eff = ln(a1/a0)/Δt_s integrates de/dt=-2(ȧ/a)e to e·(a0/a1)²
    # EXACTLY over the step (γ=5/3).  The kernel evolves the redshift across the step
    # from this z_start using aoa (z(t)=(1+z_start)exp(-aoa·t)−1), so the Compton
    # target T_cmb(z) and rates are NOT frozen at z_start — accurate for Enzo's large
    # CIC_MAXEXP steps in both the Compton-locked (high-z) and decoupled (low-z) limits.
    aoa = NaN
    if _CHEM_CFG[].hubble_expansion
        t0  = EnzoLib.session_time(h)
        a0  = EnzoLib.session_expansion_factor(h, t0)[1]
        a1  = EnzoLib.session_expansion_factor(h, t0 + dt)[1]
        dt_s = dt * TU
        (a0 > 0 && a1 > 0 && dt_s > 0) && (aoa = log(a1/a0) / dt_s)
    end

    # iterate the grids resident on this level (NOT a hardcoded grid 0) and use the
    # 0-based BaryonField slots from field_index — exactly the hydro!/gravity! slots.
    ngr = EnzoLib.session_num_grids_on_level(h, level)
    for gi in 0:ngr-1
        g = EnzoLib.problem_grid_index_on_level(h, level, gi)
        fi(ft) = EnzoLib.field_index(h, ft; grid=g)          # already 0-based
        haveD = deuterium
        iHDI = -1
        if haveD
            iHDI = try fi(18) catch; -1 end
            haveD = iHDI >= 0
        end
        iD, iTE, iGE = fi(0), fi(1), fi(2)
        iHII, iH2I = fi(9), fi(14)

        # full flat fields (incl. ghost zones), reshaped to the grid block.
        gd = Tuple(Int.(EnzoLib.problem_grid_dims(h, g)))
        GEf = EnzoLib.problem_get_field(h, iGE,  g)
        TEf = EnzoLib.problem_get_field(h, iTE,  g)
        HIIf = EnzoLib.problem_get_field(h, iHII, g)
        H2If = EnzoLib.problem_get_field(h, iH2I, g)
        HDIf = haveD ? EnzoLib.problem_get_field(h, iHDI, g) : nothing
        Df   = EnzoLib.problem_get_field(h, iD, g)
        # ACTIVE sub-block only — ghost cells are uninitialized garbage that would
        # (a) hit the per-cell subcycle cap, hanging the GPU kernel past the Metal
        # watchdog, and (b) waste work.  Enzo refreshes ghosts from the BC anyway.
        rng = (ng+1:gd[1]-ng, ng+1:gd[2]-ng, ng+1:gd[3]-ng)
        nact = map(length, rng)
        RD=reshape(Df,gd); RGE=reshape(GEf,gd); RTE=reshape(TEf,gd)
        RH=reshape(HIIf,gd); RH2=reshape(H2If,gd); RHD = haveD ? reshape(HDIf,gd) : nothing
        Da  = vec(RD[rng...]);  GEa = vec(RGE[rng...])
        HIIa= vec(RH[rng...]);  H2Ia= vec(RH2[rng...]); HDIa = haveD ? vec(RHD[rng...]) : nothing
        na = length(Da)

        # → physical CGS (active cells are well-behaved; light clamps for safety).
        # NB: floor RELATIVE to the field (eps()≈2.2e-16 is a HUGE *density* in CGS
        # — flooring with it inflates n_H to ~1e8 and over-recombines everything).
        rho  = Vector{Float64}(undef, na); eint = Vector{Float64}(undef, na)
        HII  = Vector{Float64}(undef, na); H2I  = Vector{Float64}(undef, na)
        HDI  = haveD ? Vector{Float64}(undef, na) : nothing
        rfloor = maximum(Da) * DU * 1e-20
        @inbounds for i in 1:na
            rho[i]  = (isfinite(Da[i]) && Da[i] > 0) ? Da[i]*DU : rfloor
            eint[i] = clamp(GEa[i]*VU2, 1.0, 1.0e16)         # T∈[1,~1e8] K
            HII[i]  = clamp(HIIa[i]*DU, 0.0, rho[i])
            H2I[i]  = clamp(H2Ia[i]*DU, 0.0, rho[i])
            haveD && (HDI[i] = clamp(HDIa[i]*DU, 0.0, rho[i]))
        end

        dbg = get(ENV, "CHEM_DEBUG", "0") == "1" && gi == 0
        if dbg
            j = na÷2
            Hana = 71.0*1e5/3.0856775807e24*sqrt(Om*(1+z)^3 + OL)   # analytic ΛCDM H(z) [1/s]
            @info "enzo_chem pre" z dt_s=dt*TU na rho_j=rho[j] eint_j=eint[j] HII_j=HII[j] H2I_j=H2I[j] HDI_j=(haveD ? HDI[j] : 0.0) nH=0.76*rho[j]/1.6726e-24 xHII_in=HII[j]/rho[j]/0.76 Tj=(5/3-1)*1.22*1.6726e-24*eint[j]/1.380649e-16 aoa=aoa Hana=Hana aoa_over_Hana=(isnan(aoa) ? NaN : aoa/Hana)
            flush(stderr)
        end

        # physical CGS in → solve_chem! with unit=1 (cosmology from _CHEM_CFG).
        chem_step!(rho, eint, HII, H2I, HDI; a_value=1.0/(1+z), dt=dt*TU,
                   engine=engine, backend=backend, precision=precision,
                   adot_over_a=aoa)

        dbg && (@info "enzo_chem post" xHII_out=(HII[na÷2]/rho[na÷2])/0.76 T_out=eint[na÷2]; flush(stderr))

        # → Enzo code units; update GasEnergy AND TotalEnergy by the eint delta;
        # write the results back into the ACTIVE sub-block (ghosts left untouched).
        ge_new = eint ./ VU2
        RGE[rng...] = reshape(ge_new, nact)
        RTE[rng...] = RTE[rng...] .+ reshape(ge_new .- GEa, nact)
        RH[rng...]  = reshape(HII ./ DU, nact)
        RH2[rng...] = reshape(H2I ./ DU, nact)
        haveD && (RHD[rng...] = reshape(HDI ./ DU, nact))
        EnzoLib.problem_set_field(h, iGE,  GEf;  grid=g)
        EnzoLib.problem_set_field(h, iTE,  TEf;  grid=g)
        EnzoLib.problem_set_field(h, iHII, HIIf; grid=g)
        EnzoLib.problem_set_field(h, iH2I, H2If; grid=g)
        haveD && EnzoLib.problem_set_field(h, iHDI, HDIf; grid=g)
    end
    return 1
end

# ── Arepo wiring ──────────────────────────────────────────────────────────────
# Arepo carries the two species as primitive passive-scalar abundances x (the
# :scalars field added to ArepoLib; the bridge keeps the conserved
# PConservedScalars = x*Mass in sync so they advect with the Voronoi flux).
# Arepo stores utherm as specific internal energy directly, so -- unlike RAMSES
# -- no kinetic subtraction is needed.  Requires Arepo built with
# PASSIVE_SCALARS=2 (column 1 = x_HII, column 2 = x_H2I).

"""
    arepo_chem_step!(h; dt, a_value)

Run one reduced-chemistry step on all Arepo gas cells: read rho, utherm and the
two passive-scalar abundances, call `chem_step!` (converting to/from the density-
weighted convention), and write back utherm and the abundances.  `chem_init!`
must have been called first with Arepo's code units.
"""
function arepo_chem_step!(h; dt::Real, a_value::Real, engine::Symbol=:grackle,
        backend::Symbol=:cpu, precision::Type=Float64)
    rho  = Float64.(ArepoLib.get_cell_field(h, :rho))
    eint = Float64.(ArepoLib.get_cell_field(h, :utherm))      # specific internal energy
    sc   = ArepoLib.get_cell_field(h, :scalars)               # n×{2,3} abundances
    ncol = size(sc, 2)
    HII  = rho .* Float64.(@view sc[:, 1])                    # density-weighted rho*x
    H2I  = rho .* Float64.(@view sc[:, 2])
    HDI  = ncol >= 3 ? rho .* Float64.(@view sc[:, 3]) : nothing   # HD (deuterium)

    chem_step!(rho, eint, HII, H2I, HDI; a_value=a_value, dt=dt,
               engine=engine, backend=backend, precision=precision)

    r = max.(rho, eps())
    ArepoLib.set_cell_field!(h, :utherm, eint)
    cols = HDI === nothing ? hcat(HII ./ r, H2I ./ r) : hcat(HII ./ r, H2I ./ r, HDI ./ r)
    ArepoLib.set_cell_field!(h, :scalars, cols)
    return (; ncells = length(rho))
end
