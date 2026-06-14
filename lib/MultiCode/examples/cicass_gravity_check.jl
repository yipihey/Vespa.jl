# Gravity-source diagnostic for the CICASS high-z Enzo path.
#
# The user's question: "check the potential with and without baryons at every step."
# This sets up the same CICASS Enzo IC (corrected species/energy init), evolves a
# handful of cycles with the CERTIFIED Enzo gravity, then at each step decomposes
# Enzo's GravitatingMassField (the Poisson SOURCE) into its baryon-gas and
# dark-matter-particle contributions and solves the potential three ways:
#   φ_full  : from the full GMF (gas + DM)          — what the gas actually feels
#   φ_dm    : from the DM-only overdensity
#   φ_gas   : from the gas-only overdensity
# and reports the amplitudes + cross-correlation, so we can see whether the baryon
# contribution to the potential is correctly present (right sign, right weight) and
# how it compares to the DM-sourced potential the gas is falling into.
#
# Run:
#   BACKEND=metal ENZOMODULES_GRID_LIB=.../f32/libenzomodules_grid_f32.dylib \
#     DYLD_LIBRARY_PATH=$HOME/grackle_install_f32/lib:/opt/homebrew/opt/hdf5/lib \
#     GRACKLE_DATA_FILE=... CIC_NGRID=128 CIC_NCYC=12 \
#     <julia> --project=lib/MultiCode/test lib/MultiCode/examples/cicass_gravity_check.jl

using EnzoLib, MultiCode, CICASSLib, PoissonKernels, Printf, Statistics
include(joinpath(@__DIR__, "..", "..", "EnzoLib", "examples", "sb_metal_amr.jl"))  # helpers, iD…, NG

const BOX     = parse(Float64, get(ENV, "CIC_BOX",   "0.128"))
const ZSTART  = parse(Float64, get(ENV, "CIC_ZSTART","1000.0"))
const NGRID   = envint("CIC_NGRID", 128)
const OMEGA_M = parse(Float64, get(ENV, "CIC_OMEGAM", "0.27"))
const NCYC    = envint("CIC_NCYC", 12)
const GRACKLE_DATA = get(ENV, "GRACKLE_DATA_FILE",
    joinpath(homedir(), "Research", "codes", "grackle", "input", "CloudyData_noUVB.h5"))

# periodic Poisson solve ∇²φ = src (src mean-zero), unit comoving box, via the
# certified PoissonKernels root FFT (DC mode dropped internally; plan cached).
function poisson_periodic(src::Array{Float64,3})
    phi = zeros(Float64, size(src))
    PoissonKernels.fft_poisson_root!(phi, src; G=1.0, a=1.0, boxsize=1.0, greens=:discrete7)
    phi .- sum(phi)/length(phi)
end

# GMF has its OWN ghost padding (gravity buffer ≠ baryon NumberOfGhostZones), so
# reshape to its real dims and center-extract the N³ active block.
function active_of_gmf(flat, h, N)
    gd = EnzoLib.problem_gmf_dims(h, 0)
    a = reshape(Float64.(flat), gd...)
    o = ntuple(d -> (gd[d]-N)÷2, 3)
    Array(a[o[1]+1:o[1]+N, o[2]+1:o[2]+N, o[3]+1:o[3]+N])
end

corr(a,b) = begin
    am=a.-sum(a)/length(a); bm=b.-sum(b)/length(b)
    sum(am.*bm)/sqrt(sum(am.^2)*sum(bm.^2))
end
rms(a) = std(a)

function main()
    EnzoLib.grid_available() || error("grid bridge not built")
    chem = """
    RadiativeCooling             = 1
    use_grackle                  = 1
    with_radiative_cooling       = 1
    MultiSpecies                 = 3
    CaseBRecombination           = 1
    cmb_dissociation             = 1
    cmb_recombination            = 1
    equilibrium_h2_intermediates = 1
    neutral_helium               = 1
    equilibrium_deuterium        = 1
    grackle_data_file            = $(GRACKLE_DATA)
    DualEnergyFormalism          = 1
    GreensFunctionMaxNumber      = 30
    NumberOfGhostZones           = 4
    CosmologyFinalRedshift       = 20.0
    """
    res = MultiCode.run_cicass_enzo(; boxlength=BOX, zstart=ZSTART, ngrid=NGRID,
                                    omega_m=OMEGA_M, param_extra=chem)
    h = res.handle; dims = res.dims; act = res.act; N = res.n; snap = res.snap
    try
        # inject the CICASS baryon overdensity (same as the pk pipeline)
        gδ = reshape(snap.gas_delta, N, N, N)
        ρfull = reshape(EnzoLib.problem_get_field(h, iD, 0), dims...)
        ρmean = sum(@view ρfull[act...]) / N^3
        factor = (1.0 .+ gδ) ./ (Array(@view ρfull[act...]) ./ ρmean)
        for ft in (0, 9, 14, 18)
            fi = try EnzoLib.field_index(h, ft; grid=0) catch; -1 end
            fi < 0 && continue
            full = reshape(EnzoLib.problem_get_field(h, fi, 0), dims...)
            ft == 0 ? (full[act...] = ρmean .* (1.0 .+ gδ)) :
                      (full[act...] = Array(@view full[act...]) .* factor)
            EnzoLib.problem_set_field(h, fi, vec(full); grid=0)
        end
        EnzoLib.session_rebuild(h, 0)

        eng = EnzoLib.EngineConfig(; hydro=:julia, gravity=:enzo, cooling=:enzo,
                                   comoving_expansion=:enzo, reflux=false,
                                   hooks=Dict{Symbol,Function}(:hydro=>hydro!))
        @printf("%-4s %-9s | %-10s %-10s %-10s | %-9s %-9s | %-7s %-7s\n",
                "cyc","z","rms_gasδ","rms_dmδ","rms_GMFδ",
                "rmsφ_full","rmsφ_dm","r(g,dm)","φgas/φdm")
        for cyc in 0:NCYC
            # ---- decompose the gravity source at the CURRENT state ----
            # Enzo's GravitatingMassField in cosmology is the OVERDENSITY source
            # (mean ≈ 0, in units where the mean matter density is 1) — use it
            # directly, do NOT mean-normalize.  Gas/DM contributions are additive
            # absolute overdensities (ρ − ρ̄) in the SAME code density units.
            EnzoLib.session_prepare_density(h, 0)                  # builds GMF (gas+DM)
            gmf = active_of_gmf(EnzoLib.problem_get_gravitating_mass(h, 0), h, N)
            ρb  = active_of(EnzoLib.read_density(h; grid=0), dims, N)
            ρd  = active_of(EnzoLib.deposit_particle_density(h; grid=0, periodic=true), dims, N)
            src      = gmf .- sum(gmf)/length(gmf)                 # full source (gas+DM)
            src_dm   = ρd  .- sum(ρd)/length(ρd)                   # DM-only overdensity
            src_gas  = ρb  .- sum(ρb)/length(ρb)                   # gas-only overdensity
            δb = src_gas ./ (sum(ρb)/length(ρb)); δd = src_dm ./ (sum(ρd)/length(ρd))
            # potentials in a common absolute normalization
            φfull = poisson_periodic(src)
            φdm   = poisson_periodic(src_dm)
            φgas  = poisson_periodic(src_gas)
            _, z = EnzoLib.session_cosmology(h)
            @printf("%-4d %-9.2f | %-10.3e %-10.3e %-10.3e | %-9.3e %-9.3e | %-7.4f %-7.4f\n",
                    cyc, z, rms(δb), rms(δd), rms(src),
                    rms(φfull), rms(φdm), corr(φfull, φdm), rms(φgas)/rms(φdm))
            flush(stdout)
            cyc == NCYC && break
            EnzoLib.evolve_level!(h, 0, 0.0; engine=eng, regrid=false)
            EnzoLib.session_rebuild(h, 0)
        end
        # one-shot consistency: is the GMF source = (ρgas−ρ̄gas) + (ρdm−ρ̄dm)?
        EnzoLib.session_prepare_density(h, 0)
        gmf = active_of_gmf(EnzoLib.problem_get_gravitating_mass(h, 0), h, N)
        ρb  = active_of(EnzoLib.read_density(h; grid=0), dims, N)
        ρd  = active_of(EnzoLib.deposit_particle_density(h; grid=0, periodic=true), dims, N)
        src     = gmf .- sum(gmf)/length(gmf)
        src_gas = ρb  .- sum(ρb)/length(ρb)
        src_dm  = ρd  .- sum(ρd)/length(ρd)
        model   = src_gas .+ src_dm
        @printf("\nGMF source decomposition (raw means: gmf=%.4e ρgas=%.4e ρdm=%.4e):\n",
                sum(gmf)/length(gmf), sum(ρb)/length(ρb), sum(ρd)/length(ρd))
        @printf("  corr(GMF, (ρgas−ρ̄)+(ρdm−ρ̄)) = %.5f   rms ratio = %.4f\n",
                corr(src, model), rms(src)/rms(model))
        @printf("  corr(GMF, ρdm−ρ̄ only)       = %.5f   rms ratio = %.4f\n",
                corr(src, src_dm), rms(src)/rms(src_dm))
        @printf("  baryon source fraction rms(ρgas−ρ̄)/rms(ρdm−ρ̄) = %.4e\n",
                rms(src_gas)/rms(src_dm))
    finally
        res.free()
    end
end
main()
