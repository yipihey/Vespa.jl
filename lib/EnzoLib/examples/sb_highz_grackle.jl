# High-redshift Santa Barbara run: hydro (PPMKernels) + gravity (PoissonKernels) on
# the GPU (Metal) as :julia slots, PLUS the chemistry/cooling slot driven by Grackle
# (our yipihey/grackle fork — CMB Compton heating/cooling + the new cmb_dissociation
# H2-suppression rates) through Enzo's GrackleWrapper. The initial gas temperature and
# electron fraction come from CICASS's RECFAST recombination table (thermal_state(z)),
# loaded into Enzo's CosmologySimulationInitial* params. This is the end-to-end
# "EnzoNG GPU path uses the Grackle table at high z" demonstration.
#
# Run (GPU):  BACKEND=metal ENZOMODULES_GRID_LIB=.../libenzomodules_grid_f32.dylib \
#             DYLD_LIBRARY_PATH=~/grackle_install_f32/lib:/opt/homebrew/opt/hdf5/lib \
#             HIGHZ=200 <julia> --project=lib/PPMKernels/test \
#             lib/EnzoLib/examples/sb_highz_grackle.jl [cycles]
#
# NB the SB density ICs were realized at z=63; starting at a higher z reuses that field
# (perturbations a few× too large) — fine for this chemistry/cooling integration test,
# not a science run. The dramatic CMB H2 suppression is validated separately in
# grackle/src/example/cmb_test.c (71× at z=1000).

include(joinpath(@__DIR__, "sb_metal_amr.jl"))   # GPU hydro!/gravity! + constants/helpers

using Printf

const HIGHZ = parse(Float64, get(ENV, "HIGHZ", "200"))
const GRACKLE_DATA = get(ENV, "GRACKLE_DATA_FILE",
    "/Users/tabel/Research/codes/grackle/input/CloudyData_noUVB.h5")
const RECFAST = "/Users/tabel/Projects/cicass/vbc_transfer/recfast/xeTrecfast.out"

# CICASS RECFAST thermal state (T_gas[K], x_e=n_e/n_H) at redshift z — read the table
# inline so this runner has no CICASSLib dependency.
function cicass_thermal(z)
    zs = Float64[]; xe = Float64[]; tg = Float64[]
    for (i, line) in enumerate(eachline(RECFAST))
        i == 1 && continue
        t = split(line); length(t) >= 3 || continue
        push!(zs, parse(Float64, t[1])); push!(xe, parse(Float64, t[2])); push!(tg, parse(Float64, t[3]))
    end
    p = sortperm(zs); zs, xe, tg = zs[p], xe[p], tg[p]
    interp(v) = (z <= zs[1] ? v[1] : z >= zs[end] ? v[end] :
                 (j = searchsortedfirst(zs, z); w = (z - zs[j-1])/(zs[j]-zs[j-1]); v[j-1]*(1-w)+v[j]*w))
    return (T_gas = interp(tg), x_e = interp(xe), T_cmb = 2.73*(1+z))
end

function write_highz_param(z)
    s = cicass_thermal(z)
    par = read(joinpath(SB, "SantaBarbaraCluster.enzo"), String)
    par = replace(par, r"CosmologyInitialRedshift\s*=\s*\S+" => "CosmologyInitialRedshift   = $(z)")
    par = replace(par, r"GreensFunctionMaxNumber.*" => "GreensFunctionMaxNumber   = 30\nNumberOfGhostZones        = 4")
    par *= """

    # --- high-z Grackle chemistry + cooling (CICASS thermal IC) ---
    RadiativeCooling            = 1
    use_grackle                 = 1
    with_radiative_cooling      = 1
    MultiSpecies                = 2
    CaseBRecombination          = 1
    cmb_dissociation            = 1
    equilibrium_h2_intermediates = $(get(ENV, "EQUIL", "0"))
    neutral_helium              = $(get(ENV, "HE", "0"))
    cmb_recombination           = $(get(ENV, "REC", "0"))
    grackle_data_file           = $(GRACKLE_DATA)
    DualEnergyFormalism         = 1
    CosmologySimulationInitialTemperature   = $(round(s.T_gas, digits=3))
    CosmologySimulationInitialFractionHII   = $(s.x_e)
    """
    pf = joinpath(SB, "SB_highz.enzo"); write(pf, par)
    @printf("CICASS z=%.0f: T_gas=%.1f K, x_e=%.3e, T_cmb=%.1f K  →  %s\n",
            z, s.T_gas, s.x_e, s.T_cmb, pf)
    return pf, s
end

function main_highz()
    maxcyc = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : envint("SB_MAXCYC", 4)
    EnzoLib.grid_available() || error("grid dylib not built")
    pf, s = write_highz_param(HIGHZ)
    @printf("SB high-z (z=%.0f) hydro+gravity on %s + Grackle cooling/chemistry (cmb_dissociation=1), %d cycles\n",
            HIGHZ, BE, maxcyc)
    cd(SB) do
        h = EnzoLib.session_init(pf); h == C_NULL && error("session_init failed (high-z grackle)")
        try
            eng = EnzoLib.EngineConfig(; hydro=:julia, gravity=:julia, cooling=:enzo,
                                       comoving_expansion=:enzo, reflux=false,
                                       hooks=Dict{Symbol,Function}(:hydro=>hydro!, :gravity=>gravity!))
            EnzoLib.session_rebuild(h, 0)
            m0 = EnzoLib.session_global_field_integral(h, 0)
            @printf("%-4s %-8s %-10s %-10s %-10s %-8s\n", "cyc", "a", "ρmax", "GEmean", "Δmass/M", "sec")
            for cyc in 0:maxcyc-1
                t0 = time()
                EnzoLib.evolve_level!(h, 0, 0.0; engine=eng, regrid=true)
                sec = time() - t0
                EnzoLib.session_rebuild(h, 0)
                ρ = EnzoLib.problem_get_field(h, iD, 0)
                ge = EnzoLib.problem_get_field(h, iGE, 0)
                m = EnzoLib.session_global_field_integral(h, 0)
                a = EnzoLib.session_cosmology(h)[1]
                @printf("%-4d %-8.5f %-10.3f %-10.3e %-10.1e %-8.2f\n",
                        cyc, a, maximum(ρ), sum(ge)/length(ge), abs(m-m0)/m0, sec)
                any(isnan, ρ) && (println("  NaN — abort"); break)
            end
            println("\nHIGH-Z GRACKLE RUN: completed — Grackle cooling/chemistry ran in the GPU cycle.")
        finally
            EnzoLib.free_problem(h)
        end
    end
end

main_highz()
