# Verify the v2026 ReducedChemistry BaryonField removal: print the allocated
# field set for the SB high-z grid.  REDUCED=1 turns on neutral_helium +
# equilibrium_h2_intermediates (only HII,H2I expected); REDUCED=0 = full network.
import EnzoLib
const SB = joinpath(@__DIR__, "..", "..", "..", "..", "run", "CosmologySimulation", "SantaBarbaraCluster")
const GD = get(ENV, "GRACKLE_DATA_FILE",
    joinpath(homedir(), "Research", "codes", "grackle", "input", "CloudyData_noUVB.h5"))
const REDUCED = get(ENV, "REDUCED", "0") == "1"
const DEUT    = get(ENV, "DEUT", "0") == "1"     # also track HD (3rd species)
const FTN = Dict(0=>"Density",1=>"TotalEnergy",2=>"GasEnergy",4=>"Vel1",5=>"Vel2",6=>"Vel3",
    7=>"De",8=>"HI",9=>"HII",10=>"HeI",11=>"HeII",12=>"HeIII",13=>"HM",14=>"H2I",15=>"H2II",
    16=>"DI",17=>"DII",18=>"HDI",20=>"Metal")

par = read(joinpath(SB, "SantaBarbaraCluster.enzo"), String)
par = replace(par, r"CosmologyInitialRedshift\s*=\s*\S+" => "CosmologyInitialRedshift   = 200.0")
flags = REDUCED ? "neutral_helium = 1\nequilibrium_h2_intermediates = 1\n" : ""
DEUT && REDUCED && (flags *= "equilibrium_deuterium = 1\n")
par *= """

    RadiativeCooling            = 1
    use_grackle                 = 1
    with_radiative_cooling      = 1
    MultiSpecies                = $(DEUT && REDUCED ? 3 : 2)
    $(flags)grackle_data_file           = $(GD)
    DualEnergyFormalism         = 1
    CosmologySimulationInitialTemperature   = 465.5
    CosmologySimulationInitialFractionHII   = 3.3e-4
    """
pf = joinpath(SB, "SB_fieldcheck.enzo"); write(pf, par)

EnzoLib.grid_available() || error("grid dylib not built")
cd(SB) do
    h = EnzoLib.session_init(pf); h == C_NULL && error("session_init failed")
    ft = EnzoLib.problem_field_types(h, 0)
    tag = REDUCED ? "REDUCED" : "FULL"
    println("$tag network: $(length(ft)) BaryonFields  [",
            join([get(FTN, t, string(t)) for t in ft], ", "), "]")
end
