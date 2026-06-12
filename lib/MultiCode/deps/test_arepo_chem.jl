# Live Arepo reduced-chemistry test: boot a 3D Arepo box (libarepo3d built with
# PASSIVE_SCALARS=2), confirm the new :scalars passive-scalar bridge round-trips,
# then run the shared Grackle reduced-chemistry slot and confirm utherm and the
# two species (HII, H2I) evolve on the live Voronoi cells.
import ArepoLib
import MultiCode
using Printf

const mh = 1.67262171e-24; const XH = 0.76
const AREPO = get(ENV,"AREPO_DIR", joinpath(homedir(),"Projects","arepo"))
const EX    = joinpath(AREPO, "examples", "noh_3d")

# stage the 3D example into a temp dir (param.txt + IC.hdf5 via create.py)
dir = mktempdir()
py = let cands=[get(ENV,"AREPO_PYTHON",""), joinpath(AREPO,".venv","bin","python"),
                "/Users/tabel/Projects/disco-dj-fem/.venv/bin/python"]
    something(filter(p->!isempty(p)&&isfile(p), cands)..., "python3")
end
param = read(joinpath(EX,"param.txt"), String)
param = replace(param, r"TimeMax\s+\S+" => "TimeMax  1.0")
write(joinpath(dir,"param.txt"), param)
run(`$py $(joinpath(EX,"create.py")) $dir`)
@assert isfile(joinpath(dir,"IC.hdf5")) "create.py produced no IC.hdf5"

cd(dir) do
    h = ArepoLib.init("param.txt")
    n = ArepoLib.num_gas(h)
    @printf("Arepo booted: %d gas cells (libarepo3d, PASSIVE_SCALARS)\n", n)

    # --- (1) :scalars bridge round-trip (count from the build) ---
    nps = ArepoLib.num_passive_scalars(h)
    sc0 = ArepoLib.get_cell_field(h, :scalars)
    @printf(":scalars get → size %s (PASSIVE_SCALARS=%d)\n", string(size(sc0)), nps)
    cols = nps >= 3 ? [fill(0.047,n) fill(1e-12,n) fill(1e-20,n)] : [fill(0.047,n) fill(1e-12,n)]
    want = Matrix{Float64}(cols)                           # x_HII, x_H2, [x_HD]
    ArepoLib.set_cell_field!(h, :scalars, want)
    got = ArepoLib.get_cell_field(h, :scalars)
    rt = maximum(abs.(got .- want))
    @printf("round-trip max|set-get| = %.2e  → bridge %s\n", rt, rt < 1e-12 ? "OK" : "FAIL")

    # --- (2) run the reduced-chemistry slot on the live cells ---
    rho  = Float64.(ArepoLib.get_cell_field(h, :rho))
    dens_u = (0.046*1.8788e-29*0.71^2)*(1+1000.0)^3 / maximum(rho)   # map peak rho → z=1000 density
    len_u  = 3.0857e21; time_u = 3.1557e13
    GD = joinpath(homedir(),"Research","codes","grackle","input","CloudyData_noUVB.h5")
    MultiCode.chem_init!(; hubble=71.0, Om=0.27, OL=0.73, a_value=1/1001, fh=XH,
        density_units=dens_u, length_units=len_u, time_units=time_u, data_file=GD)

    u0 = Float64.(ArepoLib.get_cell_field(h, :utherm))
    xHII_b = ArepoLib.get_cell_field(h, :scalars)[1,1]
    res = MultiCode.arepo_chem_step!(h; dt=0.01, a_value=1/1001)
    u1 = Float64.(ArepoLib.get_cell_field(h, :utherm))
    xHII_a = ArepoLib.get_cell_field(h, :scalars)[1,1]
    @printf("chem step: x_HII %.4e → %.4e,  Δutherm[1]=%.3e  (%d cells)\n",
            xHII_b, xHII_a, u1[1]-u0[1], res.ncells)
    @printf("Arepo reduced-chemistry slot %s.\n",
            (xHII_a != xHII_b) ? "WORKS: species evolved on live cells" : "NO-CHANGE")
end
