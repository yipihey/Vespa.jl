# Live RAMSES reduced-chemistry test: boot mini-RAMSES with nvar=7 (two passive
# scalars HII,H2I at vars 6,7), seed a high-z primordial gas, run the shared
# Grackle reduced-chemistry slot, and confirm the species evolve on the live
# RAMSES cells.  Requires bin64h_chem/libramses3d.dylib (NPSCAL=2).
import RamsesLib
import MultiCode
using Printf

const mh = 1.67262171e-24
const XH = 0.76

# uniform high-z box (single region); passive scalars start at 0, set below.
nml = """
  Reduced-chemistry RAMSES box (MultiCode)
  &RUN_PARAMS
  hydro=.true.
  ncontrol=1
  nrestart=0
  nsubcycle=10*1
  nstepmax=100000
  verbose=.false.
  /
  &AMR_PARAMS
  levelmin=4
  levelmax=4
  ngridtot=3000000
  ncachemax=30000
  nexpand=1
  boxlen=1.0
  /
  &INIT_PARAMS
  nregion=1
  region_type(1)='square'
  x_center=0.5
  y_center=0.5
  z_center=0.5
  length_x=10.0
  length_y=10.0
  length_z=10.0
  exp_region=10.0
  d_region=1.0
  u_region=0.0
  v_region=0.0
  w_region=0.0
  p_region=1.0
  /
  &OUTPUT_PARAMS
  foutput=0
  tout=100.0
  /
  &HYDRO_PARAMS
  gamma=1.6666667
  courant_factor=0.8
  slope_type=1
  riemann='hllc'
  /
  &REFINE_PARAMS
  interpol_var=0
  interpol_type=0
  /
  """
nmlfile = tempname()*".nml"; write(nmlfile, nml)

h = RamsesLib.init(nmlfile)
lev = 4
@printf("RAMSES booted: nvar=%d, level %d\n", RamsesLib.nvar(), lev)

# physical units: code rho=1 -> n_H(z=1000) ~ 199 cm^-3
z = 1000.0
nH = (0.046*1.8788e-29*0.71^2*XH/mh)*(1+z)^3       # ~199 cm^-3
dens_u = nH*mh/XH                                   # rho_code=1 -> this g/cm^3
len_u  = 3.0857e21; time_u = 3.1557e13              # kpc, Myr
GD = joinpath(homedir(),"Research","codes","grackle","input","CloudyData_noUVB.h5")

# seed the two species on every leaf cell at level lev: HII = rho*x_HII, H2I = rho*x_H2
ck, U = RamsesLib.get_hydro_all(h, :uold, lev)      # noct×8×nvar
noct = size(U,1)
rho_block = U[:,:,1]
xHII0 = 0.047; xH20 = 1e-12
const DEUT = get(ENV,"DEUT","0") == "1"              # also track HD (var 8)
RamsesLib.set_hydro!(h, :uold, 6, lev, ck, rho_block .* xHII0)   # HII = rho*x
RamsesLib.set_hydro!(h, :uold, 7, lev, ck, rho_block .* xH20)    # H2I
DEUT && RamsesLib.set_hydro!(h, :uold, 8, lev, ck, rho_block .* 1e-20)  # HDI

# also raise the gas energy to ~T=2728 K so the chemistry has a real temperature
vel_u = len_u/time_u; Tunits = mh*vel_u^2/1.380649e-16
eint = 2728.0/Tunits/(5/3-1)/1.22                   # specific internal energy (code)
Etot = U[:,:,1].*eint                               # E_total = rho*eint (u=0)
RamsesLib.set_hydro!(h, :uold, 5, lev, ck, Etot)

# init the shared chemistry service for these RAMSES units, run one step
MultiCode.chem_init!(; hubble=71.0, Om=0.27, OL=0.73, a_value=1/(1+z), fh=XH,
    density_units=dens_u, length_units=len_u, time_units=time_u, data_file=GD,
    deuterium=DEUT)

ck6,b6 = RamsesLib.get_hydro(h,:uold,6,lev); ck7,b7 = RamsesLib.get_hydro(h,:uold,7,lev)
ckr,br = RamsesLib.get_hydro(h,:uold,1,lev)
xHII_before = b6[1]/br[1]; xH2_before = (b7[1]/br[1])/2
@printf("before: x_HII=%.4e  x_H2=%.4e\n", xHII_before, xH2_before)

res = MultiCode.ramses_chem_step!(h, lev; dt=0.01, a_value=1/(1+z),
    density_units=dens_u, length_units=len_u, time_units=time_u, iHII=6, iH2I=7,
    iHDI = DEUT ? 8 : nothing)

ck6,b6 = RamsesLib.get_hydro(h,:uold,6,lev); ck7,b7 = RamsesLib.get_hydro(h,:uold,7,lev)
ckr,br = RamsesLib.get_hydro(h,:uold,1,lev)
xHII_after = b6[1]/br[1]; xH2_after = (b7[1]/br[1])/2
xHD_str = ""
if DEUT; ck8,b8 = RamsesLib.get_hydro(h,:uold,8,lev); xHD_str = @sprintf("  x_HD=%.4e",(b8[1]/br[1])/3); end
@printf("after : x_HII=%.4e  x_H2=%.4e%s  (%d cells)\n", xHII_after, xH2_after, xHD_str, res.ncells)
@printf("RAMSES reduced-chemistry slot %s: species evolved on live cells.\n",
        (xHII_after != xHII_before) ? "WORKS" : "NO-CHANGE")
