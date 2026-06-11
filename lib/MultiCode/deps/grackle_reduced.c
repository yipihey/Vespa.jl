/* grackle_reduced.c -- the v2026 reduced primordial chemistry as a flat C
   service, so non-Enzo MultiCode hosts (RAMSES, Arepo) can run early-universe
   chemistry+cooling with our Grackle fork while advecting only TWO species
   (HII and H2I).  Helium is forced neutral, electrons = protons, HI is
   reconstructed, and H-/H2+ are equilibrium -- all inside Grackle -- so the
   host need only carry HII and H2I.

   Build links the installed f64 Grackle (~/grackle_install, GRACKLE_FLOAT_8 ->
   gr_float == double).  All per-cell arrays are double and density-weighted
   (HII, H2I are MASS densities rho*x, in the code_units passed to _init). */
#include <stdlib.h>
#include <stdio.h>
extern "C" {
#include <grackle.h>
}

static chemistry_data *g_cd = NULL;
static code_units      g_units;
static double          g_fh = 0.76;   /* HydrogenFractionByMass */

/* Initialize the reduced network once.  density/length/time_units are the CGS
   conversion factors for the host's code units (field*density_units = g/cm^3).
   Returns 1 on success, 0 on failure (Grackle convention). */
extern "C" int grackle_reduced_init(double hubble_kmsmpc, double Om, double OL,
                                    double a_value, double fh,
                                    double density_units, double length_units,
                                    double time_units, const char *data_file)
{
  g_cd = (chemistry_data *) malloc(sizeof(chemistry_data));
  if (set_default_chemistry_parameters(g_cd) == 0) return 0;
  grackle_data->use_grackle                    = 1;
  grackle_data->with_radiative_cooling         = 1;
  grackle_data->primordial_chemistry           = 2;   /* H, He, e, H-, H2, H2+ */
  grackle_data->metal_cooling                  = 0;
  grackle_data->UVbackground                   = 0;
  grackle_data->cmb_dissociation               = 1;   /* CMB H-/H2+ photo-destr. */
  grackle_data->equilibrium_h2_intermediates   = 1;   /* H-/H2+ algebraic       */
  grackle_data->neutral_helium                 = 1;   /* He neutral, n_e=n_HII  */
  grackle_data->cmb_recombination              = 1;   /* Peebles C-factor       */
  grackle_data->cosmology_hubble_constant_now  = hubble_kmsmpc;
  grackle_data->cosmology_omega_matter_now     = Om;
  grackle_data->cosmology_omega_lambda_now     = OL;
  grackle_data->HydrogenFractionByMass         = fh;
  grackle_data->grackle_data_file              = data_file;
  g_fh = fh;

  g_units.comoving_coordinates = 0;     /* physical densities; a_value sets z */
  g_units.density_units        = density_units;
  g_units.length_units         = length_units;
  g_units.time_units           = time_units;
  g_units.a_units              = 1.0;
  g_units.a_value              = a_value;
  set_velocity_units(&g_units);
  return initialize_chemistry_data(&g_units);
}

/* Evolve n cells for dt (code time units) at expansion factor a_value.
   rho        : total gas mass density (code units)                 [in]
   e_int      : specific internal energy (code units, energy/mass)  [in/out]
   HII, H2I   : mass densities rho*x (code units)                   [in/out]
   The 7 reconstructed species (De, HI, HeI, HeII, HeIII, HM, H2II) live in
   transient scratch -- Grackle fills them from {rho, HII, H2I} each step.
   Returns 1 on success, 0 on failure. */
extern "C" int grackle_reduced_step(long n, double a_value, double dt,
                                    double *rho, double *e_int,
                                    double *HII, double *H2I)
{
  if (g_cd == NULL) return 0;
  g_units.a_value = a_value;

  double *sc = (double *) malloc((size_t)7 * n * sizeof(double));
  double *sc_e=sc+0*n,*sc_HI=sc+1*n,*sc_HeI=sc+2*n,*sc_HeII=sc+3*n,
         *sc_HeIII=sc+4*n,*sc_HM=sc+5*n,*sc_H2II=sc+6*n;
  const double tiny = 1e-20;
  for (long i = 0; i < n; i++) {
    sc_e[i]   = HII[i];                            /* n_e = n_HII */
    double hi = g_fh*rho[i] - HII[i] - H2I[i];
    sc_HI[i]  = (hi > tiny) ? hi : tiny;           /* X_H*rho - HII - H2I */
    sc_HeI[i] = (1.0 - g_fh)*rho[i];               /* all-neutral helium  */
    sc_HeII[i]=sc_HeIII[i]=sc_HM[i]=sc_H2II[i]=tiny;
  }

  grackle_field_data f;
  int dim[3]   = { (int)n, 1, 1 };
  int gstart[3]= { 0, 0, 0 };
  int gend[3]  = { (int)n - 1, 0, 0 };
  f.grid_rank = 1; f.grid_dimension = dim; f.grid_start = gstart; f.grid_end = gend;
  f.grid_dx = 0.0;
  double *zero = (double *) calloc((size_t)n, sizeof(double));
  f.density = rho; f.internal_energy = e_int;
  f.x_velocity = zero; f.y_velocity = zero; f.z_velocity = zero;
  f.HI_density=sc_HI;   f.HII_density=HII;     f.HeI_density=sc_HeI;
  f.HeII_density=sc_HeII; f.HeIII_density=sc_HeIII; f.e_density=sc_e;
  f.HM_density=sc_HM;   f.H2I_density=H2I;     f.H2II_density=sc_H2II;
  f.DI_density=NULL;    f.DII_density=NULL;    f.HDI_density=NULL;
  f.metal_density=NULL; f.dust_density=NULL;
  f.volumetric_heating_rate=NULL; f.specific_heating_rate=NULL;
  f.RT_HI_ionization_rate=NULL; f.RT_HeI_ionization_rate=NULL;
  f.RT_HeII_ionization_rate=NULL; f.RT_H2_dissociation_rate=NULL;
  f.RT_heating_rate=NULL; f.H2_self_shielding_length=NULL;
  f.H2_custom_shielding_factor=NULL; f.isrf_habing=NULL;

  int rc = solve_chemistry(&g_units, &f, dt);
  free(sc); free(zero);
  return rc;
}

/* Per-cell gas temperature [K] for diagnostics (same reconstruction). */
extern "C" int grackle_reduced_temperature(long n, double a_value,
                                           double *rho, double *e_int,
                                           double *HII, double *H2I, double *Tout)
{
  if (g_cd == NULL) return 0;
  g_units.a_value = a_value;
  double *sc = (double *) malloc((size_t)7 * n * sizeof(double));
  double *sc_e=sc+0*n,*sc_HI=sc+1*n,*sc_HeI=sc+2*n,*sc_HeII=sc+3*n,
         *sc_HeIII=sc+4*n,*sc_HM=sc+5*n,*sc_H2II=sc+6*n;
  const double tiny = 1e-20;
  for (long i = 0; i < n; i++) {
    sc_e[i]=HII[i]; double hi=g_fh*rho[i]-HII[i]-H2I[i];
    sc_HI[i]=(hi>tiny)?hi:tiny; sc_HeI[i]=(1.0-g_fh)*rho[i];
    sc_HeII[i]=sc_HeIII[i]=sc_HM[i]=sc_H2II[i]=tiny;
  }
  grackle_field_data f; int dim[3]={(int)n,1,1},gs[3]={0,0,0},ge[3]={(int)n-1,0,0};
  f.grid_rank=1; f.grid_dimension=dim; f.grid_start=gs; f.grid_end=ge; f.grid_dx=0.0;
  double *zero=(double*)calloc((size_t)n,sizeof(double));
  f.density=rho; f.internal_energy=e_int; f.x_velocity=zero; f.y_velocity=zero; f.z_velocity=zero;
  f.HI_density=sc_HI; f.HII_density=HII; f.HeI_density=sc_HeI; f.HeII_density=sc_HeII;
  f.HeIII_density=sc_HeIII; f.e_density=sc_e; f.HM_density=sc_HM; f.H2I_density=H2I;
  f.H2II_density=sc_H2II; f.DI_density=NULL; f.DII_density=NULL; f.HDI_density=NULL;
  f.metal_density=NULL; f.dust_density=NULL;
  f.volumetric_heating_rate=NULL; f.specific_heating_rate=NULL;
  f.RT_HI_ionization_rate=NULL; f.RT_HeI_ionization_rate=NULL; f.RT_HeII_ionization_rate=NULL;
  f.RT_H2_dissociation_rate=NULL; f.RT_heating_rate=NULL; f.H2_self_shielding_length=NULL;
  f.H2_custom_shielding_factor=NULL; f.isrf_habing=NULL;
  int rc = calculate_temperature(&g_units, &f, Tout);
  free(sc); free(zero);
  return rc;
}
