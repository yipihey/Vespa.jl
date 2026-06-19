/* grackle_reduced.c -- the v2026 reduced primordial chemistry as a flat C
   service, so non-Enzo MultiCode hosts (RAMSES, Arepo) can run early-universe
   chemistry+cooling with our Grackle fork while advecting only TWO species
   (HII and H2I).  Helium is forced neutral, electrons = protons, HI is
   reconstructed, and H-/H2+ are equilibrium -- all inside Grackle -- so the
   host need only carry HII and H2I.

   PRECISION-AGNOSTIC: the host (Julia) ABI is always double, but the grackle
   field_data arrays use gr_float (double for an f64 grackle build, float for
   f32).  We convert double<->gr_float at the boundary, so this same source links
   against EITHER ~/grackle_install (f64) or ~/grackle_install_f32 (f32 -- the
   precision the rest of the Vespa stack uses).  HII, H2I are MASS densities
   rho*x, in the code_units passed to _init. */
#include <stdlib.h>
#include <stdio.h>
extern "C" {
#include <grackle.h>
}

static chemistry_data *g_cd = NULL;
static code_units      g_units;
static double          g_fh = 0.76;   /* HydrogenFractionByMass */
static int             g_deut = 0;    /* deuterium (HD) tracking on/off */

/* Initialize the reduced network once.  density/length/time_units are the CGS
   conversion factors for the host's code units (field*density_units = g/cm^3).
   `deuterium`!=0 turns on the reduced D network (primordial_chemistry=3 +
   equilibrium_deuterium): the host then also advects HDI (one extra field) and
   gets the HD abundance + HD line cooling correct.  Returns 1 on success, 0 on
   failure (Grackle convention). */
extern "C" int grackle_reduced_init(double hubble_kmsmpc, double Om, double OL,
                                    double a_value, double fh,
                                    double density_units, double length_units,
                                    double time_units, const char *data_file,
                                    int deuterium)
{
  g_deut = deuterium ? 1 : 0;
  g_cd = (chemistry_data *) malloc(sizeof(chemistry_data));
  if (set_default_chemistry_parameters(g_cd) == 0) return 0;
  grackle_data->use_grackle                    = 1;
  grackle_data->with_radiative_cooling         = 1;
  grackle_data->primordial_chemistry           = g_deut ? 3 : 2;  /* +D,D+,HD if deut */
  grackle_data->equilibrium_deuterium          = g_deut;          /* advect only HD */
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
                                    double *HII, double *H2I, double *HDI)
{
  if (g_cd == NULL) return 0;
  g_units.a_value = a_value;

  /* gr_float work block (one malloc): rho,e_int,HII,H2I,HDI in/out copies +
     nsc reconstructed-species scratch + a zero-velocity array.  gr_float adapts
     to the linked grackle precision; the host arrays stay double. */
  long nsc = g_deut ? 9 : 7;
  long nf  = 5 + nsc + 1;
  gr_float *w = (gr_float *) malloc((size_t)nf * n * sizeof(gr_float));
  gr_float *gr_rho=w+0*n,*gr_e=w+1*n,*gr_HII=w+2*n,*gr_H2I=w+3*n,*gr_HDI=w+4*n;
  gr_float *sc = w + 5*n;
  gr_float *sc_e=sc+0*n,*sc_HI=sc+1*n,*sc_HeI=sc+2*n,*sc_HeII=sc+3*n,
           *sc_HeIII=sc+4*n,*sc_HM=sc+5*n,*sc_H2II=sc+6*n;
  gr_float *sc_DI = g_deut ? sc+7*n : NULL, *sc_DII = g_deut ? sc+8*n : NULL;
  gr_float *zero = w + (5+nsc)*n;
  const double tiny = 1e-20;
  for (long i = 0; i < n; i++) {
    gr_rho[i]=(gr_float)rho[i]; gr_e[i]=(gr_float)e_int[i];
    gr_HII[i]=(gr_float)HII[i]; gr_H2I[i]=(gr_float)H2I[i];
    gr_HDI[i]=(gr_float)(g_deut ? HDI[i] : 0.0); zero[i]=(gr_float)0.0;
    sc_e[i]   = gr_HII[i];                         /* n_e = n_HII */
    double hi = g_fh*rho[i] - HII[i] - H2I[i];
    sc_HI[i]  = (gr_float)((hi > tiny) ? hi : tiny);
    sc_HeI[i] = (gr_float)((1.0 - g_fh)*rho[i]);   /* all-neutral helium  */
    sc_HeII[i]=sc_HeIII[i]=sc_HM[i]=sc_H2II[i]=(gr_float)tiny;
    if (g_deut) {
      /* seed D, D+ at their cosmic partition (mass ratio 2*3.4e-5); the solver
         keeps D+ in charge-exchange equilibrium, reconstructs D from conservation */
      sc_DI[i]  = (gr_float)(6.8e-5 * sc_HI[i]);
      sc_DII[i] = (gr_float)(6.8e-5 * HII[i]);
    }
  }

  grackle_field_data f;
  int dim[3]   = { (int)n, 1, 1 };
  int gstart[3]= { 0, 0, 0 };
  int gend[3]  = { (int)n - 1, 0, 0 };
  f.grid_rank = 1; f.grid_dimension = dim; f.grid_start = gstart; f.grid_end = gend;
  f.grid_dx = 0.0;
  f.density = gr_rho; f.internal_energy = gr_e;
  f.x_velocity = zero; f.y_velocity = zero; f.z_velocity = zero;
  f.HI_density=sc_HI;   f.HII_density=gr_HII;  f.HeI_density=sc_HeI;
  f.HeII_density=sc_HeII; f.HeIII_density=sc_HeIII; f.e_density=sc_e;
  f.HM_density=sc_HM;   f.H2I_density=gr_H2I;  f.H2II_density=sc_H2II;
  f.DI_density=sc_DI;   f.DII_density=sc_DII;  f.HDI_density=g_deut?gr_HDI:NULL;
  f.metal_density=NULL; f.dust_density=NULL;
  f.volumetric_heating_rate=NULL; f.specific_heating_rate=NULL;
  f.RT_HI_ionization_rate=NULL; f.RT_HeI_ionization_rate=NULL;
  f.RT_HeII_ionization_rate=NULL; f.RT_H2_dissociation_rate=NULL;
  f.RT_heating_rate=NULL; f.H2_self_shielding_length=NULL;
  f.H2_custom_shielding_factor=NULL; f.isrf_habing=NULL;

  int rc = solve_chemistry(&g_units, &f, dt);
  /* copy the evolved in/out fields back to the host (double) arrays */
  for (long i = 0; i < n; i++) {
    e_int[i]=(double)gr_e[i]; HII[i]=(double)gr_HII[i]; H2I[i]=(double)gr_H2I[i];
    if (g_deut) HDI[i]=(double)gr_HDI[i];
  }
  free(w);
  return rc;
}

/* Per-cell gas temperature [K] for diagnostics (same reconstruction). */
extern "C" int grackle_reduced_temperature(long n, double a_value,
                                           double *rho, double *e_int,
                                           double *HII, double *H2I, double *Tout)
{
  if (g_cd == NULL) return 0;
  g_units.a_value = a_value;
  long nsc = g_deut ? 10 : 7;   /* +DI,DII,HDI scratch when deuterium on */
  /* gr_float block: rho,e_int,HII,H2I,Tout + nsc scratch + zero */
  long nf = 5 + nsc + 1;
  gr_float *w=(gr_float*)malloc((size_t)nf*n*sizeof(gr_float));
  gr_float *gr_rho=w+0*n,*gr_e=w+1*n,*gr_HII=w+2*n,*gr_H2I=w+3*n,*gr_T=w+4*n;
  gr_float *sc=w+5*n;
  gr_float *sc_e=sc+0*n,*sc_HI=sc+1*n,*sc_HeI=sc+2*n,*sc_HeII=sc+3*n,
           *sc_HeIII=sc+4*n,*sc_HM=sc+5*n,*sc_H2II=sc+6*n;
  gr_float *sc_DI=g_deut?sc+7*n:NULL,*sc_DII=g_deut?sc+8*n:NULL,*sc_HDI=g_deut?sc+9*n:NULL;
  gr_float *zero=w+(5+nsc)*n;
  const double tiny = 1e-20;
  for (long i = 0; i < n; i++) {
    gr_rho[i]=(gr_float)rho[i]; gr_e[i]=(gr_float)e_int[i];
    gr_HII[i]=(gr_float)HII[i]; gr_H2I[i]=(gr_float)H2I[i]; zero[i]=(gr_float)0.0;
    sc_e[i]=gr_HII[i]; double hi=g_fh*rho[i]-HII[i]-H2I[i];
    sc_HI[i]=(gr_float)((hi>tiny)?hi:tiny); sc_HeI[i]=(gr_float)((1.0-g_fh)*rho[i]);
    sc_HeII[i]=sc_HeIII[i]=sc_HM[i]=sc_H2II[i]=(gr_float)tiny;
    if (g_deut) { sc_DI[i]=(gr_float)tiny; sc_DII[i]=(gr_float)tiny; sc_HDI[i]=(gr_float)tiny; }
  }
  grackle_field_data f; int dim[3]={(int)n,1,1},gs[3]={0,0,0},ge[3]={(int)n-1,0,0};
  f.grid_rank=1; f.grid_dimension=dim; f.grid_start=gs; f.grid_end=ge; f.grid_dx=0.0;
  f.density=gr_rho; f.internal_energy=gr_e; f.x_velocity=zero; f.y_velocity=zero; f.z_velocity=zero;
  f.HI_density=sc_HI; f.HII_density=gr_HII; f.HeI_density=sc_HeI; f.HeII_density=sc_HeII;
  f.HeIII_density=sc_HeIII; f.e_density=sc_e; f.HM_density=sc_HM; f.H2I_density=gr_H2I;
  f.H2II_density=sc_H2II; f.DI_density=sc_DI; f.DII_density=sc_DII; f.HDI_density=sc_HDI;
  f.metal_density=NULL; f.dust_density=NULL;
  f.volumetric_heating_rate=NULL; f.specific_heating_rate=NULL;
  f.RT_HI_ionization_rate=NULL; f.RT_HeI_ionization_rate=NULL; f.RT_HeII_ionization_rate=NULL;
  f.RT_H2_dissociation_rate=NULL; f.RT_heating_rate=NULL; f.H2_self_shielding_length=NULL;
  f.H2_custom_shielding_factor=NULL; f.isrf_habing=NULL;
  int rc = calculate_temperature(&g_units, &f, gr_T);
  for (long i = 0; i < n; i++) Tout[i] = (double)gr_T[i];
  free(w);
  return rc;
}
