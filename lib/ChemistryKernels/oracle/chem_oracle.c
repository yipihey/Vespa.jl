/* chem_oracle.c — the verification backbone for ChemistryKernels.
 *
 * Exposes grackle's OWN analytic rate / cooling-coefficient functions evaluated
 * with units=1.0 (so they return the exact CGS formula value, table-free), plus
 * a reduced-network temperature.  Each ported Julia formula is diffed against the
 * matching chem_rate()/chem_cool() here over a temperature grid — a single
 * transcribed-digit typo then fails immediately at rtol~1e-12.
 *
 * The rate/cooling functions are pure formulas of (T, units, chemistry_data):
 * they read only the flags from chemistry_data, so set_default_chemistry_parameters
 * + the reduced-network flags is sufficient (no table init / no data file needed).
 *
 * Build: oracle/build_chem_oracle.sh  (links the installed grackle fork; the
 * precision of the returned doubles is the grackle build precision — use the f64
 * install for the bit-tight per-rate oracle).
 */
#include <stdlib.h>
#include <string.h>
extern "C" {
#include <grackle.h>
#include <grackle_rate_functions.h>
}

static chemistry_data g_cd;
static int g_ready = 0;

/* Configure the reduced-network flags.  caseB toggles CaseBRecombination;
 * the rate/cooling functions branch on the *_rates toggles, so expose them too. */
extern "C" void chem_set_flags(int caseB, int colexc, int colion,
                               int reccool, int brems) {
    set_default_chemistry_parameters(&g_cd);
    g_cd.primordial_chemistry           = 3;     /* full set of fns available  */
    g_cd.CaseBRecombination             = caseB;
    g_cd.collisional_excitation_rates   = colexc;
    g_cd.collisional_ionisation_rates   = colion;
    g_cd.recombination_cooling_rates    = reccool;
    g_cd.bremsstrahlung_cooling_rates   = brems;
    /* h2/three-body formulation defaults (grackle_reduced uses the defaults) */
    g_ready = 1;
}

static void ensure_ready(void) { if (!g_ready) chem_set_flags(1,1,1,1,1); }

/* ---- reduced-network gas temperature (the v2026 model) --------------------
 * Mirrors lib/MultiCode/deps/grackle_reduced.c's temperature path: advect
 * {rho, HII, H2I(, HDI)}; reconstruct HI = fh*rho - HII - H2I; helium all
 * neutral; n_e = n_HII; HM/H2II (and DI/DII/HDI scratch) = tiny; then call
 * grackle's calculate_temperature (mmw + H2 variable-gamma).  Lazily initializes
 * a separate chemistry_data for the reduced flags (table-free: metal_cooling and
 * UVbackground off, so no Cloudy file is read).  `data_file` may be "" — if a
 * non-empty path is given it is used (some grackle builds insist on a file). */
static chemistry_data g_tcd;
static code_units     g_tunits;
static int            g_tready = 0;
static double         g_tfh    = 0.76;
static int            g_tdeut  = 0;

extern "C" int chem_temperature_init(double hubble_kmsmpc, double Om, double OL,
        double a_value, double fh, double density_units, double length_units,
        double time_units, const char *data_file, int deuterium) {
    g_tdeut = deuterium ? 1 : 0;
    g_tfh   = fh;
    if (set_default_chemistry_parameters(&g_tcd) == 0) return 0;
    g_tcd.use_grackle                  = 1;
    g_tcd.with_radiative_cooling       = 1;
    g_tcd.primordial_chemistry         = g_tdeut ? 3 : 2;
    g_tcd.equilibrium_deuterium        = g_tdeut;
    g_tcd.metal_cooling                = 0;
    g_tcd.UVbackground                 = 0;
    g_tcd.cmb_dissociation             = 1;
    g_tcd.equilibrium_h2_intermediates = 1;
    g_tcd.neutral_helium               = 1;
    g_tcd.cmb_recombination            = 1;
    g_tcd.cosmology_hubble_constant_now= hubble_kmsmpc;
    g_tcd.cosmology_omega_matter_now   = Om;
    g_tcd.cosmology_omega_lambda_now   = OL;
    g_tcd.HydrogenFractionByMass       = fh;
    g_tcd.grackle_data_file            = data_file;
    g_tunits.comoving_coordinates = 0;
    g_tunits.density_units        = density_units;
    g_tunits.length_units         = length_units;
    g_tunits.time_units           = time_units;
    g_tunits.a_units              = 1.0;
    g_tunits.a_value              = a_value;
    set_velocity_units(&g_tunits);
    /* initialize_chemistry_data uses the GLOBAL grackle_data pointer */
    chemistry_data *saved = grackle_data;
    grackle_data = &g_tcd;
    int rc = initialize_chemistry_data(&g_tunits);
    grackle_data = saved;
    g_tready = (rc == 1);
    return rc;
}

extern "C" int chem_temperature(long n, double a_value, double *rho,
        double *e_int, double *HII, double *H2I, double *Tout) {
    if (!g_tready) return 0;
    g_tunits.a_value = a_value;
    long nsc = g_tdeut ? 10 : 7;
    long nf  = 5 + nsc + 1;
    double *w = (double*)malloc((size_t)nf*n*sizeof(double));
    double *gr_rho=w+0*n,*gr_e=w+1*n,*gr_HII=w+2*n,*gr_H2I=w+3*n,*gr_T=w+4*n;
    double *sc=w+5*n;
    double *sc_e=sc+0*n,*sc_HI=sc+1*n,*sc_HeI=sc+2*n,*sc_HeII=sc+3*n,
           *sc_HeIII=sc+4*n,*sc_HM=sc+5*n,*sc_H2II=sc+6*n;
    double *sc_DI=g_tdeut?sc+7*n:NULL,*sc_DII=g_tdeut?sc+8*n:NULL,*sc_HDI=g_tdeut?sc+9*n:NULL;
    double *zero=w+(5+nsc)*n;
    const double tiny = 1e-20;
    for (long i=0;i<n;i++){
        gr_rho[i]=rho[i]; gr_e[i]=e_int[i]; gr_HII[i]=HII[i]; gr_H2I[i]=H2I[i]; zero[i]=0.0;
        sc_e[i]=HII[i]; double hi=g_tfh*rho[i]-HII[i]-H2I[i];
        sc_HI[i]=(hi>tiny)?hi:tiny; sc_HeI[i]=(1.0-g_tfh)*rho[i];
        sc_HeII[i]=sc_HeIII[i]=sc_HM[i]=sc_H2II[i]=tiny;
        if (g_tdeut){ sc_DI[i]=tiny; sc_DII[i]=tiny; sc_HDI[i]=tiny; }
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
    chemistry_data *saved = grackle_data; grackle_data = &g_tcd;
    int rc = calculate_temperature(&g_tunits, &f, gr_T);
    grackle_data = saved;
    for (long i=0;i<n;i++) Tout[i]=gr_T[i];
    free(w);
    return rc;
}

/* ---- reduced-network cooling time (grackle calculate_cooling_time) ---------
 * Same reconstruction/units as chem_temperature; returns t_cool [code time
 * units] so the test can derive ė = e_int/t_cool and sanity-check edot.jl's
 * assembly (the H2/HD/Compton dom-factor bookkeeping).  Uses the analytically
 * tabulated rates from chem_temperature_init (interp error ~O(1e-3); the tight
 * gate is the Wave-5 high-bin one-zone). */
extern "C" int chem_cooling_time(long n, double a_value, double *rho,
        double *e_int, double *HII, double *H2I, double *tcool) {
    if (!g_tready) return 0;
    g_tunits.a_value = a_value;
    long nsc = g_tdeut ? 10 : 7;
    long nf  = 5 + nsc + 1;
    double *w = (double*)malloc((size_t)nf*n*sizeof(double));
    double *gr_rho=w+0*n,*gr_e=w+1*n,*gr_HII=w+2*n,*gr_H2I=w+3*n,*gr_tc=w+4*n;
    double *sc=w+5*n;
    double *sc_e=sc+0*n,*sc_HI=sc+1*n,*sc_HeI=sc+2*n,*sc_HeII=sc+3*n,
           *sc_HeIII=sc+4*n,*sc_HM=sc+5*n,*sc_H2II=sc+6*n;
    double *sc_DI=g_tdeut?sc+7*n:NULL,*sc_DII=g_tdeut?sc+8*n:NULL,*sc_HDI=g_tdeut?sc+9*n:NULL;
    double *zero=w+(5+nsc)*n;
    const double tiny = 1e-20;
    for (long i=0;i<n;i++){
        gr_rho[i]=rho[i]; gr_e[i]=e_int[i]; gr_HII[i]=HII[i]; gr_H2I[i]=H2I[i]; zero[i]=0.0;
        sc_e[i]=HII[i]; double hi=g_tfh*rho[i]-HII[i]-H2I[i];
        sc_HI[i]=(hi>tiny)?hi:tiny; sc_HeI[i]=(1.0-g_tfh)*rho[i];
        sc_HeII[i]=sc_HeIII[i]=sc_HM[i]=sc_H2II[i]=tiny;
        if (g_tdeut){ sc_DI[i]=tiny; sc_DII[i]=tiny; sc_HDI[i]=tiny; }
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
    chemistry_data *saved = grackle_data; grackle_data = &g_tcd;
    int rc = calculate_cooling_time(&g_tunits, &f, gr_tc);
    grackle_data = saved;
    for (long i=0;i<n;i++) tcool[i]=gr_tc[i];
    free(w);
    return rc;
}

/* ---- per-reaction rate (CGS, units=1) -------------------------------------- */
extern "C" double chem_rate(const char *name, double T) {
    ensure_ready();
    chemistry_data *cd = &g_cd;
    const double u = 1.0;
    if (!strcmp(name,"k1"))  return k1_rate(T,u,cd);
    if (!strcmp(name,"k2"))  return k2_rate(T,u,cd);
    if (!strcmp(name,"k3"))  return k3_rate(T,u,cd);
    if (!strcmp(name,"k4"))  return k4_rate(T,u,cd);
    if (!strcmp(name,"k5"))  return k5_rate(T,u,cd);
    if (!strcmp(name,"k6"))  return k6_rate(T,u,cd);
    if (!strcmp(name,"k7"))  return k7_rate(T,u,cd);
    if (!strcmp(name,"k8"))  return k8_rate(T,u,cd);
    if (!strcmp(name,"k9"))  return k9_rate(T,u,cd);
    if (!strcmp(name,"k10")) return k10_rate(T,u,cd);
    if (!strcmp(name,"k11")) return k11_rate(T,u,cd);
    if (!strcmp(name,"k12")) return k12_rate(T,u,cd);
    if (!strcmp(name,"k13")) return k13_rate(T,u,cd);
    if (!strcmp(name,"k14")) return k14_rate(T,u,cd);
    if (!strcmp(name,"k15")) return k15_rate(T,u,cd);
    if (!strcmp(name,"k16")) return k16_rate(T,u,cd);
    if (!strcmp(name,"k17")) return k17_rate(T,u,cd);
    if (!strcmp(name,"k18")) return k18_rate(T,u,cd);
    if (!strcmp(name,"k19")) return k19_rate(T,u,cd);
    if (!strcmp(name,"k22")) return k22_rate(T,u,cd);
    if (!strcmp(name,"k50")) return k50_rate(T,u,cd);
    if (!strcmp(name,"k51")) return k51_rate(T,u,cd);
    if (!strcmp(name,"k52")) return k52_rate(T,u,cd);
    if (!strcmp(name,"k53")) return k53_rate(T,u,cd);
    if (!strcmp(name,"k54")) return k54_rate(T,u,cd);
    if (!strcmp(name,"k55")) return k55_rate(T,u,cd);
    if (!strcmp(name,"k56")) return k56_rate(T,u,cd);
    if (!strcmp(name,"k57")) return k57_rate(T,u,cd);
    if (!strcmp(name,"k58")) return k58_rate(T,u,cd);
    return -1.0;  /* unknown name */
}

/* ---- per cooling/heating coefficient (CGS, units=1) ------------------------ */
extern "C" double chem_cool(const char *name, double T) {
    ensure_ready();
    chemistry_data *cd = &g_cd;
    const double u = 1.0;
    if (!strcmp(name,"ceHI"))    return ceHI_rate(T,u,cd);
    if (!strcmp(name,"ceHeI"))   return ceHeI_rate(T,u,cd);
    if (!strcmp(name,"ceHeII"))  return ceHeII_rate(T,u,cd);
    if (!strcmp(name,"ciHI"))    return ciHI_rate(T,u,cd);
    if (!strcmp(name,"ciHeI"))   return ciHeI_rate(T,u,cd);
    if (!strcmp(name,"ciHeII"))  return ciHeII_rate(T,u,cd);
    if (!strcmp(name,"ciHeIS"))  return ciHeIS_rate(T,u,cd);
    if (!strcmp(name,"reHII"))   return reHII_rate(T,u,cd);
    if (!strcmp(name,"reHeII1")) return reHeII1_rate(T,u,cd);
    if (!strcmp(name,"reHeII2")) return reHeII2_rate(T,u,cd);
    if (!strcmp(name,"reHeIII")) return reHeIII_rate(T,u,cd);
    if (!strcmp(name,"brem"))    return brem_rate(T,u,cd);
    if (!strcmp(name,"GAHI"))    return GAHI_rate(T,u,cd);
    if (!strcmp(name,"GAH2"))    return GAH2_rate(T,u,cd);
    if (!strcmp(name,"GAHe"))    return GAHe_rate(T,u,cd);
    if (!strcmp(name,"GAHp"))    return GAHp_rate(T,u,cd);
    if (!strcmp(name,"GAel"))    return GAel_rate(T,u,cd);
    if (!strcmp(name,"H2LTE"))   return H2LTE_rate(T,u,cd);
    if (!strcmp(name,"HDlte"))   return HDlte_rate(T,u,cd);
    if (!strcmp(name,"HDlow"))   return HDlow_rate(T,u,cd);
    return -1.0;
}
