/* grackle_tab_ref.c — libgrackle reference for BlockAMR's tabulated cooling
 * (src/cooling.jl): Grackle's Cloudy mode (primordial_chemistry = 0, metal
 * cooling, UV background).  Reads lines "nH T0 dt_Myr" on stdin, evolves one
 * cell each through local_solve_chemistry, prints "nH T0 dt_Myr e0 e1 T1 mu1"
 * (e in erg/g).  Usage: grackle_tab_ref <CloudyData.h5> <Z/Zsun> < cells.txt   */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define OMIT_LEGACY_INTERNAL_GRACKLE_FUNC
#include <grackle.h>

#define MH 1.67262171e-24
#define KB 1.3806504e-16

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s table.h5 Z_over_Zsun\n", argv[0]); return 1; }
  double Zrel = atof(argv[2]);
  grackle_verbose = 0;
  code_units u;
  u.comoving_coordinates = 0; u.a_units = 1.0; u.a_value = 1.0;
  u.density_units = MH; u.length_units = 3.0857e18; u.time_units = 3.15576e13;   /* pc, Myr */
  set_velocity_units(&u);
  chemistry_data *cd = malloc(sizeof(chemistry_data));
  set_default_chemistry_parameters(cd);
  cd->use_grackle = 1; cd->with_radiative_cooling = 1; cd->primordial_chemistry = 0;
  cd->metal_cooling = 1; cd->UVbackground = 1; cd->grackle_data_file = argv[1];
  cd->cmb_temperature_floor = 0; cd->Gamma = 5.0 / 3.0;
  chemistry_data_storage rates;
  if (local_initialize_chemistry_data(cd, &rates, &u) == 0) { fprintf(stderr, "init failed\n"); return 2; }
  double vu2 = u.velocity_units * u.velocity_units, X = cd->HydrogenFractionByMass;
  double nH, T0, dtM;
  while (scanf("%lf %lf %lf", &nH, &T0, &dtM) == 3) {
    int dim[3] = {1, 1, 1}, st[3] = {0, 0, 0}, en[3] = {0, 0, 0};
    gr_float d, e, met, vx = 0, vy = 0, vz = 0, temp, mu = 1.0;
    grackle_field_data f; memset(&f, 0, sizeof f);
    f.grid_rank = 3; f.grid_dimension = dim; f.grid_start = st; f.grid_end = en; f.grid_dx = 0.0;
    d = nH * MH / X / u.density_units;
    met = Zrel * cd->SolarMetalFractionByMass * d;
    f.density = &d; f.metal_density = &met; f.internal_energy = &e;
    f.x_velocity = &vx; f.y_velocity = &vy; f.z_velocity = &vz;
    /* e from T0 by Grackle's own temperature (iterate: T(e) = T0) */
    e = T0 * KB / ((cd->Gamma - 1.0) * 0.6 * MH) / vu2;
    for (int it = 0; it < 60; it++) {
      local_calculate_temperature(cd, &rates, &u, &f, &temp);
      e *= T0 / temp;
    }
    double e0 = e * vu2;
    if (local_solve_chemistry(cd, &rates, &u, &f, dtM) == 0) { fprintf(stderr, "solve failed\n"); return 3; }
    local_calculate_temperature(cd, &rates, &u, &f, &temp);
    mu = temp * KB / ((cd->Gamma - 1.0) * MH * e * vu2);
    printf("%.6e %.6e %.6e %.10e %.10e %.10e %.6f\n", nH, T0, dtM, e0, e * vu2, temp, mu);
  }
  return 0;
}
