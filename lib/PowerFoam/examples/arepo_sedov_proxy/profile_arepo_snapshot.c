#include <hdf5.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GAMMA_GAS (5.0 / 3.0)

static void check(herr_t status, const char *what)
{
    if(status < 0)
    {
        fprintf(stderr, "profile_arepo_snapshot: HDF5 error while %s\n", what);
        exit(2);
    }
}

static hsize_t dataset_len(hid_t file, const char *name)
{
    hid_t dset = H5Dopen2(file, name, H5P_DEFAULT);
    hid_t space = H5Dget_space(dset);
    hsize_t dims[2] = {0, 0};
    H5Sget_simple_extent_dims(space, dims, NULL);
    H5Sclose(space);
    H5Dclose(dset);
    return dims[0];
}

static void read_f64(hid_t file, const char *name, double *data)
{
    hid_t dset = H5Dopen2(file, name, H5P_DEFAULT);
    check(H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data), name);
    H5Dclose(dset);
}

static double read_attr_time(hid_t file)
{
    hid_t header = H5Gopen2(file, "Header", H5P_DEFAULT);
    hid_t attr = H5Aopen(header, "Time", H5P_DEFAULT);
    double time = 0.0;
    check(H5Aread(attr, H5T_NATIVE_DOUBLE, &time), "Header/Time");
    H5Aclose(attr);
    H5Gclose(header);
    return time;
}

static int exists(hid_t file, const char *name)
{
    htri_t ok = H5Lexists(file, name, H5P_DEFAULT);
    return ok > 0;
}

int main(int argc, char **argv)
{
    if(argc != 6)
    {
        fprintf(stderr, "usage: %s snapshot.hdf5 label profile.csv metrics.csv nbins\n", argv[0]);
        return 1;
    }

    const char *snap = argv[1];
    const char *label = argv[2];
    const char *profile_path = argv[3];
    const char *metrics_path = argv[4];
    int nbins = atoi(argv[5]);
    if(nbins <= 0)
    {
        fprintf(stderr, "profile_arepo_snapshot: nbins must be positive\n");
        return 1;
    }

    hid_t file = H5Fopen(snap, H5F_ACC_RDONLY, H5P_DEFAULT);
    if(file < 0)
    {
        fprintf(stderr, "profile_arepo_snapshot: cannot open %s\n", snap);
        return 1;
    }

    hsize_t n = dataset_len(file, "PartType0/Density");
    double *coords = malloc(3 * n * sizeof(double));
    double *vel = malloc(3 * n * sizeof(double));
    double *rho = malloc(n * sizeof(double));
    double *mass = malloc(n * sizeof(double));
    double *u = malloc(n * sizeof(double));
    double *volume = malloc(n * sizeof(double));
    double *pressure = malloc(n * sizeof(double));
    double *mass_sum = calloc(nbins, sizeof(double));
    double *rho_sum = calloc(nbins, sizeof(double));
    double *rho2_sum = calloc(nbins, sizeof(double));
    int *count = calloc(nbins, sizeof(int));
    if(!coords || !vel || !rho || !mass || !u || !volume || !pressure ||
       !mass_sum || !rho_sum || !rho2_sum || !count)
    {
        fprintf(stderr, "profile_arepo_snapshot: allocation failed\n");
        return 1;
    }

    if(exists(file, "PartType0/CenterOfMass"))
        read_f64(file, "PartType0/CenterOfMass", coords);
    else
        read_f64(file, "PartType0/Coordinates", coords);
    read_f64(file, "PartType0/Velocities", vel);
    read_f64(file, "PartType0/Density", rho);
    read_f64(file, "PartType0/Masses", mass);
    read_f64(file, "PartType0/InternalEnergy", u);
    read_f64(file, "PartType0/Volume", volume);
    if(exists(file, "PartType0/Pressure"))
        read_f64(file, "PartType0/Pressure", pressure);
    else
        for(hsize_t i = 0; i < n; i++)
            pressure[i] = (GAMMA_GAS - 1.0) * rho[i] * u[i];
    double time = read_attr_time(file);
    H5Fclose(file);

    double mass_total = 0.0, energy_total = 0.0, volume_total = 0.0, pressure_max = 0.0;
    for(hsize_t i = 0; i < n; i++)
    {
        double dx = coords[3 * i] - 0.5;
        double dy = coords[3 * i + 1] - 0.5;
        dx -= nearbyint(dx);
        dy -= nearbyint(dy);
        double r = sqrt(dx * dx + dy * dy);
        int b = (int)floor(r / 0.5 * nbins);
        if(b < 0)
            b = 0;
        if(b >= nbins)
            b = nbins - 1;
        mass_sum[b] += mass[i];
        rho_sum[b] += mass[i] * rho[i];
        rho2_sum[b] += mass[i] * rho[i] * rho[i];
        count[b]++;

        double v2 = vel[3 * i] * vel[3 * i] + vel[3 * i + 1] * vel[3 * i + 1];
        mass_total += mass[i];
        energy_total += mass[i] * u[i] + 0.5 * mass[i] * v2;
        volume_total += volume[i];
        if(pressure[i] > pressure_max)
            pressure_max = pressure[i];
    }

    double *mean = calloc(nbins, sizeof(double));
    double *scatter = calloc(nbins, sizeof(double));
    int peak = 0;
    for(int b = 0; b < nbins; b++)
    {
        if(mass_sum[b] > 0.0)
        {
            mean[b] = rho_sum[b] / mass_sum[b];
            double var = rho2_sum[b] / mass_sum[b] - mean[b] * mean[b];
            scatter[b] = sqrt(var > 0.0 ? var : 0.0);
        }
        if(mean[b] > mean[peak])
            peak = b;
    }

    double shock_radius = (peak + 0.5) * 0.5 / nbins;
    double shell_scatter = 0.0;
    int shell_bins = 0;
    for(int b = peak - 2; b <= peak + 2; b++)
        if(b >= 0 && b < nbins)
        {
            shell_scatter += scatter[b];
            shell_bins++;
        }
    shell_scatter = shell_bins > 0 ? shell_scatter / shell_bins : 0.0;

    double shell_sum = 0.0, shell_sum2 = 0.0;
    int shell_count = 0;
    for(hsize_t i = 0; i < n; i++)
    {
        double dx = coords[3 * i] - 0.5;
        double dy = coords[3 * i + 1] - 0.5;
        dx -= nearbyint(dx);
        dy -= nearbyint(dy);
        double r = sqrt(dx * dx + dy * dy);
        if(fabs(r - shock_radius) < 0.035)
        {
            shell_sum += rho[i];
            shell_sum2 += rho[i] * rho[i];
            shell_count++;
        }
    }
    double shell_density_std = 0.0;
    if(shell_count > 0)
    {
        double m = shell_sum / shell_count;
        double v = shell_sum2 / shell_count - m * m;
        shell_density_std = sqrt(v > 0.0 ? v : 0.0);
    }

    FILE *prof = fopen(profile_path, "a");
    if(!prof)
    {
        perror(profile_path);
        return 1;
    }
    for(int b = 0; b < nbins; b++)
    {
        double r = (b + 0.5) * 0.5 / nbins;
        fprintf(prof, "%s,%.17g,%.17g,%.17g,%d\n", label, r, mean[b], scatter[b], count[b]);
    }
    fclose(prof);

    FILE *met = fopen(metrics_path, "a");
    if(!met)
    {
        perror(metrics_path);
        return 1;
    }
    fprintf(met, "%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
            label, time, shock_radius, mean[peak], shell_scatter, shell_density_std,
            mass_total, energy_total, pressure_max, volume_total);
    fclose(met);

    free(coords);
    free(vel);
    free(rho);
    free(mass);
    free(u);
    free(volume);
    free(pressure);
    free(mass_sum);
    free(rho_sum);
    free(rho2_sum);
    free(count);
    free(mean);
    free(scatter);
    return 0;
}
