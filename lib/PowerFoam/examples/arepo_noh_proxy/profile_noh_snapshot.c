#include <hdf5.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define GAMMA_GAS (5.0 / 3.0)

static void check(herr_t status, const char *what)
{
    if(status < 0)
    {
        fprintf(stderr, "profile_noh_snapshot: HDF5 error while %s\n", what);
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

static double read_attr_f64(hid_t file, const char *group_name, const char *attr_name)
{
    hid_t group = H5Gopen2(file, group_name, H5P_DEFAULT);
    hid_t attr = H5Aopen(group, attr_name, H5P_DEFAULT);
    double value = 0.0;
    check(H5Aread(attr, H5T_NATIVE_DOUBLE, &value), attr_name);
    H5Aclose(attr);
    H5Gclose(group);
    return value;
}

static int exists(hid_t file, const char *name)
{
    htri_t ok = H5Lexists(file, name, H5P_DEFAULT);
    return ok > 0;
}

static double analytic_density(double r, double t)
{
    if(r < 1e-12)
        r = 1e-12;
    if(r < t / 3.0)
        return 16.0;
    return 1.0 + t / r;
}

int main(int argc, char **argv)
{
    if(argc != 7)
    {
        fprintf(stderr, "usage: %s snapshot.hdf5 label cells.csv bins.csv metrics.csv nbins\n", argv[0]);
        return 1;
    }

    const char *snap = argv[1];
    const char *label = argv[2];
    const char *cells_path = argv[3];
    const char *bins_path = argv[4];
    const char *metrics_path = argv[5];
    int nbins = atoi(argv[6]);
    if(nbins <= 0)
    {
        fprintf(stderr, "profile_noh_snapshot: nbins must be positive\n");
        return 1;
    }

    hid_t file = H5Fopen(snap, H5F_ACC_RDONLY, H5P_DEFAULT);
    if(file < 0)
    {
        fprintf(stderr, "profile_noh_snapshot: cannot open %s\n", snap);
        return 1;
    }

    hsize_t n = dataset_len(file, "PartType0/Density");
    double *coords = malloc(3 * n * sizeof(double));
    double *vel = malloc(3 * n * sizeof(double));
    double *rho = malloc(n * sizeof(double));
    double *mass = malloc(n * sizeof(double));
    double *u = malloc(n * sizeof(double));
    double *volume = malloc(n * sizeof(double));
    double *mass_sum = calloc(nbins, sizeof(double));
    double *rho_sum = calloc(nbins, sizeof(double));
    double *rho2_sum = calloc(nbins, sizeof(double));
    double *ref_sum = calloc(nbins, sizeof(double));
    int *count = calloc(nbins, sizeof(int));
    if(!coords || !vel || !rho || !mass || !u || !volume || !mass_sum ||
       !rho_sum || !rho2_sum || !ref_sum || !count)
    {
        fprintf(stderr, "profile_noh_snapshot: allocation failed\n");
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
    if(exists(file, "PartType0/Volume"))
        read_f64(file, "PartType0/Volume", volume);
    else
        for(hsize_t i = 0; i < n; i++)
            volume[i] = mass[i] / rho[i];
    double time = read_attr_f64(file, "Header", "Time");
    double box = read_attr_f64(file, "Header", "BoxSize");
    H5Fclose(file);

    double compare_radius = 0.8;
    double max_radius = 0.5 * box;
    double shock_radius = time / 3.0;
    double mass_total = 0.0, energy_total = 0.0, volume_total = 0.0;
    double l1_num = 0.0, l2_num = 0.0, wsum = 0.0;
    double post_sum = 0.0, post_sum2 = 0.0;
    int post_count = 0;

    FILE *cells = fopen(cells_path, "a");
    if(!cells)
    {
        perror(cells_path);
        return 1;
    }

    for(hsize_t i = 0; i < n; i++)
    {
        double dx = coords[3 * i] - 0.5 * box;
        double dy = coords[3 * i + 1] - 0.5 * box;
        double r = sqrt(dx * dx + dy * dy);
        double ref = analytic_density(r, time);
        double vr = 0.0;
        if(r > 1e-12)
            vr = (vel[3 * i] * dx + vel[3 * i + 1] * dy) / r;

        fprintf(cells, "%s,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                label, r, rho[i], ref, vr, u[i]);

        int b = (int)floor(r / max_radius * nbins);
        if(b < 0)
            b = 0;
        if(b >= nbins)
            b = nbins - 1;
        mass_sum[b] += mass[i];
        rho_sum[b] += mass[i] * rho[i];
        rho2_sum[b] += mass[i] * rho[i] * rho[i];
        ref_sum[b] += mass[i] * ref;
        count[b]++;

        double v2 = vel[3 * i] * vel[3 * i] + vel[3 * i + 1] * vel[3 * i + 1];
        mass_total += mass[i];
        energy_total += mass[i] * u[i] + 0.5 * mass[i] * v2;
        volume_total += volume[i];

        if(r < compare_radius)
        {
            double rel = fabs(rho[i] - ref) / ref;
            l1_num += volume[i] * rel;
            l2_num += volume[i] * rel * rel;
            wsum += volume[i];
        }
        if(r < shock_radius && r > 0.08)
        {
            post_sum += rho[i];
            post_sum2 += rho[i] * rho[i];
            post_count++;
        }
    }
    fclose(cells);

    FILE *bins = fopen(bins_path, "a");
    if(!bins)
    {
        perror(bins_path);
        return 1;
    }
    for(int b = 0; b < nbins; b++)
    {
        double r = (b + 0.5) * max_radius / nbins;
        double mean = 0.0, scatter = 0.0, ref = 0.0;
        if(mass_sum[b] > 0.0)
        {
            mean = rho_sum[b] / mass_sum[b];
            ref = ref_sum[b] / mass_sum[b];
            double var = rho2_sum[b] / mass_sum[b] - mean * mean;
            scatter = sqrt(var > 0.0 ? var : 0.0);
        }
        fprintf(bins, "%s,%.17g,%.17g,%.17g,%.17g,%d\n",
                label, r, mean, ref, scatter, count[b]);
    }
    fclose(bins);

    double l1 = wsum > 0.0 ? l1_num / wsum : 0.0;
    double l2 = wsum > 0.0 ? sqrt(l2_num / wsum) : 0.0;
    double post_mean = post_count > 0 ? post_sum / post_count : 0.0;
    double post_std = 0.0;
    if(post_count > 0)
    {
        double var = post_sum2 / post_count - post_mean * post_mean;
        post_std = sqrt(var > 0.0 ? var : 0.0);
    }

    FILE *met = fopen(metrics_path, "a");
    if(!met)
    {
        perror(metrics_path);
        return 1;
    }
    fprintf(met, "%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
            label, time, shock_radius, l1, l2, post_mean, post_std,
            mass_total, energy_total, volume_total);
    fclose(met);

    free(coords);
    free(vel);
    free(rho);
    free(mass);
    free(u);
    free(volume);
    free(mass_sum);
    free(rho_sum);
    free(rho2_sum);
    free(ref_sum);
    free(count);
    return 0;
}
