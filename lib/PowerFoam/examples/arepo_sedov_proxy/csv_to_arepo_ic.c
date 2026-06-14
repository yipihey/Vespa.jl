#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>

static void check(herr_t status, const char *what)
{
    if(status < 0)
    {
        fprintf(stderr, "csv_to_arepo_ic: HDF5 error while %s\n", what);
        exit(2);
    }
}

static void write_attr_i32(hid_t group, const char *name, const int *data, hsize_t n)
{
    hid_t space = H5Screate_simple(1, &n, NULL);
    hid_t attr = H5Acreate2(group, name, H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
    check(H5Awrite(attr, H5T_NATIVE_INT, data), name);
    H5Aclose(attr);
    H5Sclose(space);
}

static void write_attr_i32_scalar(hid_t group, const char *name, int value)
{
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(group, name, H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
    check(H5Awrite(attr, H5T_NATIVE_INT, &value), name);
    H5Aclose(attr);
    H5Sclose(space);
}

static void write_attr_f64_scalar(hid_t group, const char *name, double value)
{
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(group, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
    check(H5Awrite(attr, H5T_NATIVE_DOUBLE, &value), name);
    H5Aclose(attr);
    H5Sclose(space);
}

static void write_dataset_f64(hid_t group, const char *name, const double *data,
                              int rank, const hsize_t *dims)
{
    hid_t space = H5Screate_simple(rank, dims, NULL);
    hid_t dset = H5Dcreate2(group, name, H5T_NATIVE_DOUBLE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    check(H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data), name);
    H5Dclose(dset);
    H5Sclose(space);
}

static void write_dataset_i32(hid_t group, const char *name, const int *data,
                              int rank, const hsize_t *dims)
{
    hid_t space = H5Screate_simple(rank, dims, NULL);
    hid_t dset = H5Dcreate2(group, name, H5T_NATIVE_INT, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    check(H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data), name);
    H5Dclose(dset);
    H5Sclose(space);
}

static size_t count_lines(FILE *fp)
{
    size_t n = 0;
    int c;
    while((c = fgetc(fp)) != EOF)
        if(c == '\n')
            n++;
    rewind(fp);
    return n;
}

int main(int argc, char **argv)
{
    if(argc != 3 && argc != 4)
    {
        fprintf(stderr, "usage: %s input.csv output.hdf5 [box_size]\n", argv[0]);
        return 1;
    }
    double box_size = argc == 4 ? atof(argv[3]) : 1.0;

    FILE *fp = fopen(argv[1], "r");
    if(!fp)
    {
        perror(argv[1]);
        return 1;
    }

    size_t n = count_lines(fp);
    int *ids = malloc(n * sizeof(int));
    double *coords = calloc(3 * n, sizeof(double));
    double *vel = calloc(3 * n, sizeof(double));
    double *mass = malloc(n * sizeof(double));
    double *u = malloc(n * sizeof(double));
    if(!ids || !coords || !vel || !mass || !u)
    {
        fprintf(stderr, "csv_to_arepo_ic: allocation failed for %zu cells\n", n);
        return 1;
    }

    for(size_t i = 0; i < n; i++)
    {
        double id, area, pressure, rho;
        int scanned = fscanf(fp, " %lf , %lf , %lf , %lf , %lf , %lf , %lf , %lf , %lf , %lf",
                             &id, &coords[3 * i], &coords[3 * i + 1],
                             &area, &mass[i], &vel[3 * i], &vel[3 * i + 1],
                             &u[i], &pressure, &rho);
        if(scanned != 10)
        {
            fprintf(stderr, "csv_to_arepo_ic: malformed CSV row %zu in %s\n", i + 1, argv[1]);
            return 1;
        }
        ids[i] = (int)id;
        coords[3 * i + 2] = 0.0;
        vel[3 * i + 2] = 0.0;
    }
    fclose(fp);

    hid_t file = H5Fcreate(argv[2], H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t header = H5Gcreate2(file, "Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t part0 = H5Gcreate2(file, "PartType0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    int numpart[6] = {(int)n, 0, 0, 0, 0, 0};
    int zeros[6] = {0, 0, 0, 0, 0, 0};
    write_attr_i32(header, "NumPart_ThisFile", numpart, 6);
    write_attr_i32(header, "NumPart_Total", numpart, 6);
    write_attr_i32(header, "NumPart_Total_HighWord", zeros, 6);
    write_attr_i32(header, "MassTable", zeros, 6);
    write_attr_f64_scalar(header, "Time", 0.0);
    write_attr_f64_scalar(header, "Redshift", 0.0);
    write_attr_f64_scalar(header, "BoxSize", box_size);
    write_attr_i32_scalar(header, "NumFilesPerSnapshot", 1);
    write_attr_f64_scalar(header, "Omega0", 0.0);
    write_attr_f64_scalar(header, "OmegaB", 0.0);
    write_attr_f64_scalar(header, "OmegaLambda", 0.0);
    write_attr_f64_scalar(header, "HubbleParam", 1.0);
    write_attr_i32_scalar(header, "Flag_Sfr", 0);
    write_attr_i32_scalar(header, "Flag_Cooling", 0);
    write_attr_i32_scalar(header, "Flag_StellarAge", 0);
    write_attr_i32_scalar(header, "Flag_Metals", 0);
    write_attr_i32_scalar(header, "Flag_Feedback", 0);
    write_attr_i32_scalar(header, "Flag_DoublePrecision", 1);
    write_attr_i32_scalar(header, "PowerFoamProxy", 1);

    hsize_t dims1[1] = {n};
    hsize_t dims2[2] = {n, 3};
    write_dataset_i32(part0, "ParticleIDs", ids, 1, dims1);
    write_dataset_f64(part0, "Coordinates", coords, 2, dims2);
    write_dataset_f64(part0, "Masses", mass, 1, dims1);
    write_dataset_f64(part0, "Velocities", vel, 2, dims2);
    write_dataset_f64(part0, "InternalEnergy", u, 1, dims1);

    H5Gclose(part0);
    H5Gclose(header);
    H5Fclose(file);

    free(ids);
    free(coords);
    free(vel);
    free(mass);
    free(u);
    return 0;
}
