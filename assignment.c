#include <float.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int global_idx(int x, int y, int z, int NX, int NY) {
  return z * NX * NY + y * NX + x;
}

void read_data(const char *filename, double *data, int total_points,
               int time_steps) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    fprintf(stderr, "Error opening file: %s\n", filename);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  for (int i = 0; i < total_points * time_steps; i++) {
    fscanf(file, "%lf", &data[i]);
  }
  fclose(file);
}

void compute_local_extrema(double *sub_data, int nx, int ny, int nz, int nc,
                           int *local_min_count, int *local_max_count,
                           double *global_min, double *global_max) {
  for (int t = 0; t < nc; t++) {
    global_min[t] = DBL_MAX;
    global_max[t] = -DBL_MAX;
    local_min_count[t] = 0;
    local_max_count[t] = 0;

    for (int x = 0; x < nx; x++) {
      for (int y = 0; y < ny; y++) {
        for (int z = 0; z < nz; z++) {
          int index = global_idx(x, y, z, nx, ny) + t * (nx * ny * nz);
          double value = sub_data[index];

          if (value < global_min[t])
            global_min[t] = value;
          if (value > global_max[t])
            global_max[t] = value;

          int is_min = 1, is_max = 1;
          int dx[6] = {1, -1, 0, 0, 0, 0};
          int dy[6] = {0, 0, 1, -1, 0, 0};
          int dz[6] = {0, 0, 0, 0, 1, -1};
          for (int i = 0; i < 6; i++) {
            int nx_pos = x + dx[i], ny_pos = y + dy[i], nz_pos = z + dz[i];
            if (nx_pos >= 0 && nx_pos < nx && ny_pos >= 0 && ny_pos < ny &&
                nz_pos >= 0 && nz_pos < nz) {
              int neighbor_index = global_idx(nx_pos, ny_pos, nz_pos, nx, ny) +
                                   t * (nx * ny * nz);
              if (sub_data[neighbor_index] <= value)
                is_min = 0;
              if (sub_data[neighbor_index] >= value)
                is_max = 0;
            }
          }
          if (is_min)
            local_min_count[t]++;
          if (is_max)
            local_max_count[t]++;
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc < 10) {
    if (rank == 0)
      printf("Usage: mpirun -np P ./executable input.txt PX PY PZ NX NY NZ NC "
             "output.txt\n");
    MPI_Finalize();
    return 1;
  }

  const char *input_file = argv[1];
  int PX = atoi(argv[2]), PY = atoi(argv[3]), PZ = atoi(argv[4]);
  int NX = atoi(argv[5]), NY = atoi(argv[6]), NZ = atoi(argv[7]),
      NC = atoi(argv[8]);
  const char *output_file = argv[9];

  int total_points = NX * NY * NZ;
  int sub_nx = NX / PX, sub_ny = NY / PY, sub_nz = NZ / PZ;

  // Each process will handle sub_size number of points
  int sub_size = sub_nx * sub_ny * sub_nz;

  double *data = NULL;
  double time1 = MPI_Wtime();
  if (rank == 0) {
    data = (double *)malloc(total_points * NC * sizeof(double));
    read_data(input_file, data, total_points, NC);
  }

  double time2 = MPI_Wtime();
  double *sub_data = (double *)malloc(sub_size * NC * sizeof(double));
  MPI_Scatter(data, sub_size * NC, MPI_DOUBLE, sub_data, sub_size * NC,
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int *local_min_count = (int *)malloc(NC * sizeof(int));
  int *local_max_count = (int *)malloc(NC * sizeof(int));
  double *global_min = (double *)malloc(NC * sizeof(double));
  double *global_max = (double *)malloc(NC * sizeof(double));

  compute_local_extrema(sub_data, sub_nx, sub_ny, sub_nz, NC, local_min_count,
                        local_max_count, global_min, global_max);

  int *total_min_count = NULL, *total_max_count = NULL;
  double *final_min = NULL, *final_max = NULL;
  if (rank == 0) {
    total_min_count = (int *)malloc(NC * sizeof(int));
    total_max_count = (int *)malloc(NC * sizeof(int));
    final_min = (double *)malloc(NC * sizeof(double));
    final_max = (double *)malloc(NC * sizeof(double));
  }
  MPI_Reduce(local_min_count, total_min_count, NC, MPI_INT, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(local_max_count, total_max_count, NC, MPI_INT, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(global_min, final_min, NC, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(global_max, final_max, NC, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  double time3 = MPI_Wtime();
  double read_time = time2 - time1;
  double main_time = time3 - time2;
  double total_time = time3 - time1;
  double max_read_time, max_main_time, max_total_time;
  MPI_Reduce(&read_time, &max_read_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&main_time, &max_main_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  if (rank == 0) {
    FILE *output = fopen(output_file, "w");
    for (int t = 0; t < NC; t++) {
      fprintf(output, "(%d, %d) ", total_min_count[t], total_max_count[t]);
    }
    fprintf(output, "\n");
    for (int t = 0; t < NC; t++) {
      fprintf(output, "(%lf, %lf) ", final_min[t], final_max[t]);
    }
    fprintf(output, "\n");
    fprintf(output, "%lf %lf %lf\n", max_read_time, max_main_time,
            max_total_time);
    fclose(output);
  }

  free(sub_data);
  free(local_min_count);
  free(local_max_count);
  free(global_min);
  free(global_max);
  if (rank == 0) {
    free(data);
    free(total_min_count);
    free(total_max_count);
    free(final_min);
    free(final_max);
  }
  MPI_Finalize();
  return 0;
}
