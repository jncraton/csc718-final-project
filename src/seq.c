#include <stdio.h>
#include <stdlib.h>

#define N 100
#define ITERATIONS 400

const double G = 6.674E-11;
const double STEP_SIZE = 100.0;
const double earth_radius = 6.357E6;


double *x, *y, *z, *dx, *dy, *dz, *m;

void update_velocity(double *pos, double *vel, double *mass) {
  //F = GmM/r^2
  //a = Gm/r^2 m is mass of the other, our mass cancels
  //r^2 = distance^2
  //distance = sqrt(x^2 + y^2 + z^2)
  //r^2 = x^2 + y^2 + z^2

  for (int i = 1; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (i != j) {
        double sep = pos[j] - pos[i];
        if (sep > 1.0) {
          vel[i] += G*mass[j]/(pos[j] - pos[i]);
        }
      }
    }
  }
}

void update_position(double *pos, double *vel) {
  for (int i = 1; i < N; i++) {
    pos[i] += vel[i] * STEP_SIZE;
  }
}

void save_results() {
  static int result_set = 0;
  char buf[32];
  snprintf(buf, 32, "results/%d.csv", result_set);
  result_set++;
  
  printf("%s\n", buf);

  FILE* file = fopen(buf,"w");
  for (int i = 0; i < N; i++) {
    fprintf(file, "%f,%f,%f,%f,%f,%f\n", x[i], y[i], z[i], dx[i], dy[i], dz[i]);
  }
  fclose(file);
}

int main() {
  m = aligned_alloc(64, N*sizeof(double));
  x = aligned_alloc(64, N*sizeof(double));
  y = aligned_alloc(64, N*sizeof(double));
  z = aligned_alloc(64, N*sizeof(double));
  dx = aligned_alloc(64, N*sizeof(double));
  dy = aligned_alloc(64, N*sizeof(double));
  dz = aligned_alloc(64, N*sizeof(double));

  m[0] = 6E24;
  for (int i = 1; i<N; i++) {
    m[i] = 10000 * (double)rand() / (double)(RAND_MAX);
    dx[i] = 3000*(((double)rand() / (double)(RAND_MAX))-0.5);
    dy[i] = 6000 + 3000*(((double)rand() / (double)(RAND_MAX))-0.5);
    dz[i] = 3000*(((double)rand() / (double)(RAND_MAX))-0.5);
    x[i] = earth_radius;
    y[i] = 0;
    z[i] = 0;
  }

  for (int i = 0; i <= ITERATIONS; i++) {
    if (!(i % 20)) {
      printf("%d iterations completed. Writing results...\n", i);
      save_results();
    }

    // Update accelerations
    update_velocity(x, dx, m);
    update_velocity(y, dy, m);
    update_velocity(z, dz, m);

    // Update positions
    update_position(x, dx);
    update_position(y, dy);
    update_position(z, dz);
  }

  return 0;
}