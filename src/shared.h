#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int N = 3000;
#define ITERATIONS 400
#define LOG_EVERY  10

#define G 6.674E-11
#define STEP_SIZE 100
#define mean_particle_mass 1E13
#define earth_radius 6.357E6
#define earth_mass 6.000E24

struct Body {
  double mass;
  double radius;
  double x;
  double y;
  double z;
  double dx;
  double dy;
  double dz;
};

struct Body *bodies;

double get_radius(double mass) {
  static double earth_density = earth_mass / ((4.0 / 3.0) * 3.14159 * earth_radius * earth_radius * earth_radius);
  
  double volume = mass / earth_density;
   // Volume = (4/3) pi r^3
  // r^3 = volume * (3/4) / pi
  double r3 = volume * (3.0/4.0) / 3.14159;
  return cbrt(r3);
}

void save_results() {
  static int result_set = 0;
  char buf[32];
  snprintf(buf, 32, "results/%d.csv", result_set);
  result_set++;
  
  FILE* file = fopen(buf,"w");
  for (int i = 0; i < N; i++) {
    if (bodies[i].mass > 0) {
      fprintf(file, "%f,%f,%f,%f,%f,%f,%f,%f\n", 
        bodies[i].x,
        bodies[i].y,
        bodies[i].z,
        bodies[i].dx,
        bodies[i].dy,
        bodies[i].dz,
        bodies[i].mass,
        bodies[i].radius);
    }
  }
  fclose(file);
}
