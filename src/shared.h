#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N0 8192
int N = N0;
#define ITERATIONS 400
#define LOG_EVERY  10

#define G 6.674E-11
#define STEP_SIZE 100
#define mean_particle_mass 1E12
#define earth_radius 6.357E6
#define earth_mass 6.000E24

typedef float fptype;

struct Body {
  fptype mass;
  fptype radius;
  fptype x;
  fptype y;
  fptype z;
  fptype dx;
  fptype dy;
  fptype dz;
  char collisions;
};

struct Body *bodies;

fptype get_radius(fptype mass) {
  static fptype earth_density = earth_mass / ((4.0 / 3.0) * 3.14159 * earth_radius * earth_radius * earth_radius);
  
  fptype volume = mass / earth_density;
   // Volume = (4/3) pi r^3
  // r^3 = volume * (3/4) / pi
  fptype r3 = volume * (3.0/4.0) / 3.14159;
  return cbrt(r3);
}

void save_results(Body * bodies, int time) {
  static int result_set = 0;
  char buf[32];
  snprintf(buf, 32, "%d.csv", result_set);
  result_set++;
  
  FILE* file = fopen(buf,"w");
  for (int i = 0; i < N; i++) {
    if (bodies[i].mass > 0) {
      fprintf(file, "%f,%f,%f,%f,%f,%f,%f,%f,%d\n", 
        bodies[i].x,
        bodies[i].y,
        bodies[i].z,
        bodies[i].dx,
        bodies[i].dy,
        bodies[i].dz,
        bodies[i].mass,
        bodies[i].radius,
        time
      );
    }
  }
  fclose(file);
}
