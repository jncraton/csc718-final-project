#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int N = 400;
#define ITERATIONS 40000
#define LOG_EVERY  1000

#define G 6.674E-11
#define STEP_SIZE 1
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

void update_velocity(struct Body *bodies) {
  //F = GmM/r^2
  //a = Gm/r^2 m is mass of the other, our mass cancels
  //r^2 = distance^2
  //distance = sqrt(x^2 + y^2 + z^2)
  //r^2 = x^2 + y^2 + z^2

  for (int i = 1; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (i != j) {
        double r2 = (
          (bodies[i].x-bodies[j].x) * (bodies[i].x-bodies[j].x) +
          (bodies[i].y-bodies[j].y) * (bodies[i].y-bodies[j].y) +
          (bodies[i].z-bodies[j].z) * (bodies[i].z-bodies[j].z)
        );

        if (r2 < (bodies[i].radius + bodies[j].radius) * (bodies[i].radius + bodies[j].radius)) {
            double new_mass = bodies[i].mass + bodies[j].mass;
            bodies[j].dx = (bodies[j].dx * bodies[j].mass + bodies[i].dx * bodies[i].mass) / new_mass;
            bodies[j].dy = (bodies[j].dy * bodies[j].mass + bodies[i].dy * bodies[i].mass) / new_mass;
            bodies[j].dz = (bodies[j].dz * bodies[j].mass + bodies[i].dz * bodies[i].mass) / new_mass;
            bodies[j].mass = new_mass;
            bodies[j].radius = get_radius(new_mass);
            bodies[i] = bodies[N-1];
            N--;
        } else {
          bodies[i].dx += ((bodies[j].x - bodies[i].x) / sqrt(r2)) * 
            STEP_SIZE*G*bodies[j].mass/r2;
          bodies[i].dy += ((bodies[j].y - bodies[i].y) / sqrt(r2)) * 
            STEP_SIZE*G*bodies[j].mass/r2;
          bodies[i].dz += ((bodies[j].z - bodies[i].z) / sqrt(r2)) * 
            STEP_SIZE*G*bodies[j].mass/r2;
        }
      }
    }
  }
}

void update_position(struct Body *bodies) {
  for (int i = 1; i < N; i++) {
    bodies[i].x += bodies[i].dx * STEP_SIZE;
    bodies[i].y += bodies[i].dy * STEP_SIZE;
    bodies[i].z += bodies[i].dz * STEP_SIZE;
  }
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

int main() {
  srand(0);

  bodies = aligned_alloc(64, N*sizeof(struct Body));

  bodies[0].mass = earth_mass;
  bodies[0].radius = get_radius(earth_mass);
  for (int i = 1; i<N; i++) {
    bodies[i].mass = 1E15 * (double)rand() / (double)(RAND_MAX);
    bodies[i].radius = get_radius(bodies[i].mass);
    bodies[i].dx = 9000 + 1000*(((double)rand() / (double)(RAND_MAX))-0.5);
    bodies[i].dy = -3000 + 1000 *(((double)rand() / (double)(RAND_MAX))-0.5);
    bodies[i].dz = 1000*(((double)rand() / (double)(RAND_MAX))-0.5);
    bodies[i].x = 20000 *(((double)rand() / (double)(RAND_MAX))-0.5);
    bodies[i].y = -earth_radius - 1E6 * ((double)rand() / (double)(RAND_MAX));
    bodies[i].z = 20000 *(((double)rand() / (double)(RAND_MAX))-0.5);
  }

  double start = omp_get_wtime();

  for (int i = 0; i <= ITERATIONS; i++) {
    if (!((i - 1) % LOG_EVERY)) {
      save_results();
    }

    // Update accelerations
    update_velocity(bodies);

    // Update positions
    update_position(bodies);
  }

  printf("Total time: %f\n", omp_get_wtime() - start);

  return 0;
}