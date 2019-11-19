#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 100
#define ITERATIONS 50000
#define LOG_EVERY 2500

const double G = 6.674E-11;
const double STEP_SIZE = 100.0;
const double earth_radius = 6.357E6;
const double earth_mass = 6E24;
//const double earth_volume = (4.0 / 3.0) * 3.14159 * earth_radius * earth_radius * earth_radius;
//const double earth_density = earth_mass / earth_volume;
 
int num_bodies = N;

struct Body {
  double mass;
  double x;
  double y;
  double z;
  double dx;
  double dy;
  double dz;
};

struct Body *bodies;

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

        if (j == 0 && r2 < earth_radius*earth_radius) {
          bodies[i].mass = 0;
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
  
  printf("%s\n", buf);

  FILE* file = fopen(buf,"w");
  for (int i = 0; i < N; i++) {
    if (bodies[i].mass > 0) {
      fprintf(file, "%f,%f,%f,%f,%f,%f\n", 
        bodies[i].x,
        bodies[i].y,
        bodies[i].z,
        bodies[i].dx,
        bodies[i].dy,
        bodies[i].dz);
    }
  }
  fclose(file);
}

int main() {
  bodies = aligned_alloc(64, N*sizeof(struct Body));

  bodies[0].mass = earth_mass;
  for (int i = 1; i<N; i++) {
    bodies[i].mass = 10000 * (double)rand() / (double)(RAND_MAX);
    bodies[i].dx = 9000 + 1000*(((double)rand() / (double)(RAND_MAX))-0.5);
    bodies[i].dy = -3000 + 1000 *(((double)rand() / (double)(RAND_MAX))-0.5);
    bodies[i].dz = 1000*(((double)rand() / (double)(RAND_MAX))-0.5);
    bodies[i].x = 1000 *(((double)rand() / (double)(RAND_MAX))-0.5);;
    bodies[i].y = -earth_radius - 1E6;
    bodies[i].z = 1000 *(((double)rand() / (double)(RAND_MAX))-0.5);;
  }

  for (int i = 0; i <= ITERATIONS; i++) {
    if (!(i % LOG_EVERY)) {
      printf("%d iterations completed. Writing results...\n", i);
      save_results();
    }

    // Update accelerations
    update_velocity(bodies);

    // Update positions
    update_position(bodies);
  }

  return 0;
}