#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 100
#define ITERATIONS 400
#define LOG_EVERY 20

const double G = 6.674E-11;
const double STEP_SIZE = 1.0;
const double earth_radius = 6.357E6;

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

        if (r2 > 100.0) {
          bodies[i].dx += ((bodies[j].x - bodies[i].x) / sqrt(r2)) * 
            STEP_SIZE*G*bodies[j].mass/r2;
          bodies[i].dy += ((bodies[j].y - bodies[i].y) / sqrt(r2)) * 
            STEP_SIZE*G*bodies[j].mass/r2;
          bodies[i].dy += ((bodies[j].z - bodies[i].z) / sqrt(r2)) * 
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
    fprintf(file, "%f,%f,%f,%f,%f,%f\n", 
      bodies[i].x,
      bodies[i].y,
      bodies[i].z,
      bodies[i].dx,
      bodies[i].dy,
      bodies[i].dz);
  }
  fclose(file);
}

int main() {
  bodies = aligned_alloc(64, N*sizeof(struct Body));

  bodies[0].mass = 6E24;
  for (int i = 1; i<N; i++) {
    bodies[i].mass = 10000 * (double)rand() / (double)(RAND_MAX);
    bodies[i].dx = -300 + 100*(((double)rand() / (double)(RAND_MAX))-0.5);
    bodies[i].dy = 6000 + 100*(((double)rand() / (double)(RAND_MAX))-0.5);
    bodies[i].dz = 100*(((double)rand() / (double)(RAND_MAX))-0.5);
    bodies[i].x = -earth_radius;
    bodies[i].y = 0;
    bodies[i].z = 0;
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