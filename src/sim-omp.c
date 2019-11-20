#include "shared.h"

void update_velocity(struct Body *bodies) {
  //F = GmM/r^2
  //a = Gm/r^2 m is mass of the other, our mass cancels
  //r^2 = distance^2
  //distance = sqrt(x^2 + y^2 + z^2)
  //r^2 = x^2 + y^2 + z^2

  #pragma omp parallel for
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

#include "main.h"