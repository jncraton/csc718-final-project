int main() {
  srand(0);

  bodies = malloc(N*sizeof(struct Body));

  bodies[0].mass = earth_mass;
  bodies[0].radius = get_radius(earth_mass);
  bodies[0].dx = 29000;
  for (int i = 1; i<N; i++) {
    bodies[i].mass = mean_particle_mass * (1.0 + (double)rand() / (double)(RAND_MAX));
    bodies[i].radius = get_radius(bodies[i].mass);
    bodies[i].dx = 37000 + 1000*(((double)rand() / (double)(RAND_MAX))-0.5);
    bodies[i].dy = -4000 + 8000 *(((double)rand() / (double)(RAND_MAX))-0.5);
    bodies[i].dz = 4000 * (((double)rand() / (double)(RAND_MAX)));
    bodies[i].x = 20000 *(((double)rand() / (double)(RAND_MAX))-0.5);
    bodies[i].y = -earth_radius - 1E6 * ((double)rand() / (double)(RAND_MAX));
    bodies[i].z = 20000 *(((double)rand() / (double)(RAND_MAX))-0.5);
  }

  double start = omp_get_wtime();

  for (int i = -1; i <= ITERATIONS; i++) {
    if (!(i % LOG_EVERY)) {
      save_results(i*STEP_SIZE);
    }

    // Update accelerations
    update_velocity(bodies);

    // Update positions
    update_position(bodies);
  }

  printf("Total time: %f\n", omp_get_wtime() - start);

  return 0;
}
