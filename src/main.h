int main() {
  srand(0);

  bodies = aligned_alloc(64, N*sizeof(struct Body));

  bodies[0].mass = earth_mass;
  bodies[0].radius = get_radius(earth_mass);
  for (int i = 1; i<N; i++) {
    bodies[i].mass = mean_particle_mass * (1.0 + (double)rand() / (double)(RAND_MAX));
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
