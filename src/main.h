int main() {
  srand(0);

  bodies = (Body*) malloc(N*sizeof(struct Body));

  bodies[0].mass = earth_mass;
  bodies[0].radius = get_radius(earth_mass);
  bodies[0].dx = 29000;
  for (int i = 1; i<N; i++) {
    bodies[i].mass = mean_particle_mass * (1.0 + (fptype)rand() / (fptype)(RAND_MAX));
    bodies[i].radius = get_radius(bodies[i].mass);
    bodies[i].dx = 37000 + 1000*(((fptype)rand() / (fptype)(RAND_MAX))-0.5);
    bodies[i].dy = -4000 + 8000 *(((fptype)rand() / (fptype)(RAND_MAX))-0.5);
    bodies[i].dz = 4000 * (((fptype)rand() / (fptype)(RAND_MAX)));
    bodies[i].x = 20000 *(((fptype)rand() / (fptype)(RAND_MAX))-0.5);
    bodies[i].y = -earth_radius - 1E6 * ((fptype)rand() / (fptype)(RAND_MAX));
    bodies[i].z = 20000 *(((fptype)rand() / (fptype)(RAND_MAX))-0.5);
  }

  double inner_loops = 0;

  double start = omp_get_wtime();

  for (int i = 0; i <= ITERATIONS/LOG_EVERY; i++) {
    save_results(bodies, i*STEP_SIZE);

    update(bodies, LOG_EVERY);

    inner_loops += (N*N*LOG_EVERY)/1e9;

    for (int i = 0; i < N; i++) {
      if (bodies[i].mass == 0.0) {
        bodies[i] = bodies[N-1];
        N--;
      }
    }
  }

  save_results(bodies, (ITERATIONS/LOG_EVERY+1)*STEP_SIZE);

  float elapsed = omp_get_wtime() - start;

  printf("Total time: %f Final N: %d Inner loops: %.01f billion\n", elapsed, N, inner_loops);

  return 0;
}
