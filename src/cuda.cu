#include "shared.h"

__global__ void update_from_gravity (Body * bodies, int * N) {
  double r2, new_mass;
  int i,j;
  double earth_density = earth_mass / ((4.0 / 3.0) * 3.14159 * earth_radius * earth_radius * earth_radius);

  for (i = threadIdx.x+1; i<*N; i+=blockDim.x) {
    for (j = 0; j < *N; j++) {
      if (i != j && bodies[i].mass && bodies[j].mass) {
        r2 = (
          (bodies[i].x-bodies[j].x) * (bodies[i].x-bodies[j].x) +
          (bodies[i].y-bodies[j].y) * (bodies[i].y-bodies[j].y) +
          (bodies[i].z-bodies[j].z) * (bodies[i].z-bodies[j].z)
        );
  
        if (r2 > (bodies[i].radius + bodies[j].radius) * (bodies[i].radius + bodies[j].radius)) {
          bodies[i].dx += ((bodies[j].x - bodies[i].x) / sqrt(r2)) * 
            STEP_SIZE*G*bodies[j].mass/r2;
          bodies[i].dy += ((bodies[j].y - bodies[i].y) / sqrt(r2)) * 
            STEP_SIZE*G*bodies[j].mass/r2;
          bodies[i].dz += ((bodies[j].z - bodies[i].z) / sqrt(r2)) * 
            STEP_SIZE*G*bodies[j].mass/r2;
        } else {
            new_mass = bodies[i].mass + bodies[j].mass;
            bodies[j].dx = (bodies[j].dx * bodies[j].mass + bodies[i].dx * bodies[i].mass) / new_mass;
            bodies[j].dy = (bodies[j].dy * bodies[j].mass + bodies[i].dy * bodies[i].mass) / new_mass;
            bodies[j].dz = (bodies[j].dz * bodies[j].mass + bodies[i].dz * bodies[i].mass) / new_mass;
            bodies[j].mass = new_mass;

            double volume = new_mass / earth_density;
             // Volume = (4/3) pi r^3
            // r^3 = volume * (3/4) / pi
            double r3 = volume * (3.0/4.0) / 3.14159;

            bodies[j].radius = cbrt(r3);
            bodies[i].mass = 0.0;
        }
      }
    }
  }

  // Update positions
  for (int i = threadIdx.x; i < *N; i+=blockDim.x) {
    bodies[i].x += bodies[i].dx * STEP_SIZE;
    bodies[i].y += bodies[i].dy * STEP_SIZE;
    bodies[i].z += bodies[i].dz * STEP_SIZE;
  }

}

// Nice GPU assertion code borrowed from:
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: Error code %d %s %s %d\n", code, cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void update_velocity(struct Body *bodies) {
  //F = GmM/r^2
  //a = Gm/r^2 m is mass of the other, our mass cancels
  //r^2 = distance^2
  //distance = sqrt(x^2 + y^2 + z^2)
  //r^2 = x^2 + y^2 + z^2

  //double r2, new_mass;
  //int i,j;

  Body * cuda_bodies;
  int * cuda_N;

  gpuErrchk(cudaMalloc(&cuda_bodies, N * sizeof(Body)));
  gpuErrchk(cudaMemcpy(cuda_bodies, bodies, N * sizeof(Body), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&cuda_N, sizeof(int)));
  gpuErrchk(cudaMemcpy(cuda_N, &N, sizeof(int), cudaMemcpyHostToDevice));

  update_from_gravity<<<1,1024>>>(cuda_bodies, cuda_N);

  gpuErrchk(cudaMemcpy(bodies, cuda_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(cuda_bodies));
}

void update_position(struct Body *bodies) {
}

void update(struct Body * bodies, int iterations) {
  for (int i = 0; i < iterations; i++) {
    update_velocity(bodies);
  }
}

#include "main.h"
