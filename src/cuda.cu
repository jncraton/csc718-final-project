#include "shared.h"

__global__ void update_from_gravity (Body * bodies, int * N) {
  fptype r2, new_mass;

  int i = blockIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.x;

  if (i != j) {
    r2 = (
      (bodies[i].x-bodies[j].x) * (bodies[i].x-bodies[j].x) +
      (bodies[i].y-bodies[j].y) * (bodies[i].y-bodies[j].y) +
      (bodies[i].z-bodies[j].z) * (bodies[i].z-bodies[j].z)
    );

    fptype unit_gravity = rsqrt(r2) *rsqrt(r2) *rsqrt(r2) * STEP_SIZE * G * bodies[j].mass;

    if (r2 > (bodies[i].radius + bodies[j].radius) * (bodies[i].radius + bodies[j].radius)) {
      atomicAdd(&bodies[i].dx, (bodies[j].x - bodies[i].x) * unit_gravity);
      atomicAdd(&bodies[i].dy, (bodies[j].y - bodies[i].y) * unit_gravity);
      atomicAdd(&bodies[i].dz, (bodies[j].z - bodies[i].z) * unit_gravity);
    } else {
        new_mass = bodies[i].mass + bodies[j].mass;
        bodies[j].dx = (bodies[j].dx * bodies[j].mass + bodies[i].dx * bodies[i].mass) / new_mass;
        bodies[j].dy = (bodies[j].dy * bodies[j].mass + bodies[i].dy * bodies[i].mass) / new_mass;
        bodies[j].dz = (bodies[j].dz * bodies[j].mass + bodies[i].dz * bodies[i].mass) / new_mass;
        bodies[j].mass = new_mass;

        fptype volume = new_mass / earth_density;
         // Volume = (4/3) pi r^3
        // r^3 = volume * (3/4) / pi
        fptype r3 = volume * (3.0/4.0) / 3.14159;

        bodies[j].radius = cbrt(r3);
        bodies[i].mass = 0.0;
        bodies[i].radius = 0.0;
    }
  }
}

__global__ void update_positions (Body * bodies, int * N) {
  // Update positions

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  bodies[i].x += bodies[i].dx * STEP_SIZE;
  bodies[i].y += bodies[i].dy * STEP_SIZE;
  bodies[i].z += bodies[i].dz * STEP_SIZE;
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

void update(struct Body * bodies, int iterations) {
  Body * cuda_bodies;
  int * cuda_N;

  int threads_per_block = 1024;

  gpuErrchk(cudaMalloc(&cuda_bodies, (N + threads_per_block) * sizeof(Body)));
  gpuErrchk(cudaMemcpy(cuda_bodies, bodies, N * sizeof(Body), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&cuda_N, sizeof(int)));
  gpuErrchk(cudaMemcpy(cuda_N, &N, sizeof(int), cudaMemcpyHostToDevice));

  for (int i = 0; i < iterations; i++) {
    dim3 grid(N-1, N / threads_per_block);
    update_from_gravity<<<grid,threads_per_block>>>(cuda_bodies, cuda_N);

    int blocks = 1 + N / threads_per_block;
    update_positions<<<blocks,threads_per_block>>>(cuda_bodies, cuda_N);
  }

  gpuErrchk(cudaMemcpy(bodies, cuda_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(cuda_bodies));
}

#include "main.h"
