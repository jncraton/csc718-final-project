% N-Body Simulation
% Jon Craton
% December 10th, 2019

N-body Simulation
=================

-----

> In physics and astronomy, an N-body simulation is a simulation of a dynamical system of particles, usually under the influence of physical forces, such as gravity (Wikipedia)

Research Question
-----------------

- In October, [we discovered 20 new moons of Saturn](http://www.astronomy.com/news/2019/10/20-new-moons-discovered-orbiting-saturn)
- How does a large body disintigrating on impact form moons or ring systems around a planetoid?

Algorithm
---------

1. Apply gravitational force to each body from every other body - O(N^2)
2. Search for collisions and merge objects - O(N^2)
3. Update positions using new velocity vectors - O(N)

Sequential Implementation
=========================

Pseudocode
----------

```c
for (i = 1; i < N; i++)
  for (j = 0; j < N; j++)
      bodies[i].velocity += {Acceleration from bodies[j]};

for (j = 0; j < N; j++)
  bodies[i].position += bodies[i].velocity * delta_time;
```

Test Hardware
-------------

Component   Model                 Specs
----------  --------------------  -----------------------------------------------
CPU         Intel Core i7-8850H   6 Cores, 12 Threads, 2.6GHz clock
GPU         NVidia P3200          14 Shader Modules, 1792 Shader Units, 6 GB RAM

Performance
-----------

- Consumes resources of 1 CPU core
- Executes in 120 seconds