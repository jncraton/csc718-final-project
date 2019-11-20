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

