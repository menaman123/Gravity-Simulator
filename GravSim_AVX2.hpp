#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <time.h>
#include <memory.h>
#include <stdlib.h>
#include <immintrin.h> // Include AVX2 header

class largeBody {
public:
    double x, y; // position
    double vx, vy; // velocity
    double mass;
};

const double G = 6.67430e-11; // gravitational constant

void updateVelocity(largeBody& obj, double dt, largeBody* objects, int num_objects);

void updateVelocityAVX2(largeBody& obj, double dt, largeBody* objects, int num_objects);

void updateVelocityTHREAD(largeBody& obj, double dt, largeBody* objects, int num_objects);

void updatePosition(largeBody& obj, double dt);

void updatePositionAVX2(largeBody& obj, double dt);

void simulator(int num_objects);

void miniSimulator();

__m256d load(const double* ap);