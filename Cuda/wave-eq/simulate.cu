/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

#include "simulate.h"

#include <iostream>

using namespace std;

/* Utility function, use to do error checking.

   Use this function like this:

   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

   And to check the result of a kernel invocation:

   checkCudaCall(cudaGetLastError());
*/
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}

// The kernel that simulates an iteration of the wave equation on the GPU
__global__ void simulationKernel(double *old, double *current, double *next) {
    // determine the boundaries for this particular block

    // the main simulation loop
    for (int i = i_min; i < i_max; ++i) {
        /* simple implementation of the following wave-equation:
         * A(i, t+1) = 2 * A(i, t) - A(i, t-1) +
         * c * ( A(i-1, t) - (2 * A(i, t) - A(i+1, t)))
         */
        next[i] = 2 * current[i] - old[i] + C * (current[i-1] - (2 * current[i] - current[i + 1]));
    }
}

/*
 * Executes the entire simulation.
 *
 * Implement your code here.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * num_threads: how many threads to use (excluding the main threads)
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 */
double *simulate(const int i_max, const int t_max, const int num_threads,
                 double *old_array, double *current_array, double *next_array)
{
    for (int t = 0; t < t_max; ++t) {
        add <<< num_threads, 1 >>> (old_dev, curr_dev, next_dev);

        // swap the arrays around
        double *temp_old = old_array;
        old_array = current_array;
        current_array = next_array;
        next_array = temp_old;
    }

    return current_array;
}
