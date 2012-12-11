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

#define THREADS_PER_BLOCK 512

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
__global__ void simulationKernel(int stepsize, int max,
                                 double *old,
                                 double *current,
                                 double *next)
{
    // constant used in the equation
    double C = 0.2;

    // determine the boundaries for this particular block
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    int i_min = 1 + index * stepsize;
    int i_max = i_min + stepsize;

    /* make sure that i_max doesn't get too large
     * TODO: handle the intervals more neatly. */
    i_max = (i_max > max ? max : i_max);

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
 * num_blocks: how many threads to use (excluding the main threads)
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 */
double *simulate(const int i_max, const int t_max,
                 double *old_array, double *current_array, double *next_array)
{
    // calculate the number of blocks to be used
    // ensure it's between 1 and 65535 (the maximum number of blocks that can
    // run concurrently)
    int num_blocks = (i_max / THREADS_PER_BLOCK);
    if (num_blocks == 0)
        num_blocks = 1;
    else if (num_blocks > 65535)
        num_blocks = 65535;

    int stepsize = ((i_max - 2) / (num_blocks * THREADS_PER_BLOCK)) + 1;

    // allocate the vectors on the GPU
    double* d_old = NULL;
    checkCudaCall(cudaMalloc((void **) &d_old, i_max * sizeof(double)));
    if (d_old == NULL) {
        cout << "could not allocate memory!" << endl;
        exit(1);
    }

    double* d_current = NULL;
    checkCudaCall(cudaMalloc((void **) &d_current, i_max * sizeof(double)));
    if (d_current == NULL) {
        checkCudaCall(cudaFree(d_old));
        cout << "could not allocate memory!" << endl;
        exit(1);
    }

    double* d_next = NULL;
    checkCudaCall(cudaMalloc((void **) &d_next, i_max * sizeof(double)));
    if (d_next == NULL) {
        checkCudaCall(cudaFree(d_old));
        checkCudaCall(cudaFree(d_current));
        cout << "could not allocate memory!" << endl;
        exit(1);
    }

    // copy the data to the GPU
    checkCudaCall(cudaMemcpy(d_old, old_array, i_max*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(d_current, current_array, i_max*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(d_next, next_array, i_max*sizeof(double), cudaMemcpyHostToDevice));

    // the main loop
    double *temp_old;
    for (int t = 0; t < t_max; ++t) {
        simulationKernel <<< num_blocks, 512 >>>
            (stepsize, i_max, d_old, d_current, d_next);

        // swap the arrays around
        temp_old = d_old;
        d_old = d_current;
        d_current = d_next;
        d_next = temp_old;
    }

    // copy the data back to the main program
    checkCudaCall(cudaMemcpy(old_array, d_old, i_max*sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaCall(cudaMemcpy(current_array, d_current, i_max*sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaCall(cudaMemcpy(next_array, d_next, i_max*sizeof(double), cudaMemcpyDeviceToHost));

    return current_array;
}
