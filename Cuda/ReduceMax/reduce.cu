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

// The kernel to reduce a slice of an array
__global__ void reduceKernel(double *array, int N, int range, double *out)
{
    // determine the boundaries for this particular block
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    int end = index + range;
    
    int min = array[index]
    for (int i = index + 1; i < end; ++i) {
        // check if it's smaller than the previous minimum, and replace it if so
        if (array[i] < min) {
            min = array[i]
        }
    }

    // copy the minmum value to the output pointer
    *out = min;
}

// return the mimum value of a given array
double reduce_min(double *array, int N, )
{
    int stepsize = ((i_max - 2) / num_threads) + 1;

    // allocate the array and result pointer on the GPU
    double* dev_array = NULL;
    double* dev_result = NULL;

    checkCudaCall(cudaMalloc((void **) &dev_array, N * sizeof(double)));
    if (dev_array == NULL) {
        cout << "could not allocate memory!" << endl;
        exit(1);
    }

    checkCudaCall(cudaMalloc((void **) &dev_result, sizeof(double)));
    if (dev_result == NULL) {
        cout << "could not allocate memory!" << endl;
        cudaFree(&dev_array);
        exit(1);
    }

    // copy the data to the GPU
    checkCudaCall(cudaMemcpy(dev_array, array, N*sizeof(double), cudaMemcpyHostToDevice));

    // the main loop, where the array gets reduced treewise
    for (;;) {
        reduceKernel <<< 8, THREADS_PER_BLOCK >>>
            (stepsize, i_max, d_old, d_current, d_next);
    }

    // copy the data back to the main program
    checkCudaCall(cudaMemcpy(array, dev_array, N*sizeof(double), cudaMemcpyDeviceToHost));

    return current_array;
}
