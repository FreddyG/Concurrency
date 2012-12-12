/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

#include "reduce.h"

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

// Finds the maximum value in an array
__global__ void reduceKernel(double *array, int N, double *out)
{
    // each thread handles a chunk of the array, and writes it to block-shared
    // memory, reducing the array to a new array with THREADS_PER_BLOCK
    // elements.
    __shared__ double max_per_thread[THREADS_PER_BLOCK];
    
    int stepsize = N / THREADS_PER_BLOCK;

    int start = threadIdx.x * stepsize,
        end   = start + stepsize;

    // make sure the entire array gets checked
    if (threadIdx.x == THREADS_PER_BLOCK - 1) {
        end = N;
    }

    double max = array[start];
    for (int i = start + 1; i < end; ++i) {
        if (array[i] > max) {
            max = array[i];
        }
    }

    max_per_thread[threadIdx.x] = max;
    __syncthreads();

    // one of the threads performs a further reduction step
    max = max_per_thread[0];
    if (threadIdx.x == 0) {
        for (int i = 1; i < THREADS_PER_BLOCK; ++i) {
            if (max_per_thread[i] > max) {
                max = max_per_thread[i];
            }
        }

        out[0] = max;
    }
}

// return the mimum value of a given array
double reduce_max(double *array, int N)
{
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

    reduceKernel <<< 1, THREADS_PER_BLOCK >>>
        (dev_array, N, dev_result);

    // copy the result back to the main program
    double result[1];
    checkCudaCall(cudaMemcpy(result, dev_result, sizeof(double), cudaMemcpyDeviceToHost));

    return result[0];
}
