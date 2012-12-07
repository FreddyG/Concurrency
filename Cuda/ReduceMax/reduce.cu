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

__device__ int nextPowOfTwo(int n) {
    // 0 is also a power of 2
    if (n == 0) {
        return n;
    }

    else {
        int pow = 1;
        while ( pow < n ) {
            n *= 2;
        }

        return pow;
    }
}

// Finds the minimum value in an array
__global__ void reduceKernel(double *array, int N, double *out)
{
    // Reduction (min/max/avr/sum), works for any blockDim.x:
    int thread2;
    double temp;
    __shared__ double min[THREADS_PER_BLOCK];

	// Total number of threads, rounded up to the next power of two
    int nTotalThreads = nextPowOfTwo(blockDim.x);

    // tree-wise reduction
    while(nTotalThreads > 1)
    {
        int halfPoint = (nTotalThreads / 2);	// divide by two
        // only the first half of the threads will be active.

        if (threadIdx.x < halfPoint)
        {
            thread2 = threadIdx.x + halfPoint;

            // Skipping the fictious threads blockDim.x ... blockDim_2-1
            if (thread2 < blockDim.x)
            {
                // Get the shared value stored by another thread
                temp = min[thread2];
                if (temp < min[threadIdx.x])
                    min[threadIdx.x] = temp; 
            }
        }
        __syncthreads();

        // Reducing the binary tree size by two:
        nTotalThreads = halfPoint;
    }
}

// return the mimum value of a given array
double reduce_min(double *array, int N)
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

    // calculate the amount of blocks to be used
    // The first kernel invocation should reduce the array to one that can be
    // reduced by one block, so there should be THREADS_PER_BLOCK elements left
    int blocks = (N / 2) / THREADS_PER_BLOCK;

    reduceKernel <<< blocks, THREADS_PER_BLOCK >>>
        (stepsize, i_max, d_old, d_current, d_next);

    // copy the result back to the main program
    double *result = NULL;
    checkCudaCall(cudaMemcpy(result, dev_result, sizeof(double), cudaMemcpyDeviceToHost));

    return *result;
}
