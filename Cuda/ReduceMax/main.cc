/*
 * assign3_1.c
 *
 * Contains code for setting up and finishing the simulation.
 * NOTE: YOU SHOULD IMPLEMENT NOT HAVE TO LOOK HERE, IMPLEMENT YOUR CODE IN
 *       simulate.c.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "reduce.h"
#include "timer.h"

#define ARRAY_SIZE 10000000

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

int main(int argc, char *argv[])
{
    double *array = NULL;
    array = (double *) malloc(ARRAY_SIZE * sizeof(double));

    if (array == NULL) {
        printf("Error: malloc failed\n");
    }

    // fill the array with random numbers
    srand(0);
    for (int i = 0; i < ARRAY_SIZE; ++i)
    {
        array[i] = fRand(1, 1000);
    }

    timer_start();
    double max = reduce_max(array, ARRAY_SIZE);
    double time_taken = timer_end();
    printf("Parallel max: %f\n", max);
    printf("Time taken: %f seconds\n", time_taken);

    // find the max sequentially to validate the answer
    timer_start();
    double smax = array[0];
    for (int i = 1; i < ARRAY_SIZE; ++i)
    {
        if (array[i] > smax) {
            smax = array[i];
        }
    }

    time_taken = timer_end();
    printf("Sequential max: %f\n", smax);
    printf("Time taken: %f seconds\n", time_taken);

    return 0;
}
