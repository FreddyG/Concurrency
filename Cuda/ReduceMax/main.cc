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

#include <random>

#include "reduce.h"

#define ARRAY_SIZE 10000000

int main(int argc, char *argv[])
{
    double array[ARRAY_SIZE];

    // fill the array with random numbers
    std::default_random_engine generator;
    std::uniform_double_distribution<double> distribution(1,10000);
    for (int i = 0; i < ARRAY_SIZE; ++i)
    {
        array[i] = distribution(generator);
    }

    double min = reduce_min(array, ARRAY_SIZE);
    printf("Min: %f\n", min);

    return EXIT_SUCCESS;
}
