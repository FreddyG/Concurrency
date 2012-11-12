/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "simulate.h"

#define SIM_TAG 0


/* Add any global variables you may need. */


/* Add any functions you may need (like a worker) here. */


/*
 * Executes the entire simulation.
 *
 * Implement your code here.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 */
double *simulate(const int i_max, const int t_max)
{
    int my_num;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_num);

    MPI_Status s;
    int size;

    // first receive the array size
    MPI_Recv(&size, 1, MPI_INT, 0, SIM_TAG, MPI_COMM_WORLD, &s);

    double old[size],
           current[size];

    MPI_Recv(old, size, MPI_DOUBLE, 0, SIM_TAG, MPI_COMM_WORLD, &s);
    MPI_Recv(old, size, MPI_DOUBLE, 0, SIM_TAG, MPI_COMM_WORLD, &s);

    printf("Old[7]    : %f\n", old[7]);
    printf("Current[7]: %f\n", current[7]);

    /* You should return a pointer to the array with the final results. */
    return NULL;
}
