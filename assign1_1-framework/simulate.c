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

/* the value used for the parameter 'c' */

/*
 * Struct to pass as parameter to the 'simulate_slice' function.
 * 
 * i_min and i_max define the range of locations to work on, not including i_max
 * ( so formally, the thread will work in the interval [a, b) )
 *
 * old_array and current_array are the arrays of values at t-1 and t, respectively.
 *
 * new_array will be used to hold the return values
 */
typedef struct {
    int i_min;
    int i_max;
    double *old_array;
    double *current_array;
    double *new_array;
} ThreadArgs;


/* Add any global variables you may need. */
double C = 0.2;

/* Add any functions you may need (like a worker) here. */

/*
 * simulate_slice is meant to run as thread and will calculate the wave
 * amplitudes at t+1 in the interval [a, b) as given by the ThreadArgs parameter
 * struct. 
 * Array bounds-checking is not done; it is presumed that a proper interval has
 * been passed with the elements at i-1 and i+1 available.
 *
 * WARNING: The args parameter is expected to be freed within this function.
 */
void *simulate_slice(void *args)
{
    /* extract parameters from the parameter struct */
    ThreadArgs *t_args    = (ThreadArgs *) args;

    int i_min             = t_args->i_min;
    int i_max             = t_args->i_max;
    double *old_array     = t_args->old_array;
    double *current_array = t_args->current_array;
    double *new_array     = t_args->new_array;

    for (int i = i_min; i < i_max; ++i) {
        /* simple implementation of the following wave-equation:
         * A(i, t+1) = 2 * A(i, t) - A(i, t-1) +
         * c * ( A(i-1, t) - (2 * A(i, t) - A(i+1, t)))
         */
        new_array[i] = 2 * current_array[i] - old_array[i] +
            C * (current_array[i-1] - (2 * current_array[i] -
                                       current_array[i + 1]));
    }

    free(args);

    return NULL;
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
    pthread_t thread_pool[num_threads];
    ThreadArgs *args;

    /* initialize the attributes to use system scope */
    pthread_attr_t attrs;
    pthread_attr_init(&attrs);
    pthread_attr_setscope(&attrs, PTHREAD_SCOPE_SYSTEM);
    

    /* calculating the stepsize
     * (i_max - 2) is taken because the first and last elements of the arrays
     * are taken to be constantly 0
     */
    int stepsize = ((i_max - 2) / num_threads) + 1;

    // run the simulation for t_max timesteps
    for (int t = 0; t < t_max; ++t) {

        // Launch num_threads threads with a (roughly) equal payload.
        // Each thread gets its own slice of the array to handle.
        for (int i = 0; i < num_threads; ++i)
        {
            args = (ThreadArgs *) malloc( sizeof(ThreadArgs) );
            args->i_min = 1 + i * stepsize;
            args->i_max = args->i_min + stepsize;
            args->old_array = old_array;
            args->current_array = current_array;
            args->new_array = next_array;
            
            /* make sure that i_max doesn't get too large
             * TODO: handle the intervals more neatly. */
            args->i_max = (args->i_max > i_max ? i_max : args->i_max);

            pthread_create(&thread_pool[i], &attrs, &simulate_slice, args);
        }

        /*
         * After each timestep, you should swap the buffers around. Watch out none
         * of the threads actually use the buffers at that time.
         */
        for (int i = 0; i < num_threads; ++i) {
            pthread_join(thread_pool[i], NULL);
        }

        old_array = current_array;
        current_array = next_array;
        next_array = old_array;
    }

    /* You should return a pointer to the array with the final results. */

    return current_array;
}
