#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <signal.h>

#define BUFSIZE 128

typedef struct {
    int *inqueue;
    int *occupied,
        *nextin,
        *nextout;

    pthread_mutex_t *lock;
    pthread_cond_t *items,
                   *space;
} FilterArgs;

FilterArgs *getArgs(int *inqueue, int *occupied, int *nextin, int *nextout,
                    pthread_mutex_t *lock,
                    pthread_cond_t *items,
                    pthread_cond_t *space)
{
    FilterArgs *args = (FilterArgs *) calloc(1, sizeof(FilterArgs));

    args->inqueue  = inqueue;
    args->occupied = occupied;
    args->nextin   = nextin;
    args->nextout  = nextout;
    args->lock     = lock;
    args->items    = items;
    args->space    = space;

    return args;
}

/* insert an item into the queue specified by the args parameter, using the
 * appropriate locks and conditions */
void queue_insert(int val, FilterArgs *args)
{
    /* get a lock on the queue, and wait for the queue to decrease if it's
     * full */
    pthread_mutex_lock( args->lock );
    while (*(args->occupied) >= BUFSIZE) {
        pthread_cond_wait(args->space, args->lock);
    }

    /* insert the item into the queue, along with some plumbing */
    (args->inqueue)[*(args->nextin)] = val;
    *(args->nextin) = (*(args->nextin) + 1) % BUFSIZE;
    *(args->occupied) += 1;

    /* signal that there's at least one item in the queue, and release the
     * lock */
    pthread_cond_signal( args->items );
    pthread_mutex_unlock( args->lock );
}

/* remove and return an item into the queue specified by the args parameter,
 * using the appropriate locks and conditions */
int queue_remove(FilterArgs *args)
{
    pthread_mutex_lock( args->lock );
    while (*(args->occupied) <= 0) {
        pthread_cond_wait(args->items, args->lock);
    }

    int number = args->inqueue[*(args->nextout)];
    *(args->nextout) = (*(args->nextout) + 1) % BUFSIZE;
    *(args->occupied) -= 1;

    pthread_cond_signal( args->space );
    pthread_mutex_unlock( args->lock );

    return number;
}

/* filter thread */
void *filter(void *vargs)
{
    FilterArgs *args = (FilterArgs *) vargs;

    /* remove an item from the queue and print it out */
    int number = queue_remove(args);
    printf("%d\n", number);

    /* create a new thread for further filtering */

    /* create the output buffer, and the parameters to be passed to the filter
     * thread */
    int output[BUFSIZE];
    int occupied = 0,
        nextin   = 0,
        nextout  = 0;
    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t space = PTHREAD_COND_INITIALIZER;
    pthread_cond_t items = PTHREAD_COND_INITIALIZER;

    FilterArgs *output_args = getArgs(output, &occupied, &nextin, &nextout,
                                      &lock, &items, &space);

    /* create the first filter thread */
    pthread_t id;
    pthread_attr_t attrs;
    pthread_attr_init(&attrs);
    pthread_attr_setscope(&attrs, PTHREAD_SCOPE_SYSTEM);
    pthread_create(&id, &attrs, filter, output_args);

    while (1) {
        int new_num = queue_remove(args);

        if ((new_num % number) == 0) {
            continue;
        }

        queue_insert(new_num, output_args);
        //printf("Receiving: %d\n", new_num);
    }

    return NULL;
}

void int_handler(int param)
{
    exit(0);
}

int main(int argc, char const *argv[])
{
    signal(SIGINT, &int_handler);

    /* create the output buffer, and the parameters to be passed to the filter
     * thread */
    int output[BUFSIZE];
    int occupied = 0,
        nextin   = 0,
        nextout  = 0;
    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t space = PTHREAD_COND_INITIALIZER;
    pthread_cond_t items = PTHREAD_COND_INITIALIZER;

    FilterArgs *args = getArgs(output, &occupied, &nextin, &nextout,
                               &lock, &items, &space);

    /* create the first filter thread */
    pthread_t id;

    /* initialize the attributes to use system scope */
    pthread_attr_t attrs;
    pthread_attr_init(&attrs);
    pthread_attr_setscope(&attrs, PTHREAD_SCOPE_SYSTEM);
    pthread_create(&id, &attrs, filter, args);

    /* produce the unbounded sequence of natural numbers until the program halts 
     * */
    for (int i = 2; ; ++i) {
        //printf("Sending: %d\n", i);
        queue_insert(i, args);
    }

    return 0;
}
