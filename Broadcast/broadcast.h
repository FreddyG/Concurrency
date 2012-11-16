#ifndef BROADCAST_H_FQG0HMSH
#define BROADCAST_H_FQG0HMSH

#include <mpi.h>

/* broadcast the data to each in the communicator over a ring network */
int MYMPI_Bcast(void *buffer, int count,
                MPI_Datatype datatype, int root,
                MPI_Comm communicator);

#endif /* end of include guard: BROADCAST_H_FQG0HMSH */

