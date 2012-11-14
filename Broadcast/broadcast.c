#include "broadcast.h"
#include <stdio.h>

int MYMPI_Bcast(void *buffer, int count,
                MPI_Datatype datatype, int root,
                MPI_Comm communicator)
{
    int pnum, size;
    int tag = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &pnum);

    // the meeting point is the node (roughly) opposite the root, where the flow of data
    // will meet itself (normalised such that root = 4)
    int meeting_point = size / 2;
    //printf("Root: %d, meeting point: %d\n", root, root + meeting_point);

    // the root process sends the data to its 2 neighbours
    if (pnum == root) {
        // make 2 requests, the p and m stand for plus and minnus
        MPI_Request request_p,
                    request_m;
        MPI_Request request_arr[2] = {request_p, request_m}; // used for MPI_Waitall

        printf("Sending: %d\n", ((int *)buffer)[0]);
        MPI_Isend(buffer, count, datatype, root + 1, tag, communicator, &request_p);
        MPI_Isend(buffer, count, datatype, root - 1, tag, communicator, &request_m);

        // return 0 if both messages arrived, else 1
        return 0;
    }

    // if the process is the meeting point, receive from either side
    else if (pnum == root + meeting_point) {
        MPI_Status status_p,
                   status_m;

        MPI_Recv(buffer, count, datatype, (pnum - 1) % size, tag, communicator, &status_m);
        printf("Process %d received: %d\n", pnum, ((int*)buffer)[0]);
    }

    // if the process rank is larger than the root, receive from the lower
    // ranked process and send to the higher ranked process
    else if ((pnum - root) > meeting_point) {
        MPI_Status status_r,
                   status_s;

        MPI_Recv(buffer, count, datatype, (pnum - 1) % size, tag, communicator, &status_r);
        printf("Process %d received: %d\n", pnum, ((int *)buffer)[0]);

        if (MPI_Send(buffer, count, datatype, (pnum + 1) % size, tag, communicator) == MPI_SUCCESS) {
        } else {
            printf("Sending failed.");
            return 1;
        }
    }

    // if the process rank is smaller than the root, receive from the lower
    // ranked process and send to the higher ranked process
    else if ((pnum - root) < meeting_point) {
        MPI_Status status_r,
                   status_s;

        MPI_Recv(buffer, count, datatype, (pnum - 1) % size, tag, communicator, &status_r);
        printf("Process %d received: %d\n", pnum, ((int *)buffer)[0]);

        if (MPI_Send(buffer, count, datatype, (pnum + 1) % size, tag, communicator) == MPI_SUCCESS) {
            return 0;
        } else {
            printf("Sending failed.");
            return 1;
        }
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int pnum, size;
    int tag = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &pnum);

    int message[1];
    if (pnum = 2) {
        message[0] = 8;
    }

    MYMPI_Bcast(&message, 1, MPI_INT, 2, MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
