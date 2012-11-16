#include "broadcast.h"
#include <stdio.h>
#include <stdlib.h>

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

    // the root process sends the data to its 2 neighbours
    if (pnum == root) {
        // make 2 requests, the p and m stand for plus and minnus
        MPI_Request request_arr[2];
        MPI_Status status_arr[2];

        MPI_Isend(buffer, count, datatype, root + 1, tag, communicator, &request_arr[0]);
        MPI_Isend(buffer, count, datatype, root - 1, tag, communicator, &request_arr[1]);

        // return 0 if both messages arrived, else 1
        if (MPI_Waitall(2, request_arr, status_arr) == MPI_SUCCESS) {
            return 0;
        } else {
            printf("Error: Message(s) not received.\n");
            return 1;
        }
    }

    // if the process is the meeting point, receive from the lower ranked
    // process just because
    else if (pnum == root + meeting_point) {
        MPI_Status status_p,
                   status_m;

        MPI_Recv(buffer, count, datatype, (pnum - 1) % size, tag, communicator, &status_m);
    }

    // if the process rank is larger than the root, receive from the lower
    // ranked process and send to the higher ranked process
    else if ((pnum - root) > meeting_point) {
        MPI_Status status_r,
                   status_s;

        MPI_Recv(buffer, count, datatype, (pnum - 1) % size, tag, communicator, &status_r);

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
    int root = 2;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &pnum);

    int message[1];

    if (pnum == root) {
        message[0] = 42;
    } else {
        printf("Proof that message is unitialised: %d\n", message[0]);
    }

    if (MYMPI_Bcast(&message, 1, MPI_INT, 2, MPI_COMM_WORLD)) {
        printf("Error: Broadcast failed.\n");
    }

    if (pnum == root) {
        printf("Process %d sent: %d\n", pnum, message[0]);
    } else {
        printf("Process %d received: %d\n", pnum, message[0]);
    }

    MPI_Finalize();

    return 0;
}
