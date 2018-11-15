#include <stdio.h>
#include "mpi.h"
#include <stdlib.h>
int main(int argc, char *argv[]){
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int to = 0;
  if (argc < 2) {
    if (rank == 0) {
      printf("This program tests the speed of message passing\n");
    }
    goto die;
  }
  if (size < 2) {
    if (rank == 0) {
      printf("The program requires 2 or more nodes\n");
    }
    goto die;
  }
  sscanf(argv[1], "%d", &to);
  int len = 1;
  if (argc > 2) sscanf(argv[2], "%d", &len);
  if (len < 1 || len > 1234567) len = 1;
  int *buf = malloc(len * sizeof size);
  if (!buf) goto die;

  int i;
  // randomize
  for (i = 1; i < len; i++) {
    buf[i] = rand();
  }
  MPI_Status status;
  if (rank != 0) {
    // get a number, and send to next node
    for (i = 0; i < to; i++) {
      MPI_Recv(buf, len, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &status);
      // add some random bit so compression won't work
      *buf = *buf + 1;
      int r = rand()%len;
      buf[r] = *buf + rand() * r;
      MPI_Send(buf, len, MPI_INT, (rank+1)%size, 0, MPI_COMM_WORLD);
    }
  }
  if (rank == 0) {
    // send a number, and wait for a cycle
    *buf = 0;
    for (i = 0; i < to; i++) {
      *buf = *buf + 1;
      MPI_Send(buf, len, MPI_INT, 1, 0, MPI_COMM_WORLD);
      MPI_Recv(buf, len, MPI_INT, size-1, 0, MPI_COMM_WORLD, &status);
    }
    printf("%d\n", *buf);
  }

  free(buf);
die:
  MPI_Finalize();
  return 0;
}
