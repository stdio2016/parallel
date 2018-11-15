#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "mpi.h"

#define PI 3.1415926535

int main(int argc, char **argv) 
{
  long long i, num_intervals;
  double rect_width, area, sum, x_middle; 
  int size, rank, tag, src, dest;

  MPI_Init(&argc, &argv);
  // get my node id
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // foolproof
  if (argc < 2) {
    fprintf(stderr, "usage: %s <number of intervals>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  sscanf(argv[1],"%llu",&num_intervals);

  rect_width = PI / num_intervals;

  sum = 0;
  // divide workload
  // node 0:   1, 1+n, 1+2n,...
  // node 1:   2, 2+n, 2+2n,...
  // ...
  // node n-1: n, n+n, n+2n,...
  // where n = node count
  for(i = 1+rank; i < num_intervals + 1; i+=size) {

    /* find the middle of the interval on the X-axis. */ 

    x_middle = (i - 0.5) * rect_width;
    area = sin(x_middle) * rect_width; 
    sum = sum + area;
  } 

  tag = 0;
  dest = 0; // master node
  if (rank == dest) {
    // get result from other nodes
    for (src = 1; src < size; src++) {
      double other;
      MPI_Status status;
      MPI_Recv(&other, 1, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, &status);
      sum = sum + other;
    }
  }
  else {
    // send to master node
    MPI_Send(&sum, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
  }

  if (rank == dest) {
    // I am master node, show result
    printf("The total area is: %f\n", (float)sum);
  }

  MPI_Finalize();
  return 0;
}   
