#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

int isprime(int n) {
  int i,squareroot;
  if (n>10) {
    squareroot = (int) sqrt(n);
    for (i=3; i<=squareroot; i=i+2)
      if ((n%i)==0)
        return 0;
    return 1;
  }
  else
    return 0;
}

int main(int argc, char *argv[])
{
  long long pc,       /* prime counter */
      foundone=0; /* most recent prime found */
  long long int n, limit;

  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // foolproof
  if (argc <= 1) {
    if (rank==0) fprintf(stderr, "usage: %s <integer>\n", argv[0]);
    MPI_Finalize();
    return 0;
  }

  sscanf(argv[1],"%llu",&limit);	
  if (rank == 0) { // only show on master node
    printf("Starting. Numbers to be scanned= %lld\n",limit);
  }

  if (rank == 0) {
    pc=4;     /* Assume (2,3,5,7) are counted here */
    if (limit < 11) {
      // limit too small!
      pc=0;
      if (limit == 2) { pc=1; foundone=2; }
      if (limit >= 3) { pc=2; foundone=3; }
      if (limit >= 5) { pc=3; foundone=5; }
      if (limit >= 7) { pc=4; foundone=7; }
    }
  }
  else {
    // slave node
    pc = 0;
  }

  for (n=11+2*rank; n<=limit; n=n+2*size) {
    if (isprime(n)) {
      pc++;
      foundone = n;
    }			
  }

  long long pix, largest;
  // all nodes need to enter reduce instruction
  MPI_Reduce(&pc, &pix, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&foundone, &largest, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("Done. Largest prime is %lld Total primes %lld\n",largest,pix);
  }
  MPI_Finalize();
  return 0;
} 
