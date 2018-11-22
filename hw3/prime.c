#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

// toggle MPI
#ifndef DONT_USE_MPI
  #define USE_MPI
#endif

#ifdef USE_MPI
  #include "mpi.h"
#endif

#ifndef OLD_METHOD
  #define CAN_CHANGE_ALGORITHM
#endif

// skip multiples of 2,3,5,7
#define WHEEL_SIZE 48
#define WHEEL_PRODUCT 210
int wheel[WHEEL_SIZE] = {
    1,  11,  13,  17,  19,  23,  29,  31,
   37,  41,  43,  47,  53,  59,  61,  67,
   71,  73,  79,  83,  89,  97, 101, 103,
  107, 109, 113, 121, 127, 131, 137, 139,
  143, 149, 151, 157, 163, 167, 169, 173,
  179, 181, 187, 191, 193, 197, 199, 209
};
int skip[WHEEL_SIZE];
int dWheel[WHEEL_SIZE] = {
  10, 2, 4, 2, 4, 6, 2, 6,
   4, 2, 4, 6, 6, 2, 6, 4,
   2, 6, 4, 6, 8, 4, 2, 4,
   2, 4, 8, 6, 4, 6, 2, 4,
   6, 2, 6, 6, 4, 2, 4, 6,
   2, 6, 4, 2, 4, 2,10, 2
};

int64_t intSqrt(int64_t n) {
  if (n <= 0) return -1;
  int64_t root = (int64_t) sqrt(n);
  if ((root+1) * (root+1) < n) root += 1;
  return root;
}

int64_t minOfLL(int64_t a, int64_t b) {
  return a < b ? a : b;
}

int64_t maxOfLL(int64_t a, int64_t b) {
  return a > b ? a : b;
}

int isprime(int n) {
  int i,squareroot;
  if (n>10) {
    int rem = 0;
    squareroot = (int) sqrt(n);
    // skip divisors that are divisible by 2,3,5,7
    // this provides x2 speedup
    for (i=11; i<=squareroot; i=i+dWheel[rem]) {
      if ((n%i)==0)
        return 0;
      rem += 1;
      if (rem == WHEEL_SIZE) rem = 0;
    }
    return 1;
  }
  else
    return 0;
}

#ifdef CAN_CHANGE_ALGORITHM
/* The original isprime only works for int range
  but changing its input to long long will make isprime slower
  this version of isprime is used when prime sieve method is allowed
*/
int isprimeLL(int64_t n) {
  int64_t i,squareroot;
  if (n>10) {
    int rem = 0;
    squareroot = intSqrt(n);
    // skip divisors that are divisible by 2,3,5,7
    // this provides x2 speedup
    for (i=11; i<=squareroot; i=i+dWheel[rem]) {
      if ((n%i)==0)
        return 0;
      rem += 1;
      if (rem == WHEEL_SIZE) rem = 0;
    }
    return 1;
  }
  else
    return 0;
}

uint8_t *generatePrimes(int64_t maxN) {
  uint8_t *prime = malloc(maxN + 1);
  if (!prime) {
    fprintf(stderr, "Out of memory\n");
    exit(2);
  }
  prime[0] = prime[1] = 0;
  prime[2] = 1;
  int64_t i;
  for (i = 3; i <= maxN; i++) {
    prime[i] = i&1;
  }
  int64_t sqrtN = intSqrt(maxN);
  for (i = 3; i <= sqrtN; i += 2) {
    if (prime[i]) {
      int64_t k;
      for (k = i * i; k <= maxN; k += i) {
        prime[k] = 0;
      }
    }
  }
  return prime;
}

// Ideas from https://primesieve.org/segmented_sieve.html
uint64_t segmentedSieve(int64_t low, int64_t limit, int64_t *primes,
      int primeSize, int64_t N, uint8_t *isPrime) {
  int64_t *muls = malloc(sizeof(int64_t) * primeSize);
  int64_t sqrtN = intSqrt(N);
  int64_t segmentSize = maxOfLL(sqrtN, 32768);
  uint8_t *sieve = malloc(segmentSize);
  
  int64_t usedPrimes = 0, s;
  for (s = 3; s*s < low; s+=2) {
    if (isPrime[s]) {
      muls[usedPrimes] = (s*2) - (low-s) % (s*2);
      if (muls[usedPrimes] == s*2) muls[usedPrimes] = 0;
      usedPrimes++;
    }
  }
  
  int64_t n = low;
  if (n % 2 == 0) n += 1;
  int64_t count = 0;
  for (low = low; low <= limit; low += segmentSize) {
    int64_t high = minOfLL(low + segmentSize - 1, limit);
    memset(sieve, 1, segmentSize);
    
    // record used primes
    for (; s * s <= high; s += 2) {
      if (isPrime[s]) {
        muls[usedPrimes] = s * s - low;
        usedPrimes++;
      }
    }
    
    size_t i;
    for (i = 0; i < usedPrimes; i++) {
      // skips 2 to speedup
      int64_t p = primes[i+1] * 2, j;
      for (j = muls[i]; j < segmentSize; j += p) {
        sieve[j] = 0;
      }
      muls[i] = j - segmentSize;
    }
    for (; n <= high; n += 2) {
      count += sieve[n-low];
    }
  }
  
  free(sieve);
  free(muls);
  return count;
}
#endif // CAN_CHANGE_ALGORITHM

int main(int argc, char *argv[])
{
  long long pc,       /* prime counter */
      foundone=0; /* most recent prime found */
  long long int n, limit;

  int size = 1, rank = 0;
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  // argument check
  if (argc <= 1) {
    if (rank==0) fprintf(stderr, "usage: %s <integer>\n", argv[0]);
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
  }

  limit = 0;
  sscanf(argv[1],"%llu",&limit);	
  if (rank == 0) { // only show on master node
    printf("Starting. Numbers to be scanned= %lld\n",limit);
  }

#ifdef CAN_CHANGE_ALGORITHM
  if (limit <= 1) {
    // bad input
    foundone = 0;
    pc = 0;
  }
  else if (limit <= 10) {
    static const int primeSmall[11] = {0,0,2,3,3,5,5,7,7,7,7};
    static const int piSmall[11] = {0,0,1,2,2,3,3,4,4,4,4};
    foundone = primeSmall[limit];
    // prevent repeated counting
    pc = rank == 0 ? piSmall[limit] : 0;
  }
  else {
    // build prime table in 1 ~ sqrt(n)
    int64_t sqrtN = intSqrt(limit);
    uint8_t *table = generatePrimes(sqrtN);
    pc = 0;
    for (n = 1; n <= sqrtN; n++) {
      if (table[n]) {
        pc += 1;
      }
    }
    int64_t *primes = malloc(sizeof(int64_t) * pc);
    pc = 0;
    for (n = 1; n <= sqrtN; n++) {
      if (table[n]) {
        primes[pc++] = n;
      }
    }
    
    int64_t workDiv = (limit-sqrtN) / size;
    int64_t workHigh = rank==size-1 ? limit : sqrtN + workDiv*(rank+1);
    int64_t workLow = rank==0 ? sqrtN+1 : sqrtN+1 + workDiv*rank;
    int64_t count = segmentedSieve(workLow, workHigh, primes, pc, limit, table);
    // only forst node should include primes in [1,sqrt(n)]
    if (rank == 0) pc += count;
    else pc = count;
    if (rank == 0) {
      // find largest prime number <= limit
      foundone = limit;
      // only try odd numbers
      if (foundone%2 == 0) foundone -= 1;
      while (foundone%3 == 0 || foundone%5 == 0 || foundone%7 == 0
         || !isprimeLL(foundone)) {
        foundone -= 2;
      }
    }
    else foundone = 0;
    
    free(primes);
    free(table);
  }
#else // normal algorithm
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

  // compute skip list for this node
  int cycle = (rank+1) / WHEEL_SIZE;
  int rem = (rank+1) % WHEEL_SIZE;
  for (n=0; n<WHEEL_SIZE; n++) {
    int newn = rem + size;
    int newcycle = newn / WHEEL_SIZE;
    int newrem = newn % WHEEL_SIZE;
    skip[n] = newcycle * WHEEL_PRODUCT + wheel[newrem] - wheel[rem];
    rem = newrem;
  }
  // compute start number
  rem = (rank+1) % WHEEL_SIZE;
  n = cycle*WHEEL_PRODUCT + wheel[rem];
  
  // skip some numbers to balance workload
  for (rem = -1; n<=limit; n=n+skip[rem]) {
    if (isprime(n)) {
      pc++;
      foundone = n;
    }			
    rem++;
    if (rem == WHEEL_SIZE) rem = 0;
  }
#endif // CAN_CHANGE_ALGORITHM

  long long pix = pc, largest = foundone;
#ifdef USE_MPI
  // all nodes need to enter reduce instruction
  MPI_Reduce(&pc, &pix, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&foundone, &largest, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
#endif
  if (rank == 0) {
    printf("Done. Largest prime is %lld Total primes %lld\n",largest,pix);
  }
#ifdef USE_MPI
  MPI_Finalize();
#endif
  return 0;
} 
