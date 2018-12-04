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

inline uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t m) {
  if (a <= UINT32_MAX && b <= UINT32_MAX) return a*b%m;
  return 0;
}

inline uint64_t pow_mod_slow(uint64_t a, uint64_t b, uint64_t mod) {
  return 0;
}

uint64_t pow_mod(uint64_t a, uint64_t b, uint64_t mod) {
  if (a > UINT32_MAX || mod > UINT32_MAX) return pow_mod_slow(a, b, mod);
  int64_t r = 1;
  for (b = b; b > 0; b >>= 1) {
    if (b&1) r = r * a % mod;
    a = a * a % mod;
  }
  return r;
}

int64_t gcdLL(int64_t a, int64_t b) {
  if (b == 0) return a;
  while ((a %= b) && (b %= a)) ;
  return a + b;
}

int jacobi(int64_t a, int64_t n) {
  int ans = 1;
  if (n == 1) return 1;
  if (a < 0) {
    a = -a;
    ans = (n&3) == 1 ? 1 : -1;
  }
  if (gcdLL(a, n) != 1) return 0;
  for (;;) {
    a %= n;
    int flag = 0;
    if (a == 0) return 0;
    while ((a&1) == 0) {
      a >>= 1;
      flag ^= 1;
    }
    if (flag && ((n&7) == 3 || (n&7) == 5)) ans = -ans;
    if (a == 1) return ans;
    if ((a&3) == 3 && (n&3) == 3) ans = -ans;
    int64_t tmp = a; a = n; n = tmp;
  }
  return ans;
}

int isprimeBPSW(int64_t n) {
  int i;
  int64_t squareroot;
  if (n>10) {
    // trial division
    squareroot = intSqrt(n);
    if (squareroot>=11 && n%11==0) return 0;
    if (squareroot>=13 && n%13==0) return 0;
    if (squareroot>=17 && n%17==0) return 0;
    if (squareroot>=19 && n%19==0) return 0;
    if (squareroot * squareroot == n) return 0;
    // miller rabin base 2
    int s = 0;
    int64_t d;
    for (d = n-1; (d&1) == 0; d >>= 1) s++;
    int a = 2;
    int64_t x = pow_mod(a, d, n);
    if (x != 1 && x != n-1) {
      for (i = 1; i < s; i++) {
        x = mod_mul(x, x, n);
        if (x == n-1) break;
      }
      if (i == s) return 0;
    }
    // Jacobi
    int64_t D = 5;
    while (jacobi(D, n) != -1) {
      D += 2;
      if (jacobi(-D, n) == -1) {
        D = -D;
        break;
      }
      D += 2;
    }
    // lucas probable prime
    int64_t P = 1;
    int64_t Q = (1 - D) / 4;
    if (Q < 0) Q += n;
    if (D < 0) D += n;
    s = 0;
    for (d = n+1; (d&1) == 0; d >>= 1) s++;
    int64_t b;
    i = 0;
    for (b = d; b > 0; b >>= 1) i++;
    int64_t U = 0, V = 2, Qn = 1;
    for (i = i - 1; i >= 0; i--) {
      U = mod_mul(U, V, n);
      V = mod_mul(V, V, n) - Qn;
      if (V < 0) V += n;
      V -= Qn;
      if (V < 0) V += n;
      Qn = mod_mul(Qn, Qn, n);
      if (d>>i & 1) {
        uint64_t U2k = mod_mul(P, U, n) + V;
        uint64_t V2k = mod_mul(D, U, n) + mod_mul(P, V, n);
        if (U2k >= n) U2k -= n;
        if (U2k&1) U = U2k+n >> 1;
        else U = U2k >> 1;
        if (V2k >= n) V2k -= n;
        if (V2k&1) V = V2k+n >> 1;
        else V = V2k >> 1;
        Qn = mod_mul(Qn, Q, n);
      }
    }
    if (U != 0 && V != 0) {
      for (i = 1; i < s; i++) {
        V = mod_mul(V, V, n) - Qn;
        if (V < 0) V += n;
        V -= Qn;
        if (V < 0) V += n;
        if (V == 0) break;
        Qn = mod_mul(Qn, Qn, n);
      }
      if (i == s) return 0;
    }
    return 1;
  }
  else
    return 0;
}

int isprime(int64_t n) {
  return isprimeBPSW(n);
}

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
