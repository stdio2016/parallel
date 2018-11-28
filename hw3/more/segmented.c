#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gen.h"
#include "sieve.h"

int main(int argc, char *argv[]) {
  initSieveBitSet();
  if (argc < 2) {
    fprintf(stderr, "usage: %s <number>\n", argv[0]);
    return 0;
  }
  long long n = atoll(argv[1]);
  printf("n=%lld\n", n);
  long long sqrtn = (long long) sqrt(n);
  int64_t pi;
  uint32_t *P = generatePrimes(sqrtn, &pi);
  
  long long goodSize = getSieveSize(sqrtn * 4);
  printf("goodSize=%lld\n", goodSize);
  struct Sieve *s = createSieve(goodSize, pi, P, 0);
  int64_t low;
  int64_t pc = 0;
  for (low = 0; low <= n; low += goodSize) {
    int64_t high = low + goodSize;
    if (high > n) high = n+1;
    initSieve(s, low, high);
    int j;
    for (j = 4; j <= pi; j++) {
      sieveCrossoff(s, P[j], j);
    }
    uint64_t *sv = s->sieve;
    uint8_t *sb = s->sieve;
    int send = s->sieveSize>>3;
    for (j = 0; j < send; j++) {
      pc += __builtin_popcountll(sv[j]);
    }
    /*for (j = 0; j < s->sieveSize; j++) {
      int k;
      int goo[8]={1,7,11,13,17,19,23,29};
      for (k=0;k<8;k++) {
        if (sb[j]>>k&1) printf("%lld,",low+goo[k]+j*30);
      }
    }
    puts("");*/
  }
  printf("pi(n) = %lld\n", pc + pi - 1);
  return 0;
}
