// just want to imitate https://github.com/kimwalisch/primecount/blob/master/src/Sieve.cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sieve.h"

uint8_t unsetLargeSieveB[240*8];
uint8_t unsetSmallSieveB[240*8];
uint64_t *unsetLargeSieve = (uint64_t *) unsetLargeSieveB;
uint64_t *unsetSmallSieve = (uint64_t *) unsetSmallSieveB;
struct Wheel wheel30[30] = {
  {1,0}, {0,0}, {5,1}, {4,1}, {3,1}, {2,1}, {1,1}, {0,1}, {3,2}, {2,2},
  {1,2}, {0,2}, {1,3}, {0,3}, {3,4}, {2,4}, {1,4}, {0,4}, {1,5}, {0,5},
  {3,6}, {2,6}, {1,6}, {0,6}, {5,7}, {4,7}, {3,7}, {2,7}, {1,7}, {0,7}
};

void initSieveBitSet() {
  int i = 0;
  for (i = 0; i < 29; i++) {
    unsetLargeSieveB[i*8] = (1<<wheel30[i+1].idx) - 1;
  }
  unsetLargeSieveB[29*8] = 0xff;
  for (i = 0; i < 8; i++) {
    int j, k;
    for (j = 0; j < 30; j++) {
      for (k = 0; k < i; k++) unsetLargeSieveB[(i*30+j)*8+k] = 0xff;
      unsetLargeSieveB[(i*30+j)*8+i] = unsetLargeSieveB[j*8];
      for (k = i+1; k < 8; k++) unsetLargeSieveB[(i*30+j)*8+k] = 0;
    }
  }
}

struct Sieve *createSieve(uint64_t sieveSize, uint32_t primeCount, uint32_t *primes, uint64_t start) {
  struct Sieve *s = malloc(sizeof(struct Sieve));
  s->sieveSize = sieveSize/30;
  s->primeSize = primeCount;
  s->sieve = malloc(sieveSize/30);
  s->primes = malloc((primeCount+1) * sizeof(struct Wheel));
  s->start = start;
  int64_t i;
  for (i = 4; i <= primeCount; i++) {
    int64_t p = primes[i];
    int64_t q = start / p + 1;
    int idx = wheel30[q%30].idx;
    idx += wheel30[p%30].idx << 3;
    q += wheel30[q%30].mul;
    s->primes[i].mul = (p * q) / 30;
    s->primes[i].idx = idx;
  }
  return s;
}

uint64_t getSieveSize(uint64_t size) {
  if (size == 0) size = 240;
  if (size % 240) size += 240 - size % 240;
  return size;
}

void initSieve(struct Sieve *s, uint64_t from, uint64_t to) {
  memset(s->sieve, 0xff, s->sieveSize);
  uint64_t size = to - from;
  if (size < s->sieveSize*30) {
    s->sieveSize = getSieveSize(size)/30;
    uint64_t *b = (uint64_t *) s->sieve;
    uint64_t last = size-1;
    b[last/240] &= unsetLargeSieve[last%240];
  }
  s->start = from;
}

#define UNSET_BIT_CASE(cas,n,m,a) {\
    if (b >= end) { w.idx = cas; break; }\
  case cas:\
    cnt += *b>>n & 1; *b &= ~(1<<n); b += m*prime+a;\
  }
#define UNSET_BIT_P(n, p) do { cnt += b[p]>>n & 1; b[p] &= ~(1<<n); } while (0)

#define CASES(bas,a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7) {\
    UNSET_BIT_CASE(bas+0, a0, 6, b0);\
    UNSET_BIT_CASE(bas+1, a1, 4, b1);\
    UNSET_BIT_CASE(bas+2, a2, 2, b2);\
    UNSET_BIT_CASE(bas+3, a3, 4, b3);\
    UNSET_BIT_CASE(bas+4, a4, 2, b4);\
    UNSET_BIT_CASE(bas+5, a5, 4, b5);\
    UNSET_BIT_CASE(bas+6, a6, 6, b6);\
    UNSET_BIT_CASE(bas+7, a7, 2, b7);\
  }

#define UNROLL(a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7) do {\
    UNSET_BIT_P(a0,  0*prime+b0);\
    UNSET_BIT_P(a1,  6*prime+b1);\
    UNSET_BIT_P(a2, 10*prime+b2);\
    UNSET_BIT_P(a3, 12*prime+b3);\
    UNSET_BIT_P(a4, 16*prime+b4);\
    UNSET_BIT_P(a5, 18*prime+b5);\
    UNSET_BIT_P(a6, 22*prime+b6);\
    UNSET_BIT_P(a7, 28*prime+b7);\
  } while (0)

int64_t sieveCrossoff(struct Sieve *s, uint64_t prime, uint64_t i) {
  struct Wheel w = s->primes[i];
  int64_t cnt = 0;
  if (w.mul >= s->sieveSize) {
    w.mul -= s->sieveSize;
  }
  else {
    uint8_t *b = (uint8_t *) s->sieve;
    uint8_t *end = &b[s->sieveSize];
    b = &b[w.mul];
    prime /= 30;
    switch (w.idx) {
      while (1) { /* case 0~7 */
        CASES(0*8 , 0,1,2,3,4,5,6,7 , 0,0,0,0,0,0,0,1);
        for (; b + prime*28 < end; b += 30*prime+1)
          UNROLL(0,1,2,3,4,5,6,7 , 0,0,0,0,0,0,0,0);
      }
      break;
      
      while (7) { /* case 8~15 */
        CASES(1*8 , 1,5,4,0,7,3,2,6 , 1,1,1,0,1,1,1,1);
        for (; b + prime*28+6 < end; b += 30*prime+7)
          UNROLL(1,5,4,0,7,3,2,6 , 0,1,2,3,3,4,5,6);
      }
      break;
      
      while (11) { /* case 16~23 */
        CASES(2*8 , 2,4,0,6,1,7,3,5 , 2,2,0,2,0,2,2,1);
        for (; b + prime*28+10 < end; b += 30*prime+11)
          UNROLL(2,4,0,6,1,7,3,5 , 0,2,4,4,6,6,8,10);
      }
      break;
      
      while (13) { /* case 24~31 */
        CASES(3*8 , 3,0,6,5,2,1,7,4 , 3,1,1,2,1,1,3,1);
        for (; b + prime*28+12 < end; b += 30*prime+13)
          UNROLL(3,0,6,5,2,1,7,4 , 0,3,4,5,7,8,9,12);
      }
      break;
      
      while (17) { /* case 32~39 */
        CASES(4*8 , 4,7,1,2,5,6,0,3 , 3,3,1,2,1,3,3,1);
        for (; b + prime*28+16 < end; b += 30*prime+17)
          UNROLL(4,7,1,2,5,6,0,3 , 0,3,6,7,9,10,13,16);
      }
      break;
      
      while (19) { /* case 40~47 */
        CASES(5*8 , 5,3,7,1,6,0,4,2 , 4,2,2,2,2,2,4,1);
        for (; b + prime*28+18 < end; b += 30*prime+19)
          UNROLL(5,3,7,1,6,0,4,2 , 0,4,6,8,10,12,14,18);
      }
      break;
      
      while (23) { /* case 48~55 */
        CASES(6*8 , 6,2,3,7,0,4,5,1 , 5,3,1,4,1,3,5,1);
        for (; b + prime*28+22 < end; b += 30*prime+23)
          UNROLL(6,2,3,7,0,4,5,1 , 0,5,8,9,13,14,17,22);
      }
      break;
      
      while (29) { /* case 56~63 */
        CASES(7*8 , 7,6,5,4,3,2,1,0 , 6,4,2,4,2,4,6,1);
        for (; b + prime*28+28 < end; b += 30*prime+29)
          UNROLL(7,6,5,4,3,2,1,0 , 0,6,10,12,16,18,22,28);
      }
      break;
    }
    w.mul = (uint32_t)(b - end);
  }
  s->primes[i].mul = w.mul;
  s->primes[i].idx = w.idx;
  return cnt;
}
