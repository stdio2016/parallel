#pragma once
#include <stdint.h>

struct Wheel {
  uint32_t mul;
  uint32_t idx;
};

struct Sieve {
  void *sieve;
  struct Wheel *primes;
  uint32_t sieveSize;
  uint32_t primeSize;
  uint64_t start;
};

void initSieveBitSet(void);
struct Sieve *createSieve(uint64_t sieveSize, uint32_t primeCount, uint32_t *primes, uint64_t start);
uint64_t getSieveSize(uint64_t size);
void initSieve(struct Sieve *s, uint64_t from, uint64_t to);
int64_t sieveCrossoff(struct Sieve *s, uint64_t prime, uint64_t i);
