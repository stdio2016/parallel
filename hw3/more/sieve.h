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
