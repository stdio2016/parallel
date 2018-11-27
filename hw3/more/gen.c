#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "gen.h"

uint32_t *generatePrimes(int64_t maxN, int64_t *size) {
  uint8_t *s = malloc(maxN + 1);
  if (!s) {
    fprintf(stderr, "Out of memory\n");
    exit(2);
  }
  s[0] = s[1] = 0; s[2] = 1;
  int64_t i, k;
  int64_t sqrtN = (int) sqrt(maxN);
  for (i = 3; i <= maxN; i++) s[i] = i&1;
  for (i = 3; i <= sqrtN; i += 2) {
    if (s[i]) {
      for (k = i * i; k <= maxN; k += i) s[k] = 0;
    }
  }
  int64_t cnt = 0;
  for (i = 2; i <= maxN; i++) cnt += s[i];
  *size = cnt;
  uint32_t *prime = malloc((cnt+1) * sizeof(uint32_t));
  prime[0] = 0;
  cnt = 0;
  for (i = 2; i <= maxN; i++) {
    if (s[i]) prime[++cnt] = i;
  }
  free(s);
  return prime;
}
