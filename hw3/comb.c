#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

int *PiTable;
int *MuTable;
int *FTable;
uint32_t *SmallPrimes;
int PiSqrtN;
int64_t Steps;

//#define DEBUG

static inline void countStep() {
#ifdef DEBUG
  Steps++;
#endif
}

int64_t intSqrt(int64_t n) {
  if (n <= 0) return -1;
  int64_t root = (int64_t) sqrt(n);
  if ((root+1) * (root+1) <= n) root += 1;
  return root;
}

int64_t intCbrt(int64_t n) {
  if (n <= 0) return -1;
  int64_t root = (int64_t) cbrt(n);
  root += 1;
  if (root * root * root > n) root -= 1;
  return root;
}

int64_t minOfLL(int64_t a, int64_t b) {
  return a < b ? a : b;
}

int64_t maxOfLL(int64_t a, int64_t b) {
  return a > b ? a : b;
}

void buildTable(int64_t n) {
  PiTable = malloc((n+1) * sizeof(int));
  MuTable = malloc((n+1) * sizeof(int));
  FTable = malloc((n+1) * sizeof(int));
  PiTable[0] = PiTable[1] = 0;
  PiTable[2] = 1;
  int64_t i, k;
  for (i = 3; i <= n; i++) PiTable[i] = 1;
  for (i = 0; i <= n; i++) MuTable[i] = 1;
  MuTable[1] = 1;
  FTable[0] = FTable[1] = 0;
  int pisqrtN = 0;
  for (i = 2; i*i <= n; i += 1) {
    if (PiTable[i]) {
      ++pisqrtN;
      for (k = i*i; k <= n; k += i) {
        if (PiTable[k]) FTable[k] = pisqrtN;
        PiTable[k] = 0;
      }
    }
  }
  for (i = 2; i <= n; i++) {
    if (PiTable[i]) {
      for (k = i; k <= n; k += i) {
        MuTable[k] = -MuTable[k];
      }
      for (k = i*i; k <= n; k += i*i) {
        MuTable[k] = 0;
      }
    }
  }
  
  pisqrtN = 0;
  for (i = 0; i <= n; i++) pisqrtN += PiTable[i];
  SmallPrimes = malloc(pisqrtN * sizeof(uint32_t));
  pisqrtN = 0;
  for (i = 2; i <= n; i++) {
    if (PiTable[i]) {
      SmallPrimes[pisqrtN] = i;
      FTable[i] = pisqrtN+1;
      pisqrtN++;
    }
  }
  PiSqrtN = pisqrtN;
  for (i = 1; i <= n; i++) {
    PiTable[i] += PiTable[i-1];
  }
}

int64_t phiEasy(int64_t x) {
  int64_t sqrtX = intSqrt(x);
  int64_t n, sum = 0;
  for (n = 1; n <= sqrtX; n++) {
    sum += MuTable[n] * (x / n);
    countStep();
  }
  return sum;
}

int64_t phiHard(int64_t x) {
  int64_t sqrtX = intSqrt(x);
  int64_t x14 = intSqrt(sqrtX);
  int64_t b, sum = 0;
  uint32_t *phiTable = malloc((sqrtX+1) * sizeof(uint32_t));
  for (b = 0; b <= sqrtX; b++) phiTable[b] = b;
  
  for (b = 1; b <= PiTable[x14]; b++) {
    int64_t i;
    int64_t pb = SmallPrimes[b-1];
    for (i = sqrtX/pb + 1; i <= sqrtX; i ++) {
      if (FTable[i] > b) {
        int64_t con = (int64_t)phiTable[x / (i*pb)] * MuTable[i];
        sum += con;
      }
      countStep();
    }
    i = sqrtX;
    int64_t j;
    for (j = sqrtX / pb; j >= 0; j--) {
      int64_t pj = phiTable[j], iM = j * pb;
      for (; i >= iM; i--) {
        phiTable[i] -= pj;
        countStep();
      }
    }
  }
  free(phiTable);
  return -sum;
}

int64_t phiA(int64_t x) {
  int64_t x13 = intCbrt(x);
  int64_t p13 = PiTable[x13];
  countStep();
  return (PiSqrtN - p13) * (PiSqrtN - p13 - 1) / 2;
}

int64_t phiB(int64_t x) {
  int64_t x13 = intCbrt(x);
  int64_t x14 = intSqrt(intSqrt(x));
  int64_t p13 = PiTable[x13];
  int64_t sum = 0;
  int64_t b;
  for (b = PiTable[x14] + 1; b <= p13; b++) {
    int64_t pb = SmallPrimes[b-1];
    sum += PiSqrtN - PiTable[x / (pb*pb)];
    countStep();
  }
  return sum;
}

int64_t phiC(int64_t x) {
  int64_t x13 = intCbrt(x);
  int64_t x14 = intSqrt(intSqrt(x));
  int64_t p13 = PiTable[x13];
  int64_t sum = 0;
  int64_t b;
  for (b = PiTable[x14] + 1; b <= p13; b++) {
    int64_t pb = SmallPrimes[b-1];
    int64_t cMax = PiTable[x / (pb*pb)];
    int64_t pi = PiTable[x / (SmallPrimes[cMax-1] * pb)];
    int64_t cMin = PiTable[x / (pb*SmallPrimes[pi])];
    int64_t cEnd = PiTable[x / (x13*pb)];
    while (cMin >= cEnd) {
      //printf("b=%lld c=%lld~%lld pi=%lld ", b,cMin+1,cMax,pi);
      //printf("get %lld\n", (pi - (b-1) + 1) * (cMax - cMin));
      sum += (pi - (b-1) + 1) * (cMax - cMin);
      pi++;
      cMax = cMin;
      cMin = PiTable[x / (pb*SmallPrimes[pi])];
      countStep();
    }
    sum += (pi - (b-1) + 1) * (cMax - cEnd);
    countStep();
  }
  return sum;
}

int64_t phiD(int64_t x) {
  int64_t x13 = intCbrt(x);
  int64_t x14 = intSqrt(intSqrt(x));
  int64_t p13 = PiTable[x13];
  int64_t sum = 0;
  int64_t b, c;
  for (b = PiTable[x14] + 1; b <= p13; b++) {
    int64_t pb = SmallPrimes[b-1];
    int64_t cMax = PiTable[x / (x13*pb)];
    for (c = b+1; c <= cMax; c++) {
      int64_t pc = SmallPrimes[c-1];
      sum += PiTable[x / (pc*pb)] - (b-1) + 1;
      countStep();
    }
  }
  return sum;
}

int64_t pi(int64_t x) {
  int64_t pe = phiEasy(x);
  int64_t ph = phiHard(x);
  int64_t pa = phiA(x);
  int64_t pb = phiB(x);
  int64_t pc = phiC(x);
  int64_t pd = phiD(x);
#ifdef DEBUG
  printf("easy        = %lld\n", pe);
  printf("hard        = %lld\n", ph);
  printf("area A      = %lld\n", pa);
  printf("area B      = %lld\n", pb);
  printf("area C      = %lld\n", pc);
  printf("area D      = %lld\n", pd);
  printf("pi(sqrt(x)) = %d\n", PiSqrtN);
#endif
  return pe + ph + pa + pb + pc + pd + PiSqrtN - 1;
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
  
  if (limit <= 0) {
    pc = 0;
  }
  else {
#ifdef DEBUG
    printf("sqrt(x)     = %lld\n", intSqrt(limit));
#endif
    buildTable(intSqrt(limit));
    pc = pi(limit);
    printf("ans=%lld steps=%lld\n", pc, Steps);
  }
  return 0;
}
