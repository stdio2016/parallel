#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int *PiTable;
int *MuTable;
int *FTable;
int64_t *SmallPrimes;
int PiSqrtN;
int64_t Steps;

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
  SmallPrimes = malloc(pisqrtN * sizeof(int64_t));
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

double Ei(double x) {
  double gamma = 0.577215664901532;
  int n;
  double sum = 0;
  double term = -2 * exp(x*0.5);
  double pp = 0;
  for (n = 1; n < 100; n++) {
    term *= -x / (2*n);
    if (n&1) {
      pp += 1.0 / n;
    }
    double s = term * pp;
    if (fabs(s) < 1.0e-15) {
      break;
    }
    sum += s;
  }
  return gamma + log(fabs(x)) + sum;
}

double RiemannR(double x) {
  double lnx = log(x), sum;
  int n;
  for (n = 1; n < 200; n++) {
    if (MuTable[n]) {
      double li = Ei(lnx/n) * MuTable[n] / n;
      sum += li;
    }
  }
  return sum;
}

int main() {
  double x;
  buildTable(200);
  scanf("%lf", &x);
  printf("%f", RiemannR(x));
  return 0;
}
