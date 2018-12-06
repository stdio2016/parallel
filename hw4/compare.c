#include <stdio.h>

int main(int argc, char *argv[]) {
  if (argc < 3) return 0;
  FILE *f = fopen(argv[1], "r");
  FILE *g = fopen(argv[2], "r");
  int nl = 0;
  while (nl < 4) {
    if (fgetc(f) == '\n') nl++;
  }
  nl = 0;
  while (nl < 4) {
    if (fgetc(g) == '\n') nl++;
  }
  int i;
  float f1, f2;
  float sum = 0, maxerr = 0;
  int n = 0;
  while (fscanf(f, "%f", &f1) == 1) {
    fscanf(g, "%f", &f2);
    n++;
    float a;
    if (f1 > f2) a = f1 - f2;
    else a = f2 - f1;
    sum += a;
    if (a > maxerr) maxerr = a;
  }
  printf("average error: %f\n", sum / n);
  printf("largest error: %f\n", maxerr);
  return 0;
}
