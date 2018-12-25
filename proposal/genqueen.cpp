#include <stdio.h>

int main() {
  int n;
  scanf("%d", &n);
  puts("dlx");
  for (int i = 1; i <= n; i++) {
    printf("col -%d min 1 max 1\n", i);
  }
  for (int i = 1; i <= n; i++) {
    printf("col |%d min 1 max 1\n", i);
  }
  for (int i = 1; i <= n*2-1; i++) {
    printf("col /%d min 0 max 1\n", i);
  }
  for (int i = 1; i <= n*2-1; i++) {
    printf("col \\%d min 0 max 1\n", i);
  }
  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= n; j++) {
      int a = i + j - 1;
      int b = j - i + n;
      printf("row r%dc%d -%d |%d /%d \\%d\n", i,j, i,j,a,b);
    }
  }
}
