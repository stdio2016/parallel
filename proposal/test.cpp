#include <cstdio>
#include "dlx.h"

int main(void)
{
  DLXSolver d;
  int N = 12;
  for(int i=0;i<N;i++) d.addColumn(100+i, 1, 1);
  for(int i=0;i<N;i++) d.addColumn(200+i, 1, 1);
  for(int i=0;i<N*2-1;i++) d.addColumn(300+i, 0, 1);
  for(int i=0;i<N*2-1;i++) d.addColumn(400+i, 0, 1);
  for (int i=0;i<N;i++) {
    for (int j=0;j<N;j++) {
      d.addRow(i*100+j+101);
      d.addCell(i*N+j, i);
      d.addCell(i*N+j, N+j);
      d.addCell(i*N+j, N+N+ (i+j));
      d.addCell(i*N+j, N+N+N*2-1+ (i-j+N-1));
    }
  }
  d.maxTry = 1000000;
  d.setMaxLv(N);
  d.dlx();
  printf("%d\n", d.solCount);
  return 0;
}
