#include <cstdio>
#include "dlx.h"

// solve N queen problem
int main(void)
{
  DLXSolver d;
  int N = 13;
  MatrixColumn *cols = new MatrixColumn[1+N*2+(2*N-1)*2];
  MatrixCell *nodes = new MatrixCell[N*N*5];
  int horId = 1;
  int verId = horId + N;
  int diagId = verId + N;
  int diagId2 = diagId + (2*N-1);
  int colN = diagId2 + (2*N-1);
  for (int i=0; i<colN; i++) {
    cols[i].setRight(&cols[0]);
    cols[i].col = &cols[i];
    cols[i].row = i;
    if (i >= diagId) cols[i].min = 0; // optional column
  }
  cols[0].max = 0; // root node
  for (int i=0;i<N;i++) {
    for (int j=0;j<N;j++) {
      int nid = (i*N+j)*5;
      nodes[nid].col = &cols[0];
      nodes[nid+1].col = &cols[horId + i];
      nodes[nid+2].col = &cols[verId + j];
      nodes[nid+3].col = &cols[diagId + i+j];
      nodes[nid+4].col = &cols[diagId2 + i-j+N-1];
      for (int k = 0; k < 5; k++) {
        nodes[nid+k].setRight(&nodes[nid]);
        nodes[nid+k].setDown(nodes[nid+k].col);
        nodes[nid+k].row = (i+1)*100 + (j+1);
        nodes[nid+k].col->size += 1;
      }
    }
  }
  d.maxTry = -1;
  d.setMaxLv(N);
  d.setRoot(cols);
  d.setRowCount(N*N);
  d.dlx();
  printf("%d\n", d.solCount);
  delete [] cols;
  delete [] nodes;
  return 0;
}
