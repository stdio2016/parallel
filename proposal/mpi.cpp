#include <iostream>
#include <fstream>
#include <mpi.h>
#include "dlx.h"
#include "loader.h"

void mpifinalize() {
  MPI_Finalize();
}

int Proc_rank, Proc_size;

struct MyDummyCout {
  template <class T>
  MyDummyCout &operator<<(T a) {
    if (Proc_rank == 0) std::cout << a;
    return *this;
  }
} cout;

int main(int argc, char *argv[]) {
  int maxDepth = 2;
  MPI_Init(&argc, &argv);
  atexit(mpifinalize);
  MPI_Comm_size(MPI_COMM_WORLD, &Proc_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &Proc_rank);
  
  // load dlx file
  if (Proc_size < 2) {
    cout << "this program requires at least 2 processes to start\n";
    return 1;
  }
  if (argc <= 1) {
    cout <<"usage: "<<argv[0]<<" <dlx file> <depth>\n";
    return 1;
  }
  DLXLoader loader;
  std::ifstream f(argv[1]);
  if (!f) {
    cout << "cannot open file \"" << argv[1] << "\"\n";
    return 1;
  }
  loader.readFile(f);
  f.close();
  
  // prepare solver
  DLXSolver d;
  int rowCount = loader.rows.size();
  int *rows = new int[rowCount];
  d.maxTry = -1;
  d.setMaxLv(rowCount);
  d.setRoot(loader.root);
  d.setRowCount(rowCount);
  
  MPI_Status status;
  if (Proc_rank == 0) {
    d.maxSol = 1;
    d.setMaxLv(maxDepth);
    int i;
    int more;
    int total = 0;
    for (i = 1; i < Proc_size && (more = d.dlx()); i++) {
      for (int j = 0; j < d.solutions[1]; j++) {
        rows[j] = d.solRows[j];
      }
      MPI_Send(rows, d.solutions[1], MPI_INT, i, 0, MPI_COMM_WORLD);
      d.solCount = 0;
    }
    if (!more) {
      // too many processes
      for (int j = i; j < Proc_size; j++) {
        MPI_Send(NULL, 0, MPI_INT, j, 0, MPI_COMM_WORLD);
      }
    }
    while (more || i > 1) {
      int ans;
      MPI_Recv(&ans, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
      total += ans;
      more = d.dlx();
      if (more) {
        for (int j = 0; j < d.solutions[1]; j++) {
          rows[j] = d.solRows[j];
        }
        MPI_Send(rows, d.solutions[1], MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
      }
      else {
        MPI_Send(NULL, 0, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
        i--;
      }
      d.solCount = 0;
    }
    cout << "solution count: " << total << "\n";
  }
  else {
    char name[100];
    sprintf(name, "out%d.txt", Proc_rank);
    std::ofstream fout(name);
    MPI_Recv(rows, rowCount, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    int actualLen;
    MPI_Get_count(&status, MPI_INT, &actualLen);
    while (actualLen > 0) {
      for (int i = 0; i < actualLen; i++) {
        d.enterBranch(rows[i]);
      }
      d.minLv = actualLen;
      d.solCount = 0;
      d.ended = false;
      d.dlx();
      /*for (int i = 0; i < d.solCount; i++) {
        fout << "sol:";
        for (int j = d.solutions[i]; j < d.solutions[i+1]; j++) {
          fout << " " << d.solRows[j];
        }
        fout << "\n";
      }*/
      for (int i = 0; i < actualLen; i++) {
        d.leaveBranch();
      }
      MPI_Send(&d.solCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      MPI_Recv(rows, rowCount, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, MPI_INT, &actualLen);
    }
  }
  
  // cleanup
  delete [] loader.root;
  delete [] loader.cells;
  return 0;
}
