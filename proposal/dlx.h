#pragma once
#ifndef DLX_INCLUDED
#define DLX_INCLUDED
#include <vector>

struct MatrixColumn;

struct MatrixCell {
  MatrixCell *up, *down;
  MatrixCell *left;
  MatrixCell *right;
  MatrixColumn *col;
  int row;
  MatrixCell(): up(this), down(this), left(this), right(this),
    col(NULL), row(0)
  {
  }
  void setRight(MatrixCell *n) {
    right = n;
    left = n->left;
    n->left->right = this;
    n->left = this;
  }
  void setDown(MatrixCell *n) {
    down = n;
    up = n->up;
    n->up->down = this;
    n->up = this;
  }
};

struct MatrixColumn : MatrixCell {
  int size;
  int value, min, max;
  MatrixColumn(): MatrixCell(),
    size(0), value(0), min(1), max(1)
  {
  }
};

struct DLXSolver {
  long long tried, maxTry;
  int solCount, maxSol;
  int lv, maxLv, minLv;
  MatrixColumn *root;
  struct StackFrame {
    MatrixCell *n;
    MatrixColumn *c;
    int choice;
  };
  std::vector<StackFrame> stack;
  int removedRowCount;
  std::vector<MatrixCell *> removedRow;
  std::vector<int> solutions;
  std::vector<int> solRows;
  bool ended;

  int dlx();
  void showSolution();

  DLXSolver() {
    tried = maxTry = 0;
    solCount = 0;
    solutions.push_back(0);
    maxSol = -1;
    lv = maxLv = minLv = 0;
    root = NULL;
    removedRowCount = 0;
    ended = false;
  }
  void setMaxLv(int n);
  int minfit(MatrixColumn **result);
  void setRowCount(int n);
  void setRoot(MatrixColumn *node);

  void unlinkRow(MatrixCell *n, bool includeN);
  void relinkRow(MatrixCell *n, bool includeN);
  void cover(MatrixCell *n);
  void uncover(MatrixCell *n);
  
  void enterBranch(int index);
  void leaveBranch();
};

inline void DLXSolver::unlinkRow(MatrixCell *n, bool includeN) {
  //printf("unlink %d\n", n->row);
  MatrixCell *e = n;
  if (!includeN) n = n->right;

  do {
    n->up->down = n->down;
    n->down->up = n->up;
    n->col->size -= 1;
    n = n->right;
  } while (n != e) ;
}

inline void DLXSolver::relinkRow(MatrixCell *n, bool includeN) {
  //printf("relink %d\n", n->row);
  MatrixCell *e = n;
  if (!includeN) n = n->right;

  do {
    n->up->down = n;
    n->down->up = n;
    n->col->size += 1;
    n = n->right;
  } while (n != e) ;
}

inline void DLXSolver::cover(MatrixCell *n) {
  //printf("cover %d\n", n->row);
  MatrixCell *n2, *n3;
  MatrixColumn *c2;
  unlinkRow(n, true);
  n2 = n;
  do {
    c2 = n2->col;
    c2->value += 1;
    if (c2->value == c2->max) {
      c2->left->right = c2->right;
      c2->right->left = c2->left;
      n3 = c2->down;
      while (n3 != c2) {
        unlinkRow(n3, false);
        n3 = n3->down;
      }
    }
    n2 = n2->right;
  } while (n2 != n) ;
}

inline void DLXSolver::uncover(MatrixCell *n) {
  //printf("uncover %d\n", n->row);
  MatrixCell *n2, *n3;
  MatrixColumn *c2;
  n2 = n;
  do {
    n2 = n2->left;
    c2 = n2->col;
    if (c2->value == c2->max) {
      n3 = c2->up;
      while (n3 != c2) {
        relinkRow(n3, false);
        n3 = n3->up;
      }
      c2->left->right = c2;
      c2->right->left = c2;
    }
    c2->value -= 1;
  } while (n2 != n) ;
}

#endif
