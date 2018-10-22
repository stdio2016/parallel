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
  MatrixCell(int id, MatrixColumn *c) {
    col = c;
    row = id;
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
  MatrixColumn(int id, int min, int max):
    MatrixCell(id,NULL), size(0), value(0), min(min), max(max) {}
};

struct DLXSolver {
  long long tried, maxTry;
  int solCount;
  int lv, maxLv;
  MatrixColumn *root;
  std::vector<MatrixColumn *> cols;
  std::vector<MatrixCell *> rows;
  std::vector<MatrixCell *> cells;
  struct StackFrame {
    MatrixCell *n;
    MatrixColumn *c;
    int choice;
  };
  std::vector<StackFrame> stack;
  int removedRowCount;
  std::vector<MatrixCell *> removedRow;

  int dlx();
  void showSolution();

  DLXSolver() {
    tried = maxTry = 0;
    solCount = 0;
    lv = maxLv = 0;
    root = new MatrixColumn(-1, 0, 0);
    root->left = root->right = root->up = root->down = root;
    removedRowCount = 0;
  }
  void addColumn(int id, int min, int max);
  void addRow(int id);
  void addCell(int row, int col);
  void setMaxLv(int n);
};

#endif
