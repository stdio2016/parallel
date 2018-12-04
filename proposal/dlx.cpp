#include <cstdio>
#include "dlx.h"

int DLXSolver::minfit(MatrixColumn **result) {
  unsigned minfit = -1; // -1 converted to UINT_MAX!
  MatrixColumn *c = NULL;
  for (MatrixColumn *col = (MatrixColumn*)root->right;
      col != root; col = (MatrixColumn *)col->right) {
    int value = col->value;
    // satisfied
    if (value <= col->max && value >= col->min) continue;
    // unsatisfiable
    if (value > col->max || value + col->size < col->min) {
      return -1;
    }

    // min fit
    if (unsigned(col->size) < minfit) {
      c = col;
      minfit = col->size;
    }
  }
  if (minfit == unsigned(-1)) return -2; // a solution
  *result = c;
  return 0;
}

int DLXSolver::dlx() {
  while (tried != maxTry) {
    tried++;
    MatrixColumn *c;
    MatrixCell *n;
    int has = minfit(&c);
    int rrc = removedRowCount;
    if (lv > maxLv || has == -2) {
      showSolution();
      has = -1;
    }
    if (has == -1) {
      if (lv == 0) return 0;
      --lv;
      n = stack[lv].n;
      c = stack[lv].c;
      rrc = stack[lv].choice;

      // recover
      uncover(n);
      n = n->down;
      //puts("leave");
    }
    else {
      n = c->down;
    }

    // get deeper
    while (n == c) {
      while (removedRowCount > rrc) {
        relinkRow(removedRow[--removedRowCount], true);
      }

      // pop stack
      if (lv == 0) return 0;
      --lv;
      n = stack[lv].n;
      c = stack[lv].c;
      rrc = stack[lv].choice;

      // recover
      uncover(n);
      n = n->down;
      //puts("leave");
    }

    // push stack
    removedRow[removedRowCount++] = n;
    StackFrame sf = {n, c, rrc};
    stack[lv] = sf;
    lv++;
    //puts("enter");
    cover(n);
  }
  return 1;
}

void DLXSolver::showSolution() {
  /*for (int i = 0; i < lv; i++) {
    printf("%d ", stack[i].n->row);
  }
  puts("");*/
  solCount++;
}

void DLXSolver::setMaxLv(int n) {
  maxLv = n;
  stack.resize(n+1);
}

void DLXSolver::setRowCount(int n) {
  removedRow.resize(n);
}

void DLXSolver::setRoot(MatrixColumn *node) {
  root = node;
  root->min = root->max = 0;
}
