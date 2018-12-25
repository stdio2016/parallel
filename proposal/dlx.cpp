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
  if (ended) return 0;
  while (tried != maxTry && solCount != maxSol) {
    tried++;
    MatrixColumn *c;
    MatrixCell *n;
    int has = minfit(&c);
    int rrc = removedRowCount;
    if (lv >= maxLv || has == -2) {
      showSolution();
      has = -1;
    }
    if (has == -1) {
      if (lv <= minLv) { ended = true; return solCount == maxSol; }
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
      if (lv <= minLv) { ended = true; return solCount == maxSol; }
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
  if (solCount >= solutions.size()-1) {
    solutions.push_back(solutions[solCount] + lv);
  }
  else {
    solutions[solCount+1] = solutions[solCount] + lv;
  }
  int from = solutions[solCount];
  if (solRows.size() < from + lv) {
    solRows.resize(from+lv);
  }
  for (int i = 0; i < lv; i++) {
    solRows[from+i] = stack[i].n->row;
  }
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

void DLXSolver::enterBranch(int row) {
  MatrixColumn *c;
  MatrixCell *n;
  int has = minfit(&c);
  int rrc = removedRowCount;
  n = c->down;
  while (n->row != row) {
    removedRow[removedRowCount++] = n;
    unlinkRow(n, true);
    n = n->down;
  }
  removedRow[removedRowCount++] = n;
  StackFrame sf = {n, c, rrc};
  stack[lv] = sf;
  lv++;
  //puts("enter");
  cover(n);
}

void DLXSolver::leaveBranch() {
  --lv;
  MatrixCell *n = stack[lv].n;
  int rrc = stack[lv].choice;

  // recover
  uncover(n);

  while (removedRowCount > rrc) {
    relinkRow(removedRow[--removedRowCount], true);
  }
}
