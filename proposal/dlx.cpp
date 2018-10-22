#include <cstdio>
#include "dlx.h"

void unlinkRow(MatrixCell *n, bool includeN) {
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

void relinkRow(MatrixCell *n, bool includeN) {
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

void cover(MatrixCell *n) {
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

void uncover(MatrixCell *n) {
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

int minfit(MatrixColumn *root, MatrixColumn **result) {
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
  if (rows.size() != removedRow.size()) removedRow.resize(rows.size());
  while (tried != maxTry) {
    tried++;
    MatrixColumn *c;
    MatrixCell *n;
    int has = minfit(root, &c);
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
    stack[lv++] = {n, c, rrc};
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

void DLXSolver::addColumn(int id, int min, int max) {
  MatrixColumn *c = new MatrixColumn(id, min, max);
  cols.push_back(c);
  c->up = c->down = c;
  c->setRight(root);
}

void DLXSolver::addRow(int id) {
  MatrixCell *r = new MatrixCell(id, root);
  rows.push_back(r);
  r->setDown(root);
  r->left = r->right = r;
}

void DLXSolver::addCell(int row, int col) {
  MatrixCell *n = new MatrixCell(rows[row]->row, cols[col]);
  cells.push_back(n);
  n->setDown(cols[col]);
  n->setRight(rows[row]);
  cols[col]->size += 1;
}

void DLXSolver::setMaxLv(int n) {
  maxLv = n;
  stack.resize(n+1);
}
