#include <cstdio>
#include "dlx.h"

void unlinkRow(MatrixCell *n, bool includeN) {
  MatrixCell *e = n;
  if (!includeN) n = n->right;

  while (n != e) {
    n->up->down = n->down;
    n->down->up = n->up;
    n->col->size -= 1;
    n = n->right;
  }
}

void relinkRow(MatrixCell *n, bool includeN) {
  MatrixCell *e = n;
  if (!includeN) n = n->right;

  while (n != e) {
    n->up->down = n;
    n->down->up = n;
    n->col->size += 1;
    n = n->right;
  }
}

void cover(MatrixCell *n) {
  MatrixCell *n2, *n3;
  MatrixColumn *c2;
  n2 = n;
  do {
    c2 = n2->col;
    c2->value += 1;
    if (c2->value == c2->max) {
      c2->body.left->right = c2->body.right;
      c2->body.right->left = c2->body.left;
      n3 = c2->body.down;
      while (n3 != &c2->body) {
        unlinkRow(n3, false);
        n3 = n3->down;
      }
    }
    n2 = n2->right;
  } while (n2 != n) ;
}

void uncover(MatrixCell *n) {
  MatrixCell *n2, *n3;
  MatrixColumn *c2;
  n2 = n;
  do {
    n2 = n2->left;
    c2 = n2->col;
    if (c2->value == c2->max) {
      n3 = c2->body.up;
      while (n3 != &c2->body) {
        relinkRow(n3, false);
        n3 = n3->up;
      }
      c2->body.left->right = &c2->body;
      c2->body.right->left = &c2->body;
    }
    c2->value -= 1;
  } while (n2 != n) ;
}

int minfit(MatrixColumn *root, MatrixColumn **result) {
  unsigned minfit = -1; // -1 converted to UINT_MAX!
  MatrixColumn *c = NULL;
  for (MatrixColumn *col = root->body.rightCol; col != root; col = col->body.rightCol) {
    int value = col->value;
    // satisfied
    if (value <= col->max && value >= col->min) continue;
    // unsatisfiable
    if (value > col->max || value + col->size < col->min) {
      return -2;
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

int main(void)
{
  return 0;
}
