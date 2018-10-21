#pragma once
#ifndef DLX_INCLUDED
#define DLX_INCLUDED

struct MatrixColumn;

struct MatrixCell {
  MatrixCell *up, *down;
  union { MatrixCell *left; MatrixColumn *leftCol; };
  union { MatrixCell *right; MatrixColumn *rightCol; };
  union { MatrixCell *row; int rowid; };
  union { MatrixColumn *col; int colid; };
};

struct MatrixColumn {
  struct MatrixCell body; // must be at first
  int size;
  int value, min, max;
};

#endif
