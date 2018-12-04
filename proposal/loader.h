#include <vector>
#include <string>
#include <map>
#include <iostream>
#include "dlx.h"

struct DLXLoader {
  struct DLXColumn {
    std::string name;
    int min, max;
    DLXColumn(): min(1), max(1) {}
  };
  struct DLXRow {
    std::string name;
    std::vector<int> cells;
  };
  std::vector<DLXColumn> columns;
  std::map<std::string, int> columnLookup;
  std::vector<DLXRow> rows;
  
  MatrixCell *cells;
  MatrixColumn *root;
  
  int cellCount;
  
  void readFile(std::istream &in);
};

/*
 * Format:
 *  dlx
 *  col colName min 1 max 1
 *  col colName min 0 max 1
 *  row rowName colName...
 *  row rowName colName...
 */
