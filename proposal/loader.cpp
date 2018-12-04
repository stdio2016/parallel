#include "loader.h"
#include <sstream>

void DLXLoader::readFile(std::istream &in) {
  std::string line;
  int nodeN = 0, lineno = 1;
  std::getline(in, line);
  if (line.compare("dlx") != 0) return;
  while (std::getline(in, line)) {
    lineno++;
    std::stringstream tok(line);
    std::string name;
    if (!(tok >> name)) continue; // empty line
    if (name.size() > 0 && name[0] == '%') continue; // comment
    if (name.compare("col") == 0) {
      DLXColumn col;
      tok >> col.name;
      int id = columns.size();
      columnLookup[col.name] = id;
      if (tok >> name && name.compare("min") == 0) {
        tok >> col.min;
      }
      if (tok >> name && name.compare("max") == 0) {
        tok >> col.max;
      }
      columns.push_back(col);
    }
    else if (name.compare("row") == 0) {
      DLXRow row;
      tok >> row.name;
      while (tok >> name) {
        std::map<std::string, int>::iterator it = columnLookup.find(name);
        if (it != columnLookup.end()) {
          row.cells.push_back(it->second);
          nodeN++;
        }
        else {
          std::cout << "Warning at line " << lineno << ": "
          << "column \"" << name << "\" is not defined\n";
        }
      }
      rows.push_back(row);
      nodeN++;
    }
  }
  
  int colN = columns.size();
  root = new MatrixColumn[colN + 1];
  for (int i = 0; i < colN; i++) {
    root[i+1].setRight(root);
    root[i+1].col = &root[i+1];
    root[i+1].row = i;
    root[i+1].min = columns[i].min;
    root[i+1].max = columns[i].max;
  }
  root->col = root;
  root->row = -1;
  root->min = 0;
  root->max = 0;
  
  cells = new MatrixCell[nodeN];
  int id = 0;
  for (int i = 0; i < rows.size(); i++) {
    cells[id].col = root;
    cells[id].row = i;
    cells[id].setDown(root);
    root->size += 1;
    id++;
    for (int j = 0; j < rows[i].cells.size(); j++) {
      cells[id+j].col = &root[1 + rows[i].cells[j]];
      cells[id+j].row = i;
      cells[id+j].setRight(&cells[id-1]);
      cells[id+j].setDown(cells[id+j].col);
      cells[id+j].col->size += 1;
    }
    id += rows[i].cells.size();
  }
  cellCount = nodeN;
}
