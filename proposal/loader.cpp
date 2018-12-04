#include "loader.h"
#include <sstream>

void DLXLoader::readFile(std::istream &in) {
  std::string line;
  std::getline(in, line);
  if (line.compare("dlx") != 0) return;
  while (std::getline(in, line)) {
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
      std::cout << "col " << col.name << " min " << col.min << " max " << col.max << "\n";
      columns.push_back(col);
    }
    else if (name.compare("row") == 0) {
      DLXRow row;
      tok >> row.name;
      std::cout << "row " << row.name;
      while (tok >> name) {
        std::cout << ' ' << name;
        std::map<std::string, int>::iterator it = columnLookup.find(name);
        if (it != columnLookup.end()) {
          row.cells.push_back(it->second);
        }
      }
      std::cout << "\n";
      rows.push_back(row);
    }
  }
}
