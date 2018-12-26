#include <cstdio>
#include <set>
#include <algorithm>

typedef std::pair<int,int> coord;

char name[12] = {
  'I', 'L', 'N', 'Y', 'P', 'U',
  'V', 'W', 'Z', 'T', 'F', 'X'
};

struct SS {
  std::pair<int, int> p[5];  
  void rotate() {
    int maxx = 0;
    for (int i = 0; i < 5; i++)
      maxx = std::max(maxx, p[i].first);
    for (int i = 0; i < 5; i++) {
      p[i] = coord{p[i].second, maxx-p[i].first};
    }
    std::sort(p, p+5);
  }
  
  void mirror() {
    int maxx = 0;
    for (int i = 0; i < 5; i++)
      maxx = std::max(maxx, p[i].first);
    for (int i = 0; i < 5; i++) {
      p[i] = coord{maxx-p[i].first, p[i].second};
    }
    std::sort(p, p+5);
  }
};
bool operator<(SS a,SS b) {
  for (int i = 0; i < 5; i++) {
    if (a.p[i] < b.p[i]) return true;
    if (a.p[i] > b.p[i]) return false;
  }
  return false;
}

coord shape[12][5] = {
  {{0,0},{1,0},{2,0},{3,0},{4,0}},
  {{0,0},{1,0},{2,0},{3,0},{3,1}},
  {{0,0},{1,0},{2,0},{2,1},{3,1}},
  {{0,0},{1,0},{2,0},{2,1},{3,0}},
  {{0,0},{1,0},{1,1},{2,0},{2,1}},
  {{0,0},{0,1},{1,0},{2,0},{2,1}},
  
  {{0,0},{1,0},{2,0},{2,1},{2,2}},
  {{0,0},{1,0},{1,1},{2,1},{2,2}},
  {{0,0},{1,0},{1,1},{1,2},{2,2}},
  {{0,0},{1,0},{1,1},{1,2},{2,0}},
  {{0,1},{1,0},{1,1},{1,2},{2,0}},
  {{0,1},{1,0},{1,1},{1,2},{2,1}}
};

int main() {
  int height = 6, width = 10;
  FILE *f = fopen("pento.txt", "wb");
  fprintf(f, "dlx\n");
  for (int i = 1; i <= height; i++) {
    for (int j = 1; j <= width; j++) {
      fprintf(f, "col r%dc%d\n", i, j);
    }
  }
  for (int i = 0; i < 12; i++) {
    fprintf(f, "col %c\n", name[i]);
  }
  
  for (int i = 0; i < 12; i++) {
    std::set<SS> rots;
    SS cur;
    for (int j = 0; j < 5; j++) cur.p[j] = shape[i][j];
    for (int r = 0; r < 4; r++) {
      rots.insert(cur);
      cur.rotate();
    }
    cur.mirror();
    for (int r = 0; r < 4; r++) {
      rots.insert(cur);
      cur.rotate();
    }
    
    int id = 1;
    for (SS r: rots) {
      int mx = 0, my = 0;
      for (int j = 0; j < 5; j++) {
        my = std::max(my, r.p[j].first);
        mx = std::max(mx, r.p[j].second);
      }
      for (int y = 1; y <= height - my; y++) {
        for (int x = 1; x <= width - mx; x++) {
          fprintf(f, "row %c%dr%dc%d %c", name[i], id, y, x, name[i]);
          for (int j = 0; j < 5; j++) {
            fprintf(f, " r%dc%d", y + r.p[j].first, x + r.p[j].second);
          }
          fputc('\n', f);
        }
      }
      id++;
    }
  }
}
