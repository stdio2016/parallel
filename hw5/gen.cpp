#include <cstdlib>
#include <fstream>
int main(int argc, char *argv[]) {
	if (argc<2) return 1;
	std::ofstream f("input");
	int n = atoi(argv[1]);
	f << n<< '\n';
	unsigned seed = argc>2 ? atoi(argv[2]) : 0;
	for (int i=0;i<n;i++) {
		int r=rand_r(&seed)&255;
		int g=rand_r(&seed)&255;
		int b=rand_r(&seed)&255;
		f<<r<<' '<<g<<' '<<b<<'\n';
	}
	f.close();
	return 0;
}
