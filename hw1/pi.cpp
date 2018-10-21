#include <iostream>
#include <cstdlib>
#include <climits>
#include <ctime>
#include <pthread.h>

struct PiThreadPack {
  pthread_t pid;
  unsigned seed;
  long long toss;
  long long inCircle;
};
PiThreadPack *pack;

void *piRunner(void *param) {
  // get param
  PiThreadPack *arg = (PiThreadPack *) param;
  long long toss = arg->toss;
  long long number_in_circle = 0;

  // random number generator
  unsigned seed = arg->seed;

  // Monte Carlo loop
  while (toss-->0) {
    double x = double(rand_r(&seed)) / double(RAND_MAX); // 0 to 1 inclusive
    double y = double(rand_r(&seed)) / double(RAND_MAX); // 0 to 1 inclusive
    x = (x - 0.5) * 2.0; // -1 to 1
    y = (y - 0.5) * 2.0; // -1 to 1
    double distance_squared = x*x + y*y;
    if (distance_squared <= 1)
      number_in_circle++;
  }

  // output result
  arg->inCircle = number_in_circle;
  return param;
}

int main(int argc, char *argv[])
{
  // parse arguments, not parallelizable
  int threadCount = 1;
  if (argc > 1) threadCount = atoi(argv[1]);
  long long int toss = 1;
  if (argc > 2) toss = atoll(argv[2]);

  // randomize
  srand(time(NULL));

  // create thread
  long long total = toss;
  pack = new PiThreadPack[threadCount];
  for (int i = 0; i < threadCount; i++) {
    // distribute workload
    long long dist = total / (threadCount - i);
    pack[i].toss = dist;
    total -= dist;

    // set random seed
    pack[i].seed = rand();
    pack[i].inCircle = 0;
    pthread_create(&pack[i].pid, NULL, piRunner, &pack[i]);
  }

  // join thread
  long long inCircle = 0;
  for (int i = 0; i < threadCount; i++) {
    pthread_join(pack[i].pid, NULL);
    inCircle += pack[i].inCircle;
  }
  delete[] pack;

  double pi = double(inCircle)/toss * 4;
  std::cout.precision(10);
  std::cout << pi << '\n';
  return 0;
}
