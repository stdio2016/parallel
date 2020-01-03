#include "wtime.h"
#include <time.h>
#ifndef DOS
#ifndef _WIN32
#include <sys/time.h>
#endif
#endif

#ifdef _WIN32
#include <windows.h>
void wtime(double *t)
{
  static long long sec = -1;
  LARGE_INTEGER freq, count;
  if (QueryPerformanceCounter(&count) && QueryPerformanceFrequency(&freq)) {
    if (sec < 0) sec = count.QuadPart / freq.QuadPart;
    *t = (double)count.QuadPart / (double)freq.QuadPart - sec;
  }
  else *t = 0.0; // something goes wrong!
}
#else
void wtime(double *t)
{
  static int sec = -1;
  struct timeval tv;
  gettimeofday(&tv, (void *)0);
  if (sec < 0) sec = tv.tv_sec;
  *t = (tv.tv_sec - sec) + 1.0e-6*tv.tv_usec;
}
#endif

    
