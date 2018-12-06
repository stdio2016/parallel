#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAXPOINTS 1000000
#define MAXSTEPS 1000000
#define MINPOINTS 20
#define PI 3.14159265

#define THREAD_COUNT 1024

void check_param(void);
void init_line(void);
void update (void);
void printfinal (void);

int nsteps,                 	/* number of time steps */
    tpoints, 	     		/* total points along string */
    rcode;                  	/* generic return code */
float  values[MAXPOINTS+2], 	/* values at time t */
       oldval[MAXPOINTS+2], 	/* values at time (t-dt) */
       newval[MAXPOINTS+2]; 	/* values at time (t+dt) */


/**********************************************************************
 *	Checks input values from parameters
 *********************************************************************/
void check_param(void)
{
   char tchar[20];

   /* check number of points, number of iterations */
   while ((tpoints < MINPOINTS) || (tpoints > MAXPOINTS)) {
      printf("Enter number of points along vibrating string [%d-%d]: "
           ,MINPOINTS, MAXPOINTS);
      scanf("%s", tchar);
      tpoints = atoi(tchar);
      if ((tpoints < MINPOINTS) || (tpoints > MAXPOINTS))
         printf("Invalid. Please enter value between %d and %d\n", 
                 MINPOINTS, MAXPOINTS);
   }
   while ((nsteps < 1) || (nsteps > MAXSTEPS)) {
      printf("Enter number of time steps [1-%d]: ", MAXSTEPS);
      scanf("%s", tchar);
      nsteps = atoi(tchar);
      if ((nsteps < 1) || (nsteps > MAXSTEPS))
         printf("Invalid. Please enter value between 1 and %d\n", MAXSTEPS);
   }

   printf("Using points = %d, steps = %d\n", tpoints, nsteps);

}

/**********************************************************************
 *     Initialize points on line
 *********************************************************************/
void init_line(void)
{
   int i, j;
   float x, fac, k, tmp;

   /* Calculate initial values based on sine curve */
   fac = 2.0 * PI;
   k = 0.0; 
   tmp = tpoints - 1;
   for (j = 1; j <= tpoints; j++) {
      x = k/tmp;
      values[j] = sin (fac * x);
      k = k + 1.0;
   } 

   /* Initialize old values array */
   for (i = 1; i <= tpoints; i++) 
      oldval[i] = values[i];
}

/**********************************************************************
 *      Calculate new values using wave equation
 *********************************************************************/
void do_math(int i)
{
   float dtime, c, dx, tau, sqtau;

   dtime = 0.3;
   c = 1.0;
   dx = 1.0;
   tau = (c * dtime / dx);
   sqtau = tau * tau;
   newval[i] = (2.0 * values[i]) - oldval[i] + (sqtau *  (-2.0)*values[i]);
}

/**********************************************************************
 *     Update all values along line a specified number of times
 *********************************************************************/
void update()
{
   int i, j;

   /* Update values for each time step */
   for (i = 1; i<= nsteps; i++) {
      /* Update points along line for this time step */
      for (j = 1; j <= tpoints; j++) {
         /* global endpoints */
         if ((j == 1) || (j  == tpoints))
            newval[j] = 0.0;
         else
            do_math(j);
      }

      /* Update old values with new values */
      for (j = 1; j <= tpoints; j++) {
         oldval[j] = values[j];
         values[j] = newval[j];
      }
   }
}

/* update one value for each thread */
__global__ void updateKernel(float *gpuMem, int nsteps, int tpoints) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  float dtime, c, dx, tau, sqtau;

  dtime = 0.3;
  c = 1.0;
  dx = 1.0;
  tau = (c * dtime / dx);
  sqtau = tau * tau;
  
  /* thread might be unused */
  if (j >= tpoints-1) return ;
  float value = gpuMem[j];
  float oldval = value;
  
  /* update values for each time step */
  int i;
  for (i = 1; i <= nsteps; i++) {
    float newval;
    newval = (2.0f * value) - oldval + (sqtau *  (-2.0f)*value);
    oldval = value;
    value = newval;
  }
  
  gpuMem[j] = value;
}

/**********************************************************************
 *     Print final results
 *********************************************************************/
void printfinal()
{
   int i;

   for (i = 1; i <= tpoints; i++) {
      printf("%6.4f ", values[i]);
      if (i%10 == 0)
         printf("\n");
   }
}

void TellError(int err) {
  if (err == cudaErrorInvalidValue) puts("invalid value");
  if (err == cudaErrorInvalidDevicePointer) puts("invalid device pointer");
  if (err == cudaErrorInvalidMemcpyDirection) puts("invalid memcpy direction");
}

/**********************************************************************
 *	Main program
 *********************************************************************/
int main(int argc, char *argv[])
{
  if (argc < 3) return 1;
	sscanf(argv[1],"%d",&tpoints);
	sscanf(argv[2],"%d",&nsteps);
	check_param();
  
  /* calculate block size and allocate gpu memory */
  int blockCount = (tpoints + (THREAD_COUNT)) / (THREAD_COUNT);
  int size = blockCount * (THREAD_COUNT);
  int usedSize = (tpoints + 1);
  float *gpuMem;
  int e;
  e = cudaMalloc((void**)&gpuMem, size * sizeof(float));
  if (e != cudaSuccess) TellError(e);
  
	printf("Initializing points on the line...\n");
	init_line();
  e = cudaMemcpy(gpuMem, values+1, usedSize * sizeof(float), cudaMemcpyHostToDevice);
  if (e != cudaSuccess) TellError(e);
  
	printf("Updating all points for all time steps...\n");
	updateKernel<<<blockCount, THREAD_COUNT>>>(gpuMem, nsteps, tpoints);
  e = cudaMemcpy(values+1, gpuMem, usedSize * sizeof(float), cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) TellError(e);

	printf("Printing final results...\n");
	printfinal();
  cudaFree(gpuMem);
	printf("\nDone.\n\n");
	
	return 0;
}
