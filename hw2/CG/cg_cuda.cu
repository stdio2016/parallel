#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "globals.h"
#include "randdp.h"
#include "timers.h"

//---------------------------------------------------------------------
/* common / main_int_mem / */
static int colidx[NZ];
static int rowstr[NA+1];
static int iv[NA];
static int arow[NA];
static int acol[NAZ];

/* common / main_flt_mem / */
static double aelt[NAZ];
static double a[NZ];
static double x[NA+2];
static double z[NA+2];
static double p[NA+2];
static double q[NA+2];
static double r[NA+2];

/* gpu memory */
static int *gpu_colidx;
static int *gpu_rowstr;
static double *gpu_a;
static double *gpu_x;
static double *gpu_z;
static double *gpu_p;
static double *gpu_q;
static double *gpu_r;
static double *gpu_sum, cpu_sum[NA+2];

/* common / partit_size / */
static int naa;
static int nzz;
static int firstrow;
static int lastrow;
static int firstcol;
static int lastcol;

/* common /urando/ */
static double amult;
static double tran;

/* common /timers/ */
static logical timeron;
//---------------------------------------------------------------------


//---------------------------------------------------------------------
static void conj_grad(int colidx[],
                      int rowstr[],
                      double x[],
                      double z[],
                      double a[],
                      double p[],
                      double q[],
                      double r[],
                      double *rnorm);
static void makea(int n,
                  int nz,
                  double a[],
                  int colidx[],
                  int rowstr[],
                  int firstrow,
                  int lastrow,
                  int firstcol,
                  int lastcol,
                  int arow[],
                  int acol[][NONZER+1],
                  double aelt[][NONZER+1],
                  int iv[]);
static void sparse(double a[],
                   int colidx[],
                   int rowstr[],
                   int n,
                   int nz,
                   int nozer,
                   int arow[],
                   int acol[][NONZER+1],
                   double aelt[][NONZER+1],
                   int firstrow,
                   int lastrow,
                   int nzloc[],
                   double rcond,
                   double shift);
static void sprnvc(int n, int nz, int nn1, double v[], int iv[]);
static int icnvrt(double x, int ipwr2);
static void vecset(int n, double v[], int iv[], int *nzv, int i, double val);
//---------------------------------------------------------------------


int main(int argc, char *argv[])
{
  int i, j, k, it;

  double zeta;
  double rnorm;
  double norm_temp1, norm_temp2;

  double t, mflops, tmax;
  //char Class;
  logical verified;
  double zeta_verify_value, epsilon, err;

  char *t_names[T_last];

  for (i = 0; i < T_last; i++) {
    timer_clear(i);
  }
  
  timer_start(T_init);

  firstrow = 0;
  lastrow  = NA-1;
  firstcol = 0;
  lastcol  = NA-1;

  zeta_verify_value = VALID_RESULT;
  
  printf("\nCG start...\n\n");
  printf(" Size: %11d\n", NA);
  printf(" Iterations: %5d\n", NITER);
  printf("\n");

  naa = NA;
  nzz = NZ;

  //---------------------------------------------------------------------
  // Inialize random number generator
  //---------------------------------------------------------------------
  tran    = 314159265.0;
  amult   = 1220703125.0;
  zeta    = randlc(&tran, amult);

  //---------------------------------------------------------------------
  //  
  //---------------------------------------------------------------------
  makea(naa, nzz, a, colidx, rowstr, 
        firstrow, lastrow, firstcol, lastcol, 
        arow, 
        (int (*)[NONZER+1])(void*)acol, 
        (double (*)[NONZER+1])(void*)aelt,
        iv);

  //---------------------------------------------------------------------
  // Note: as a result of the above call to makea:
  //      values of j used in indexing rowstr go from 0 --> lastrow-firstrow
  //      values of colidx which are col indexes go from firstcol --> lastcol
  //      So:
  //      Shift the col index vals from actual (firstcol --> lastcol ) 
  //      to local, i.e., (0 --> lastcol-firstcol)
  //---------------------------------------------------------------------
  for (j = 0; j < lastrow - firstrow + 1; j++) {
    for (k = rowstr[j]; k < rowstr[j+1]; k++) {
      colidx[k] = colidx[k] - firstcol;
    }
  }

  // move data to gpu
  cudaMalloc(&gpu_colidx, sizeof(colidx));
  cudaMalloc(&gpu_rowstr, sizeof(rowstr));
  cudaMalloc(&gpu_a, sizeof(a));
  cudaMalloc(&gpu_x, sizeof(x));
  cudaMalloc(&gpu_z, sizeof(z));
  cudaMalloc(&gpu_p, sizeof(p));
  cudaMalloc(&gpu_q, sizeof(q));
  cudaMalloc(&gpu_r, sizeof(r));
  cudaMalloc(&gpu_sum, sizeof(cpu_sum));
  cudaMemcpy(gpu_colidx, colidx, sizeof(colidx), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_rowstr, rowstr, sizeof(rowstr), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_a, a, sizeof(a), cudaMemcpyHostToDevice);

  //---------------------------------------------------------------------
  // set starting vector to (1, 1, .... 1)
  //---------------------------------------------------------------------
  for (i = 0; i < NA+1; i++) {
    x[i] = 1.0;
  }
  for (j = 0; j < lastcol - firstcol + 1; j++) {
    q[j] = 0.0;
    z[j] = 0.0;
    r[j] = 0.0;
    p[j] = 0.0;
  }

  zeta = 0.0;

  //---------------------------------------------------------------------
  //---->
  // Do one iteration untimed to init all code and data page tables
  //---->                    (then reinit, start timing, to niter its)
  //---------------------------------------------------------------------
  for (it = 1; it <= 1; it++) {
    //---------------------------------------------------------------------
    // The call to the conjugate gradient routine:
    //---------------------------------------------------------------------
    conj_grad(colidx, rowstr, x, z, a, p, q, r, &rnorm);

    //---------------------------------------------------------------------
    // zeta = shift + 1/(x.z)
    // So, first: (x.z)
    // Also, find norm of z
    // So, first: (z.z)
    //---------------------------------------------------------------------
    norm_temp1 = 0.0;
    norm_temp2 = 0.0;
    for (j = 0; j < lastcol - firstcol + 1; j++) {
      norm_temp1 = norm_temp1 + x[j] * z[j];
      norm_temp2 = norm_temp2 + z[j] * z[j];
    }

    norm_temp2 = 1.0 / sqrt(norm_temp2);

    //---------------------------------------------------------------------
    // Normalize z to obtain x
    //---------------------------------------------------------------------
    for (j = 0; j < lastcol - firstcol + 1; j++) {     
      x[j] = norm_temp2 * z[j];
    }
  } // end of do one iteration untimed


  //---------------------------------------------------------------------
  // set starting vector to (1, 1, .... 1)
  //---------------------------------------------------------------------
  for (i = 0; i < NA+1; i++) {
    x[i] = 1.0;
  }

  zeta = 0.0;

  timer_stop(T_init);

  printf(" Initialization time = %15.3f seconds\n", timer_read(T_init));

  timer_start(T_bench);

  //---------------------------------------------------------------------
  //---->
  // Main Iteration for inverse power method
  //---->
  //---------------------------------------------------------------------
  for (it = 1; it <= NITER; it++) {
    //---------------------------------------------------------------------
    // The call to the conjugate gradient routine:
    //---------------------------------------------------------------------
    if (timeron) timer_start(T_conj_grad);
    conj_grad(colidx, rowstr, x, z, a, p, q, r, &rnorm);
    if (timeron) timer_stop(T_conj_grad);

    //---------------------------------------------------------------------
    // zeta = shift + 1/(x.z)
    // So, first: (x.z)
    // Also, find norm of z
    // So, first: (z.z)
    //---------------------------------------------------------------------
    norm_temp1 = 0.0;
    norm_temp2 = 0.0;
    for (j = 0; j < lastcol - firstcol + 1; j++) {
      norm_temp1 = norm_temp1 + x[j]*z[j];
      norm_temp2 = norm_temp2 + z[j]*z[j];
    }

    norm_temp2 = 1.0 / sqrt(norm_temp2);

    zeta = SHIFT + 1.0 / norm_temp1;
    if (it == 1) 
      printf("\n   iteration           ||r||                 zeta\n");
    printf("    %5d       %20.14E%20.13f\n", it, rnorm, zeta);

    //---------------------------------------------------------------------
    // Normalize z to obtain x
    //---------------------------------------------------------------------
    for (j = 0; j < lastcol - firstcol + 1; j++) {
      x[j] = norm_temp2 * z[j];
    }
  } // end of main iter inv pow meth

  timer_stop(T_bench);

  //---------------------------------------------------------------------
  // End of timed section
  //---------------------------------------------------------------------

  t = timer_read(T_bench);

  printf("\nComplete...\n");

  epsilon = 1.0e-10;
  err = fabs(zeta - zeta_verify_value) / zeta_verify_value;
  if (err <= epsilon) {
    verified = true;
    printf(" VERIFICATION SUCCESSFUL\n");
    printf(" Zeta is    %20.13E\n", zeta);
    printf(" Error is   %20.13E\n", err);
  } else {
    verified = false;
    printf(" VERIFICATION FAILED\n");
    printf(" Zeta                %20.13E\n", zeta);
    printf(" The correct zeta is %20.13E\n", zeta_verify_value);
  }
  
  printf("\n\nExecution time : %lf seconds\n\n", t);
  
  return 0;
}

// thanks to Nvidia paper
// N. Bell, M. Garland, "Efficient sparse matrix-vector multiplication on cuda",
// Nvidia Technical Report NVR-2008–004 Tech. Rep., 2008
__global__ void mat_vec_mul(int n,
    const int * __restrict__ colidx,
    const int * __restrict__ rowstr,
    const double * __restrict__ a,
    const double * __restrict__ pp,
    double * __restrict__ q) {
  int gid, tid, j, k, k_end, lane;
  __shared__ volatile double sum[256];
  gid = threadIdx.x + blockIdx.x * blockDim.x;
  j = gid >> 5;
  lane = gid & 31;

  tid = threadIdx.x;
  sum[tid] = 0.0;
  if (j >= n) return;

  k_end = rowstr[j+1];
  for (k = rowstr[j] + lane; k < k_end; k += 32) {
    sum[tid] = sum[tid] + a[k]*pp[colidx[k]];
  }
  // in the same warp, no sync needed
  if (lane < 16) sum[tid] += sum[tid + 16];
  if (lane < 8) sum[tid] += sum[tid + 8];
  if (lane < 4) sum[tid] += sum[tid + 4];
  if (lane < 2) sum[tid] += sum[tid + 2];
  if (lane < 1) sum[tid] += sum[tid + 1];
  if (lane == 0) q[j] = sum[tid];
}

__device__ void reduce_block_sum(double sum, double *result) {
  __shared__ double s_sum[1024];
  int i, step;
  i = threadIdx.x;
  s_sum[i] = sum;
  __syncthreads();
  for (step = blockDim.x>>1; step >= 1; step >>= 1) {
    if (i < step) s_sum[i] += s_sum[i+step];
    __syncthreads();
  }
  if (i == 0) result[blockIdx.x] = s_sum[0];
}

__global__ void dot_prod(int n, double *a, double *b, double *result) {
  int i, step;
  double sum = 0.0;
  i = threadIdx.x + blockIdx.x * blockDim.x;
  step = blockDim.x * gridDim.x;
  while (i < n) {
    sum += a[i] * b[i];
    i += step;
  }
  reduce_block_sum(sum, result);
}

__global__ void dist_vec(int n, double *a, double *b, double *result) {
  int i, step;
  double sum = 0.0;
  i = threadIdx.x + blockIdx.x * blockDim.x;
  step = blockDim.x * gridDim.x;
  while (i < n) {
    double d = a[i] - b[i];
    sum += d * d;
    i += step;
  }
  reduce_block_sum(sum, result);
}

__global__ void vec_scale_add(int n, double *a, double scale, double *b, double *result) {
  int i;
  i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;
  result[i] = a[i] * scale + b[i];
}

//---------------------------------------------------------------------
// Floaging point arrays here are named as in spec discussion of 
// CG algorithm
//---------------------------------------------------------------------
static void conj_grad(int colidx[], // const gpu
                      int rowstr[], // const gpu
                      double x[], // in cpu
                      double z[], // out to cpu
                      double a[], // const gpu
                      double p[], // gpu
                      double q[], // gpu
                      double r[], // gpu
                      double *rnorm)
{
  int j, k;
  int cgit, cgitmax = 25;
  int rowcnt, colcnt;
  int rowblk, colblk;
  double d, sum, rho, rho0, alpha, beta;

  rho = 0.0;
  rowcnt = lastrow - firstrow + 1;
  colcnt = lastcol - firstcol + 1;
  rowblk = (rowcnt + 7) / 8;
  colblk = (colcnt + 255) / 256;

  //---------------------------------------------------------------------
  // Initialize the CG algorithm:
  //---------------------------------------------------------------------
  /*for (j = 0; j < naa+1; j++) {
    q[j] = 0.0;
    z[j] = 0.0;
    r[j] = x[j];
    p[j] = r[j];
  }*/
  cudaMemcpy(gpu_x, x, sizeof(double) * (naa+1), cudaMemcpyHostToDevice);
  cudaMemset(gpu_q, 0, sizeof(double) * (naa+1));
  cudaMemset(gpu_z, 0, sizeof(double) * (naa+1));
  cudaMemcpy(gpu_r, x, sizeof(double) * (naa+1), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_p, x, sizeof(double) * (naa+1), cudaMemcpyHostToDevice);

  //---------------------------------------------------------------------
  // rho = r.r
  // Now, obtain the norm of r: First, sum squares of r elements locally...
  //---------------------------------------------------------------------
  /*for (j = 0; j < lastcol - firstcol + 1; j++) {
    rho = rho + r[j]*r[j];
  }*/
  dot_prod<<<colblk/4+1, 256>>>(colcnt, gpu_r, gpu_r, gpu_sum);
  cudaMemcpy(cpu_sum, gpu_sum, sizeof(double) * colblk, cudaMemcpyDeviceToHost);
  for (j = 0; j < colblk/4+1; j++) {
    rho = rho + cpu_sum[j];
  }

  //---------------------------------------------------------------------
  //---->
  // The conj grad iteration loop
  //---->
  //---------------------------------------------------------------------
  for (cgit = 1; cgit <= cgitmax; cgit++) {
    //---------------------------------------------------------------------
    // q = A.p
    // The partition submatrix-vector multiply: use workspace w
    //---------------------------------------------------------------------
    //
    // NOTE: this version of the multiply is actually (slightly: maybe %5) 
    //       faster on the sp2 on 16 nodes than is the unrolled-by-2 version 
    //       below.   On the Cray t3d, the reverse is true, i.e., the 
    //       unrolled-by-two version is some 10% faster.  
    //       The unrolled-by-8 version below is significantly faster
    //       on the Cray t3d - overall speed of code is 1.5 times faster.

    /*for (j = 0; j < lastrow - firstrow + 1; j++) {
      sum = 0.0;
      for (k = rowstr[j]; k < rowstr[j+1]; k++) {
        sum = sum + a[k]*p[colidx[k]];
      }
      q[j] = sum;
    }*/
    mat_vec_mul<<<rowblk, 256>>>(rowcnt, gpu_colidx, gpu_rowstr, gpu_a, gpu_p, gpu_q);

    //---------------------------------------------------------------------
    // Obtain p.q
    //---------------------------------------------------------------------
    d = 0.0;
    /*for (j = 0; j < lastcol - firstcol + 1; j++) {
      d = d + p[j]*q[j];
    }*/
    dot_prod<<<colblk/4+1, 256>>>(colcnt, gpu_p, gpu_q, gpu_sum);
    cudaMemcpy(cpu_sum, gpu_sum, sizeof(double) * colblk, cudaMemcpyDeviceToHost);
    for (j = 0; j < colblk/4+1; j++) {
      d = d + cpu_sum[j];
    }

    //---------------------------------------------------------------------
    // Obtain alpha = rho / (p.q)
    //---------------------------------------------------------------------
    alpha = rho / d;

    //---------------------------------------------------------------------
    // Save a temporary of rho
    //---------------------------------------------------------------------
    rho0 = rho;

    //---------------------------------------------------------------------
    // Obtain z = z + alpha*p
    // and    r = r - alpha*q
    //---------------------------------------------------------------------
    rho = 0.0;
    /*for (j = 0; j < lastcol - firstcol + 1; j++) {
      z[j] = z[j] + alpha*p[j];  
      r[j] = r[j] - alpha*q[j];
    }*/
    vec_scale_add<<<colblk, 256>>>(colcnt, gpu_p, alpha, gpu_z, gpu_z);
    vec_scale_add<<<colblk, 256>>>(colcnt, gpu_q, -alpha, gpu_r, gpu_r);
            
    //---------------------------------------------------------------------
    // rho = r.r
    // Now, obtain the norm of r: First, sum squares of r elements locally...
    //---------------------------------------------------------------------
    /*for (j = 0; j < lastcol - firstcol + 1; j++) {
      rho = rho + r[j]*r[j];
    }*/
    dot_prod<<<colblk/4+1, 256>>>(colcnt, gpu_r, gpu_r, gpu_sum);
    cudaMemcpy(cpu_sum, gpu_sum, sizeof(double) * colblk, cudaMemcpyDeviceToHost);
    for (j = 0; j < colblk/4+1; j++) {
      rho = rho + cpu_sum[j];
    }

    //---------------------------------------------------------------------
    // Obtain beta:
    //---------------------------------------------------------------------
    beta = rho / rho0;

    //---------------------------------------------------------------------
    // p = r + beta*p
    //---------------------------------------------------------------------
    /*for (j = 0; j < lastcol - firstcol + 1; j++) {
      p[j] = r[j] + beta*p[j];
    }*/
    vec_scale_add<<<colblk, 256>>>(colcnt, gpu_p, beta, gpu_r, gpu_p);
  } // end of do cgit=1,cgitmax

  //---------------------------------------------------------------------
  // Compute residual norm explicitly:  ||r|| = ||x - A.z||
  // First, form A.z
  // The partition submatrix-vector multiply
  //---------------------------------------------------------------------
  sum = 0.0;
  /*for (j = 0; j < lastrow - firstrow + 1; j++) {
    d = 0.0;
    for (k = rowstr[j]; k < rowstr[j+1]; k++) {
      d = d + a[k]*z[colidx[k]];
    }
    r[j] = d;
  }*/
  mat_vec_mul<<<rowblk, 256>>>(rowcnt, gpu_colidx, gpu_rowstr, gpu_a, gpu_z, gpu_r);

  //---------------------------------------------------------------------
  // At this point, r contains A.z
  //---------------------------------------------------------------------
  /*for (j = 0; j < lastcol-firstcol+1; j++) {
    d   = x[j] - r[j];
    sum = sum + d*d;
  }*/
  dist_vec<<<colblk/4+1, 256>>>(colcnt, gpu_x, gpu_r, gpu_sum);
  cudaMemcpy(cpu_sum, gpu_sum, sizeof(double) * colblk, cudaMemcpyDeviceToHost);
  for (j = 0; j < colblk/4+1; j++) {
    sum = sum + cpu_sum[j];
  }

  *rnorm = sqrt(sum);

  cudaMemcpy(z, gpu_z, sizeof(double) * colcnt, cudaMemcpyDeviceToHost);
}


//---------------------------------------------------------------------
// generate the test problem for benchmark 6
// makea generates a sparse matrix with a
// prescribed sparsity distribution
//
// parameter    type        usage
//
// input
//
// n            i           number of cols/rows of matrix
// nz           i           nonzeros as declared array size
// rcond        r*8         condition number
// shift        r*8         main diagonal shift
//
// output
//
// a            r*8         array for nonzeros
// colidx       i           col indices
// rowstr       i           row pointers
//
// workspace
//
// iv, arow, acol i
// aelt           r*8
//---------------------------------------------------------------------
static void makea(int n,
                  int nz,
                  double a[],
                  int colidx[],
                  int rowstr[],
                  int firstrow,
                  int lastrow,
                  int firstcol,
                  int lastcol,
                  int arow[],
                  int acol[][NONZER+1],
                  double aelt[][NONZER+1],
                  int iv[])
{
  int iouter, ivelt, nzv, nn1;
  int ivc[NONZER+1];
  double vc[NONZER+1];

  //---------------------------------------------------------------------
  // nonzer is approximately  (int(sqrt(nnza /n)));
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // nn1 is the smallest power of two not less than n
  //---------------------------------------------------------------------
  nn1 = 1;
  do {
    nn1 = 2 * nn1;
  } while (nn1 < n);

  //---------------------------------------------------------------------
  // Generate nonzero positions and save for the use in sparse.
  //---------------------------------------------------------------------
  for (iouter = 0; iouter < n; iouter++) {
    nzv = NONZER;
    sprnvc(n, nzv, nn1, vc, ivc);
    vecset(n, vc, ivc, &nzv, iouter+1, 0.5);
    arow[iouter] = nzv;
    
    for (ivelt = 0; ivelt < nzv; ivelt++) {
      acol[iouter][ivelt] = ivc[ivelt] - 1;
      aelt[iouter][ivelt] = vc[ivelt];
    }
  }

  //---------------------------------------------------------------------
  // ... make the sparse matrix from list of elements with duplicates
  //     (iv is used as  workspace)
  //---------------------------------------------------------------------
  sparse(a, colidx, rowstr, n, nz, NONZER, arow, acol, 
         aelt, firstrow, lastrow,
         iv, RCOND, SHIFT);
}


//---------------------------------------------------------------------
// rows range from firstrow to lastrow
// the rowstr pointers are defined for nrows = lastrow-firstrow+1 values
//---------------------------------------------------------------------
static void sparse(double a[],
                   int colidx[],
                   int rowstr[],
                   int n,
                   int nz,
                   int nozer,
                   int arow[],
                   int acol[][NONZER+1],
                   double aelt[][NONZER+1],
                   int firstrow,
                   int lastrow,
                   int nzloc[],
                   double rcond,
                   double shift)
{
  int nrows;

  //---------------------------------------------------
  // generate a sparse matrix from a list of
  // [col, row, element] tri
  //---------------------------------------------------
  int i, j, j1, j2, nza, k, kk, nzrow, jcol;
  double size, scale, ratio, va;
  logical cont40;

  //---------------------------------------------------------------------
  // how many rows of result
  //---------------------------------------------------------------------
  nrows = lastrow - firstrow + 1;

  //---------------------------------------------------------------------
  // ...count the number of triples in each row
  //---------------------------------------------------------------------
  for (j = 0; j < nrows+1; j++) {
    rowstr[j] = 0;
  }

  for (i = 0; i < n; i++) {
    for (nza = 0; nza < arow[i]; nza++) {
      j = acol[i][nza] + 1;
      rowstr[j] = rowstr[j] + arow[i];
    }
  }

  rowstr[0] = 0;
  for (j = 1; j < nrows+1; j++) {
    rowstr[j] = rowstr[j] + rowstr[j-1];
  }
  nza = rowstr[nrows] - 1;

  //---------------------------------------------------------------------
  // ... rowstr(j) now is the location of the first nonzero
  //     of row j of a
  //---------------------------------------------------------------------
  if (nza > nz) {
    printf("Space for matrix elements exceeded in sparse\n");
    printf("nza, nzmax = %d, %d\n", nza, nz);
    exit(EXIT_FAILURE);
  }

  //---------------------------------------------------------------------
  // ... preload data pages
  //---------------------------------------------------------------------
  for (j = 0; j < nrows; j++) {
    for (k = rowstr[j]; k < rowstr[j+1]; k++) {
      a[k] = 0.0;
      colidx[k] = -1;
    }
    nzloc[j] = 0;
  }

  //---------------------------------------------------------------------
  // ... generate actual values by summing duplicates
  //---------------------------------------------------------------------
  size = 1.0;
  ratio = pow(rcond, (1.0 / (double)(n)));

  for (i = 0; i < n; i++) {
    for (nza = 0; nza < arow[i]; nza++) {
      j = acol[i][nza];

      scale = size * aelt[i][nza];
      for (nzrow = 0; nzrow < arow[i]; nzrow++) {
        jcol = acol[i][nzrow];
        va = aelt[i][nzrow] * scale;

        //--------------------------------------------------------------------
        // ... add the identity * rcond to the generated matrix to bound
        //     the smallest eigenvalue from below by rcond
        //--------------------------------------------------------------------
        if (jcol == j && j == i) {
          va = va + rcond - shift;
        }

        cont40 = false;
        for (k = rowstr[j]; k < rowstr[j+1]; k++) {
          if (colidx[k] > jcol) {
            //----------------------------------------------------------------
            // ... insert colidx here orderly
            //----------------------------------------------------------------
            for (kk = rowstr[j+1]-2; kk >= k; kk--) {
              if (colidx[kk] > -1) {
                a[kk+1]  = a[kk];
                colidx[kk+1] = colidx[kk];
              }
            }
            colidx[k] = jcol;
            a[k]  = 0.0;
            cont40 = true;
            break;
          } else if (colidx[k] == -1) {
            colidx[k] = jcol;
            cont40 = true;
            break;
          } else if (colidx[k] == jcol) {
            //--------------------------------------------------------------
            // ... mark the duplicated entry
            //--------------------------------------------------------------
            nzloc[j] = nzloc[j] + 1;
            cont40 = true;
            break;
          }
        }
        if (cont40 == false) {
          printf("internal error in sparse: i=%d\n", i);
          exit(EXIT_FAILURE);
        }
        a[k] = a[k] + va;
      }
    }
    size = size * ratio;
  }

  //---------------------------------------------------------------------
  // ... remove empty entries and generate final results
  //---------------------------------------------------------------------
  for (j = 1; j < nrows; j++) {
    nzloc[j] = nzloc[j] + nzloc[j-1];
  }

  for (j = 0; j < nrows; j++) {
    if (j > 0) {
      j1 = rowstr[j] - nzloc[j-1];
    } else {
      j1 = 0;
    }
    j2 = rowstr[j+1] - nzloc[j];
    nza = rowstr[j];
    for (k = j1; k < j2; k++) {
      a[k] = a[nza];
      colidx[k] = colidx[nza];
      nza = nza + 1;
    }
  }
  for (j = 1; j < nrows+1; j++) {
    rowstr[j] = rowstr[j] - nzloc[j-1];
  }
  nza = rowstr[nrows] - 1;
}


//---------------------------------------------------------------------
// generate a sparse n-vector (v, iv)
// having nzv nonzeros
//
// mark(i) is set to 1 if position i is nonzero.
// mark is all zero on entry and is reset to all zero before exit
// this corrects a performance bug found by John G. Lewis, caused by
// reinitialization of mark on every one of the n calls to sprnvc
//---------------------------------------------------------------------
static void sprnvc(int n, int nz, int nn1, double v[], int iv[])
{
  int nzv, ii, i;
  double vecelt, vecloc;

  nzv = 0;

  while (nzv < nz) {
    vecelt = randlc(&tran, amult);

    //---------------------------------------------------------------------
    // generate an integer between 1 and n in a portable manner
    //---------------------------------------------------------------------
    vecloc = randlc(&tran, amult);
    i = icnvrt(vecloc, nn1) + 1;
    if (i > n) continue;

    //---------------------------------------------------------------------
    // was this integer generated already?
    //---------------------------------------------------------------------
    logical was_gen = false;
    for (ii = 0; ii < nzv; ii++) {
      if (iv[ii] == i) {
        was_gen = true;
        break;
      }
    }
    if (was_gen) continue;
    v[nzv] = vecelt;
    iv[nzv] = i;
    nzv = nzv + 1;
  }
}


//---------------------------------------------------------------------
// scale a double precision number x in (0,1) by a power of 2 and chop it
//---------------------------------------------------------------------
static int icnvrt(double x, int ipwr2)
{
  return (int)(ipwr2 * x);
}


//---------------------------------------------------------------------
// set ith element of sparse vector (v, iv) with
// nzv nonzeros to val
//---------------------------------------------------------------------
static void vecset(int n, double v[], int iv[], int *nzv, int i, double val)
{
  int k;
  logical set;

  set = false;
  for (k = 0; k < *nzv; k++) {
    if (iv[k] == i) {
      v[k] = val;
      set  = true;
    }
  }
  if (set == false) {
    v[*nzv]  = val;
    iv[*nzv] = i;
    *nzv     = *nzv + 1;
  }
}

