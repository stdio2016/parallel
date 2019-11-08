#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int isprime(int n) {
    int i,squareroot;
    if (n>10) {
        squareroot = (int) sqrt(n);
        for (i=3; i<=squareroot; i=i+2)
            if ((n%i)==0)
                return 0;
        return 1;

    }
    else
        return 0;

}

__global__ void gpu_prime(int limit, int *ans) {
	int segn = blockDim.x * gridDim.x;
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int pc = 0, foundone = 0;
	limit = (limit-1)/2; // consider only odd numbers
	int n = int(float(limit) * id / segn);
	int hi = int(float(limit) * (id+1) / segn) - 1;
	if (id == segn-1) hi = limit;
	n = n*2+1;
	hi = hi*2+1;
	if (n == 1) n = 3; // 1 is not prime
	if (n > hi) return;
	int sq = round(sqrt(float(n))), fact = 3;
	while (n <= hi) {
		if (fact > sq) {
			n += 2;
			fact = 3;
			sq = round(sqrt(float(n)));
			pc += 1;
			foundone = n-2;
		}
		else if (n % fact == 0) {
			n += 2;
			fact = 3;
		}
		else fact += 2;
	}
	atomicAdd(&ans[1], pc);
	atomicMax(&ans[0], foundone);
}

void try_size(int *gpu_ans, int limit, int bs, int ts) {
    int ans[2] = {/*max prime*/ 0, /*count*/ 1};
    cudaMemcpy(gpu_ans, ans, sizeof ans, cudaMemcpyHostToDevice);
    int t1 = clock();
    gpu_prime<<<bs, ts>>>(limit, gpu_ans);
    cudaMemcpy(ans, gpu_ans, sizeof ans, cudaMemcpyDeviceToHost);
    int t2 = clock();
    printf("Done. Largest prime is %d Total primes %d\n",ans[0], ans[1]);
    printf("block size = %d thread size = %d GPU time %fs\n", bs, ts, double(t2-t1)/CLOCKS_PER_SEC);
}

int main(int argc, char *argv[])
{
    int pc,       /* prime counter */
        foundone; /* most recent prime found */
    long long int n, limit;

    sscanf(argv[1],"%llu",&limit);
    printf("Starting. Numbers to be scanned= %lld\n",limit);
    int t1 = clock();

    pc=4;     /* Assume (2,3,5,7) are counted here */

    for (n=11; n<=limit; n=n+2) {
        if (isprime(n)) {
            pc++;
            foundone = n;

        }

    }
    int t2 = clock();
    printf("Done. Largest prime is %d Total primes %d\n",foundone, pc);
    printf("CPU time %fs\n", double(t2-t1)/CLOCKS_PER_SEC);
    int *gpu_ans;
    cudaMalloc(&gpu_ans, sizeof(int[2]));
    printf("CUDA version:\n");
    // 160 block 32 thread is best for my GPU
    // around 22 times acceleration
    try_size(gpu_ans, limit, 160, 32);
    cudaFree(&gpu_ans);
    return 0;
}
