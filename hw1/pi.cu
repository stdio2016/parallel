#include <stdio.h>
#include <pthread.h>
#include <sys/time.h>
int cpu_cores;
long long num_tosses;
int gpu_cores;

struct toss_t {
	int id;
	pthread_t pid;
	long long num_tosses;
	long long in_circle;
};

// from gcc rand
inline __device__ __host__ unsigned my_rand(unsigned *seed) {
	return *seed = 1103515245u * *seed + 12345u;
}
const unsigned MY_RAND_MAX = ~0u;

long long Timer() {
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec * 1000LL + t.tv_usec / 1000;
}

void *cpu_toss(void *args) {
	toss_t *me = (toss_t *) args;

	long long num_tosses = me->num_tosses;
	float x = 0.0f, y = 0.0f;
	long long in_circle = 0;
	unsigned seed = me->id;

	for (long long i = 0; i < num_tosses; i++) {
		x = my_rand(&seed) / (float) MY_RAND_MAX;
		y = my_rand(&seed) / (float) MY_RAND_MAX;
		if (x * x + y * y <= 1.0f) in_circle += 1;
	}
	me->in_circle = in_circle;
	return me;
}

__global__ void gpu_toss(long long num_tosses, long long *result) {
	int num_threads = blockDim.x * gridDim.x;
	int pid = blockDim.x * blockIdx.x + threadIdx.x;
	long long work = num_tosses / num_threads;
	if (pid < num_tosses - work * num_threads) {
		// remaining work
		work += 1;
	}
	long long in_circle = 0;
	unsigned seed = pid;
	float x, y;
	for (long long i = 0; i < work; i++) {
		x = my_rand(&seed) / (float) MY_RAND_MAX;
		y = my_rand(&seed) / (float) MY_RAND_MAX;
		if (x * x + y * y <= 1.0f) in_circle += 1;
	}
	result[pid] = in_circle;
}

int main(int argc, char *argv[]) {
	if (argc < 3) {
		fprintf(stderr, "usage: ./pi <threads> <tosses> [<blocks>]\n");
		return 1;
	}
	cpu_cores = atoi(argv[1]);
	num_tosses = atoll(argv[2]);
	gpu_cores = 0;
	if (argc == 4) gpu_cores = atoi(argv[3]);
	if (cpu_cores <= 0 || gpu_cores < 0) {
		fprintf(stderr, "thread count must not be negative\n");
		return 1;
	}

	long long answer = 0;
	int t1;
	if (gpu_cores > 0) { // GPU CUDA parallel
		int total_cores = cpu_cores * gpu_cores;
		long long *gpu_answer;
		cudaMalloc(&gpu_answer, sizeof(long long) * total_cores);
		t1 = Timer();
		gpu_toss<<<cpu_cores, gpu_cores>>>(num_tosses, gpu_answer);
		long long *arr = new long long[total_cores];
		cudaMemcpy(arr, gpu_answer, sizeof(long long) * total_cores, cudaMemcpyDeviceToHost);
		for (int i = 0; i < total_cores; i++)
			answer += arr[i];
		delete[] arr;
	}
	else { // CPU parallel
		toss_t *jobs = new toss_t[cpu_cores];
		long long remain = num_tosses;
		t1 = Timer();
		for (int i = 0; i < cpu_cores; i++) {
			jobs[i].id = i;
			long long use = remain / (cpu_cores - i);
			jobs[i].num_tosses = use;
			remain -= use;
			jobs[i].in_circle = 0;
			pthread_create(&jobs[i].pid, NULL, cpu_toss, &jobs[i]);
		}
		for (int i = 0; i < cpu_cores; i++) {
			pthread_join(jobs[i].pid, NULL);
			answer += jobs[i].in_circle;
		}
		delete[] jobs;
	}
	double pi = 4.0 * answer / num_tosses;
	int t2 = Timer();
	printf("Pi: %f\n", pi);
	printf("use time: %fs\n", float(t2-t1)/1000);
	return 0;
}
