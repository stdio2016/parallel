#include <stdio.h>
#include <math.h>

#define PI 3.1415926535
#define BLOCK_SIZE 160
#define THREAD_SIZE 32

__global__ void integrate_kern(long long num_intervals, double *result) {
    long long i;
    double rect_width, area, sum, x_middle;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    rect_width = PI / num_intervals;

    sum = 0;
    for(i = 1 + idx; i < num_intervals + 1; i += size) {

        /* find the middle of the interval on the X-axis. */ 

        x_middle = (i - 0.5) * rect_width;
        area = sin(x_middle) * rect_width; 
        sum = sum + area;

    } 

    result[idx] = sum;
}

int main(int argc, char **argv) 
{
    long long i, num_intervals, size;
    double sum; 
    double *gpu_sum, *array;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <number of intervals>\n", argv[0]);
        return 1;
    }

    sscanf(argv[1],"%llu",&num_intervals);

    size = BLOCK_SIZE * THREAD_SIZE;
    cudaMalloc(&gpu_sum, sizeof(double) * size);
    array = (double *) malloc(sizeof(double) * size);

    integrate_kern<<<BLOCK_SIZE, THREAD_SIZE>>>(num_intervals, gpu_sum);
    cudaMemcpy(array, gpu_sum, sizeof(double) * size, cudaMemcpyDeviceToHost);

    sum = 0;
    for (i = 0; i < size; i++) {
        sum += array[i];
    }
    printf("The total area is: %f\n", (float)sum);

    free(array);
    cudaFree(gpu_sum);
    return 0;

}   
