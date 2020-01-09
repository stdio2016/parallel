#include<stdio.h>
#define BUF_SIZE 20 * 1048576

__global__ void get_nl_count(char *buf, int n, int *out) {
    int tid = threadIdx.x;
    __shared__ volatile int cnt[256];
    for (int i = tid + blockDim.x * blockIdx.x, j = blockIdx.x; j < n; i += blockDim.x * gridDim.x, j += gridDim.x) {
        cnt[tid] = buf[i] == '\n';
        //__syncthreads();
        //if (tid < 512) cnt[tid] += cnt[tid+512];
        //__syncthreads();
        //if (tid < 256) cnt[tid] += cnt[tid+256];
        __syncthreads();
        if (tid < 128) cnt[tid] += cnt[tid+128];
        __syncthreads();
        if (tid < 64) cnt[tid] += cnt[tid+64];
        __syncthreads();
        if (tid < 32) {
            cnt[tid] += cnt[tid+32];
            cnt[tid] += cnt[tid+16];
            cnt[tid] += cnt[tid+8];
            cnt[tid] += cnt[tid+4];
            cnt[tid] += cnt[tid+2];
            cnt[tid] += cnt[tid+1];
            if (tid == 0) out[j] = cnt[0];
        }
        __syncthreads();
    }
}

__global__ void get_line_pos(char *buf, int n, int *pos, int *out) {
    int tid = threadIdx.x;
    __shared__ int cnt[256], cnt2[256];
    for (int i = tid + blockDim.x * blockIdx.x, j = blockIdx.x; j < n; i += blockDim.x * gridDim.x, j += gridDim.x) {
        int b = buf[i] == '\n';
        cnt[tid] = b;
        if (tid == 0) cnt[0] += pos[j];
        // reduce sum
        __syncthreads();
        #pragma unroll
        for (int dd = 0; dd < 8; dd += 2) {
            cnt2[tid] = tid >= 1<<dd ? cnt[tid] + cnt[tid-(1<<dd)] : cnt[tid];
            __syncthreads();
            cnt[tid] = tid >= 2<<dd ? cnt2[tid] + cnt2[tid-(2<<dd)] : cnt2[tid];
            __syncthreads();
        }
        if (b) out[cnt[tid]-1] = i;
    }
}

__global__ void parse(char *buf, int *bmp, int n) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = gid; i < n - 1; i += blockDim.x * gridDim.x) {
        int pos = bmp[i] + 1;
        int red = 0, green = 0, blue = 0;
        while (buf[pos] <= ' ' && buf[pos] != '\n') { pos += 1; }
        while (buf[pos] >= '0' && buf[pos] <= '9') {
            red = red * 10 + (buf[pos] - '0');
            pos += 1;
        }
        while (buf[pos] <= ' ' && buf[pos] != '\n') { pos += 1; }
        while (buf[pos] >= '0' && buf[pos] <= '9') {
            green = green * 10 + (buf[pos] - '0');
            pos += 1;
        }
        while (buf[pos] <= ' ' && buf[pos] != '\n') { pos += 1; }
        while (buf[pos] >= '0' && buf[pos] <= '9') {
            blue = blue * 10 + (buf[pos] - '0');
            pos += 1;
        }
        bmp[i] = red | green<<8 | blue<<16;
    }
}

__global__ void hist_kern(int *bmp, int n, int *out) {
    __shared__ int sr[256], sg[256], sb[256];
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    sr[threadIdx.x] = 0;
    sg[threadIdx.x] = 0;
    sb[threadIdx.x] = 0;
    __syncthreads();
    for (int i = gid; i < n; i += blockDim.x * gridDim.x) {
        int rgb = bmp[i];
        atomicAdd(&sr[rgb & 255], 1);
        atomicAdd(&sg[rgb >> 8 & 255], 1);
        atomicAdd(&sb[rgb >> 16 & 255], 1);
    }
    __syncthreads();
    out[gid] += sr[threadIdx.x];
    out[gid + 256 * gridDim.x] += sg[threadIdx.x];
    out[gid + 256 * gridDim.x * 2] += sb[threadIdx.x];
}

int main() {
    if (cudaSetDevice(0) != cudaSuccess) return 2;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("using GPU %s\n", prop.name);
    int smcount = prop.multiProcessorCount;

    cudaEvent_t e, e2;
    cudaEventCreate(&e);
    cudaEventCreate(&e2);

    char *gpu_buf, *buf;
    cudaHostAlloc(&buf, BUF_SIZE + 100, cudaHostAllocDefault);
    cudaMalloc(&gpu_buf, BUF_SIZE + 100);

    int *gpu_what, *what;
    cudaMalloc(&gpu_what, sizeof(int) * 1048576);

    int *hist = new int[256*3*4*smcount], *gpu_hist;
    cudaMalloc(&gpu_hist, sizeof(int) * (256*3*4*smcount));
    cudaMemset(gpu_hist, 0, sizeof(int) * (256*3*4*smcount));
    int histogram_results[768] = {0};

    what = new int[1048576];
    FILE *inFile = fopen("input", "rb");
    FILE *outFile = fopen("yyyyyy.out", "wb");
    if (!inFile || !outFile) return 1;
    printf("open file\n");
    unsigned n = 0;
    if (fscanf(inFile, "%u", &n) != 1 || n%3 != 0) return 1;

    int *bmp = new int[n/3], *gpu_bmp;
    cudaMalloc(&gpu_bmp, sizeof(int) * BUF_SIZE);

    fgets(buf, BUF_SIZE-1, inFile);
    int bytes_read, tail = 1;
    unsigned sum = 0;
    buf[0] = '\n';
    cudaEventRecord(e, 0);
    while ((bytes_read = fread(buf + tail, 1, BUF_SIZE - tail, inFile)) > 0) {
        printf("read %d bytes\n", bytes_read);
        int blocks = (tail + bytes_read-1) / 256 + 1;

        for (int i = bytes_read + tail; i < blocks*256; i++) buf[i] = 0;

        cudaMemcpy(gpu_buf, buf, BUF_SIZE, cudaMemcpyHostToDevice);
        get_nl_count<<<8*smcount, 256>>>(gpu_buf, blocks, gpu_what);
        cudaMemcpy(what, gpu_what, sizeof(int) * blocks, cudaMemcpyDeviceToHost);

        unsigned sum2 = sum;
        for (int i = 0; i < blocks; i++) sum2 += what[i];
        for (int i = 1; i < blocks; i++) what[i] += what[i-1];
        for (int i = blocks-1; i > 0; i--) what[i] = what[i-1];
        what[0] = 0;

        cudaMemcpy(gpu_what, what, sizeof(int) * blocks, cudaMemcpyHostToDevice);
        get_line_pos<<<8*smcount, 256>>>(gpu_buf, blocks, gpu_what, gpu_bmp);
        //cudaDeviceSynchronize();

        //cudaMemcpy(bmp + sum, gpu_bmp, sizeof(int) * (sum2 - sum), cudaMemcpyDeviceToHost);
        parse<<<8*smcount, 256>>>(gpu_buf, gpu_bmp, sum2 - sum);
        //cudaDeviceSynchronize();

        hist_kern<<<4*smcount, 256>>>(gpu_bmp, sum2-1 - sum, gpu_hist);

        printf("sum = %u sum2 = %u\n", sum, sum2);
        sum = sum2-1;
        
        if (bytes_read + tail < BUF_SIZE) break;
        for (tail = 1; tail <= BUF_SIZE; tail++) {
            if (buf[BUF_SIZE - tail] == '\n') break;
        }
        if (tail > BUF_SIZE) tail = 0;
        for (int i = 0; i < tail; i++) buf[i] = buf[i + BUF_SIZE-tail];
    }

    cudaMemcpy(hist, gpu_hist, sizeof(int) * (256*3*4*smcount), cudaMemcpyDeviceToHost);
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < 4*smcount; i++) {
            for (int j = 0; j < 256; j++) histogram_results[j+c*256] += hist[j+i*256+c*256*20];
        }
    }

    int sum2 = 0;
    for (int i = 0; i < 768; i++) {
        if (i%256 == 0 && i != 0) fputc('\n', outFile);
        fprintf(outFile, "%d ", histogram_results[i]);
        sum2 += histogram_results[i];
    }

    cudaEventRecord(e2, 0);
    float t;
    cudaEventSynchronize(e);
    cudaEventSynchronize(e2);
    cudaEventElapsedTime(&t, e, e2);
    printf("%d %d %f %d\n", n, sum, t, sum2);

    fclose(inFile);
    fclose(outFile);
    delete[] what;
    cudaFree(gpu_what);
    cudaFreeHost(buf);
}
