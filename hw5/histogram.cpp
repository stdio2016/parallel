#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <fstream>
#include <iostream>

#include <CL/cl.h>
#include <sys/time.h>
#define THREAD_COUNT 250

// OpenCL context
int good = 0;
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel;
cl_mem str_cl, his_cl, range_cl;

long long Timer() {
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec * 1000LL + t.tv_usec / 1000;
}

int initCL() {
	// get platform
	cl_uint num = 1;
	cl_int err;
	err = clGetPlatformIDs(1, &platform, &num);
	if (err != CL_SUCCESS || num < 1) {
		std::cerr << "unable to get platform\n";
		return 0;
	}

	// get device
	num = 1;
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num);
	if (err != CL_SUCCESS || num < 1) {
		std::cerr << "unable to get device ID\n";
		return 0;
	}

	// create context
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		std::cerr << "unable to create context\n";
		return 0;
	}
	good = 1;

	// create command queue
	queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
	if (err != CL_SUCCESS) {
		std::cerr << "unable to create command queue\n";
		return 0;
	}
	good = 2;
	return 1;
}

cl_program loadKernel(const char *name) {
	std::ifstream f(name, std::ios_base::in);
	if (!f) {
		std::cerr << "program \"" << name << "\" not found\n";
		return 0;
	}

	// get file size
	f.seekg(0, std::ios_base::end);
	size_t len = f.tellg();
	if (len > 102400) {
		std::cerr << "program \"" << name << "\" is too big! limit is 100KB\n";
		return 0;
	}
	f.seekg(0, std::ios_base::beg);

	// read file
	char *code = new char[len];
	f.read(code, len);
	const char *codes = &code[0];

	// compile program
	cl_int err;
	cl_program prog = clCreateProgramWithSource(context, 1, &codes, &len, &err);
	delete [] code;
	if (err != CL_SUCCESS) {
		std::cerr << "cannot create program \"" << name << "\"\n";
		return 0;
	}
	err = clBuildProgram(prog, 0, NULL, "-cl-std=CL2.0", NULL, NULL);
	if (err != CL_SUCCESS) {
		std::cerr << "program \"" << name << "\" has errors:\n";
		size_t len;
		err = clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
		if (err != CL_SUCCESS) return 0;
		code = new char[len];
		err = clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, len, code, &len);
		if (err == CL_SUCCESS) {
			std::cerr << code << "\n";
		}
		delete [] code;
		clReleaseProgram(prog);
		return 0;
	}
	return prog;
}

void release() {
	if (good >= 4) {
		if (str_cl) clReleaseMemObject(str_cl);
		if (his_cl) clReleaseMemObject(his_cl);
		if (range_cl) clReleaseMemObject(range_cl);
		clReleaseKernel(kernel);
	}
	if (good >= 3) clReleaseProgram(program);
	if (good >= 2) clReleaseCommandQueue(queue);
	if (good >= 1) clReleaseContext(context);
}

int main(int argc, char const *argv[])
{
	atexit(release);
	long long times[6] = { 0 };
	long long t[6] = { Timer() };
	// opencl init and load kernel
	cl_int err1, err2, err3;
	if (initCL() == false) exit(1);
	program = loadKernel("histogram.cl");
	if (program == 0) {
		return 1;
	}
	good = 3;
	kernel = clCreateKernel(program, "histogram", &err1);
	if (err1 != CL_SUCCESS) {
		std::cerr << "unable to create kernel\n";
		return 1;
	}
	good = 4;

	// open file
	FILE *inFile = fopen("input", "rb");
	std::ofstream outFile("yyyyyy.out", std::ios_base::out);

	// create buffer in host
	size_t input_size = 1<<27;
	size_t his_size = 256 * 3 * sizeof(unsigned int) * THREAD_COUNT;
	char *fileBuf1 = (char *) clSVMAlloc(context, CL_MEM_READ_WRITE, input_size, 0);
	char *fileBuf2 = (char *) clSVMAlloc(context, CL_MEM_READ_WRITE, input_size, 0);
	char *fileBuf[2] = {fileBuf1, fileBuf2};
	unsigned int * histogram_partial = new unsigned int[256 * 3 * THREAD_COUNT];
	unsigned int * histogram_results = (unsigned int *) calloc(sizeof(int[256 * 3]), 1);
	size_t range_size = sizeof(int[THREAD_COUNT+1]);
	int *rangeArr = new int[THREAD_COUNT+1];

	// create buffer in device
	//str_cl = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size, NULL, &err1);
	his_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, his_size, NULL, &err2);
	range_cl = clCreateBuffer(context, CL_MEM_READ_ONLY, range_size, NULL, &err3);
	if (err2 != CL_SUCCESS || err3 != CL_SUCCESS) {
		std::cerr << "failed to create buffer\n";
		return 1;
	}
	good = 5;
	t[1] = Timer();
	std::cout << "initialize took " << t[1] - t[0] << " ms\n";
	t[0] = t[1];

	// read file
	size_t read_size, read_offset = 0;
	int n;
	fscanf(inFile, "%d", &n);
	fgetc(inFile);
	n = 0;
	clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, fileBuf1, input_size, 0, NULL, NULL);
	read_size = fread(fileBuf1+read_offset, 1, input_size-read_offset, inFile);
	t[2] = Timer();
	times[2] = t[2] - t[1];
	t[0] = t[2];
	while (read_size > 0) {
		char *str = fileBuf[n%2];
		char *str2 = fileBuf[(n+1)%2];
		// split data
		read_size += read_offset;
		rangeArr[0] = 0;
		int division = read_size / THREAD_COUNT;
		for (int i = 1; i <= THREAD_COUNT; i++) {
			// find newline
			int pos = division * i;
			while (pos > 0 && str[pos-1] != '\n') {
				pos--;
			}
			rangeArr[i] = pos;
		}
		read_offset = read_size - rangeArr[THREAD_COUNT];
		clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, str2, input_size, 0, NULL, NULL);
		memmove(str2, str+read_size-read_offset, read_offset);
		t[1] = Timer();

		// send data
		//err1 = clEnqueueWriteBuffer(queue, str_cl, CL_TRUE, 0, read_size, fileBuf, 0, NULL, NULL);
		err1 = clEnqueueSVMUnmap(queue, str, 0, NULL, NULL);
		err2 = clEnqueueWriteBuffer(queue, range_cl, CL_TRUE, 0, range_size, rangeArr, 0, NULL, NULL);
		if (err1 != CL_SUCCESS || err2 != CL_SUCCESS) {
			std::cerr << "failed to send data\n";
		}
		t[2] = Timer();

		// call kernel
		clSetKernelArgSVMPointer(kernel, 0, str);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &range_cl);
		clSetKernelArg(kernel, 2, sizeof(cl_mem), &his_cl);
		size_t work_offset[1] = { 0 };
		size_t work_size[1] = { THREAD_COUNT };
		size_t local_size[1] = { 1 };
		err1 = clEnqueueNDRangeKernel(
			queue, kernel, 1, work_offset, work_size, local_size, 0, NULL, NULL
		);
		if (err1 != CL_SUCCESS) {
			std::cerr << "failed to start kernel\n";
		}

		// read file
		read_size = fread(str2+read_offset, 1, input_size-read_offset, inFile);
		t[3] = Timer();
		clFinish(queue);
		t[4] = Timer();

		// receive result
		err1 = clEnqueueReadBuffer(queue, his_cl, CL_TRUE, 0, his_size, histogram_partial, 0, NULL, NULL);
		if (err1 != CL_SUCCESS) {
			std::cerr << "failed to read buffer\n";
		}
		t[5] = Timer();

		// merge result
		for (int i = 0; i < THREAD_COUNT; i++) {
			for (int j = 0; j < 768; j++) {
				histogram_results[j] += histogram_partial[i*768 + j];
			}
		}
		std::cout << "round " << n << " finished offset " << read_offset <<"\n";

		// profiling
		for (int i = 0; i < 5; i++) times[i] += t[i+1] - t[i];
		t[0] = Timer();
		times[5] += t[0] - t[5];
		n++;
	}
	char *str = fileBuf[n%2];
	// last line will be skipped
	int off = 0;
	for (int i = 0; i < read_offset; i++) {
		int num = 0;
		while (str[i] >= '0' && str[i] <= '9') {
			num = num*10 + (str[i]-'0');
			i++;
		}
		histogram_results[off<<8 | num]++;
		off = (off+1)%3;
	}
	times[0] += Timer() - t[0];
	std::cout << "read   took " << times[2] << " ms\n";
	std::cout << "split  took " << times[0] << " ms\n";
	std::cout << "send   took " << times[1] << " ms\n";
	std::cout << "kernel took " << (times[2]+times[3]) << " ms\n";
	std::cout << "recv   took " << times[4] << " ms\n";
	std::cout << "merge  took " << times[5] << " ms\n";

	for(unsigned int i = 0; i < 256 * 3; ++i) {
		if (i % 256 == 0 && i != 0)
			outFile << std::endl;
		outFile << histogram_results[i]<< ' ';
	}

	fclose(inFile);
	outFile.close();
	return 0;
}
