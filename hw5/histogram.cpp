#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <fstream>
#include <iostream>

#include <CL/cl.h>

// OpenCL context
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;

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

	// create command queue
	queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
	if (err != CL_SUCCESS) {
		std::cerr << "unable to create command queue\n";
		return 0;
	}
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
	err = clBuildProgram(prog, 0, NULL, "", NULL, NULL);
	if (err == CL_BUILD_PROGRAM_FAILURE) {
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
		return 0;
	}
	else if (err != CL_SUCCESS) {
		std::cerr << "cannot build program \"" << name << "\"\n";
		return 0;
	}
	return prog;
}

int main(int argc, char const *argv[])
{
	// opencl init and load kernel
	cl_int err1, err2;
	if (initCL() == false) exit(1);
	cl_program histogramProg = loadKernel("histogram.cl");
	if (histogramProg == 0) {
		return 1;
	}
	cl_kernel kernel = clCreateKernel(histogramProg, "histogram", &err1);
	if (err1 != CL_SUCCESS) {
		std::cerr << "unable to create kernel\n";
		return 1;
	}

	unsigned int * histogram_results;
	unsigned int a, input_size;
	std::fstream inFile("input", std::ios_base::in);
	std::ofstream outFile("yyyyyy.out", std::ios_base::out);

	inFile >> input_size;
	unsigned char *image = new unsigned char[input_size];
	for ( unsigned i = 0; i < input_size; i++ ) {
		inFile >> a;
		image[i] = a;
	}

	// create buffer
	size_t his_size = 256 * 3 * sizeof(unsigned int);
	cl_mem image_cl = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size, NULL, &err1);
	cl_mem his_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, his_size, NULL, &err2);
	if (err1 != CL_SUCCESS || err2 != CL_SUCCESS) {
		std::cerr << "failed to create buffer\n";
		return 1;
	}

	// send data
	histogram_results = (unsigned int *) calloc(his_size, 1);
	err1 = clEnqueueWriteBuffer(queue, image_cl, CL_TRUE, 0, input_size, image, 0, NULL, NULL);
	err2 = clEnqueueWriteBuffer(queue, his_cl, CL_TRUE, 0, his_size, histogram_results, 0, NULL, NULL);
	if (err1 != CL_SUCCESS || err2 != CL_SUCCESS) {
		std::cerr << "failed to send data\n";
		return 1;
	}

	// call kernel
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &image_cl);
	clSetKernelArg(kernel, 1, sizeof(unsigned int), &input_size);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &his_cl);
	size_t work_offset[1] = { 0 };
	size_t work_size[1] = { 3 };
	err1 = clEnqueueNDRangeKernel(
		queue, kernel, 1, work_offset, work_size, NULL, 0, NULL, NULL
	);
	if (err1 != CL_SUCCESS) {
		std::cerr << "failed to run kernel, or kernel crashed\n";
		return 1;
	}

	// receive result
	err1 = clEnqueueReadBuffer(queue, his_cl, CL_TRUE, 0, his_size, histogram_results, 0, NULL, NULL);
	for(unsigned int i = 0; i < 256 * 3; ++i) {
		if (i % 256 == 0 && i != 0)
			outFile << std::endl;
		outFile << histogram_results[i]<< ' ';
	}

	inFile.close();
	outFile.close();
	return 0;
}
