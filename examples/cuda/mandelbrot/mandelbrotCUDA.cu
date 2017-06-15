// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cuda.h>
#include <iostream>
//for writing image
//#include "examples/opencl/mandelbrot/pngwriter.cpp"

using namespace std;

//###########################################################################
//Kernels
//###########################################################################
template<typename T>
__global__ void mandelbrot(char *out, int *width, int *height) {
	unsigned int xDim = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yDim = blockIdx.y * blockDim.y + threadIdx.y;

	//index of the output array, multiplied by 3 for R,G,B values
	int arrayIndex = 3 * (*width) * yDim + xDim * 3;

	float xPoint = ((float)(xDim) / (*width)) * 3.25f - 2.0f;
	float yPoint = ((float)(yDim) / (*height)) * 2.5f - 1.25f;

	//for calculation of complex number
	float x = 0.0;
	float y = 0.0;

	int iterationCount = 0;
	int numIterations = 256;
	//terminating condition x^2+y^2 < 4 or iterations >= numIterations
	while (y*y + x*x <= 4 && iterationCount<(numIterations)) {
		float xTemp = x*x - y*y + xPoint;
		y = 2 * x*y + yPoint;
		x = xTemp;
		iterationCount++;
	}

	if (iterationCount == (numIterations)) {
		out[arrayIndex] = 0;
		out[arrayIndex + 1] = 0;
		out[arrayIndex + 2] = 0;
	}
	else {
		out[arrayIndex] = iterationCount;
		out[arrayIndex + 1] = iterationCount;
		out[arrayIndex + 2] = iterationCount;
	}
}

//###########################################################################
//Main
//###########################################################################

int main(int argc, char*argv[]) {

	if (argc != 3) {
		std::cout << "Usage: " << argv[0] << "width height" << std::endl;
		exit(1);
	}

	int width = atoi(argv[1]);
	int height = atoi(argv[2]);
	const int bytes = sizeof(char) * width * height * 3;

	//Malloc Host
	char *image_host;
	cudaMallocHost((void**)&image_host, bytes);

	//Malloc Device
	char *image_device;
	cudaMalloc((void**)&image_device, bytes);

	dim3 block(16,16,1);
	dim3 grid;

	grid.x = width / block.x;
	grid.y = (height / (block.y));
	grid.z = 1;

	/*
	 * Kernel launch
	 */
	mandelbrot<char><<<grid, block>>>(image_device, &width, &height);
	cudaDeviceSynchronize();

	/*
	 * Copy result back
	 */
	cudaMemcpy(image_host, image_device, bytes, cudaMemcpyDeviceToHost);

	//write images to file
	//std::shared_ptr<std::vector<char>> img_data;

	/*
	 * Save result to png
	 */

	//write images to file
	/*
	std::shared_ptr<std::vector<char>> img_data;

	img_data = std::make_shared <std::vector <char> >
		(image_host, image_host+bytes);

	save_png(image_host, width, height, "Mandelbrot_img.png");
	*/

	/*
	 * Free
	 */
	 cudaFreeHost(image_host);
	 cudaFree(image_device);

	return 0;
}