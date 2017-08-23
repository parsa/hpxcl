// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "cuda_tests.hpp"

/*
 * The following set of unit tests are to test the 
 * functionality of the kernel code of HPXCL CUDA
 */

#define DATASIZE (sizeof("Hello World!"))

const char in[] = { ('H'  - static_cast<char>( 0)), 
                              ('e'  - static_cast<char>( 1)), 
                              ('l'  - static_cast<char>( 2)), 
                              ('l'  - static_cast<char>( 3)), 
                              ('o'  - static_cast<char>( 4)), 
                              (' '  - static_cast<char>( 5)), 
                              ('W'  - static_cast<char>( 6)), 
                              ('o'  - static_cast<char>( 7)), 
                              ('r'  - static_cast<char>( 8)), 
                              ('l'  - static_cast<char>( 9)), 
                              ('d'  - static_cast<char>(10)), 
                              ('!'  - static_cast<char>(11)), 
                              ('\0' - static_cast<char>(12)) };

/*
* Define some kernels
*/
static const char hello_world_src[] =
		"                                                                                                        "
		"extern \"C\"  __global__ void hello_world(char *in, char *out, int length){ 	   					   \n"
		"		int index = blockIdx.x * blockDim.x + threadIdx.x;											   \n"
		"		if(index < length)																			   \n"
		"			out[index] = in[index];																	   \n"
		"}                                             							                               \n";

static const char invalid_hello_world_src[] =
		"                                                                                                        "
		"extern \"C\"  __global__ void hello_world(char *in, char *out, int length){ 	   					   \n"
		"		int index = blockIdx.x * blockDim.x + threadIdx.x;											   \n"
		"		if(index < length)																			   \n"
		"			out[TestIndex] = in[index];																	   \n"
		"}                                             							                               \n";

/*
*
* Define new buffers and run the kernel and verify execution
*
*/
static void run_kernel (hpx::cuda::program program) {
	
	int len = DATASIZE;
	char *out;
	out = (char*) malloc (DATASIZE);

	hpx::cuda::buffer inBuffer = cudaDevice.create_buffer(DATASIZE);
	hpx::cuda::buffer outBuffer = cudaDevice.create_buffer(DATASIZE);
	hpx::cuda::buffer lenBuffer = cudaDevice.create_buffer(sizeof(int));

	std::vector<hpx::cuda::buffer> args;
	//Generate the grid and block dim
	hpx::cuda::server::program::Dim3 grid;
	hpx::cuda::server::program::Dim3 block;

	block.x = DATASIZE;

	grid.x = 1;

	//Test if buffer is initialized properly and equal number of bytes
	// are issued
	{
		size_t inBuffer_size = inBuffer.size().get();
		HPX_TEST_EQ(inBuffer_size, DATASIZE);

		size_t outBuffer_size = outBuffer.size().get();
		HPX_TEST_EQ(outBuffer_size, DATASIZE);

		size_t lenBuffer_size = lenBuffer.size().get();
		HPX_TEST_EQ(lenBuffer_size, sizeof(int));
	}

	//Set input value to the kernel
	{
		auto future1 = inBuffer.enqueue_write(0, DATASIZE, in);
		future1.get();
		auto future2 = outBuffer.enqueue_write(0, DATASIZE, out);
		future2.get();
		auto future3 = lenBuffer.enqueue_write(0, sizeof(int), &len);
		future3.get();

		args.push_back(inBuffer);
		args.push_back(outBuffer);
		args.push_back(lenBuffer);

		//Synchronize data transfer before new data is written
		hpx::wait_all(args);
	}

	//run the program on the device
	{
		auto future = program.run(args, "hello_world", grid, block, args);
		future.get();		
	}


}

/*
*
* Defines the list of tests for Testing program compilation 
* and kernel run
*
*/
static void cuda_test(hpx::cuda::device cudaDevice, hpx::cuda::device remote_device){

	//Standard test for program compilation and build
	{
		//Test if a program can be created
		hpx::cuda::program program = cudaDevice.create_program_with_source(hello_world_src);
		
		std::vector<std::string> flags;
		
		//Test if the program can be compiled
		program.build(flags, "hello_world");

		//Running the kernel

	}

	//Standard test for program compilation and build with arguments
	{
		//Test if a program can be created
		hpx::cuda::program program = cudaDevice.create_program_with_source(hello_world_src);

		//Compile with the kernal
		std::vector<std::string> flags;
		std::string mode = "--gpu-architecture=compute_";
		mode.append(
			std::to_string(cudaDevice.get_device_architecture_major().get()));
		mode.append(
			std::to_string(cudaDevice.get_device_architecture_minor().get()));

		flags.push_back(mode);

		//Test if the program can be compiled
		program.build(flags, "hello_world");

		//Running the kernel
		
	}

	//Standard test for program compilation and build with invalid kernel source name
	{
		//Test if a program can be created
		hpx::cuda::program program = cudaDevice.create_program_with_source(hello_world_src);

		//Compile with the kernal
		std::vector<std::string> flags;
		std::string mode = "--gpu-architecture=compute_";
		mode.append(
			std::to_string(cudaDevice.get_device_architecture_major().get()));
		mode.append(
			std::to_string(cudaDevice.get_device_architecture_minor().get()));

		flags.push_back(mode);

		//Test if the program can be compiled
		bool caught_exception = false;
		//The exception must be caught
		try{
			program.build(flags, "helloWorld");
		} catch (hpx::exception e){
            hpx::cout << "Build error:" << hpx::endl;
            hpx::cout << e.what() << hpx::endl << hpx::endl;
            caught_exception = true;
        }
        HPX_TEST(caught_exception);		
	}


	//Standard test for program compilation and build from source file
	{
		//Test if a program can be created
		hpx::cuda::program program = cudaDevice.create_program_with_file("hello_world.cu");

		//Compile with the kernal
		std::vector<std::string> flags;
		std::string mode = "--gpu-architecture=compute_";
		mode.append(
			std::to_string(cudaDevice.get_device_architecture_major().get()));
		mode.append(
			std::to_string(cudaDevice.get_device_architecture_minor().get()));

		flags.push_back(mode);

		//Test if the program can be compiled
		program.build(flags, "hello_world");

		//Running the kernel
		
	}

	//Test invalid error detection
	{
		//Test if a program can be created
		hpx::cuda::program program = cudaDevice.create_program_with_source(invalid_hello_world_src);

		//Compile with the kernal
		std::vector<std::string> flags;
		std::string mode = "--gpu-architecture=compute_";
		mode.append(
			std::to_string(cudaDevice.get_device_architecture_major().get()));
		mode.append(
			std::to_string(cudaDevice.get_device_architecture_minor().get()));

		flags.push_back(mode);

		//Test if the program can be compiled
		bool caught_exception = false;
		//The exception must be caught
		try{
			program.build(flags, "hello_world");
		} catch (hpx::exception e){
            hpx::cout << "Build error:" << hpx::endl;
            hpx::cout << e.what() << hpx::endl << hpx::endl;
            caught_exception = true;
        }
        HPX_TEST(caught_exception);				
	}

}
