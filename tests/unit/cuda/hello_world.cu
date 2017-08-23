// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
extern "C" __global__ void hello_world(char *in, char *out, int length){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < length)
		out[index] = in[index];
}