// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "cuda_tests.hpp"

#define DATASIZE sizeof(sizeof("Hello World!"))

/*
 * This test is meant to verify the buffer read and buffer write functionality.
 */
static void cuda_test(hpx::cuda::device local_device, hpx::cuda::device remote_device)
{
	char bufferData[] = "Hello World";
	hpx::cuda::buffer local_buffer = local_device.create_buffer(DATASIZE);
	hpx::cuda::buffer local_buffer2 = local_device.create_buffer(DATASIZE);
	hpx::cuda::buffer remote_buffer = remote_device.create_buffer(DATASIZE);

	//Test if buffer initialization worked
	{
		size_t localBuffer_size  = local_buffer.size().get();
		HPX_TEST_EQ(localBuffer_size, DATASIZE);

		size_t remoteBuffer_size = remote_buffer.size().get();
		HPX_TEST_EQ(remoteBuffer_size, DATASIZE);
	}

	//Test if the buffer write works with no offset
	{
		auto data_write_test = local_buffer.enqueue_write(0, sizof(bufferData), bufferData);
		data_write_test.get();
	}

	//Test multiple buffer writes
	{
		auto future1 = local_buffer.enqueue_write(0, sizeof(bufferData), bufferData);
		auto future2 = local_buffer2.enqueue_write(0, sizeof(bufferData), bufferData);

		std::vector<hpx::future<void> > futures;
        futures.push_back(std::move(future1));
        futures.push_back(std::move(future2));

        hpx::when_all(futures).get();
	}

	//Test read function
	{
		auto read_test = local_buffer.enqueue_read(0, DATASIZE);
		COMPARE_STRING_RESULT(read_test.get(), bufferData);
	}

	// Test local continuity
	{
		auto write_data = local_buffer.enqueue_write(0, sizof(bufferData), bufferData);
		auto future2 = write_data.then(
            [](hpx::future<void> fut){
                return true;   
            }
        );
        HPX_TEST(future2.get());
	}
}