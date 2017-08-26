// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "cuda_tests.hpp"


/*
 * This test is meant to verify information works
 */
static void cuda_test(hpx::cuda::device local_device,
					  hpx::cuda::device remote_device)
{

		//Test if the device Id information can be obtained for a local device
		{
			// Test for valid device client
			HPX_TEST(local_device.get_device_id().get());
		}
		
		//Test if the device Id information can be obtained for a remote device
		{
			// Test for valid device client
			HPX_TEST(remote_device.get_device_id().get());
		}

}
