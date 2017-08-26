// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "cuda_tests.hpp"


/*
 * This test is meant to verify hpxcl is able to get devices in the localities works
 */
static void cuda_test(hpx::cuda::device local_device,
					  hpx::cuda::device remote_device)
{
	// Try to get remote devices
    std::vector<hpx::cuda::device> remote_devices
            = hpx::cuda::get_remote_devices(1, 0).get();
    std::vector<hpx::cuda::device> local_devices
            = hpx::cuda::get_local_devices(1, 0).get();
	
	
	{
		HPX_ASSERT(!remote_devices.empty());
		HPX_TEST(local_devices.size() > device_id);
	}
	
	{
		HPX_ASSERT(!local_devices.empty());
		HPX_TEST(remote_devices.size() > device_id);
	}

}