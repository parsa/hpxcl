// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <hpx/include/iostreams.hpp>

#include <hpxcl/cuda.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

// the main test function
static void cuda_test(hpx::cuda::device, hpx::cuda::device);


#define COMPARE_RESULT( result_data, correct_result )                           \
{                                                                               \
    auto lhs = result_data;                                                     \
    auto rhs = correct_result;                                                  \
    HPX_TEST_EQ(lhs.size(), rhs.size());                                        \
    std::string correct_string = to_string(rhs);                                \
    std::string result_string = to_string(lhs);                                 \
    HPX_TEST_EQ(correct_string, result_string);                                 \
}																				

#define COMPARE_STRING_RESULT (lhs, rhs)										\
{																				\
	HPX_TEST_EQ(lhs,rhs);														\
}																				

static std::vector<hpx::cuda::device> init(variables_map & vm)
{

    std::size_t device_id = 0;

    if (vm.count("deviceid"))
        device_id = vm["deviceid"].as<std::size_t>();

    // Try to get remote devices
    std::vector<hpx::cuda::device> remote_devices
            = hpx::cuda::get_remote_devices(1, 0).get();
    std::vector<hpx::cuda::device> local_devices
            = hpx::cuda::get_local_devices(1, 0).get();
    // If no remote devices present, get local device
    if(remote_devices.empty()){
        hpx::cout << "WARNING: No remote devices found." << hpx::endl;
        remote_devices = local_devices;
    }
    HPX_ASSERT(!remote_devices.empty());
    HPX_ASSERT(!local_devices.empty());
    HPX_TEST(local_devices.size() > device_id);
    HPX_TEST(remote_devices.size() > device_id);

    // Choose device
    hpx::cuda::device local_device  = local_devices[device_id];
    hpx::cuda::device remote_device = remote_devices[device_id];

    // Print info
    hpx::cout << "Local device:" << hpx::endl;
    //print_testdevice_info(local_device, device_id, local_devices.size());
    hpx::cout << "Remote device:" << hpx::endl;
    //print_testdevice_info(remote_device, device_id, remote_devices.size());

    // return the devices
    std::vector<hpx::cuda::device> devices;
    devices.push_back(local_device);
    devices.push_back(remote_device);
    return devices;

}


int hpx_main(variables_map & vm)
{
    {
        auto devices = init(vm);
        hpx::cout << hpx::endl;
        cuda_test(devices[0], devices[1]);
    }

    hpx::finalize();
    return hpx::util::report_errors();
}


///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");
    cmdline.add_options()
        ( "deviceid"
        , value<std::size_t>()->default_value(0)
        , "the ID of the device we will run our tests on") ;

    return hpx::init(cmdline, argc, argv);
}