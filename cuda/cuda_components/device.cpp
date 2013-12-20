// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/device.hpp"

HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::managed_component<
	hpx::cuda::server::device
	> cuda_device_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(cuda_device_type,device);

HPX_REGISTER_ACTION(
    cuda_device_type::wrapped_type::calculate_pi_action,
    cuda_device_calculate_pi_action);
HPX_REGISTER_ACTION(
    cuda_device_type::wrapped_type::get_cuda_info_action,
    cuda_device_get_cuda_info_action);
HPX_REGISTER_ACTION(
    cuda_device_type::wrapped_type::set_device_action,
    cuda_device_set_device_action);
HPX_REGISTER_ACTION(
    cuda_device_type::wrapped_type::get_all_devices_action,
    cuda_device_get_all_devices_action);

