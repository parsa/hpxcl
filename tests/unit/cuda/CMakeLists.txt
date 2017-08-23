# Copyright (c)       2017 Madhavan Seshadri
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    buffer
	kernel
   )

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  set(${test}_FLAGS ${${test}_FLAGS} DEPENDENCIES ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_cuda_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     FOLDER "Tests/Unit/cuda")

  add_hpx_unit_test("cuda" ${test})
  if(DEFINED ENV{CIRCLECI})
    message(STATUS "WARNING: CircleCI detected. Disabling test ${test}_remote ...")
  else()
		  #add_hpx_unit_test("cuda" ${test}_remote EXECUTABLE ${test}
		  #                                      LOCALITIES 2
		  #                                     THREADS_PER_LOCALITY 2)
  endif()

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.cuda.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.cuda
                              tests.unit.cuda.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.cuda.${test}
                              ${test}_cuda_test_exe)
endforeach()