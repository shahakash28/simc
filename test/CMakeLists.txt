find_package(SEAL 3.6.4 REQUIRED)
include_directories(${SEAL_INCLUDE_DIRS})

ADD_SUBDIRECTORY(LinearLayer)

#Testing macro
macro (add_test_executable_with_lib _name libs)
	add_executable(test_${_name} "${_name}.cpp")
	target_link_libraries(test_${_name} ${EMP-OT_LIBRARIES} SEAL::seal LinearLayer)
endmacro()

macro (add_test_case _name)
	add_test_executable_with_lib(${_name} "")
  	add_test(NAME ${_name} COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_${_name}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")
endmacro()

macro (add_test_case_with_run _name)
	add_test_executable_with_lib(${_name} "")
	add_test(NAME ${_name} COMMAND "./run" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_${_name}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")
endmacro()

# Test cases
add_test_case_with_run(bit)
add_test_case_with_run(int)
IF(${ENABLE_FLOAT})
add_test_case_with_run(float)
ENDIF(${ENABLE_FLOAT})
add_test_case_with_run(circuit_file)
add_test_case_with_run(example)
add_test_case_with_run(repeat)
add_test_case_with_run(msi_relu)
add_test_case_with_run(msi_relu_final)
add_test_case_with_run(msi_linearlayer)
add_test_case_with_run(msi_convlayer)
add_test_case_with_run(msi_microbenchmark)
add_test_case_with_run(msi_relu_preprocess)
add_test_case_with_run(components)
add_test_case_with_run(msi_relu_integrate)
##add_test_case_with_run(fc_field)
