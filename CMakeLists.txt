cmake_minimum_required (VERSION 2.8)
project (NISTfit)

# We heavily use C++11, C++11 support is required
set (CMAKE_CXX_STANDARD 11)

# All source files to be compiled shall have the .cpp file extension and reside in the src directory
file(GLOB_RECURSE APP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# All global includes shall reside in the include directory
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

# 
if (NISTFIT_EXAMPLES)
	add_executable(simple ${CMAKE_CURRENT_SOURCE_DIR}/examples/simple.cpp ${APP_SOURCES})
endif()