cmake_minimum_required (VERSION 3.0)
project (NISTfit)

# We heavily use C++11, C++11 support is required
set (CMAKE_CXX_STANDARD 11)

# All source files to be compiled shall have the .cpp file extension and reside in the src directory
file(GLOB_RECURSE APP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# All global includes shall reside in the include directory
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

# Munge the arguments for g++ (might not be the best way, but seems to work)
if (${CMAKE_GENERATOR} MATCHES  "Unix Makefile")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -fPIC")
endif()

if (NISTFIT_EXAMPLES)
    # Add Eigen include here since we are going to use our version of Eigen
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
    add_executable(simple ${CMAKE_CURRENT_SOURCE_DIR}/examples/simple.cpp ${APP_SOURCES})
    if (NISTFIT_NTHREAD_MAX)
        message(STATUS "Set the number of threads")
        add_definitions(-DNTHREAD_MAX=${NISTFIT_NTHREAD_MAX})
    endif()
elseif (NISTFIT_STATIC_LIBRARY OR NISTFIT_PYBIND11)
    add_library(NISTfit STATIC ${APP_SOURCES})
    # Add target include directories for easy linking with other applications
    target_include_directories(NISTfit PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
endif()

if (NISTFIT_PYBIND11)

    # ----------------------------
    # Build pybind11 python module
    # ----------------------------
    
    # Add Eigen include here since we are going to use our version of Eigen
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
    add_subdirectory("${CMAKE_SOURCE_DIR}/externals/pybind11")
    add_definitions(-DPYBIND11)
    pybind11_add_module(PolyFitter "${CMAKE_SOURCE_DIR}/src/pybind11_interface.cxx")
    target_link_libraries (PolyFitter PUBLIC NISTfit)
endif()