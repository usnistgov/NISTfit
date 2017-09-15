cmake_minimum_required (VERSION 3.0)
project (NISTfit)

# We heavily use C++11, C++11 support is required
set (CMAKE_CXX_STANDARD 11)

# All source files to be compiled shall have the .cpp file extension and reside in the src directory
file(GLOB_RECURSE APP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# All global includes shall reside in the include directory
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/externals/ThreadPool2")

macro(add_nix_libraries target)
    # Add DL and pthreads
    FIND_PACKAGE ( Threads REQUIRED )
    target_link_libraries (${target} PRIVATE ${CMAKE_DL_LIBS} PRIVATE ${CMAKE_THREAD_LIBS_INIT})
endmacro()

macro(add_libraries target)
    # Check for the existence of OpenMP and enable it as needed; see also http://stackoverflow.com/a/12404666/1360263
    find_package(OpenMP)
    if (OPENMP_FOUND)
        target_compile_options(${target} PUBLIC ${OpenMP_CXX_FLAGS})
        set_property(TARGET ${target} APPEND PROPERTY LINK_FLAGS ${OpenMP_CXX_FLAGS})
    endif()
    if (UNIX)
        add_nix_libraries(${target})
    endif()
endmacro()    

if (NISTFIT_EXAMPLES)
    # Add Eigen include here since we are going to use our version of Eigen
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
    add_executable(simple ${CMAKE_CURRENT_SOURCE_DIR}/examples/simple.cpp ${APP_SOURCES})
    if (NISTFIT_NTHREAD_MAX)
        message(STATUS "Set the number of threads")
        add_definitions(-DNTHREAD_MAX=${NISTFIT_NTHREAD_MAX})
    endif()
    add_libraries(simple)
elseif (NISTFIT_STATIC_LIBRARY)
    add_library(NISTfit STATIC ${APP_SOURCES})
    # Add target include directories for easy linking with other applications
    target_include_directories(NISTfit PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_include_directories(NISTfit PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
    target_include_directories(NISTfit PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/ThreadPool2")
    add_libraries(NISTfit)
elseif (NISTFIT_PYBIND11)
    # ----------------------------
    # Build pybind11 python module
    # ----------------------------
    
    # Add Eigen include here since we are going to use our version of Eigen
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
    add_subdirectory("${CMAKE_SOURCE_DIR}/externals/pybind11")
    add_definitions(-DPYBIND11)
    pybind11_add_module(NISTfit ${APP_SOURCES} "${CMAKE_SOURCE_DIR}/src/pybind11_interface.cxx")
    add_libraries(NISTfit)
else()
    message(FATAL_ERROR "You must select a build option")
endif()

