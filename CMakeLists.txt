cmake_minimum_required(VERSION 3.5)
project(AADC_EXAMPLES2)    

include(../options.cmake)

if (${MSVC})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

include_directories(../3rdparty/Eigen)
include_directories(../3rdparty)

include_directories(../3rdparty/nlohmann)
add_executable(par generic.cpp)
