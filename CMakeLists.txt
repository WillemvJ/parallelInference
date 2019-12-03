cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(parallelInference)

find_package(Torch REQUIRED)

add_executable(parallelInference parallelInference.cpp)

target_compile_options(parallelInference PRIVATE "-Wall") 
target_link_libraries(parallelInference "${TORCH_LIBRARIES}")
set_target_properties(parallelInference PROPERTIES
	CXX_STANDARD 17
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++1z")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=skylake -O3 -std=c++1z")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg") 
#set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg") 
