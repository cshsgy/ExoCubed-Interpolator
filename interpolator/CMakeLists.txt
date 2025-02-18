cmake_minimum_required(VERSION 3.18)
project(CudaCumsum CUDA CXX)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

# Enable debug printing in CUDA kernels
option(ENABLE_DEBUG_PRINT "Enable debug printing in CUDA kernels" OFF)

if(ENABLE_DEBUG_PRINT)
    add_definitions(-DENABLE_DEBUG_PRINT=1)
else()
    add_definitions(-DENABLE_DEBUG_PRINT=0)
endif()

# Add executable
add_executable(test
    test.cu
    interpolator_forw.cu
)

target_link_libraries(test PRIVATE 
    ${TORCH_LIBRARIES}
    cublas
)

# Enable separable compilation
set_target_properties(test PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)