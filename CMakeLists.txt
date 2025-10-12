cmake_minimum_required(VERSION 3.18)
project(VictorNet LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

enable_language(CUDA)

# Sources
set(SOURCES
    src/cnn/CNN.cpp
    src/cnn/ConvolutionalLayer.cpp
    src/MainApplication.cpp
    src/core/Tensor.cpp
    src/kernels/Conv3D.cu
    src/tests/TestDeviceMemory.cu
)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDA REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-expt-relaxed-constexpr)
add_executable(VictorNet ${SOURCES})

# Enable separable compilation for CUDA
set_target_properties(VictorNet PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Include directories
target_include_directories(VictorNet PRIVATE
    ${CUDA_INCLUDE_DIRS}
    src
)



# Link libraries if needed (e.g., TracyClient, CUDA libraries are linked automatically)
#target_link_libraries(VictorNet PRIVATE Eigen3::Eigen)

# Add compile definitions if needed
#target_compile_definitions(VictorNet PRIVATE TRACY_ENABLE)

