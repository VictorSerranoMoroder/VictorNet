cmake_minimum_required(VERSION 3.16)
project(VictorNet LANGUAGES CXX)
#project(VictorNet LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

add_subdirectory(src/external/tracy)


add_executable(VictorNet 
    src/MainApplication.cpp
    src/tokens/Tensor.cpp
    )
target_include_directories(VictorNet PRIVATE src/)
target_link_libraries(VictorNet PRIVATE TracyClient)
target_compile_definitions(VictorNet PRIVATE TRACY_ENABLE)