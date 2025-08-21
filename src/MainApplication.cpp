#include "tests/TestDeviceMemory.hpp"
#include <core/Tensor.hpp>

#include <tests/TestDeviceMemory.hpp>
#include <kernels/Conv3D.hpp>

void testA()
{
    core::Tensor tensor_in{"/workspaces/VictorNet/data/toyota-corolla.jpg"};   
    tests::ConvolutionScalarData scalar{
        tensor_in.get_height(),
        tensor_in.get_width(),
        3,
        3,
        1,
        1,
        0
    };
    tensor_in.print_to_image("outputA.jpg");
    core::Tensor output = test_device_memory(tensor_in, scalar);
    output.print_to_image("outputB.jpg");
}

void testB()
{
    core::Tensor tensor_in{"/workspaces/VictorNet/data/toyota-corolla.jpg"};
    
    //Edge Detection
    float val_kernel_edge_rgb[3 * 3 * 3] = {
        // Row 1 (top)
        -1,  -1,  -1,     -2,  -2,  -2,     -1,  -1,  -1,
        // Row 2 (middle)
        0,   0,   0,      0,   0,   0,      0,   0,   0,
        // Row 3 (bottom)
        1,   1,   1,      2,   2,   2,      1,   1,   1
    };


    core::Tensor tensor_kernel{val_kernel_edge_rgb,3,3,3};
    kernels::ConvolutionScalarData scalar{
        tensor_in.get_height(),
        tensor_in.get_width(),
        3,
        3,
        1,
        1,
        0
    };
    kernels::launch_conv3d_kernel(tensor_in, tensor_kernel, scalar).print_to_image("outputConv.jpg");
}

int main()
{
    //testA();
    testB();

    return 0;
}