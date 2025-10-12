#include "cnn/ConvolutionalLayer.hpp"
#include <core/Tensor.hpp>
#include <sstream>
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

void testC()
{
    core::Tensor tensor_in{"/workspaces/VictorNet/data/toyota-corolla.jpg"};
    tensor_in.print_to_image("ImageTest.jpg");
    const auto ret = cnn::ConvolutionalLayer{cnn::ConvolutionalLayerSettings{55}, 24000}.perform_convolution(tensor_in);
    return;
    for (std::size_t i{}; i < ret.size(); i++)
    {
        std::ostringstream name;
        name << "Output" << (i + 1) << ".jpg";
        ret.at(i).print_to_image(name.str().c_str());
    }
}

int main()
{
    //testA();
    testB();
    testC();

    return 0;
}