#include "core/device/DeviceTensor.hpp"
#include <tests/TestDeviceMemory.hpp>

#include <core/device/DeviceTensor.hpp>
#include <core/Tensor.hpp>
#include <iostream>

namespace tests
{

    __global__ void test_kernel(float* input, ConvolutionScalarData scalar)
    {
        int w = blockIdx.x * blockDim.x + threadIdx.x;  // width
        int h = blockIdx.y * blockDim.y + threadIdx.y;  // height
        int c = blockIdx.z; // channel

        // (HWC format)
        int index = (h * scalar.input_w + w) * scalar.channels + c;
        input[index] = input[index] * 0.5;
    }

    core::Tensor test_device_memory(const core::Tensor& in_tensor, ConvolutionScalarData scalar)
    {
        core::device::DeviceTensor cu_t{in_tensor};

        int out_height = scalar.input_h;
        int out_width  = scalar.input_w;

        dim3 blockDim(16, 16);
        dim3 gridDim((out_width + blockDim.x - 1) / blockDim.x,
                    (out_height + blockDim.y - 1) / blockDim.y,
                    scalar.channels);

        std::cout << "Running Kernel" << std::endl;
        test_kernel<<<gridDim,blockDim>>>(cu_t.get_cuda_tensor().get_device(), scalar);

        cu_t.get_cuda_tensor().sync_to_host();

        return cu_t.get_tensor();
    }
}