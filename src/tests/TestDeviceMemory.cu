#include <tests/TestDeviceMemory.hpp>

#include <core/device/CudaTensor.hpp>
#include <core/Tensor.hpp>

namespace src::tests
{

    __global__ void test_kernel(std::uint8_t* input)
    {

    }

    core::Tensor test_device_memory(core::Tensor in_tensor, ConvolutionScalarData scalar)
    {
        core::Tensor out_tensor{};
        core::device::CudaTensor cu_t{in_tensor};
        core::device::CudaTensor cu_output{in_tensor};

        out_tensor.set_data(cu_output.get_data(), cu_output.get_height(), cu_output.get_width()); 

        return out_tensor;
    }
}