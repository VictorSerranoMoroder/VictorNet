#pragma once

#include <cstdint>

#include <core/device/CudaTensor.hpp>
#include <core/device/CudaObject.hpp>


namespace src::kernels
{
    using namespace src::core::device;

    struct ConvolutionScalarData {
            std::size_t input_h;
            std::size_t input_w;
            std::size_t kernel_dim;
            std::size_t channels = 3;
            std::size_t stride = 1;
            std::size_t dilation = 1;
            std::size_t padding = 0;

            std::size_t get_output_dim()
            {
                return ((input_h + 2*padding - dilation * (kernel_dim-1)-1)/stride)+1;
            }
        };
    
    void launch_conv3d_kernel(CudaObject<CudaTensor, std::uint8_t>& input, 
                                CudaObject<CudaTensor, std::uint8_t>& output, 
                                CudaObject<CudaTensor, std::uint8_t>& kernel, 
                                ConvolutionScalarData scalar);
}