#pragma once

#include <core/Tensor.hpp>


namespace kernels
{
    struct ConvolutionScalarData {
            std::uint32_t input_h;
            std::uint32_t input_w;
            std::uint32_t kernel_dim;
            std::uint32_t channels = 3;
            std::uint32_t stride = 1;
            std::uint32_t dilation = 1;
            std::uint32_t padding = 0;

            constexpr std::size_t get_output_height() const
            {
                return ((input_h + 2 * padding - dilation * (kernel_dim - 1) - 1) / stride) + 1;
            }

            constexpr std::size_t get_output_width() const
            {
                return ((input_w + 2 * padding - dilation * (kernel_dim - 1) - 1) / stride) + 1;
            }
        };
    
    core::Tensor launch_conv3d_kernel(const core::Tensor& input, 
                                const core::Tensor& kernel, 
                                ConvolutionScalarData scalar);
}