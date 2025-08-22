#pragma once

#include <core/Tensor.hpp>

namespace kernels
{
    struct MaxPoolingScalarData
    {
        std::uint32_t input_h;
        std::uint32_t input_w;
        std::uint32_t window_dim;
        std::uint32_t channels = 3;
        std::uint32_t stride = 1;
        std::uint32_t dilation = 1;
        std::uint32_t padding = 0;
    };


    core::Tensor launch_maxpooling_kernel(const core::Tensor& input,
                                MaxPoolingScalarData scalar);
}