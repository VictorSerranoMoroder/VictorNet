#pragma once

#include <cmath>

#include <core/Tensor.hpp>

namespace kernels
{
    struct MaxPoolingScalarData
    {
        std::size_t pool_size;
        std::size_t stride;
        std::size_t input_w;
        std::size_t input_h;
        std::size_t channels = 3;
        std::uint32_t dilation = 1;
        std::uint32_t padding = 0;

        constexpr std::size_t get_output_height() const noexcept {
            return (input_h - pool_size) / stride + 1;
        }

        constexpr std::size_t get_output_width() const noexcept {
            return (input_w - pool_size) / stride + 1;
        }
    };

    core::Tensor launch_maxpooling_kernel(const core::Tensor& input,
                                MaxPoolingScalarData scalar);
}