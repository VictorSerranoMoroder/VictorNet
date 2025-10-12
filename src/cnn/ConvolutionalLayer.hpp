#pragma once

#include "core/Tensor.hpp"
#include <unordered_map>
#include <vector>

namespace cnn
{
    struct ConvolutionalLayerSettings
    {
        std::uint32_t kernel_dim;
        std::uint32_t channels = 3;
        std::uint32_t stride = 1;
        std::uint32_t dilation = 1;
        std::uint32_t padding = 0;
    };

    /// @brief
    ///
    /// @details
    class ConvolutionalLayer
    {
        public:
        ConvolutionalLayer(const ConvolutionalLayerSettings& settings, std::size_t kernel_num);

        // Copyable
        ConvolutionalLayer(const ConvolutionalLayer&) = default;
        ConvolutionalLayer& operator=(const ConvolutionalLayer&) = default;

        // Movable
        ConvolutionalLayer(ConvolutionalLayer&&) = default;
        ConvolutionalLayer& operator=(ConvolutionalLayer&&) = default;

        std::vector<core::Tensor> perform_convolution(const core::Tensor& input);

        private:
            ConvolutionalLayerSettings settings_;
            std::vector<core::Tensor> kernel_list_;
    };
}