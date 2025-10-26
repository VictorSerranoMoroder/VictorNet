#pragma once

#include "cnn/layers/BaseLayer.hpp"
#include "core/Tensor.hpp"
#include "kernels/MaxPooling.hpp"
#include <unordered_map>
#include <vector>

namespace cnn::layers
{
    struct ConvolutionalLayerSettings
    {
        std::uint32_t kernel_dim;
        std::uint32_t stride;
        std::uint32_t channels = 3;
        std::uint32_t dilation = 1;
        std::uint32_t padding = 0;
    };

    /// @brief
    ///
    /// @details
    class ConvolutionalLayer : public BaseLayer
    {
        public:
        ConvolutionalLayer(const ConvolutionalLayerSettings& settings, std::size_t kernel_num);

        // Copyable
        ConvolutionalLayer(const ConvolutionalLayer&) = default;
        ConvolutionalLayer& operator=(const ConvolutionalLayer&) = default;

        // Movable
        ConvolutionalLayer(ConvolutionalLayer&&) = default;
        ConvolutionalLayer& operator=(ConvolutionalLayer&&) = default;

        virtual std::vector<core::Tensor> run_layer(std::vector<core::Tensor>&& input) override;

        void perform_convolution(core::Tensor&& input);


        private:
            ConvolutionalLayerSettings settings_;
            std::vector<core::Tensor> kernel_list_;
            std::vector<core::Tensor> activation_map_list_;
    };
}