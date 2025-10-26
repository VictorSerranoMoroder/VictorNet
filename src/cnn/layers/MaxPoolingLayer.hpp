#pragma once

#include <cnn/layers/BaseLayer.hpp>
#include <vector>

#include <core/Tensor.hpp>
#include <kernels/MaxPooling.hpp>
namespace cnn::layers
{
    /// @brief
    ///
    /// @details
    class MaxPoolingLayer : public BaseLayer
    {
        public:
        explicit MaxPoolingLayer(kernels::MaxPoolingScalarData scalar);

        // Copyable
        MaxPoolingLayer(const MaxPoolingLayer&) = default;
        MaxPoolingLayer& operator=(const MaxPoolingLayer&) = default;

        // No Movable
        MaxPoolingLayer(MaxPoolingLayer&&) = default;
        MaxPoolingLayer& operator=(MaxPoolingLayer&&) = default;

        virtual std::vector<core::Tensor> run_layer(std::vector<core::Tensor>&& input) override;

        void perform_max_pooling(core::Tensor&& input);

        private:
        kernels::MaxPoolingScalarData scalar_;
        std::vector<core::Tensor> activation_map_list_;
    };
}