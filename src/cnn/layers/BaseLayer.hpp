#pragma once

#include <core/Tensor.hpp>

namespace cnn::layers
{
    /// @brief
    ///
    /// @details

    class BaseLayer
    {
        public:
        BaseLayer() = default;

        // Copyable
        BaseLayer(const BaseLayer&) = default;
        BaseLayer& operator=(const BaseLayer&) = default;

        // Movable
        BaseLayer(BaseLayer&&) = default;
        BaseLayer& operator=(BaseLayer&&) = default;

        virtual std::vector<core::Tensor> run_layer(std::vector<core::Tensor>&& input) = 0;
    };
}