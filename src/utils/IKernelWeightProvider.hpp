#pragma once

#include "core/Tensor.hpp"
namespace utils
{
    /// @brief
    ///
    /// @details
    class IKernelWeightProvider
    {
        public:
        IKernelWeightProvider() = default;

        // Copyable
        IKernelWeightProvider(const IKernelWeightProvider&) = default;
        IKernelWeightProvider& operator=(const IKernelWeightProvider&) = default;

        // Movable
        IKernelWeightProvider(IKernelWeightProvider&&) = default;
        IKernelWeightProvider& operator=(IKernelWeightProvider&&) = default;

        virtual core::Tensor initialize_weights(std::size_t dimension) = 0;
    };
}