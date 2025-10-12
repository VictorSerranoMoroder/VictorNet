#pragma once

#include <random>

#include <core/Tensor.hpp>
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

        protected:
        std::mt19937 random_generator_{std::random_device{}()};
    };
}