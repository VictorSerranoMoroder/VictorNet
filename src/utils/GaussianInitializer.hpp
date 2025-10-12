#pragma once

#include <utils/IKernelWeightProvider.hpp>

namespace utils
{
    /// @brief
    ///
    /// @details
    class GaussianInitializer : public IKernelWeightProvider
    {
        public:
        GaussianInitializer() = default;

        // Copyable
        GaussianInitializer(const GaussianInitializer&) = default;
        GaussianInitializer& operator=(const GaussianInitializer&) = default;

        // Movable
        GaussianInitializer(GaussianInitializer&&) = default;
        GaussianInitializer& operator=(GaussianInitializer&&) = default;

        virtual core::Tensor initialize_weights(std::size_t dimension) override
        {
            // TODO
        }
    };
}