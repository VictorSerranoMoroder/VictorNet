#pragma once

#include <random>
#include <vector>

#include <utils/IKernelWeightProvider.hpp>
namespace utils
{
    /// @brief Implements Kaiming (He) initialization for convolutional kernels.
    ///
    /// @details
    /// He initialization, also known as Kaiming Initialization addresses a common issue introduced by ReLu activations.
    /// Since ReLu sets all negative activations to 0 (`y=max(0,x)`) on average only about half of the neurons will remain active
    /// or usable. This effectively halves the network capability of feature detection and distribution.
    ///
    /// To compensate for this, Kaiming initialization scales the weight variance by a factor relative to the number of input connections:
    ///
    /// Var(Weight) = 2 / fan_in
    ///
    /// This preserves the expected variance of activations between layers and prevents gradients from vanishing or exploding during
    /// training
    ///
    /// Typical usage:
    ///   W ~ N(0, sqrt(2 / fan_in))   // Normal distribution
    ///   W ~ U(-sqrt(6 / fan_in), sqrt(6 / fan_in))   // Uniform distribution
    ///
    class KaimingInitializer : public IKernelWeightProvider
    {
        public:
        KaimingInitializer() = default;

        // Copyable
        KaimingInitializer(const KaimingInitializer&) = default;
        KaimingInitializer& operator=(const KaimingInitializer&) = default;

        // Movable
        KaimingInitializer(KaimingInitializer&&) = default;
        KaimingInitializer& operator=(KaimingInitializer&&) = default;

        virtual core::Tensor initialize_weights(std::size_t dimension) override
        {
            std::vector<float> out(dimension);

            float stddev = std::sqrt(2.0f / dimension*dimension*3);
            init_rgb_normal(out);

            return core::Tensor{out,
                static_cast<std::uint32_t>(dimension),
                static_cast<std::uint32_t>(dimension)};
        }

        private:

        void init_rgb_normal(std::vector<float>& weights, float stddev = 0.01f)
        {
            std::mt19937 gen(42);  // fixed seed for reproducibility
            std::normal_distribution<float> dist(0.0f, stddev);

            for (auto& value : weights)
            {
                value = dist(gen);
            }
        }
    };
}