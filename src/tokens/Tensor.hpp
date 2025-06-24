#pragma once

#include <cstdint>

namespace src::tokens
{
    /// @brief
    constexpr std::uint32_t RGB_CHANNELS = 3;

    enum composition_type{
        STBI_default = 0, // only used for desired_channels
        STBI_grey       = 1,
        STBI_grey_alpha = 2,
        STBI_rgb        = 3,
        STBI_rgb_alpha  = 4
    };

    /// @brief
    ///
    /// @details
    class Tensor
    {
        public:
        Tensor() = default;

        /// @brief Image serializer constructor
        ///
        /// @param[in] fname    Source image path
        Tensor(const char* fname);

        /// @brief 3D Tensor constructor
        ///
        /// @param[in] data     Serialized data that represents a tensor
        /// @param[in] width    Tensor X
        /// @param[in] height   Tensor Y
        Tensor(float* data, std::uint32_t height, std::uint32_t width, std::uint32_t channels_ = RGB_CHANNELS);

        /// @brief 
        ///
        ~Tensor() = default;

        // Copyable
        Tensor(const Tensor&) = delete;
        Tensor& operator=(const Tensor&) = delete;

        // Movable
        Tensor(Tensor&&) = default;
        Tensor& operator=(Tensor&&) = default;

        /// @brief Generates an image from tensor internal data values
        ///
        /// @param[in] fname    Name given to the new image file
        void print_to_image(const char* fname);

        /// @brief Getter function for tensor data
        ///
        /// @return     Tensor data values
        float* get_data();

        private:

        float* data_;
        std::uint32_t width_;
        std::uint32_t height_;
        std::uint32_t channels_;
    };
}