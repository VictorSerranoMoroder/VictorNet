#pragma once

#include <cstdint>
#include <memory>

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
        Tensor(std::uint8_t* data, std::uint32_t height, std::uint32_t width, std::uint32_t channels_ = RGB_CHANNELS);

        
        // Default destructor
        ~Tensor() = default;
        
        // Copyable
        Tensor(const Tensor&);
        Tensor& operator=(const Tensor&);
        
        // Movable
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(Tensor&&) noexcept;

        /// @brief Generates an image from tensor internal data values
        ///
        /// @param[in] fname    Name given to the new image file
        void print_to_image(const char* fname) const;

        /// @brief Getter function for tensor data
        ///
        /// @warning This function should ONLY be used with measure as it breaks ptr ownership of std::unique_ptr
        ///
        /// @return     Tensor data values
        std::uint8_t* get_data();

        /// @brief Getter function for tensor value count
        ///
        /// @return     Number of elements that form the tensor
        std::uint32_t get_value_count();

        /// @brief Getter function of width
        ///
        /// @return     width value
        std::uint32_t get_width();

        /// @brief Getter function of height
        ///
        /// @return     height value
        std::uint32_t get_height();

        /// @brief Getter function of channels
        ///
        /// @return     channel value
        std::uint32_t get_channels();

        private:

        std::unique_ptr<std::uint8_t[]> data_;
        std::uint32_t width_;
        std::uint32_t height_;
        std::uint32_t channels_;
    };
}