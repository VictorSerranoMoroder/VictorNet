#pragma once

#include <cstdint>

#include <core/device/IDeviceStorable.hpp>

namespace core
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
    class Tensor : public core::device::IDeviceStorable
    {
        public:
        Tensor() = delete;

        /// @brief Image serializer constructor
        ///
        /// @param[in] fname    Source image path
        Tensor(const char* fname);

        /// @brief 3D Tensor constructor
        ///
        /// @param[in] data     Serialized data that represents a tensor
        /// @param[in] width    Tensor X
        /// @param[in] height   Tensor Y
        Tensor(float* data, std::uint32_t height, std::uint32_t width, std::uint32_t channels = RGB_CHANNELS);

        
        Tensor(std::uint32_t height, std::uint32_t width, std::uint32_t channels = RGB_CHANNELS);

        // Default destructor
        ~Tensor() = default;
        
        // Copyable
        Tensor(const Tensor&) = default;
        Tensor& operator=(const Tensor&) = default;
        
        // Movable
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(Tensor&&) noexcept;

        /// @brief Generates an image from tensor internal data values
        ///
        /// @param[in] fname    Name given to the new image file
        void print_to_image(const char* fname) const;

        /// @brief Getter function for tensor element size
        ///
        /// @return     Number of bytes that are stored
        virtual std::uint32_t get_size() const override;

        /// @brief Getter function for tensor element count
        ///
        /// @return     Number of elements that form the tensor
        virtual std::uint32_t get_count() const override;

        /// @brief Getter function of width
        ///
        /// @return     width value
        std::uint32_t get_width() const;

        /// @brief Getter function of height
        ///
        /// @return     height value
        std::uint32_t get_height() const ;

        /// @brief Getter function of channels
        ///
        /// @return     channel value
        std::uint32_t get_channels() const;

        void set_data(float* data, std::uint32_t height, std::uint32_t width, std::uint32_t channels = RGB_CHANNELS);

        /// @brief Getter function for tensor data
        ///
        /// @return     Tensor data values
        virtual float* get_data() const override;

        protected:


        private:
        float* data_;
        std::uint32_t width_;
        std::uint32_t height_;
        std::uint32_t channels_;
    };
}