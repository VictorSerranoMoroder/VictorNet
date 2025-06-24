#include <cstdint>
#include <iostream>
#include <tokens/Tensor.hpp>

extern "C" {
    #define STB_IMAGE_IMPLEMENTATION
    #include "stb_image/stb_image.h"
    #define STB_IMAGE_WRITE_IMPLEMENTATION
    #include "stb_image/stb_image_write.h"
}

namespace src::tokens
{
    Tensor::Tensor(const char* fname)
    {
        int width{}, height{}, channels{};
        stbi_info(fname, &width, &height, &channels);
        std::uint8_t *data = stbi_load(fname, &width, &height, &channels, composition_type::STBI_rgb);

        width_ = width;
        height_ = height;
        channels_ = channels;

        if (data != nullptr) 
        {
            // Allocate new data
            data_ = new float[width_ * height_ * channels_];
            // Fill data with image info
            for (size_t i = 0; i < width_ * height_ * channels_; ++i) {
                data_[i] = static_cast<float>(data[i]);
            }       
            delete (data);
        }
        else 
        {
            std::cerr << "Failed to load image: " << stbi_failure_reason() << "\n";
        }
    }

    void Tensor::print_to_image(const char* fname)
    {
        if (data_ != nullptr)
        {
            std::uint8_t* data = new std::uint8_t[width_ * height_ * channels_];
            for (size_t i = 0; i < width_ * height_ * channels_; ++i) {
                data[i] = static_cast<std::uint8_t>(data_[i]);
            }  
            stbi_write_jpg(fname, width_, height_, channels_, data, 100);
        }
    }
}