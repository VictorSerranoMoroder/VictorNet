#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

#include <core/Tensor.hpp>
//#include <external/tracy/public/tracy/Tracy.hpp>

extern "C" {
    #define STB_IMAGE_IMPLEMENTATION
    #include "external/stb_image/stb_image.h"
    #define STB_IMAGE_WRITE_IMPLEMENTATION
    #include "external/stb_image/stb_image_write.h"
}

namespace core
{
    Tensor::Tensor(const char* fname)
    {
        int width{}, height{}, channels{};
        stbi_info(fname, &width, &height, &channels);
        std::uint8_t* data = stbi_load(fname, &width, &height, &channels, composition_type::STBI_rgb);

        width_ = width;
        height_ = height;
        channels_ = channels;

        if (data != nullptr) 
        {
            data_ = new float[get_count()];
            for (size_t i = 0; i < get_count(); ++i)
            {
                data_[i] = static_cast<float>(data[i]) / 255.0f;
            }

            stbi_image_free(data); // correctly free stb-allocated memory
        }
        else 
        {
            std::cerr << "Failed to load image: " << stbi_failure_reason() << "\n";
        }
    }

    Tensor::Tensor(float* data, std::uint32_t height, std::uint32_t width, std::uint32_t channels)
    : data_{nullptr}
    , height_ {height}
    , width_{width}
    , channels_{channels}
    {
        set_data(data, height, width, channels);
    }

    Tensor::Tensor(std::uint32_t height, std::uint32_t width, std::uint32_t channels)
    : data_{nullptr}
    , height_ {height}
    , width_{width}
    , channels_{channels}
    {
        data_ = new float[get_count()];
    }

    void Tensor::print_to_image(const char* fname) const
    {
        //ZoneScopedN("Print");
        if (data_ != nullptr)
        {
            std::uint8_t* output = new std::uint8_t[get_count()];

            for (std::size_t i = 0; i < get_count(); i++)
            {
                float val = std::clamp(data_[i], 0.0f, 1.0f);
                output[i] = static_cast<std::uint8_t>(val * 255.0f);
            }
            std::cout << "Printing Image" << std::endl;
            stbi_write_jpg(fname, width_, height_, channels_, output, 100);
        }
        else 
        {
            std::cerr << "Failed to load image: " << stbi_failure_reason() << "\n";
        }
    }

    void Tensor::set_data(float* data, std::uint32_t height, std::uint32_t width, std::uint32_t channels)
    {
        if (data != nullptr)
        {
            height_ = height;
            width_ = width;
            channels_ = channels;

            delete[] data_;

            data_ = new float[get_count()];
            for (size_t i = 0; i < get_count(); ++i)
            {
                data_[i] = static_cast<float>(data[i]) / 255.0f;
            }
        }
        else 
        {
            std::cerr << "Illegal Operation, cannot copy null data" << "\n"; 
        }
    }

    float* Tensor::get_data() const
    {
        return data_;
    }

    std::uint32_t Tensor::get_size() const
    {
        return channels_ * width_ * height_ * sizeof(float);
    }

    std::uint32_t Tensor::get_count() const
    {
        return channels_ * width_ * height_;    
    }

    std::uint32_t Tensor::get_width() const
    {
        return width_;
    }

    std::uint32_t Tensor::get_height() const
    {
        return height_;
    }

    std::uint32_t Tensor::get_channels() const
    {
        return channels_;
    }
}