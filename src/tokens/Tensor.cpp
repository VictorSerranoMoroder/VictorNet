#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <tokens/Tensor.hpp>
#include <tracy/Tracy.hpp>

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
        ZoneScopedN("ParseImage");
        int width{}, height{}, channels{};
        stbi_info(fname, &width, &height, &channels);
        std::uint8_t* data = stbi_load(fname, &width, &height, &channels, composition_type::STBI_rgb);

        width_ = width;
        height_ = height;
        channels_ = channels;

        if (data != nullptr) 
        {
            data_ = std::unique_ptr<std::uint8_t[]>{new std::uint8_t[get_value_count()]};
            std::memcpy(data_.get(), data, get_value_count());

            stbi_image_free(data); // correctly free stb-allocated memory
        }
        else 
        {
            std::cerr << "Failed to load image: " << stbi_failure_reason() << "\n";
        }
    }

    Tensor::Tensor(std::uint8_t* data, std::uint32_t height, std::uint32_t width, std::uint32_t channels_)
    : data_{nullptr}
    , height_ {height}
    , width_{width}
    , channels_{channels_}
    {
        // data_ = new float[width_ * height_ * channels_];
        // // Fill data with image info
        // for (size_t i = 0; i < width_ * height_ * channels_; ++i) {
        //     data_[i] = static_cast<float>(data[i]);
        // }       
        // delete (data);
    }

    Tensor::Tensor(const Tensor& r)
    : data_ {}
    , channels_ {r.channels_}
    , height_ {r.height_}
    , width_ {r.width_}
    {
        data_ = std::unique_ptr<std::uint8_t[]>{ new std::uint8_t[get_value_count()]};
        std::memcpy(data_.get(), r.data_.get(), get_value_count());
    }

    Tensor& Tensor::operator=(const Tensor& r)
    {
        // Self-assignment guard
        if (this == &r) return *this;

        channels_ = r.channels_;
        height_ = r.height_;
        width_ = r.width_;
        data_ = std::unique_ptr<std::uint8_t[]>{ new std::uint8_t[get_value_count()]};
        std::memcpy(data_.get(), r.data_.get(), get_value_count());
        return *this;
    }

    Tensor::Tensor(Tensor&& other) noexcept
        : data_{std::move(other.data_)}
        , channels_{other.channels_}
        , height_{other.height_}
        , width_{other.width_}
    {
        other.channels_ = other.height_ = other.width_ = 0;
    }

    Tensor& Tensor::operator=(Tensor&& other) noexcept
    {
        if (this == &other) return *this;

        data_ = std::move(other.data_);
        channels_ = other.channels_;
        height_ = other.height_;
        width_ = other.width_;

        other.channels_ = other.height_ = other.width_ = 0;
        
        return *this;
    }

    void Tensor::print_to_image(const char* fname) const
    {
        ZoneScopedN("Print");
        if (data_ != nullptr)
        {
            auto d = reinterpret_cast<std::uint8_t*>(data_.get());
            stbi_write_jpg(fname, width_, height_, channels_, d, 100);
        }
        else 
        {
            std::cerr << "Failed to load image: " << stbi_failure_reason() << "\n";
        }
    }

    std::uint8_t* Tensor::get_data() 
    {
        return data_.get();
    }

    std::uint32_t Tensor::get_value_count()
    {
        return channels_ * width_ * height_;    
    }

    std::uint32_t Tensor::get_width()
    {
        return width_;
    }

    std::uint32_t Tensor::get_height()
    {
        return height_;
    }

    std::uint32_t Tensor::get_channels()
    {
        return channels_;
    }
}