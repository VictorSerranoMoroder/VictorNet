#pragma once
#include <cstdint>

#include <core/device/CudaObject.hpp>
#include <core/Tensor.hpp>


namespace src::core::device
{
    class CudaTensor : public CudaObject<CudaTensor, std::uint8_t>
    {
        public:
    
        struct ScalarData {
            std::uint32_t input_x;     // input width
            std::uint32_t input_y;     // input height
            std::uint32_t kernel_dim;  // kernel size (assumes square)
            std::uint32_t channels;    // number of channels
            std::uint32_t stride;      // number of indexes the kernel moves per step
        };
    
        CudaTensor(std::uint32_t count);
        CudaTensor(std::uint8_t* values, std::uint32_t c, std::uint32_t x, std::uint32_t y);
        CudaTensor(core::Tensor& host_tensor);
    
        void set_data(std::uint8_t* data, std::uint32_t c, std::uint32_t x, std::uint32_t y);
    
        std::uint8_t* get_data();
    
        int get_count();
    
        int get_size();

        std::uint32_t get_width() const;

        std::uint32_t get_height() const;

        std::uint32_t get_channels() const;
    
        private:
        std::uint32_t x_;
        std::uint32_t y_;
        std::uint32_t c_;
    };
}