#pragma once
#include <cstdint>

#include <core/device/CudaObject.hpp>
#include <core/Tensor.hpp>


namespace src::core::device
{
    class CudaTensor : public CudaObject<CudaTensor, std::uint8_t>
    {
        public:
    
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