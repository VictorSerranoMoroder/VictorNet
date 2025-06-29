#include <core/device/CudaTensor.hpp>
#include <cstdint>

namespace src::core::device
{ 

    CudaTensor::CudaTensor(std::uint32_t count)
    : CudaObject<CudaTensor, std::uint8_t> {count}
    , x_ {}
    , y_ {}
    , c_ {}
    {

    }


    CudaTensor::CudaTensor(std::uint8_t* values, std::uint32_t c, std::uint32_t x, std::uint32_t y)
    : CudaObject<CudaTensor, std::uint8_t> {values, c*x*y, sizeof(std::uint8_t)*c*x*y}
    , x_ {x}
    , y_ {y}
    , c_ {c}
    {
    }

    CudaTensor::CudaTensor(core::Tensor& host_tensor)
    : CudaObject<CudaTensor, std::uint8_t> {host_tensor.get_data(), host_tensor.get_channels()*host_tensor.get_width()*host_tensor.get_height(), 
        sizeof(std::uint8_t)*host_tensor.get_channels()*host_tensor.get_width()*host_tensor.get_height()}
    , x_ {host_tensor.get_width()}
    , y_ {host_tensor.get_height()}
    , c_ {host_tensor.get_channels()}
    {
    }

    void CudaTensor::set_data(std::uint8_t* data, std::uint32_t c, std::uint32_t x, std::uint32_t y)
    {
        x_ = x;
        y_ = y;
        c_ = c;
        CudaObject::set_data(data, get_size());
    }

    std::uint8_t* CudaTensor::get_data()
    {
        return CudaObject::get_host();
    }

    int CudaTensor::get_count()
    {
        return c_ * x_ * y_;
    }

    int CudaTensor::get_size()
    {
        return sizeof(std::uint8_t) * get_count();
    }

    std::uint32_t CudaTensor::get_width() const
    {
        return x_;
    }

    std::uint32_t CudaTensor::get_height() const
    {
        return y_;
    }

    std::uint32_t CudaTensor::get_channels() const
    {
        return c_;
    }
}