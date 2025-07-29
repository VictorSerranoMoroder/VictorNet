#pragma once


#include "core/Tensor.hpp"
#include "core/device/CudaObject.hpp"
namespace core::device {
    /// @brief
    ///
    /// @details
    class DeviceTensor : public core::Tensor, public CudaObject<core::Tensor, float>
    {
        public:
        DeviceTensor(const core::Tensor& tensor)
        : core::Tensor{tensor}
        , core::device::CudaObject<core::Tensor, float>{tensor.get_data(), tensor.get_count()}
        {
        }
    
        // Copyable
        DeviceTensor(const DeviceTensor&) = default;
        DeviceTensor& operator=(const DeviceTensor&) = default;

        // Movable
        DeviceTensor(DeviceTensor&&) = default;
        DeviceTensor& operator=(DeviceTensor&&) = default;
    };
}