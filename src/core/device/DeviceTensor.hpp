#pragma once

#include <core/Tensor.hpp>
#include <core/device/CudaObject.hpp>

namespace core::device {
    /// @brief
    ///
    /// @details
    class DeviceTensor
    {
        public:
        DeviceTensor(const core::Tensor& tensor)
        : tensor_{tensor}
        , cuda_tensor_ {tensor.get_data().get(), tensor.get_count()}
        {
        }

        // Not Copyable
        DeviceTensor(const DeviceTensor&) = delete;
        DeviceTensor& operator=(const DeviceTensor&) = delete;

        // Movable
        DeviceTensor(DeviceTensor&&) = default;
        DeviceTensor& operator=(DeviceTensor&&) = default;

        const core::Tensor& get_tensor()
        {
            return tensor_;
        }

        CudaObject<float>& get_cuda_tensor()
        {
            return cuda_tensor_;
        }

        private:

        core::Tensor tensor_;
        CudaObject<float> cuda_tensor_;
    };
}