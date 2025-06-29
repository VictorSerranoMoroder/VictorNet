#pragma once

#include <core/device/CudaTensor.hpp>
#include <core/device/CudaObject.hpp>


namespace src::kernels
{
    using namespace src::core::device;
    
    void launch_conv3d_kernel(CudaObject<CudaTensor, std::uint8_t>& input, 
                                CudaObject<CudaTensor, std::uint8_t>& output, 
                                CudaObject<CudaTensor, std::uint8_t>& kernel, 
                                CudaTensor::ScalarData scalar);
}