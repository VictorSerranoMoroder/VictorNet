
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <list>
#include <core/Tensor.hpp>
#include <core/device/CudaTensor.hpp>
#include <vector>

#include <kernels/Conv3D.hpp>

int main()
{
    char* fname = "/workspaces/VictorNet/data/toyota-corolla.jpg";
    std::list<src::core::Tensor> l{};
    std::cout << "Filling Tensor" << std::endl;

    for (std::size_t i = 0; i < 10; i++)
    {
        l.emplace_back(fname);
    }

    std::cout << "Filling CudaTensor" << std::endl;
    std::vector<src::core::device::CudaTensor> cu_l{};
    for(auto& tensor : l)
    {
        cu_l.push_back(tensor);
    }

    std::uint8_t k_values [3 * 3] = {
        1, 1,  1,
        1,  1, 1,
        1, 1,  1
    };

    src::core::Tensor kernel{k_values, 3, 3, 1};
    src::core::device::CudaTensor cu_kernel{kernel};
    

    std::cout << "Performing Convolution" << std::endl;
    std::vector<src::core::Tensor> result{};
    for (std::size_t i = 0; i < 10; i++)
    {
        src::core::device::CudaTensor ret{
        new std::uint8_t[3 * l.begin()->get_height() * l.begin()->get_width()],
        3, 
        224, 
        224};
        src::core::device::CudaTensor::ScalarData scalar{
            224, 
            224,
            3, 
            3, 
            1};
        src::kernels::launch_conv3d_kernel(
            cu_l.at(i),
            ret, 
            cu_kernel, 
            scalar
        );
        result.emplace_back(ret.get_data(), ret.get_height(), ret.get_width());
    }
    
    for (const auto& tensor : result)
    {
        tensor.print_to_image("Ouput.png");
    }

    l.clear();
    cu_l.clear();

    return 0;
}