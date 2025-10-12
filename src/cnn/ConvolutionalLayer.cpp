
#include "utils/KaimingInitializer.hpp"
#include <kernels/Conv3D.hpp>
#include <algorithm>

#include <cnn/ConvolutionalLayer.hpp>
#include <core/Tensor.hpp>

#include <sstream>

namespace cnn
{
    ConvolutionalLayer::ConvolutionalLayer(const ConvolutionalLayerSettings& settings, std::size_t kernel_num)
    : settings_ {settings}
    , kernel_list_{kernel_num}
    {
        std::generate(
            kernel_list_.begin(),
            kernel_list_.end(),
        [&settings]()
            {
                return utils::KaimingInitializer{}.initialize_weights(settings.kernel_dim);
            }
        );
        /*
        for (std::size_t i{}; i < kernel_list_.size(); i++)
        {
            std::ostringstream name{};
            name << "Kernel" << (i + 1) << ".jpg";
            kernel_list_.at(i).print_to_image(name.str().c_str());
        }
        */
    }

    std::vector<core::Tensor> ConvolutionalLayer::perform_convolution(const core::Tensor& input)
    {
        std::vector<core::Tensor> activation_map{};
        activation_map.reserve(kernel_list_.size());

        for (const auto& kernel : kernel_list_)
        {
            activation_map.emplace_back(kernels::launch_conv3d_kernel(input, kernel,
                {
                    input.get_height(),
                    input.get_width(),
                    settings_.kernel_dim,
                    settings_.channels,
                    settings_.stride,
                    settings_.dilation,
                    settings_.padding
                }));
        }

        return activation_map;
    }

}