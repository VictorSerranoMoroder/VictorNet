#include "kernels/MaxPooling.hpp"
#include <cnn/layers/MaxPoolingLayer.hpp>

namespace cnn::layers
{

    MaxPoolingLayer::MaxPoolingLayer(kernels::MaxPoolingScalarData scalar)
    : BaseLayer{}
    , scalar_{scalar}
    {

    }

    std::vector<core::Tensor> MaxPoolingLayer::run_layer(std::vector<core::Tensor>&& input)
    {
        for (auto&& map : input)
        {
            perform_max_pooling(std::forward<core::Tensor>(map));
        }
        return activation_map_list_;
    }

    void MaxPoolingLayer::perform_max_pooling (core::Tensor&& input)
    {
        scalar_.input_h = input.get_height();
        scalar_.input_w = input.get_width();
        activation_map_list_.emplace_back(kernels::launch_maxpooling_kernel(input, scalar_));
        scalar_.input_h = 0;
        scalar_.input_w = 0;
    }
}