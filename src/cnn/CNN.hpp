#pragma once

#include <cstddef>
#include <utility>
#include <list>
#include <type_traits>

#include <sstream>

#include "cnn/layers/BaseLayer.hpp"
#include "cnn/layers/ConvolutionalLayer.hpp"
#include "cnn/layers/MaxPoolingLayer.hpp"
#include "core/Tensor.hpp"
#include "kernels/MaxPooling.hpp"

#include <cnn/layers/ConvolutionalLayer.hpp>
namespace cnn
{
    struct RunInitializer
    {
        const core::Tensor& tensor;
    };

    /// @brief
    ///
    /// @details
    class CNN
    {
        public:
        CNN() = default;

        // Copyable
        CNN(const CNN&) = delete;
        CNN& operator=(const CNN&) = delete;

        // Movable
        CNN(CNN&&) = delete;
        CNN& operator=(CNN&&) = delete;

        template <class TClass>
        //Use SFINAE to try to substitute TClass, if that leads to false in is_base_of,
        //the compiler just ignores that overload instead of erroring out.
        std::enable_if_t<std::is_base_of_v<cnn::layers::BaseLayer, std::decay_t<TClass>>, CNN&>
        operator<<(TClass&& layer)
        {
            // Need to strip reference and possible const qualifiers using std::decay
            // to work correctly with both lvalues and rvalues.
            // This is usually found in "perfect forwarding + storage" patterns
            layers_.push_back(std::make_unique<std::decay_t<TClass>>(std::forward<TClass>(layer)));
            return *this;
        }

        CNN& operator<<(const RunInitializer& input)
        {
            run_network_from_source(input.tensor);
            return *this;
        }

        cnn::layers::ConvolutionalLayer add_conv_layer(std::uint32_t kernel_size, std::uint32_t stride, std::uint32_t kernel_num)
        {
            return cnn::layers::ConvolutionalLayer
            {
                cnn::layers::ConvolutionalLayerSettings{kernel_size, stride}
                , kernel_num
            };
        }

        cnn::layers::MaxPoolingLayer add_max_pooling_layer(std::size_t pool_size, std::size_t stride)
        {
            return cnn::layers::MaxPoolingLayer
            {
                kernels::MaxPoolingScalarData
                {
                    pool_size,
                    stride
                }
            };
        }

        RunInitializer run_from(const core::Tensor& input)
        {
            return RunInitializer{input};
        }

        private:
        void run_network_from_source(const core::Tensor& tensor)
        {
            std::vector<core::Tensor> map_list{};
            map_list.push_back(tensor);
            for (auto& layer : layers_)
            {
                map_list = layer->run_layer(std::move(map_list));
            }

            for (std::size_t i{}; i < map_list.size(); i++)
            {
                std::ostringstream name{};
                name << "Out" << (i + 1) << ".jpg";
                map_list.at(i).print_to_image(name.str().c_str());
            }
        }

        std::list<std::unique_ptr<layers::BaseLayer>> layers_;
    };
}