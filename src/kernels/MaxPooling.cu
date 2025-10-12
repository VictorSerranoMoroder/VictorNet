#include <cfloat>
#include <kernels/MaxPooling.hpp>

#include <cstddef>

#include <core/device/DeviceTensor.hpp>

namespace kernels
{
    __global__ void maxpooling(float* input, float* output, MaxPoolingScalarData scalar)
    {
        const int out_w = blockIdx.x * blockDim.x + threadIdx.x;  // width
        const int out_h = blockIdx.y * blockDim.y + threadIdx.y;  // height
        const int c     = blockIdx.z;                             // channel

        const std::size_t out_h_max = scalar.get_output_height();
        const std::size_t out_w_max = scalar.get_output_width();

        // Bounds check
        if (out_w >= out_w_max || out_h >= out_h_max || c >= scalar.channels) return;

        float max_val = input[0];

        // Pooling window
        for (std::size_t ph = 0; ph < scalar.pool_size; ++ph)
        {
            for (std::size_t pw = 0; pw < scalar.pool_size; ++pw)
            {
                int in_h = out_h * scalar.pool_size + ph;
                int in_w = out_w * scalar.pool_size + pw;

                if (in_h < scalar.input_h && in_w < scalar.input_w)
                {
                    std::size_t idx = (in_h * scalar.input_w + in_w) * scalar.channels + c;
                    max_val = fmaxf(max_val, input[idx]);
                }
            }
        }

        // Write pooled value
        std::size_t out_idx = (out_h * out_w_max + out_w) * scalar.channels + c;
        output[out_idx] = max_val;
    }

    core::Tensor launch_maxpooling_kernel(const core::Tensor& input,
                                MaxPoolingScalarData scalar)
    {
        core::device::DeviceTensor cu_input{input};

        int out_height = scalar.get_output_height();
        int out_width  = scalar.get_output_width();

        std::cout << "width:" <<out_width << std::endl;
        std::cout << "height:" <<out_height << std::endl;

        core::Tensor out_tensor{out_height, out_width, scalar.channels};
        core::device::DeviceTensor cu_output{out_tensor};

        dim3 blockDim(16, 16);
        dim3 gridDim((out_width + blockDim.x - 1) / blockDim.x,
                    (out_height + blockDim.y - 1) / blockDim.y,
                    scalar.channels);

        if (scalar.channels <= 0 || out_width <= 0 || out_height <= 0) {
            std::cerr << "Invalid kernel launch dimensions: "
                    << "channels=" << scalar.channels
                    << ", out_width=" << out_width
                    << ", out_height=" << out_height << "\n";
            std::exit(EXIT_FAILURE);
        }

        maxpooling<<<gridDim, blockDim>>>(
            cu_input.get_cuda_tensor().get_device()
            ,cu_output.get_cuda_tensor().get_device()
            ,scalar);

        cu_output.get_cuda_tensor().sync_to_host();

        return cu_output.get_tensor();
    }
}