#include <kernels/MaxPooling.hpp>


namespace kernels
{
    __global__ void maxpooling(float* input, float* output, MaxPoolingScalarData scalar)
    {
        int out_w = blockIdx.x * blockDim.x + threadIdx.x;
        int out_h = blockIdx.y * blockDim.y + threadIdx.y;
        int c     = blockIdx.z;

        if (out_w >= scalar.out_width || out_h >= scalar.out_height || c >= scalar.channels)
            return;

        // top-left corner of pooling window in input tensor
        int in_w_start = out_w * scalar.stride;
        int in_h_start = out_h * scalar.stride;

        float max_val{};
        for (std::size_t i = 0; i < scalar.window_dim; i++) // For each row
        {
            for (std::size_t j = 0; j < scalar.window_dim; j++) // For each pixel
            {
                int in_h = in_h_start + i;
                int in_w = in_w_start + j;

                if (in_h < scalar.in_height && in_w < scalar.in_width)
                {
                    int input_idx = (in_h * scalar.in_width + in_w) * scalar.channels + c;
                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }
        }
        int output_idx = (out_h * scalar.out_width + out_w) * scalar.channels + c;
        output[output_idx] = max_val;
    }

    core::Tensor launch_maxpooling_kernel(const core::Tensor& input,
                                MaxPoolingScalarData scalar)
    {

    }
}