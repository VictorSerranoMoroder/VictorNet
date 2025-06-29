#include <cstdint>
#include <iostream>

#include <core/device/CudaTensor.hpp>

namespace src::kernels
{
    using namespace src::core::device;

    __global__ void image_convolution(std::uint8_t* input, std::uint8_t* output, std::uint8_t* kernel, 
                        CudaTensor::ScalarData scalar)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;  // width
        int y = blockIdx.y * blockDim.y + threadIdx.y;  // height
        int c = blockIdx.z;                             // channel

        int k_size = scalar.kernel_dim;
        int stride = scalar.stride;

        int in_w = scalar.input_x;
        int in_h = scalar.input_y;

        int out_w = (in_w - k_size) / stride + 1;
        int out_h = (in_h - k_size) / stride + 1;

        if (x < out_w && y < out_h)
        {
            float sum = 0.0f;

            for (int ky = 0; ky < k_size; ++ky) // loop over kernel rows
            {
                for (int kx = 0; kx < k_size; ++kx) // loop over kernel columns
                {
                    int in_x = x * stride + kx;
                    int in_y = y * stride + ky;

                    int input_index = c * (in_h * in_w) + in_y * in_w + in_x;
                    int kernel_index = ky * k_size + kx;

                    float val = input[input_index];
                    float k = kernel[kernel_index];
                    sum += val * k;
                }
            }

            output[c * (out_h * out_w) + y *out_w + x] = static_cast<std::uint8_t>(sum);
        }
    }

    __host__ void launch_conv3d_kernel(CudaObject<CudaTensor, std::uint8_t>& input, 
                                CudaObject<CudaTensor, std::uint8_t>& output, 
                                CudaObject<CudaTensor, std::uint8_t>& kernel, 
                                CudaTensor::ScalarData scalar)
    {
        int out_width  = scalar.input_x - scalar.kernel_dim + 1;
        int out_height = scalar.input_y - scalar.kernel_dim + 1;

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

        image_convolution<<<gridDim, blockDim>>>(input.get_device(), output.get_device(), kernel.get_device(), scalar);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
        {
            std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << "\n";
            std::exit(EXIT_FAILURE);
        }
        output.sync_to_host();
    }
}

