
#include <kernels/Conv3D.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>

#include <core/device/CudaTensor.hpp>

namespace src::kernels
{
    __global__ void image_convolution(std::uint8_t* input, std::uint8_t* output, std::uint8_t* kernel, 
                                    ConvolutionScalarData scalar)
    {
        int w = blockIdx.x * blockDim.x + threadIdx.x;  // width
        int h = blockIdx.y * blockDim.y + threadIdx.y;  // height
        int c = blockIdx.z; // channel

        const std::size_t input_h = scalar.input_h;
        const std::size_t input_w = scalar.input_w;
        const std::size_t kernel_dim = scalar.kernel_dim;
        const std::size_t channels = scalar.channels;
        const std::size_t padding = 0;  // Not supported atm
        const std::size_t dilation = scalar.dilation;
        const std::size_t stride = scalar.stride;

        const std::size_t output_dim = ((input_h + 2*padding - dilation * (kernel_dim-1)-1)/stride)+1;
    
        float kernel_sum{};
        for (std::size_t kh = 0; kh < kernel_dim; ++kh)
        {
            for (std::size_t kw = 0; kw < kernel_dim; ++kw)
            {
                kernel_sum += kernel[(kh * kernel_dim + kw) * channels + c];
            }
        }

        float max_possible_val = kernel_sum * 255.0f;
        if (fabs(max_possible_val) < 1e-5f) max_possible_val = 1.0f;  // avoid div/0
        const std::size_t min_possible_val = 0;
        float sum{};

        // Run through kernel
        for (std::size_t kernel_h = 0; kernel_h < kernel_dim; ++kernel_h) // For each kernel row
        {
            for (std::size_t kernel_w = 0; kernel_w < kernel_dim; ++kernel_w) // For each kernel value
            {
                int current_h = h * stride + kernel_h * dilation - padding; // Compute the corresponding input row 
                int current_w = w * stride + kernel_w * dilation - padding; // Compute the corresponding input column 
                
                // Bounds check to avoid out-of-bounds memory access.
                if (current_h >= 0 && current_h < input_h && current_w >= 0 && current_w < (int)input_w)
                {
                    // Compute the linear index into the input array (HWC format)
                    std::size_t input_idx = (current_h * input_w + current_w) * channels + c;
                    // Compute the linear index into the kernel array (HWC format)
                    std::size_t kernel_idx = (kernel_h * kernel_dim + kernel_w) * channels + c;
                    // Accumulate the convolution result
                    sum += input[input_idx] * kernel[kernel_idx];
                }
            }
        }
        if (w < output_dim && h < output_dim && c < channels)
        {
            std::uint8_t normalized_val = static_cast<std::uint8_t>(255.0f * (sum - min_possible_val) / (max_possible_val - min_possible_val));
            std::size_t out_idx = (h * output_dim * channels) + (w * channels) + c;
            output[out_idx] = normalized_val;
        }
    }

    __host__ src::core::Tensor launch_conv3d_kernel(
            src::core::Tensor& input, 
            src::core::Tensor& kernel, 
            ConvolutionScalarData scalar
        )
    {
        core::device::CudaTensor cu_input{input};
        core::device::CudaTensor cu_output{scalar.get_output_dim() * scalar.get_output_dim() * scalar.channels};
        core::device::CudaTensor cu_kernel{kernel};

        int out_height = scalar.input_h - scalar.kernel_dim + 1;
        int out_width  = scalar.input_w - scalar.kernel_dim + 1;

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

        image_convolution<<<gridDim, blockDim>>>(cu_input.get_device(), cu_output.get_device(), cu_kernel.get_device(), scalar);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
        {
            std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << "\n";
            std::exit(EXIT_FAILURE);
        }
        
        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            std::cerr << "CUDA sync error: " << cudaGetErrorString(sync_err) << "\n";
            std::exit(EXIT_FAILURE);
        }

        cu_output.sync_to_host();

        core::Tensor output{
            cu_output.get_data(),
            static_cast<uint32_t>(scalar.get_output_dim()), 
            static_cast<uint32_t>(scalar.get_output_dim()), scalar.channels};
        return output;
    }
}

