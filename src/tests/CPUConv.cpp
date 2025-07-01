

#include <core/Tensor.hpp>
#include <core/device/CudaTensor.hpp>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <kernels/Conv3D.hpp>

int main()
{
    std::uint8_t input[5 * 5 * 3] = {
        // Row 0
        1, 10,  5,   2, 20,  4,   3, 30,  3,   4, 40,  2,   5, 50,  1,
        // Row 1
        6, 15, 10,   7, 25,  9,   8, 35,  8,   9, 45,  7,  10, 55,  6,
        // Row 2
        11, 20, 15,  12, 30, 14,  13, 40, 13,  14, 50, 12,  15, 60, 11,
        // Row 3
        16, 25, 20,  17, 35, 19,  18, 45, 18,  19, 55, 17,  20, 65, 16,
        // Row 4
        21, 30, 25,  22, 40, 24,  23, 50, 23,  24, 60, 22,  25, 70, 21
    };

    std::uint8_t kernel[3 * 3 * 3] = {
        // Row 0
        0, 0, 0,   1, 1, 1,   0, 0, 0,
        // Row 1
        1, 1, 1,   5, 5, 5,   1, 1, 1,
        // Row 2
        0, 0, 0,   1, 1, 1,   0, 0, 0
    };

    const std::size_t input_h = 5;
    const std::size_t input_w = 5;
    const std::size_t kernel_dim = 3;
    const std::size_t channels = 3;
    const std::size_t padding = 0;
    const std::size_t dilation = 1;
    const std::size_t stride = 1;

    const std::size_t output_dim = ((input_h + 2*padding - dilation * (kernel_dim-1)-1)/stride)+1;
    
    float kernel_sum[channels]{};
    for (std::size_t h = 0; h < kernel_dim; ++h)
    for (std::size_t w = 0; w < kernel_dim; ++w)
        for (std::size_t c = 0; c < channels; ++c)
            kernel_sum[c] += kernel[(h * kernel_dim + w) * channels + c];

    std::uint8_t output[output_dim*output_dim*channels];
 
    for (std::size_t h = 0; h < output_dim; ++h) // For each row
    {
        for (std::size_t w = 0; w < output_dim; ++w) // For each px as center
        {
            for (std::size_t c = 0; c < channels; ++c) // For each channel 
            {
                const float max_possible_val = kernel_sum[c] * 255.0f;
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
                std::uint8_t normalized_val = static_cast<std::uint8_t>(255.0f * (sum - min_possible_val) / (max_possible_val - min_possible_val));
                std::cout << "Raw value:" << sum << " Normalized value:" << (int)normalized_val << std::endl;
                output[(h * output_dim + w) * channels + c] = normalized_val;
            }
        }
    }

    return 0;
}