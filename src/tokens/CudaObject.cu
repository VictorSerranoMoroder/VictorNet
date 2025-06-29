
#include <vector>
#include <iostream>
#include <memory>
#include <type_traits>

template <class TClass, class TData, std::enable_if_t<std::is_trivially_copyable_v<TData>, bool> = true>
class CudaObject 
{
    public:

    CudaObject(TData* data, std::uint32_t count, std::uint64_t size)
    : host_data_ {nullptr}
    , device_data_ {nullptr}
    {
        host_data_ = new TData[count];
        if (data)
        {
            std::copy(data, data + count, host_data_);
        }
        allocate_device_memory(size);
    }

    ~CudaObject()
    {
        delete[] host_data_;
        cudaFree(device_data_);
    }

    // Not Copyable
    CudaObject(const CudaObject&) = delete;
    CudaObject& operator=(const CudaObject&) = delete;

    // Movable
    CudaObject(CudaObject&&) = default;
    CudaObject& operator=(CudaObject&&) = default;

    __host__ TData* get_host()
    {
        return host_data_;
    }

    __host__ TData* get_device()
    {
        return device_data_;
    }

    __host__ void sync_to_device() 
    {
        cudaMemcpy(device_data_, host_data_, static_cast<TClass*>(this)->get_size(), cudaMemcpyHostToDevice);
    }

    __host__ void sync_to_host() 
    {
        cudaMemcpy(host_data_, device_data_, static_cast<TClass*>(this)->get_size(), cudaMemcpyDeviceToHost);
    }

    protected:

    void allocate_device_memory(std::uint64_t size)
    {
        cudaMalloc(&device_data_, size);
        cudaMemcpy(device_data_, host_data_, size, cudaMemcpyHostToDevice);
    }

    void set_data(float* data, std::uint32_t size)
    {
        memcpy(host_data_, data, size);
        sync_to_device();
    }

    private:

    TData* host_data_;
    TData* device_data_;
};

class CudaTensor3D : public CudaObject<CudaTensor3D, float>
{
    public:

    struct ScalarData {
        int input_x;     // input width
        int input_y;     // input height
        int kernel_dim;  // kernel size (assumes square)
        int channels;    // number of channels
        int stride;      // number of indexes the kernel moves per step
    };

    CudaTensor3D(float* values, std::uint32_t c, std::uint32_t x, std::uint32_t y)
    : CudaObject<CudaTensor3D, float> {values, c*x*y, sizeof(float)*c*x*y}
    , x_ {x}
    , y_ {y}
    , c_ {c}
    {
    }

    void set_data(float* data, std::uint32_t c, std::uint32_t x, std::uint32_t y)
    {
        x_ = x;
        y_ = y;
        c_ = c;
        CudaObject::set_data(data, get_size());
    }

    float* get_data()
    {
        return CudaObject::get_host();
    }

    int get_count()
    {
        return c_ * x_ * y_;
    }

    int get_size()
    {
        return sizeof(float) * get_count();
    }

    private:
        
    std::uint32_t x_;
    std::uint32_t y_;
    std::uint32_t c_;
};

__global__ void image_convolution(float* input, float* output, float* kernel, 
                        CudaTensor3D::ScalarData scalar)
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

        output[c * (out_h * out_w) + y *out_w + x] = sum;
    }
}

__host__ void launch_conv3d_kernel(CudaObject<CudaTensor3D, float>& input, 
                            CudaObject<CudaTensor3D, float>& output, 
                            CudaObject<CudaTensor3D, float>& kernel, 
                            CudaTensor3D::ScalarData scalar)
{
    int out_width  = scalar.input_x - scalar.kernel_dim + 1;
    int out_height = scalar.input_y - scalar.kernel_dim + 1;

    dim3 blockDim(16, 16);
    dim3 gridDim((out_width + blockDim.x - 1) / blockDim.x,
                 (out_height + blockDim.y - 1) / blockDim.y,
                 scalar.channels);

    image_convolution<<<gridDim, blockDim>>>(input.get_device(), output.get_device(), kernel.get_device(), scalar);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }
    output.sync_to_host();
}

int main()
{

    float input[3 * 5 * 5] = {
        // Channel 0
        1, 2, 3, 4, 5,
        6, 7, 8, 9,10,
        11,12,13,14,15,
        16,17,18,19,20,
        21,22,23,24,25,

        // Channel 1
        10,20,30,40,50,
        15,25,35,45,55,
        20,30,40,50,60,
        25,35,45,55,65,
        30,40,50,60,70,

        // Channel 2
        5, 4, 3, 2, 1,
        10, 9, 8, 7, 6,
        15,14,13,12,11,
        20,19,18,17,16,
        25,24,23,22,21
    };
    CudaTensor3D cu_input{input, 3, 5, 5};

    float kernel[3 * 3] = {
        0, -1,  0,
        -1,  5, -1,
        0, -1,  0
    };
    CudaTensor3D cu_kernel{kernel, 3, 3, 3};

    float output[3 * 3 * 3] = {};
    CudaTensor3D cu_output{output, 3, 3, 3};

    launch_conv3d_kernel(cu_input, cu_output, cu_kernel, CudaTensor3D::ScalarData{5,5,3,3,1});
    
    std::cout << cu_output.get_data()[0] << std::endl;
    std::cout << cu_output.get_data()[9] << std::endl;
    std::cout << cu_output.get_data()[18] << std::endl;
}


