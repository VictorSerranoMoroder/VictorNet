
#include <type_traits>
#include <cstdint>


namespace src::core::device
{
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
}