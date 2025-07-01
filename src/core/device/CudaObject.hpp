#pragma once
#include <type_traits>
#include <cstdint>
#include <cstring>
#include <iostream>   
#include <cuda_runtime.h>  

namespace src::core::device
{
    template <class TClass, class TData, std::enable_if_t<std::is_trivially_copyable_v<TData>, bool> = true>
    class CudaObject 
    {
        public:

        CudaObject(std::uint32_t count)
        : host_data_ {nullptr}
        , device_data_ {nullptr}
        {
            allocate_device_memory(sizeof(TData) * count);
        }
    
        CudaObject(TData* data, std::uint32_t count, std::uint64_t size)
        : host_data_ {nullptr}
        , device_data_ {nullptr}
        {
            host_data_ = new TData[count];
            if (data)
            {
                std::memcpy(host_data_, data,  size);
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
            if (device_data_)
            {
                safe_cudaMemcpy(device_data_, host_data_, static_cast<TClass*>(this)->get_size(), MemcpyType::HostToDevice);
            }
            else 
            {
                allocate_device_memory(static_cast<TClass*>(this)->get_size());
            }
        }
    
        __host__ void sync_to_host() 
        {
            if (host_data_)
            {
                safe_cudaMemcpy(host_data_, device_data_, static_cast<TClass*>(this)->get_size(), MemcpyType::DeviceToHost);
            }
            else 
            {
                host_data_ = new TData[static_cast<TClass*>(this)->get_count()];
                safe_cudaMemcpy(host_data_, device_data_, static_cast<TClass*>(this)->get_size(), MemcpyType::DeviceToHost);
            }
        }
    
        protected:
    
        void allocate_device_memory(std::uint64_t size)
        {
            safe_cudaMalloc(size);
            if (host_data_)
            {  
                safe_cudaMemcpy(device_data_, host_data_, size, MemcpyType::HostToDevice);
            }
        }
    
        void set_data(TData* data, std::uint32_t size)
        {
            std::memcpy(host_data_, data, size);
            sync_to_device();
        }
    
        private:

        enum class MemcpyType
        {
            HostToDevice,
            DeviceToHost
        };

        void safe_cudaMemcpy(TData* dest, TData* orig, std::uint64_t size, MemcpyType type)
        {
            cudaError_t err{};
            if (type == MemcpyType::HostToDevice)
            {
                std::cout << "cudaMemcpyHostToDevice" << std::endl;
                err = cudaMemcpy(dest, orig, size, cudaMemcpyHostToDevice);
            }
            else 
            {
                std::cout << "cudaMemcpyDeviceToHost" << std::endl;
                err = cudaMemcpy(dest, orig, size, cudaMemcpyDeviceToHost);
            }

            if (err != cudaSuccess) {
                std::cerr << "safe_cudaMemcpy failed: " << cudaGetErrorString(err) << "\n";
                std::exit(EXIT_FAILURE);
            }
        }

        void safe_cudaMalloc(std::uint32_t size)
        {
            cudaError_t err{};
            std::cout << "cudaMalloc" << std::endl;
            err = cudaMalloc(&device_data_, size);
            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
                std::exit(EXIT_FAILURE);
            }
        }

        TData* host_data_;
        TData* device_data_;
    };
}