#pragma once
#include <type_traits>
#include <cstdint>
#include <cstring>
#include <iostream>   
#include <cuda_runtime.h>  

#include <core/device/IDeviceStorable.hpp>

namespace core::device
{
    template <class TClass, class TData, std::enable_if_t<std::is_trivially_copyable_v<TData> 
        && std::is_base_of_v<IDeviceStorable, TClass>, bool> = true>
    class CudaObject 
    {
        public:
    
        CudaObject(TData* data, std::uint32_t count)
        : element_num_ {count} 
        , host_data_ {data}
        , device_data_ {nullptr}
        {
            allocate_device_memory(get_byte_size());
        }
    
        ~CudaObject()
        {
            std::cout << "CudaObject destructor called" << std::endl;
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
                safe_cudaMemcpy(get_byte_size(), MemcpyType::HostToDevice);
            }
            else 
            {
                allocate_device_memory(get_byte_size());
            }
        }
    
        __host__ void sync_to_host() 
        {
            if (host_data_)
            {
                safe_cudaMemcpy(get_byte_size(), MemcpyType::DeviceToHost);
            }
            else 
            {
                host_data_ = new TData[element_num_];
                safe_cudaMemcpy(get_byte_size(), MemcpyType::DeviceToHost);
            }
        }
    
        protected:
    
        void allocate_device_memory(std::uint64_t size)
        {
            safe_cudaMalloc(size);
            if (host_data_)
            {  
                safe_cudaMemcpy(size, MemcpyType::HostToDevice);
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

        void safe_cudaMemcpy(std::uint64_t size, MemcpyType type)
        {
            cudaError_t err{};
            if (type == MemcpyType::HostToDevice)
            {
                std::cout << "cudaMemcpyHostToDevice" << std::endl;
                err = cudaMemcpy(device_data_, host_data_, size, cudaMemcpyHostToDevice);
            }
            else 
            {
                std::cout << "cudaMemcpyDeviceToHost" << std::endl;
                err = cudaMemcpy(host_data_, device_data_, size, cudaMemcpyDeviceToHost);
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

        std::uint32_t get_byte_size()
        {
            return element_num_ * sizeof(TData);
        }

        TClass* get_derived()
        {
            return static_cast<TClass*>(*this);
        }

        std::uint32_t element_num_;
        TData* host_data_;
        TData* device_data_;
    };
}