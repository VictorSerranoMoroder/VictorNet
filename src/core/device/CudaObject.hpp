#pragma once

#include <cstddef>
#include <type_traits>
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>

#include <core/device/IDeviceStorable.hpp>

namespace core::device
{
    /// @brief RAII wrapper for Device memory
    ///
    /// @tparam TData   Element type, must be trivially copyable
    ///
    /// @details
    /// This class manages a device buffer allocated and freed with CUDA while maintaining sync with a raw pointer to a
    /// host buffer provided on construction. It features:
    ///
    /// - Safe allocation and deallocation of Device memory.
    /// - Explicit synchronization methods to copy data between host and device.
    /// - Host memory borrowed on construction, host memory is not owned by this class.
    ///
    template <class TData, std::enable_if_t<std::is_trivially_copyable_v<TData>, bool> = true>
    class CudaObject
    {
        public:


        /// @brief Construct a CudaObject wrapping host memory and allocating device memory.
        ///
        /// @param data Pointer to host memory (must remain valid while CudaObject exists).
        /// @param count Number of elements in the buffer.
        CudaObject(TData* data, std::size_t count)
        : element_num_ {count}
        , host_data_ {data}
        , device_data_ {nullptr}
        {
            allocate_device_memory(get_byte_size());
        }

        /// @brief Destructor. Frees GPU memory allocated with cudaMalloc.
        ~CudaObject()
        {
            std::cout << "CudaObject destructor called" << std::endl;
            cudaFree(device_data_);
        }

        // Not Copyable
        CudaObject(const CudaObject&) = delete;
        CudaObject& operator=(const CudaObject&) = delete;


        /// @brief Move constructor. Transfers ownership of GPU memory.
        CudaObject(CudaObject&& other) noexcept
        : element_num_{other.element_num_}
        , host_data_{other.host_data_}
        , device_data_{other.device_data_}
        {
            other.device_data_ = nullptr;
            other.host_data_ = nullptr; // optional: avoid dangling reference
            other.element_num_ = 0;
        }


        /// @brief Move assignment. Frees current GPU memory and takes ownership from @p other.
        CudaObject& operator=(CudaObject&& other) noexcept
        {
            if (this != &other) {
                if (device_data_)
                    cudaFree(device_data_);
                element_num_ = other.element_num_;
                host_data_ = other.host_data_;
                device_data_ = other.device_data_;

                other.device_data_ = nullptr;
                other.host_data_ = nullptr;
                other.element_num_ = 0;
            }
            return *this;
        }

        /// @brief Get raw pointer to host memory.
        /// @note Caller must ensure host memory is valid.
        /// @return raw pointer to host memory.
        __host__ TData* get_host()
        {
            return host_data_;
        }

        /// @brief Get raw pointer to device memory.
        /// @return raw pointer to device memory
        __host__ TData* get_device()
        {
            return device_data_;
        }

        /// @brief Copy host buffer into device memory.
        /// @note If device memory is not yet allocated, it will be allocated.
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

        /// @brief Copy device buffer back into host memory.
        /// @note Host pointer must be valid and large enough.
        __host__ void sync_to_host()
        {
            safe_cudaMemcpy(get_byte_size(), MemcpyType::DeviceToHost);
        }

        protected:

        /// @brief Allocate device memory and upload initial host data.
        void allocate_device_memory(std::size_t size)
        {
            safe_cudaMalloc(size);
            if (host_data_)
            {
                safe_cudaMemcpy(size, MemcpyType::HostToDevice);
            }
        }

        void set_data(TData* data, std::size_t size)
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

        void safe_cudaMemcpy(std::size_t size, MemcpyType type)
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

            if (err != cudaSuccess)
            {
                std::cerr << "safe_cudaMemcpy failed: " << cudaGetErrorString(err) << "\n";
                std::exit(EXIT_FAILURE);
            }
        }

        void safe_cudaMalloc(std::size_t size)
        {
            cudaError_t err{};
            std::cout << "cudaMalloc" << std::endl;
            err = cudaMalloc(&device_data_, size);
            if (err != cudaSuccess)
            {
                std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
                std::exit(EXIT_FAILURE);
            }
        }

        std::size_t get_byte_size()
        {
            return element_num_ * sizeof(TData);
        }

        std::size_t element_num_;
        TData* host_data_;
        TData* device_data_;
    };
}