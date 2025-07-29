#pragma once

#include <cstdint>

namespace core::device {
    /// @brief
    ///
    /// @details
    class IDeviceStorable
    {
        public:
        IDeviceStorable() = default;
    
        // Copyable
        IDeviceStorable(const IDeviceStorable&) = default;
        IDeviceStorable& operator=(const IDeviceStorable&) = default;
    
        // Movable
        IDeviceStorable(IDeviceStorable&&) = default;
        IDeviceStorable& operator=(IDeviceStorable&&) = default;

        virtual float* get_data() const = 0;

        protected:
        virtual std::uint32_t get_size() const = 0;
        virtual std::uint32_t get_count() const = 0;
    };
}