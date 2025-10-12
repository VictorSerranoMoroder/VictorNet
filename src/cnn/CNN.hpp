#pragma once

namespace cnn
{
    /// @brief
    ///
    /// @details
    class CNN
    {
        public:
        CNN() = default;

        // Copyable
        CNN(const CNN&) = delete;
        CNN& operator=(const CNN&) = delete;

        // Movable
        CNN(CNN&&) = delete;
        CNN& operator=(CNN&&) = delete;

        private:
    };
}