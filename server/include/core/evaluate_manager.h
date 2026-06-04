// Mint-C - Web-based digit recognition tool with C++ CNN backend
// Copyright (C) 2026 Chrollis
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <mint.h>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <thread>
#include <vector>

namespace server {

struct evaluation_status {
    enum class state_type {
        idle,
        running,
        cancelling,
        finished,
        cancelled,
        error,
    } state = state_type::idle;

    int processed_samples = 0;
    int total_samples = 0;
    double current_accuracy = 0.0;
    std::string error_message;
    double final_accuracy = 0.0;
    nlohmann::json per_class_accuracy;
};

class evaluate_manager {
public:
    static evaluate_manager& instance();

    evaluate_manager(const evaluate_manager&) = delete;
    evaluate_manager& operator=(const evaluate_manager&) = delete;
    evaluate_manager(evaluate_manager&&) = delete;
    evaluate_manager& operator=(evaluate_manager&&) = delete;

    bool start_evaluation(
        const std::string& dataset_name,
        size_t offset = 0,
        size_t count = 0,
        size_t batch_size = 32);

    void cancel_evaluation();

    evaluation_status status() const;

private:
    evaluate_manager() = default;
    ~evaluate_manager();

    void evaluation_thread_func(
        std::string dataset_name,
        size_t offset,
        size_t count,
        size_t batch_size);

    std::atomic<bool> cancel_requested_{false};
    std::unique_ptr<std::thread> worker_;
    mutable std::mutex status_mutex_;
    mutable evaluation_status status_;

    mutable double accum_correct_sum_ = 0.0;
    mutable size_t accum_count_ = 0;
};

}  // namespace server