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

struct training_status {
    enum class state_type {
        idle,
        running,
        cancelling,
        finished,
        cancelled,
        error,
    } state = state_type::idle;

    int current_epoch = 0;
    int total_epochs = 0;
    int processed_samples = 0;
    int total_samples = 0;
    double current_loss = 0.0;
    double current_accuracy = 0.0;
    std::vector<double> loss_history;
    std::vector<double> accuracy_history;
    std::string error_message;
};

class train_manager {
public:
    static train_manager& instance();

    train_manager(const train_manager&) = delete;
    train_manager& operator=(const train_manager&) = delete;
    train_manager(train_manager&&) = delete;
    train_manager& operator=(train_manager&&) = delete;

    bool start_training(
        const std::vector<mint::mnist_sample>& samples,
        const nlohmann::json& hyperparams);

    void cancel_training();

    training_status status() const;

    bool accept_trained_model();

    bool discard_trained_model();

private:
    train_manager() = default;
    ~train_manager();

    void training_thread_func(
        const std::vector<mint::mnist_sample>& samples,
        const nlohmann::json& hyperparams);

    std::atomic<bool> cancel_requested_{false};
    std::unique_ptr<std::thread> worker_;
    std::unique_ptr<mint::model> trained_model_;

    mutable std::mutex status_mutex_;
    mutable training_status status_;

    mutable double accum_loss_sum_ = 0.0;
    mutable double accum_acc_sum_ = 0.0;
    mutable size_t accum_count_ = 0;
};

}  // namespace server