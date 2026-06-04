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

#include <core/config.h>
#include <core/logger.h>
#include <core/model_manager.h>
#include <core/train_manager.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>

namespace server {

train_manager& train_manager::instance() {
    static train_manager mng;
    return mng;
}

train_manager::~train_manager() {
    cancel_training();
    if (worker_ && worker_->joinable()) {
        worker_->join();
    }
}

bool train_manager::start_training(
    const std::vector<mint::mnist_sample>& samples,
    const nlohmann::json& hyperparams) {
    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        if (status_.state == training_status::state_type::running ||
            status_.state == training_status::state_type::cancelling) {
            logger::instance().error("Training already in progress");
            return false;
        }
    }

    if (samples.empty()) {
        logger::instance().error("Empty training samples");
        return false;
    }

    if (worker_ && worker_->joinable()) {
        worker_->join();
    }

    double lr = hyperparams.value("learning_rate", 0.001);
    int batch_size = hyperparams.value("batch_size", 32);
    int epochs = hyperparams.value("epochs", 10);

    if (lr <= 0 || batch_size <= 0 || epochs <= 0) {
        logger::instance().error("Invalid hyperparameters");
        return false;
    }

    auto samples_copy =
        std::make_shared<std::vector<mint::mnist_sample>>(samples);

    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        status_.state = training_status::state_type::running;
        status_.current_epoch = 0;
        status_.total_epochs = epochs;
        status_.processed_samples = 0;
        status_.total_samples = static_cast<int>(samples.size());
        status_.current_loss = 0.0;
        status_.current_accuracy = 0.0;
        status_.loss_history.clear();
        status_.accuracy_history.clear();
        status_.error_message.clear();
        accum_loss_sum_ = 0.0;
        accum_acc_sum_ = 0.0;
        accum_count_ = 0;
    }

    cancel_requested_ = false;

    worker_ = std::make_unique<std::thread>(
        &train_manager::training_thread_func, this, std::move(*samples_copy),
        hyperparams);

    logger::instance().info(
        "Training started: epochs=" + std::to_string(epochs) + ", batch_size=" +
        std::to_string(batch_size) + ", lr=" + std::to_string(lr) +
        ", samples=" + std::to_string(samples.size()));
    return true;
}

void train_manager::cancel_training() {
    if (status().state == training_status::state_type::running) {
        cancel_requested_ = true;
        logger::instance().info("Training cancel requested");
    }
}

training_status train_manager::status() const {
    std::lock_guard<std::mutex> lock(status_mutex_);

    if (accum_count_ > 0) {
        status_.current_loss = accum_loss_sum_ / accum_count_;
        status_.current_accuracy = accum_acc_sum_ / accum_count_;
        accum_loss_sum_ = 0.0;
        accum_acc_sum_ = 0.0;
        accum_count_ = 0;
    }

    return status_;
}

bool train_manager::accept_trained_model() {
    std::lock_guard<std::mutex> lock(status_mutex_);
    if (!trained_model_) {
        logger::instance().error("No trained model available to accept");
        return false;
    }

    std::filesystem::path temp_dir = config::instance().temp_dir();
    std::filesystem::create_directories(temp_dir);
    std::ostringstream oss;
    oss << "model_trained_" << std::this_thread::get_id() << "_" << std::hex
        << std::chrono::steady_clock::now().time_since_epoch().count()
        << ".bin";
    std::filesystem::path temp_path = temp_dir / oss.str();

    try {
        trained_model_->save(temp_path);
        bool ok = model_manager::instance().load_from_bin(temp_path);
        std::filesystem::remove(temp_path);
        if (ok) {
            logger::instance().info(
                "Trained model accepted and loaded into main model");
            trained_model_.reset();
            status_.state = training_status::state_type::idle;
            return true;
        } else {
            logger::instance().error(
                "Failed to load trained model from temporary file");
            return false;
        }
    } catch (const std::exception& e) {
        logger::instance().error(
            std::string("Exception while accepting model: ") + e.what());
        std::filesystem::remove(temp_path);
        return false;
    }
}

bool train_manager::discard_trained_model() {
    std::lock_guard<std::mutex> lock(status_mutex_);
    if (!trained_model_) {
        logger::instance().error("No trained model available to discard");
        return false;
    }
    trained_model_.reset();
    status_.state = training_status::state_type::idle;
    logger::instance().info("Trained model discarded");
    return true;
}

void train_manager::training_thread_func(
    const std::vector<mint::mnist_sample>& samples,
    const nlohmann::json& hyperparams) {
    double lr = hyperparams.value("learning_rate", 0.001);
    int batch_size = hyperparams.value("batch_size", 32);
    int epochs = hyperparams.value("epochs", 10);

    auto model = model_manager::instance().clone_model();
    if (!model) {
        logger::instance().info(
            "train_manager:  "
            "Failed to clone model for training");
        std::lock_guard<std::mutex> lock(status_mutex_);
        status_.state = training_status::state_type::error;
        status_.error_message = "Failed to clone model";
        return;
    }

    std::vector<int> indices(samples.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(
        static_cast<unsigned>(
            std::chrono::steady_clock::now().time_since_epoch().count()));

    auto preprocess =
        [](const Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor>& src) {
            return mint::reconstruct_to_matrixxd(src);
        };

    for (int epoch = 0; epoch < epochs && !cancel_requested_; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        double epoch_loss_sum = 0.0;
        double epoch_acc_sum = 0.0;
        int batches = 0;

        for (size_t i = 0; i < indices.size(); i += batch_size) {
            if (cancel_requested_) break;

            std::vector<std::vector<Eigen::MatrixXd>> batch_input;
            std::vector<uint8_t> batch_labels;
            size_t end = std::min(i + batch_size, indices.size());

            for (size_t j = i; j < end; ++j) {
                const auto& sample = samples[indices[j]];
                batch_input.push_back({preprocess(sample.img)});
                batch_labels.push_back(sample.lbl);
            }

            nlohmann::json result;
            try {
                result = model->train(batch_input, batch_labels, lr);
            } catch (const std::exception& e) {
                logger::instance().info(
                    "train_manager:  " + std::string("Training exception: ") +
                    e.what());
                std::lock_guard<std::mutex> lock(status_mutex_);
                status_.state = training_status::state_type::error;
                status_.error_message = e.what();
                return;
            }

            double batch_loss = result.value("loss", 0.0);
            double batch_acc = result.value("accuracy", 0.0);
            epoch_loss_sum += batch_loss * (end - i);
            epoch_acc_sum += batch_acc * (end - i);
            batches++;

            {
                std::lock_guard<std::mutex> lock(status_mutex_);
                status_.processed_samples += static_cast<int>(end - i);
                accum_loss_sum_ += batch_loss * (end - i);
                accum_acc_sum_ += batch_acc * (end - i);
                accum_count_ += (end - i);
            }
        }

        if (cancel_requested_) break;

        double epoch_loss = epoch_loss_sum / samples.size();
        double epoch_acc = epoch_acc_sum / samples.size();

        {
            std::lock_guard<std::mutex> lock(status_mutex_);
            status_.current_epoch = epoch + 1;
            status_.loss_history.push_back(epoch_loss);
            status_.accuracy_history.push_back(epoch_acc);
        }

        logger::instance().info(
            "train_manager:  "
            "Epoch " +
            std::to_string(epoch + 1) + "/" + std::to_string(epochs) +
            " | Loss: " + std::to_string(epoch_loss) +
            " | Acc: " + std::to_string(epoch_acc));
    }

    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        if (cancel_requested_) {
            status_.state = training_status::state_type::cancelled;
            logger::instance().info(
                "train_manager:  "
                "Training cancelled by user");
        } else {
            status_.state = training_status::state_type::finished;
            logger::instance().info(
                "train_manager:  "
                "Training completed successfully");
        }
    }

    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        if (!cancel_requested_) {
            trained_model_ = std::move(model);
        }
    }
}

}  // namespace server