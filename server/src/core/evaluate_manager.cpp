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
#include <core/evaluate_manager.h>
#include <core/logger.h>
#include <core/model_manager.h>
#include <data/dataset_manager.h>
#include <data/image_processor.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>
#include <thread>

namespace server {

evaluate_manager& evaluate_manager::instance() {
    static evaluate_manager mng;
    return mng;
}

evaluate_manager::~evaluate_manager() {
    cancel_evaluation();
    if (worker_ && worker_->joinable()) {
        worker_->join();
    }
}

bool evaluate_manager::start_evaluation(
    const std::string& dataset_name,
    size_t offset,
    size_t count,
    size_t batch_size) {
    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        if (status_.state == evaluation_status::state_type::running ||
            status_.state == evaluation_status::state_type::cancelling) {
            logger::instance().error("Evaluation already in progress");
            return false;
        }
    }

    auto& cfg = config::instance();
    std::filesystem::path ds_dir = cfg.dataset_dir() / dataset_name;
    std::filesystem::path img_path = ds_dir / "images.idx3-ubyte";
    std::filesystem::path lbl_path = ds_dir / "labels.idx1-ubyte";

    if (!std::filesystem::exists(img_path) ||
        !std::filesystem::exists(lbl_path)) {
        logger::instance().error("Dataset files missing: " + dataset_name);
        return false;
    }

    uint32_t total_samples = 0;
    try {
        total_samples =
            dataset_manager::instance().get_sample_count(img_path, lbl_path);
    } catch (const std::exception& e) {
        logger::instance().error(
            "Cannot read dataset: " + std::string(e.what()));
        return false;
    }

    if (offset >= total_samples) {
        logger::instance().error("Offset exceeds dataset size");
        return false;
    }
    if (count == 0 || offset + count > total_samples) {
        count = total_samples - offset;
    }

    if (count == 0) {
        logger::instance().error("No samples to evaluate");
        return false;
    }

    if (worker_ && worker_->joinable()) {
        worker_->join();
    }

    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        status_.state = evaluation_status::state_type::running;
        status_.processed_samples = 0;
        status_.total_samples = static_cast<int>(count);
        status_.current_accuracy = 0.0;
        status_.error_message.clear();
        status_.final_accuracy = 0.0;
        status_.per_class_accuracy = nlohmann::json::object();
        accum_correct_sum_ = 0.0;
        accum_count_ = 0;
    }

    cancel_requested_ = false;

    worker_ = std::make_unique<std::thread>(
        &evaluate_manager::evaluation_thread_func, this, dataset_name, offset,
        count, batch_size);

    logger::instance().info(
        "Evaluation started: dataset=" + dataset_name + ", offset=" +
        std::to_string(offset) + ", count=" + std::to_string(count) +
        ", batch_size=" + std::to_string(batch_size));
    return true;
}

void evaluate_manager::cancel_evaluation() {
    if (status().state == evaluation_status::state_type::running) {
        cancel_requested_ = true;
        logger::instance().info("Evaluation cancel requested");
    }
}

evaluation_status evaluate_manager::status() const {
    std::lock_guard<std::mutex> lock(status_mutex_);
    if (accum_count_ > 0) {
        status_.current_accuracy = accum_correct_sum_ / accum_count_;
        accum_correct_sum_ = 0.0;
        accum_count_ = 0;
    }
    return status_;
}

void evaluate_manager::evaluation_thread_func(
    std::string dataset_name, size_t offset, size_t count, size_t batch_size) {
    auto& cfg = config::instance();
    std::filesystem::path ds_dir = cfg.dataset_dir() / dataset_name;
    std::filesystem::path img_path = ds_dir / "images.idx3-ubyte";
    std::filesystem::path lbl_path = ds_dir / "labels.idx1-ubyte";

    std::vector<mint::mnist_sample> samples;
    try {
        samples = dataset_manager::instance().load_dataset(
            img_path, lbl_path, static_cast<uint32_t>(offset),
            static_cast<uint32_t>(count));
    } catch (const std::exception& e) {
        logger::instance().error(
            "Failed to load dataset: " + std::string(e.what()));
        std::lock_guard<std::mutex> lock(status_mutex_);
        status_.state = evaluation_status::state_type::error;
        status_.error_message = e.what();
        return;
    }

    if (samples.empty()) {
        logger::instance().error("No samples loaded");
        std::lock_guard<std::mutex> lock(status_mutex_);
        status_.state = evaluation_status::state_type::error;
        status_.error_message = "No samples loaded";
        return;
    }

    auto model = model_manager::instance().clone_model();
    if (!model) {
        logger::instance().error("Failed to clone model for evaluation");
        std::lock_guard<std::mutex> lock(status_mutex_);
        status_.state = evaluation_status::state_type::error;
        status_.error_message = "Failed to clone model";
        return;
    }

    auto preprocess =
        [](const Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor>& src) {
            return mint::reconstruct_to_matrixxd(src);
        };

    size_t total = samples.size();
    size_t processed = 0;
    double total_correct = 0.0;
    std::vector<int> class_correct(10, 0);
    std::vector<int> class_total(10, 0);

    for (size_t start = 0; start < total && !cancel_requested_;
         start += batch_size) {
        size_t end = std::min(start + batch_size, total);
        std::vector<std::vector<Eigen::MatrixXd>> batch_input;
        std::vector<uint8_t> batch_labels;
        batch_input.reserve(end - start);
        batch_labels.reserve(end - start);

        for (size_t i = start; i < end; ++i) {
            const auto& sample = samples[i];
            batch_input.push_back({preprocess(sample.img)});
            batch_labels.push_back(sample.lbl);
        }

        try {
            auto predictions = model->predict(batch_input);
            size_t batch_correct = 0;
            for (size_t j = 0; j < predictions.size(); ++j) {
                uint8_t pred = predictions[j].first;
                uint8_t true_label = batch_labels[j];
                class_total[true_label]++;
                if (pred == true_label) {
                    class_correct[true_label]++;
                    total_correct++;
                    batch_correct++;
                }
            }
            processed = end;

            {
                std::lock_guard<std::mutex> lock(status_mutex_);
                status_.processed_samples = static_cast<int>(processed);
                accum_correct_sum_ += static_cast<double>(batch_correct);
                accum_count_ += (end - start);
            }
        } catch (const std::exception& e) {
            logger::instance().error(
                "Evaluation batch error: " + std::string(e.what()));
            std::lock_guard<std::mutex> lock(status_mutex_);
            status_.state = evaluation_status::state_type::error;
            status_.error_message = e.what();
            return;
        }
    }

    if (cancel_requested_) {
        std::lock_guard<std::mutex> lock(status_mutex_);
        status_.state = evaluation_status::state_type::cancelled;
        logger::instance().info("Evaluation cancelled");
        return;
    }

    double final_accuracy = total_correct / total;
    nlohmann::json per_class = nlohmann::json::object();
    for (int i = 0; i < 10; ++i) {
        if (class_total[i] > 0) {
            per_class[std::to_string(i)] =
                static_cast<double>(class_correct[i]) / class_total[i];
        } else {
            per_class[std::to_string(i)] = nullptr;
        }
    }

    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        status_.state = evaluation_status::state_type::finished;
        status_.final_accuracy = final_accuracy;
        status_.per_class_accuracy = per_class;
    }

    logger::instance().info(
        "Evaluation finished: accuracy=" + std::to_string(final_accuracy));
}

}  // namespace server