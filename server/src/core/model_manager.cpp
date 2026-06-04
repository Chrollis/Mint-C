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

#include <chrono>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

namespace server {

model_manager& model_manager::instance() {
    static model_manager mng;
    return mng;
}

bool model_manager::load_from_bin(const std::filesystem::path& path) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto& logger = server::logger::instance();

    try {
        if (!std::filesystem::exists(path)) {
            logger.error("Model binary not found: " + path.string());
            return false;
        }
        model_.load(path);
        model_ready_ = true;
        logger.info("Model loaded from " + path.string());
        return true;
    } catch (const std::exception& e) {
        logger.error(
            "Failed to load model from " + path.string() + ": " + e.what());
        model_ready_ = false;
        return false;
    }
}

bool model_manager::save_to_bin(const std::filesystem::path& path) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto& logger = server::logger::instance();

    if (!model_ready_) {
        logger.error("Cannot save model: model not ready");
        return false;
    }

    try {
        std::filesystem::create_directories(path.parent_path());
        model_.save(path);
        logger.info("Model saved to " + path.string());
        return true;
    } catch (const std::exception& e) {
        logger.error(
            "Failed to save model to " + path.string() + ": " + e.what());
        return false;
    }
}

bool model_manager::init_from_json(const nlohmann::json& config) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto& logger = server::logger::instance();

    try {
        model_ = mint::model::from_json(config);
        model_ready_ = true;
        logger.info("Model initialized from JSON");
        return true;
    } catch (const std::exception& e) {
        logger.error(
            "Failed to initialize model from JSON: " + std::string(e.what()));
        model_ready_ = false;
        return false;
    }
}

nlohmann::json model_manager::export_to_json() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (!model_ready_) {
        return nlohmann::json::object();
    }
    try {
        return model_.to_json();
    } catch (const std::exception& e) {
        auto& logger = server::logger::instance();
        logger.error("Failed to export model JSON: " + std::string(e.what()));
        return nlohmann::json::object();
    }
}

std::vector<std::pair<uint8_t, double>> model_manager::predict(
    const std::vector<std::vector<Eigen::MatrixXd>>& input) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto& logger = server::logger::instance();

    if (!model_ready_) {
        logger.warning("Predict called but model not ready");
        return {};
    }

    if (input.empty()) {
        logger.warning("Predict called with empty input");
        return {};
    }

    try {
        return model_.predict(input);
    } catch (const std::exception& e) {
        logger.error("Prediction failed: " + std::string(e.what()));
        return {};
    }
}

std::unique_ptr<mint::model> model_manager::clone_model() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto& logger = server::logger::instance();

    if (!model_ready_) {
        logger.error("Cannot clone model: model not ready");
        return nullptr;
    }

    try {
        std::filesystem::path temp_dir = config::instance().temp_dir();
        std::filesystem::create_directories(temp_dir);
        std::ostringstream oss;
        oss << "model_cloned_" << std::this_thread::get_id() << "_" << std::hex
            << std::chrono::steady_clock::now().time_since_epoch().count()
            << ".bin";
        std::filesystem::path temp_path = temp_dir / oss.str();

        model_.save(temp_path);
        auto cloned = std::make_unique<mint::model>();
        cloned->load(temp_path);
        std::filesystem::remove(temp_path);

        logger.info("Model cloned successfully");
        return cloned;
    } catch (const std::exception& e) {
        logger.error("Failed to clone model: " + std::string(e.what()));
        return nullptr;
    }
}

}  // namespace server