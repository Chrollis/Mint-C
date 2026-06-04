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

#include <data/dataset_manager.h>

#include <core/logger.h>
#include <mint.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

namespace server {

dataset_manager& dataset_manager::instance() {
    static dataset_manager dm;
    return dm;
}

nlohmann::json dataset_manager::save_samples(
    const std::vector<mint::mnist_sample>& samples,
    const std::filesystem::path& images_path,
    const std::filesystem::path& labels_path,
    bool append) {
    auto& logger = logger::instance();
    nlohmann::json result;

    if (samples.empty()) {
        result["success"] = false;
        result["message"] = "No samples to save";
        logger.warning("save_samples called with empty samples");
        return result;
    }

    if (samples.size() > 60000) {
        result["success"] = false;
        result["message"] = "Total samples exceed 60000 limit";
        logger.error("Save samples: total exceed 60000");
        return result;
    }

    try {
        if (append && std::filesystem::exists(images_path) &&
            std::filesystem::exists(labels_path)) {
            auto existing =
                mint::mnist::read(images_path, labels_path, 0, 60000);
            if (existing.size() + samples.size() > 60000) {
                result["success"] = false;
                result["message"] = "Total samples would exceed 60000 limit";
                logger.error("Save samples: total would exceed 60000");
                return result;
            }
            existing.insert(existing.end(), samples.begin(), samples.end());
            mint::mnist::write(images_path, labels_path, existing);
            result["success"] = true;
            result["message"] = "Appended " + std::to_string(samples.size()) +
                                " samples, total " +
                                std::to_string(existing.size());
            logger.info(
                "Appended " + std::to_string(samples.size()) + " samples to " +
                images_path.string());
        } else {
            mint::mnist::write(images_path, labels_path, samples);
            result["success"] = true;
            result["message"] =
                "Saved " + std::to_string(samples.size()) + " samples";
            logger.info(
                "Saved " + std::to_string(samples.size()) + " samples to " +
                images_path.string());
        }
    } catch (const std::exception& e) {
        result["success"] = false;
        result["message"] = std::string("Save failed: ") + e.what();
        logger.error("save_samples exception: " + std::string(e.what()));
    }
    return result;
}

nlohmann::json dataset_manager::merge_datasets(
    const std::vector<std::pair<std::filesystem::path, std::filesystem::path>>&
        files,
    const std::filesystem::path& output_images_path,
    const std::filesystem::path& output_labels_path) {
    auto& logger = logger::instance();
    nlohmann::json result;
    std::vector<mint::mnist_sample> all_samples;

    if (files.empty()) {
        result["success"] = false;
        result["message"] = "No dataset pairs provided";
        return result;
    }

    for (const auto& [img_path, lbl_path] : files) {
        try {
            auto samples = mint::mnist::read(img_path, lbl_path, 0, 60000);
            all_samples.insert(
                all_samples.end(), samples.begin(), samples.end());
            logger.info(
                "Merged " + std::to_string(samples.size()) + " samples from " +
                img_path.string());
        } catch (const std::exception& e) {
            result["success"] = false;
            result["message"] =
                "Failed to read " + img_path.string() + ": " + e.what();
            logger.error("merge_datasets read error: " + std::string(e.what()));
            return result;
        }
        if (all_samples.size() > 60000) {
            result["success"] = false;
            result["message"] = "Total samples exceed 60000 limit (got " +
                                std::to_string(all_samples.size()) + ")";
            logger.error("merge_datasets total exceeds 60000");
            return result;
        }
    }

    try {
        mint::mnist::write(output_images_path, output_labels_path, all_samples);
        result["success"] = true;
        result["message"] = "Merged " + std::to_string(all_samples.size()) +
                            " samples into " + output_images_path.string();
        logger.info("Merged dataset saved to " + output_images_path.string());
    } catch (const std::exception& e) {
        result["success"] = false;
        result["message"] = std::string("Write failed: ") + e.what();
        logger.error("merge_datasets write error: " + std::string(e.what()));
    }
    return result;
}

std::vector<mint::mnist_sample> dataset_manager::load_dataset(
    const std::filesystem::path& images_path,
    const std::filesystem::path& labels_path,
    uint32_t offset,
    uint32_t count) {
    auto& logger = logger::instance();
    try {
        auto samples =
            mint::mnist::read(images_path, labels_path, offset, count);
        logger.info(
            "Loaded " + std::to_string(samples.size()) + " samples from " +
            images_path.string() + " (offset=" + std::to_string(offset) + ")");
        return samples;
    } catch (const std::exception& e) {
        logger.error("load_dataset failed: " + std::string(e.what()));
        return {};
    }
}

uint32_t dataset_manager::get_sample_count(
    const std::filesystem::path& images_path,
    const std::filesystem::path& labels_path) {
    std::ifstream img_file(images_path, std::ios::binary);
    std::ifstream lbl_file(labels_path, std::ios::binary);
    if (!img_file.is_open() || !lbl_file.is_open()) {
        logger::instance().error(
            "Cannot open files for counting: " + images_path.string() + " or " +
            labels_path.string());
        return 0;
    }
    try {
        uint32_t count = mint::mnist::read_header_and_get_count(
            img_file, lbl_file, images_path, labels_path);
        return count;
    } catch (const std::exception& e) {
        logger::instance().error(
            "get_sample_count failed: " + std::string(e.what()));
        return 0;
    }
}

}  // namespace server