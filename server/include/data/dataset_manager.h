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
#include <Eigen/Dense>
#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <vector>

namespace server {

class dataset_manager {
public:
    static dataset_manager& instance();

    nlohmann::json save_samples(
        const std::vector<mint::mnist_sample>& samples,
        const std::filesystem::path& images_path,
        const std::filesystem::path& labels_path,
        bool append = false);

    nlohmann::json merge_datasets(
        const std::vector<
            std::pair<std::filesystem::path, std::filesystem::path>>& files,
        const std::filesystem::path& output_images_path,
        const std::filesystem::path& output_labels_path);

    std::vector<mint::mnist_sample> load_dataset(
        const std::filesystem::path& images_path,
        const std::filesystem::path& labels_path,
        uint32_t offset = 0,
        uint32_t count = 60000);

    uint32_t get_sample_count(
        const std::filesystem::path& images_path,
        const std::filesystem::path& labels_path);

private:
    dataset_manager() = default;
};

}  // namespace server