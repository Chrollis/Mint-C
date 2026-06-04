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
#include <string>
#include <utility>
#include <vector>

namespace server {

std::string base64_encode(const std::vector<uint8_t>& data);

class image_processor {
public:
    static image_processor& instance();

    nlohmann::json process_file(const std::filesystem::path& filepath);

    nlohmann::json process_base64(const std::string& base64_str);

private:
    image_processor() = default;

    nlohmann::json process_impl(
        const std::vector<uint8_t>& original_img,
        int img_w,
        int img_h,
        const std::vector<Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor>>&
            digit_mats,
        const std::vector<mint::rectangle>& rects);

    std::string mat_to_mnist_base64(
        const Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor>& mat);

    std::pair<int, double> predict_digit(
        const Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor>& mat);
};

}  // namespace server