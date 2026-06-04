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

#include <Eigen/Dense>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace mint {

struct mnist_sample {
    Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor> img;
    uint8_t lbl;
};

namespace mnist {

uint32_t read_header_and_get_count(
    std::ifstream& img_file,
    std::ifstream& lbl_file,
    const std::filesystem::path& img_path,
    const std::filesystem::path& lbl_path);

std::vector<mnist_sample> read(
    const std::filesystem::path& img_path,
    const std::filesystem::path& lbl_path,
    uint32_t offset = 0,
    uint32_t count = 1024);

void write(
    const std::filesystem::path& img_path,
    const std::filesystem::path& lbl_path,
    const std::vector<mnist_sample>& samples);

}  // namespace mnist

}  // namespace mint