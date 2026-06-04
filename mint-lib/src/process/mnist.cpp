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

#include <process/mnist.h>
#include <process/utils.h>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>

namespace mint {
namespace mnist {

uint32_t read_header_and_get_count(
    std::ifstream& img_file,
    std::ifstream& lbl_file,
    const std::filesystem::path& img_path,
    const std::filesystem::path& lbl_path) {
    uint32_t magic;
    img_file.read(reinterpret_cast<char*>(&magic), 4);
    if (!img_file)
        throw std::runtime_error(
            "read_header_and_get_count: Failed to read magic from image "
            "file: " +
            img_path.string());
    if (be32_to_host(magic) != 2051u)
        throw std::runtime_error(
            "read_header_and_get_count: Invalid MNIST image magic (expected "
            "2051) in " +
            img_path.string());

    lbl_file.read(reinterpret_cast<char*>(&magic), 4);
    if (!lbl_file)
        throw std::runtime_error(
            "read_header_and_get_count: Failed to read magic from label "
            "file: " +
            lbl_path.string());
    if (be32_to_host(magic) != 2049u)
        throw std::runtime_error(
            "read_header_and_get_count: Invalid MNIST label magic (expected "
            "2049) in " +
            lbl_path.string());

    uint32_t img_items, lbl_items;
    img_file.read(reinterpret_cast<char*>(&img_items), 4);
    lbl_file.read(reinterpret_cast<char*>(&lbl_items), 4);
    if (!img_file || !lbl_file)
        throw std::runtime_error(
            "read_header_and_get_count: Failed to read item counts from " +
            img_path.string() + " or " + lbl_path.string());
    img_items = be32_to_host(img_items);
    lbl_items = be32_to_host(lbl_items);
    if (img_items != lbl_items)
        throw std::runtime_error(
            "read_header_and_get_count: Image count (" +
            std::to_string(img_items) + ") and label count (" +
            std::to_string(lbl_items) + ") mismatch");

    uint32_t rows, cols;
    img_file.read(reinterpret_cast<char*>(&rows), 4);
    img_file.read(reinterpret_cast<char*>(&cols), 4);
    if (!img_file)
        throw std::runtime_error(
            "read_header_and_get_count: Failed to read image dimensions from " +
            img_path.string());

    rows = be32_to_host(rows);
    cols = be32_to_host(cols);
    if (rows != 28u || cols != 28u)
        throw std::runtime_error(
            "read_header_and_get_count: Unexpected image size " +
            std::to_string(rows) + "x" + std::to_string(cols) +
            " (expected 28x28) in " + img_path.string());
    return img_items;
}

std::vector<mnist_sample> read(
    const std::filesystem::path& img_path,
    const std::filesystem::path& lbl_path,
    uint32_t offset,
    uint32_t count) {
    std::ifstream img_file(img_path, std::ios::binary);
    std::ifstream lbl_file(lbl_path, std::ios::binary);
    if (!img_file)
        throw std::runtime_error(
            "read: Cannot open image file: " + img_path.string());
    if (!lbl_file)
        throw std::runtime_error(
            "read: Cannot open label file: " + lbl_path.string());

    uint32_t total_items =
        read_header_and_get_count(img_file, lbl_file, img_path, lbl_path);
    if (offset >= total_items) return {};

    uint32_t to_read = std::min(count, total_items - offset);
    if (to_read == 0) return {};

    const std::streamoff img_skip =
        static_cast<std::streamoff>(offset) * 28 * 28;
    const std::streamoff lbl_skip = static_cast<std::streamoff>(offset);

    img_file.seekg(img_skip, std::ios::cur);
    lbl_file.seekg(lbl_skip, std::ios::cur);
    if (!img_file)
        throw std::runtime_error(
            "read: Seek error in image file " + img_path.string());
    if (!lbl_file)
        throw std::runtime_error(
            "read: Seek error in label file " + lbl_path.string());

    std::vector<mnist_sample> samples;
    samples.reserve(to_read);

    for (uint32_t i = 0; i < to_read; ++i) {
        uint8_t lbl_byte;
        lbl_file.read(reinterpret_cast<char*>(&lbl_byte), 1);
        if (!lbl_file)
            throw std::runtime_error("read: Failed to read label data");

        if (lbl_byte > 9)
            throw std::invalid_argument(
                "read: Label " + std::to_string(lbl_byte) +
                " out of range [0,9] from " + lbl_path.string());

        Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor> img;
        img_file.read(reinterpret_cast<char*>(img.data()), 28 * 28);
        if (!img_file)
            throw std::runtime_error(
                "read: Failed to read image data at index " +
                std::to_string(offset + i) + " from " + img_path.string());

        samples.push_back({std::move(img), lbl_byte});
    }

    return samples;
}

void write(
    const std::filesystem::path& img_path,
    const std::filesystem::path& lbl_path,
    const std::vector<mnist_sample>& samples) {
    if (samples.empty()) return;
    if (samples.size() > 60000)
        throw std::overflow_error(
            "write: Too many samples for MNIST format (max 60000), got " +
            std::to_string(samples.size()));

    std::ofstream img_file(img_path, std::ios::binary);
    std::ofstream lbl_file(lbl_path, std::ios::binary);
    if (!img_file)
        throw std::runtime_error(
            "write: Cannot create image file: " + img_path.string());
    if (!lbl_file)
        throw std::runtime_error(
            "write: Cannot create label file: " + lbl_path.string());

    img_file.exceptions(std::ios::failbit | std::ios::badbit);
    lbl_file.exceptions(std::ios::failbit | std::ios::badbit);

    uint32_t magic = host_to_be32(2051u);
    img_file.write(reinterpret_cast<const char*>(&magic), 4);

    uint32_t count = host_to_be32(static_cast<uint32_t>(samples.size()));
    img_file.write(reinterpret_cast<const char*>(&count), 4);

    uint32_t rows = host_to_be32(28u);
    uint32_t cols = host_to_be32(28u);
    img_file.write(reinterpret_cast<const char*>(&rows), 4);
    img_file.write(reinterpret_cast<const char*>(&cols), 4);

    magic = host_to_be32(2049u);
    lbl_file.write(reinterpret_cast<const char*>(&magic), 4);
    lbl_file.write(reinterpret_cast<const char*>(&count), 4);

    for (const auto& s : samples) {
        if (s.lbl > 9)
            throw std::invalid_argument(
                "write: Label " + std::to_string(s.lbl) +
                " out of range [0,9]");
        uint8_t lbl_byte = s.lbl;
        lbl_file.write(reinterpret_cast<const char*>(&lbl_byte), 1);

        img_file.write(reinterpret_cast<const char*>(s.img.data()), 28 * 28);
    }

    img_file.flush();
    lbl_file.flush();
}

}  // namespace mnist
}  // namespace mint