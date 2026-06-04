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

#include <process/mnist.h>
#include <process/utils.h>
#include <Eigen/Dense>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace mint {
namespace image {

struct image_data {
    std::vector<uint8_t> px;
    int w;
    int h;
};

mnist_sample png_base64_to_mnist_sample(
    const std::string& base64_png, uint8_t label);

image_data load_image_from_file(const std::filesystem::path& filepath);

image_data load_image_from_base64(const std::string& base64_str);

bool resize_uint8_linear(
    const uint8_t* src,
    int src_w,
    int src_h,
    uint8_t* dst,
    int dst_w,
    int dst_h);

Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor> preprocess_to_mnist_size(
    const Eigen::
        Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& mat);

struct integral_image {
    std::vector<int64_t> sum;
    std::vector<int64_t> sum_sq;
    int w;
    int h;

    integral_image(const std::vector<uint8_t>& img, int width, int height);

    void block_sum(
        int x0, int y0, int x1, int y1, int64_t& out_sum, int64_t& out_sum_sq)
        const;
};

std::vector<uint8_t> sauvola_binarize(
    const std::vector<uint8_t>& img,
    int w,
    int h,
    int window_radius = -1,
    double k = 0.2,
    double R = 128.0);

void ensure_white_background(std::vector<uint8_t>& img, int w, int h);

std::vector<double> gaussian_kernel(int radius, double sigma);

std::vector<uint8_t> gaussian_blur(
    const std::vector<uint8_t>& img,
    int w,
    int h,
    int kernel_size = 3,
    double sigma = -1.0);

std::vector<uint8_t> preprocess_to_mnist_color(
    const std::vector<uint8_t>& img, int w, int h);

void split_digits(
    const image_data& gray_image,
    std::vector<image_data>& digits,
    std::vector<rectangle>& rects,
    uint8_t threshold = 128,
    double width_ratio = 1.5,
    int pad = 4);

void split_merged_digits(
    const image_data& merged_digits,
    std::vector<image_data>& digits,
    std::vector<rectangle>& rects,
    uint8_t threshold = 128,
    double height_ratio = 0.2);

void sort_digits_by_reading_order(
    std::vector<Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor>>&
        digit_matrices,
    std::vector<rectangle>& rects,
    double height_ratio = 0.25);

image_data process_from_file(
    const std::filesystem::path& filepath,
    std::vector<Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor>>&
        digit_matrices,
    std::vector<rectangle>& rects);

image_data process_from_base64(
    const std::string& base64_str,
    std::vector<Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor>>&
        digit_matrices,
    std::vector<rectangle>& rects);

}  // namespace image

}  // namespace mint