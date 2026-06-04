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

#include <core/logger.h>
#include <core/model_manager.h>
#include <data/image_processor.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include <stb_image_write.h>

#include <string>
#include <utility>
#include <vector>

namespace server {

std::string base64_encode(const std::vector<uint8_t>& data) {
    static const char* encode_table =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";
    std::string result;
    for (size_t i = 0; i < data.size(); i += 3) {
        int a = data[i];
        int b = (i + 1 < data.size()) ? data[i + 1] : 0;
        int c = (i + 2 < data.size()) ? data[i + 2] : 0;
        result.push_back(encode_table[(a >> 2) & 0x3F]);
        result.push_back(encode_table[((a << 4) | (b >> 4)) & 0x3F]);
        result.push_back(
            (i + 1 < data.size()) ? encode_table[((b << 2) | (c >> 6)) & 0x3F]
                                  : '=');
        result.push_back((i + 2 < data.size()) ? encode_table[c & 0x3F] : '=');
    }
    return result;
}

image_processor& image_processor::instance() {
    static image_processor prc;
    return prc;
}

std::string image_processor::mat_to_mnist_base64(
    const Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor>& mat) {
    std::vector<uint8_t> png;
    stbi_write_png_to_func(
        [](void* ctx, void* data, int size) {
            auto& vec = *reinterpret_cast<std::vector<uint8_t>*>(ctx);
            vec.insert(vec.end(), (uint8_t*)data, (uint8_t*)data + size);
        },
        &png, 28, 28, 1, mat.data(), 28);
    std::string b64 = base64_encode(png);
    return "data:image/png;base64," + b64;
}

std::pair<int, double> image_processor::predict_digit(
    const Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor>& mat) {
    Eigen::MatrixXd dmat = mint::reconstruct_to_matrixxd(mat);
    std::vector<std::vector<Eigen::MatrixXd>> input;
    input.push_back({dmat});
    auto results = model_manager::instance().predict(input);
    if (results.empty()) return {0, 0.0};
    return {results[0].first, results[0].second};
}

nlohmann::json image_processor::process_impl(
    const std::vector<uint8_t>& processed_img,
    int img_w,
    int img_h,
    const std::vector<Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor>>&
        digit_mats,
    const std::vector<mint::rectangle>& rects) {
    std::vector<int> digits;
    std::vector<double> confidences;
    std::vector<std::string> mnist_b64s;

    for (const auto& mat : digit_mats) {
        auto pred = predict_digit(mat);
        digits.push_back(pred.first);
        confidences.push_back(pred.second);
        mnist_b64s.push_back(mat_to_mnist_base64(mat));
    }

    std::vector<uint8_t> png;
    stbi_write_png_to_func(
        [](void* ctx, void* data, int size) {
            auto& vec = *reinterpret_cast<std::vector<uint8_t>*>(ctx);
            vec.insert(vec.end(), (uint8_t*)data, (uint8_t*)data + size);
        },
        &png, img_w, img_h, 1, processed_img.data(), img_w);
    std::string img_b64 = "data:image/png;base64," + base64_encode(png);

    nlohmann::json result;
    result["success"] = true;
    result["image_base64"] = img_b64;

    nlohmann::json digits_array = nlohmann::json::array();
    for (size_t i = 0; i < digits.size(); ++i) {
        nlohmann::json item;
        item["digit"] = digits[i];
        item["confidence"] = confidences[i];
        item["bbox"] = {
            {"x", rects[i].x},
            {"y", rects[i].y},
            {"w", rects[i].w},
            {"h", rects[i].h}};
        item["mnist_base64"] = mnist_b64s[i];
        digits_array.push_back(item);
    }
    result["digits"] = digits_array;
    return result;
}

nlohmann::json image_processor::process_file(
    const std::filesystem::path& filepath) {
    auto& logger = logger::instance();
    try {
        std::vector<Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor>> digit_mats;
        std::vector<mint::rectangle> rects;
        auto img_data =
            mint::image::process_from_file(filepath, digit_mats, rects);
        mint::image::sort_digits_by_reading_order(digit_mats, rects);
        return process_impl(
            img_data.px, img_data.w, img_data.h, digit_mats, rects);
    } catch (const std::exception& e) {
        logger.error("process_file exception: " + std::string(e.what()));
        return {{"success", false}, {"error", e.what()}};
    }
}

nlohmann::json image_processor::process_base64(const std::string& base64_str) {
    auto& logger = logger::instance();
    try {
        std::vector<Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor>> digit_mats;
        std::vector<mint::rectangle> rects;
        auto img_data =
            mint::image::process_from_base64(base64_str, digit_mats, rects);
        mint::image::sort_digits_by_reading_order(digit_mats, rects);
        return process_impl(
            img_data.px, img_data.w, img_data.h, digit_mats, rects);
    } catch (const std::exception& e) {
        logger.error("process_base64 exception: " + std::string(e.what()));
        return {{"success", false}, {"error", e.what()}};
    }
}

}  // namespace server