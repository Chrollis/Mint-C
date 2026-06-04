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

#include <process/image.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include <stb_image.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace mint {
namespace image {

mnist_sample png_base64_to_mnist_sample(
    const std::string& base64_png, uint8_t label) {
    if (label > 9) {
        throw std::invalid_argument("Label must be in range [0,9]");
    }

    std::string data_url = base64_png;
    size_t comma = data_url.find(',');
    if (comma != std::string::npos) {
        data_url = data_url.substr(comma + 1);
    }
    std::vector<uint8_t> png_data = mint::base64_decode(data_url);

    int w = 0, h = 0, channels = 0;
    std::unique_ptr<uint8_t, decltype(&stbi_image_free)> img(
        stbi_load_from_memory(
            png_data.data(), static_cast<int>(png_data.size()), &w, &h,
            &channels, 1),
        stbi_image_free);

    if (!img) {
        throw std::runtime_error("Failed to decode PNG from base64");
    }

    if (w != 28 || h != 28) {
        throw std::invalid_argument(
            "PNG must be 28x28 pixels, got " + std::to_string(w) + "x" +
            std::to_string(h));
    }

    Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor> mat;
    std::memcpy(mat.data(), img.get(), 28 * 28 * sizeof(uint8_t));

    return {mat, label};
}

image_data load_image_from_file(const std::filesystem::path& filepath) {
    image_data result;
    int w, h, c;
    std::unique_ptr<uint8_t, decltype(&stbi_image_free)> data(
        stbi_load(filepath.string().c_str(), &w, &h, &c, 1), stbi_image_free);
    if (!data) {
        throw std::runtime_error(
            "load_image_from_file: Failed to load image from " +
            filepath.string() + " - " + stbi_failure_reason());
    }
    result.w = w;
    result.h = h;
    if (w <= 0 || h <= 0)
        throw std::invalid_argument(
            "load_image_from_file: Image size must be > 0 but from " +
            filepath.string());
    result.px.assign(data.get(), data.get() + w * h);
    return result;
}

image_data load_image_from_base64(const std::string& base64_str) {
    std::vector<uint8_t> raw_data = base64_decode(base64_str);
    image_data result;
    int w, h, c;
    std::unique_ptr<uint8_t, decltype(&stbi_image_free)> data(
        stbi_load_from_memory(
            raw_data.data(), static_cast<int>(raw_data.size()), &w, &h, &c, 1),
        stbi_image_free);
    if (!data) {
        throw std::runtime_error(
            std::string(
                "load_image_from_base64: Failed to decode base64 image - ") +
            stbi_failure_reason());
    }
    result.w = w;
    result.h = h;
    if (w <= 0 || h <= 0)
        throw std::invalid_argument(
            "load_image_from_base64: Image size must be > 0");
    result.px.assign(data.get(), data.get() + w * h);
    return result;
}

bool resize_uint8_linear(
    const uint8_t* src,
    int src_w,
    int src_h,
    uint8_t* dst,
    int dst_w,
    int dst_h) {
    if (!src || !dst || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        return false;
    }
    float scale_x = static_cast<float>(src_w) / dst_w;
    float scale_y = static_cast<float>(src_h) / dst_h;

    for (int dy = 0; dy < dst_h; ++dy) {
        for (int dx = 0; dx < dst_w; ++dx) {
            float sx = (dx + 0.5f) * scale_x - 0.5f;
            float sy = (dy + 0.5f) * scale_y - 0.5f;

            sx = std::max(0.0f, std::min(sx, static_cast<float>(src_w - 1)));
            sy = std::max(0.0f, std::min(sy, static_cast<float>(src_h - 1)));

            int x0 = static_cast<int>(sx);
            int x1 = std::min(x0 + 1, src_w - 1);
            int y0 = static_cast<int>(sy);
            int y1 = std::min(y0 + 1, src_h - 1);

            float u = sx - x0;
            float v = sy - y0;

            float v00 = src[y0 * src_w + x0];
            float v01 = src[y1 * src_w + x0];
            float v10 = src[y0 * src_w + x1];
            float v11 = src[y1 * src_w + x1];

            float value = (1 - u) * (1 - v) * v00 + (1 - u) * v * v01 +
                          u * (1 - v) * v10 + u * v * v11;

            dst[dy * dst_w + dx] = static_cast<uint8_t>(value + 0.5f);
        }
    }

    return true;
}

Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor> preprocess_to_mnist_size(
    const Eigen::
        Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& mat) {
    int src_w = static_cast<int>(mat.cols());
    int src_h = static_cast<int>(mat.rows());
    if (src_w <= 0 || src_h <= 0) {
        throw std::invalid_argument(
            "preprocess_to_mnist_size: empty image (src_w=" +
            std::to_string(src_w) + ", src_h=" + std::to_string(src_h) + ")");
    }

    double scale = 20.0 / std::max(src_w, src_h);
    int dst_w = static_cast<int>(src_w * scale);
    int dst_h = static_cast<int>(src_h * scale);
    dst_w = std::max(1, dst_w);
    dst_h = std::max(1, dst_h);

    std::vector<uint8_t> src_pixels(src_w * src_h);
    for (int y = 0; y < src_h; ++y)
        for (int x = 0; x < src_w; ++x) src_pixels[y * src_w + x] = mat(y, x);

    std::vector<uint8_t> scaled(dst_w * dst_h);

    if (!resize_uint8_linear(
            src_pixels.data(), src_w, src_h, scaled.data(), dst_w, dst_h)) {
        throw std::runtime_error(
            "preprocess_to_mnist_size: failed to resize from " +
            std::to_string(src_w) + "x" + std::to_string(src_h) + " to " +
            std::to_string(dst_w) + "x" + std::to_string(dst_h));
    }

    Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor> result =
        Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor>::Zero();
    int offset_x = (28 - dst_w) / 2;
    int offset_y = (28 - dst_h) / 2;
    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            result(offset_y + y, offset_x + x) = scaled[y * dst_w + x];
        }
    }
    return result;
}

integral_image::integral_image(
    const std::vector<uint8_t>& img, int width, int height)
    : w(width),
      h(height),
      sum((width + 1) * (height + 1), 0),
      sum_sq((width + 1) * (height + 1), 0) {
    for (int y = 0; y < h; ++y) {
        int64_t row_sum = 0, row_sum_sq = 0;
        for (int x = 0; x < w; ++x) {
            int idx = y * w + x;
            uint8_t val = img[idx];
            row_sum += val;
            row_sum_sq += val * val;
            int pos = (y + 1) * (w + 1) + (x + 1);
            sum[pos] = sum[y * (w + 1) + (x + 1)] + row_sum;
            sum_sq[pos] = sum_sq[y * (w + 1) + (x + 1)] + row_sum_sq;
        }
    }
}

void integral_image::block_sum(
    int x0, int y0, int x1, int y1, int64_t& out_sum, int64_t& out_sum_sq)
    const {
    x0 = std::max(0, x0);
    y0 = std::max(0, y0);
    x1 = std::min(w, x1);
    y1 = std::min(h, y1);

    int64_t A = sum[y0 * (w + 1) + x0];
    int64_t B = sum[y0 * (w + 1) + x1];
    int64_t C = sum[y1 * (w + 1) + x0];
    int64_t D = sum[y1 * (w + 1) + x1];
    out_sum = D - B - C + A;

    int64_t Asq = sum_sq[y0 * (w + 1) + x0];
    int64_t Bsq = sum_sq[y0 * (w + 1) + x1];
    int64_t Csq = sum_sq[y1 * (w + 1) + x0];
    int64_t Dsq = sum_sq[y1 * (w + 1) + x1];
    out_sum_sq = Dsq - Bsq - Csq + Asq;
}

std::vector<uint8_t> sauvola_binarize(
    const std::vector<uint8_t>& img,
    int w,
    int h,
    int window_radius,
    double k,
    double R) {
    if (window_radius <= 0) {
        int short_side = std::min(w, h);
        window_radius = short_side / 8;
        window_radius = std::clamp(window_radius, 5, 31);
        if (window_radius % 2 == 0) window_radius++;
    }
    int window_size = 2 * window_radius + 1;

    integral_image integral(img, w, h);
    std::vector<uint8_t> result(w * h, 0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int x0 = x - window_radius;
            int y0 = y - window_radius;
            int x1 = x + window_radius + 1;
            int y1 = y + window_radius + 1;

            int64_t sum, sum_sq;
            integral.block_sum(x0, y0, x1, y1, sum, sum_sq);
            int area = (x1 - x0) * (y1 - y0);
            double mean = static_cast<double>(sum) / area;
            double stddev =
                std::sqrt((static_cast<double>(sum_sq) / area) - mean * mean);

            double threshold = mean * (1.0 + k * (stddev / R - 1.0));
            result[y * w + x] = (img[y * w + x] >= threshold) ? 255 : 0;
        }
    }
    return result;
}

void ensure_white_background(std::vector<uint8_t>& img, int w, int h) {
    if (w * h == 0) return;
    std::vector<uint8_t> sorted = img;
    std::nth_element(
        sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
    uint8_t median = sorted[sorted.size() / 2];
    if (median < 128) {
        for (uint8_t& v : img) v = static_cast<uint8_t>(255) - v;
    }
}

std::vector<double> gaussian_kernel(int radius, double sigma) {
    int size = 2 * radius + 1;
    std::vector<double> kernel(size);
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        double val = std::exp(-(i * i) / (2.0 * sigma * sigma));
        kernel[i + radius] = val;
        sum += val;
    }

    for (double& v : kernel) v /= sum;
    return kernel;
}

static double auto_adjust_sigma(int kernel_size) {
    return 0.3 * (kernel_size / 2.0 - 1.0) + 0.8;
}

static void horizontal_convolve(
    const std::vector<uint8_t>& src,
    std::vector<double>& dst,
    int w,
    int h,
    const std::vector<double>& kernel,
    int radius) {
    dst.assign(w * h, 0.0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double sum = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                int sx = x + k;
                if (sx < 0)
                    sx = -sx - 1;
                else if (sx >= w)
                    sx = w - (sx - w) - 1;
                uint8_t val = src[y * w + sx];
                sum += val * kernel[k + radius];
            }
            dst[y * w + x] = sum;
        }
    }
}

static void vertical_convolve(
    const std::vector<double>& src,
    std::vector<uint8_t>& dst,
    int w,
    int h,
    const std::vector<double>& kernel,
    int radius) {
    dst.assign(w * h, 0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double sum = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                int sy = y + k;
                if (sy < 0)
                    sy = -sy - 1;
                else if (sy >= h)
                    sy = h - (sy - h) - 1;
                double val = src[sy * w + x];
                sum += val * kernel[k + radius];
            }
            dst[y * w + x] = static_cast<uint8_t>(std::clamp(sum, 0.0, 255.0));
        }
    }
}

std::vector<uint8_t> gaussian_blur(
    const std::vector<uint8_t>& img,
    int w,
    int h,
    int kernel_size,
    double sigma) {
    if (kernel_size % 2 == 0) kernel_size++;
    int radius = kernel_size / 2;
    if (sigma <= 0.0) sigma = auto_adjust_sigma(kernel_size);

    std::vector<double> kernel1D = gaussian_kernel(radius, sigma);

    std::vector<double> temp(w * h);
    std::vector<uint8_t> result(w * h);

    horizontal_convolve(img, temp, w, h, kernel1D, radius);
    vertical_convolve(temp, result, w, h, kernel1D, radius);

    return result;
}

std::vector<uint8_t> preprocess_to_mnist_color(
    const std::vector<uint8_t>& img, int w, int h) {
    if (w <= 0 || h <= 0)
        throw std::runtime_error(
            "preprocess_to_mnist_color: invalid image dimensions (w=" +
            std::to_string(w) + ", h=" + std::to_string(h) + ")");
    std::vector<uint8_t> blurred = gaussian_blur(img, w, h);
    ensure_white_background(blurred, w, h);
    auto binary = sauvola_binarize(blurred, w, h);
    for (auto& b : binary) b = static_cast<uint8_t>(255) - b;
    return binary;
}

void split_digits(
    const image_data& gray_image,
    std::vector<image_data>& digits,
    std::vector<rectangle>& rects,
    uint8_t threshold,
    double width_ratio,
    int pad) {
    const int w = gray_image.w;
    const int h = gray_image.h;
    if (w <= 0 || h <= 0) {
        throw std::invalid_argument("split_digits: invalid image dimensions");
    }
    const int total_pixels = w * h;
    if (total_pixels == 0) return;

    std::vector<int> labels(total_pixels, -1);
    union_find uf;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = y * w + x;
            if (gray_image.px[idx] < threshold) continue;

            int left_label = -1, up_label = -1, upleft_label = -1,
                upright_label = -1;
            if (x > 0 && labels[idx - 1] >= 0) left_label = labels[idx - 1];
            if (y > 0 && labels[idx - w] >= 0) up_label = labels[idx - w];
            if (y > 0 && x > 0 && labels[idx - w - 1] >= 0)
                upleft_label = labels[idx - w - 1];
            if (y > 0 && x + 1 < w && labels[idx - w + 1] >= 0)
                upright_label = labels[idx - w + 1];

            std::vector<int> neighbors;
            if (left_label >= 0) neighbors.push_back(left_label);
            if (up_label >= 0) neighbors.push_back(up_label);
            if (upleft_label >= 0) neighbors.push_back(upleft_label);
            if (upright_label >= 0) neighbors.push_back(upright_label);

            if (neighbors.empty()) {
                int new_label = uf.add();
                labels[idx] = new_label;
            } else {
                int root = uf.find(neighbors[0]);
                for (size_t i = 1; i < neighbors.size(); ++i) {
                    int r = uf.find(neighbors[i]);
                    if (r != root) uf.unite(root, r);
                    root = uf.find(root);
                }
                labels[idx] = root;
            }
        }
    }

    std::unordered_map<int, std::vector<std::pair<int, int>>> component_pixels;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = y * w + x;
            if (labels[idx] < 0) continue;
            int root = uf.find(labels[idx]);
            component_pixels[root].emplace_back(x, y);
        }
    }

    struct component_info {
        image_data img;
        rectangle rect;
        int orig_w;
    };
    std::vector<component_info> components;

    for (const auto& pair : component_pixels) {
        const auto& pixels = pair.second;
        if (pixels.empty()) continue;

        int min_x = w, min_y = h, max_x = -1, max_y = -1;
        for (const auto& pt : pixels) {
            min_x = std::min(min_x, pt.first);
            min_y = std::min(min_y, pt.second);
            max_x = std::max(max_x, pt.first);
            max_y = std::max(max_y, pt.second);
        }
        int box_w = max_x - min_x + 1;
        int box_h = max_y - min_y + 1;
        int area = box_w * box_h;
        if (area <= 6 || area >= total_pixels / 2) continue;

        rectangle padded;
        padded.x = std::max(0, min_x - pad);
        padded.y = std::max(0, min_y - pad);
        padded.w = std::min(w - padded.x, box_w + 2 * pad);
        padded.h = std::min(h - padded.y, box_h + 2 * pad);

        image_data comp_img;
        comp_img.w = padded.w;
        comp_img.h = padded.h;
        comp_img.px.assign(padded.w * padded.h, 0);

        for (const auto& pt : pixels) {
            int src_x = pt.first;
            int src_y = pt.second;
            uint8_t val = gray_image.px[src_y * w + src_x];
            int dst_x = src_x - padded.x;
            int dst_y = src_y - padded.y;
            comp_img.px[dst_y * padded.w + dst_x] = val;
        }

        components.push_back({std::move(comp_img), padded, box_w});
    }

    if (components.empty()) {
        digits.clear();
        rects.clear();
        return;
    }

    std::vector<int> widths;
    for (const auto& comp : components) widths.push_back(comp.orig_w);
    std::nth_element(
        widths.begin(), widths.begin() + widths.size() / 2, widths.end());
    int median_w = widths[widths.size() / 2];

    digits.clear();
    rects.clear();

    for (const auto& comp : components) {
        if (comp.orig_w > median_w * width_ratio) {
            std::vector<image_data> sub_digits;
            std::vector<rectangle> sub_rects_local;
            split_merged_digits(
                comp.img, sub_digits, sub_rects_local, threshold, 0.2);

            for (size_t i = 0; i < sub_digits.size(); ++i) {
                rectangle global = comp.rect;
                global.x += sub_rects_local[i].x;
                global.y += sub_rects_local[i].y;
                global.w = sub_rects_local[i].w;
                global.h = sub_rects_local[i].h;
                digits.push_back(std::move(sub_digits[i]));
                rects.push_back(global);
            }
        } else {
            digits.push_back(comp.img);
            rects.push_back(comp.rect);
        }
    }
}

void split_merged_digits(
    const image_data& merged_digits,
    std::vector<image_data>& digits,
    std::vector<rectangle>& rects,
    uint8_t threshold,
    double height_ratio) {
    const int w = merged_digits.w;
    const int h = merged_digits.h;
    if (w <= 0 || h <= 0) {
        digits.clear();
        rects.clear();
        return;
    }

    std::vector<int> vproj(w, 0);
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            if (merged_digits.px[y * w + x] >= threshold) ++vproj[x];
        }
    }

    std::vector<int> split_cols;
    for (int i = 2; i < w - 2; ++i) {
        int left_avg = (vproj[i - 2] + vproj[i - 1]) / 2;
        int right_avg = (vproj[i + 1] + vproj[i + 2]) / 2;
        int local_min = vproj[i];
        if (local_min < left_avg && local_min < right_avg &&
            local_min < std::min(left_avg, right_avg) * height_ratio) {
            split_cols.push_back(i);
        }
    }

    if (!split_cols.empty()) {
        std::vector<int> merged;
        merged.push_back(split_cols[0]);
        for (size_t i = 1; i < split_cols.size(); ++i) {
            if (split_cols[i] - merged.back() > 3)
                merged.push_back(split_cols[i]);
        }

        digits.clear();
        rects.clear();
        int start_x = 0;
        for (int sc : merged) {
            int end_x = sc;
            int sub_w = end_x - start_x + 1;
            if (sub_w > 2 && h > 3) {
                image_data sub;
                sub.w = sub_w;
                sub.h = h;
                sub.px.resize(sub_w * h);
                for (int y = 0; y < h; ++y) {
                    for (int x = 0; x < sub_w; ++x) {
                        sub.px[y * sub_w + x] =
                            merged_digits.px[y * w + (start_x + x)];
                    }
                }
                digits.push_back(std::move(sub));
                rects.push_back({start_x, 0, sub_w, h});
            }
            start_x = sc + 1;
        }
        if (start_x < w) {
            int sub_w = w - start_x;
            if (sub_w > 2 && h > 3) {
                image_data sub;
                sub.w = sub_w;
                sub.h = h;
                sub.px.resize(sub_w * h);
                for (int y = 0; y < h; ++y) {
                    for (int x = 0; x < sub_w; ++x) {
                        sub.px[y * sub_w + x] =
                            merged_digits.px[y * w + (start_x + x)];
                    }
                }
                digits.push_back(std::move(sub));
                rects.push_back({start_x, 0, sub_w, h});
            }
        }
    } else {
        digits = {merged_digits};
        rects = {{0, 0, w, h}};
    }
}

void sort_digits_by_reading_order(
    std::vector<Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor>>&
        digit_matrices,
    std::vector<rectangle>& rects,
    double height_ratio) {
    if (rects.empty()) return;

    std::vector<int> heights;
    heights.reserve(rects.size());
    for (const auto& r : rects) heights.push_back(r.h);
    std::nth_element(
        heights.begin(), heights.begin() + heights.size() / 2, heights.end());
    int median_h = heights[heights.size() / 2];
    int tolerance = static_cast<int>(median_h * height_ratio);

    std::vector<int> center_y(rects.size());
    for (size_t i = 0; i < rects.size(); ++i) {
        center_y[i] = rects[i].y + rects[i].h / 2;
    }

    std::vector<size_t> idx(rects.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return center_y[a] < center_y[b];
    });

    std::vector<std::vector<size_t>> rows;
    for (size_t i : idx) {
        if (rows.empty()) {
            rows.push_back({i});
            continue;
        }
        size_t last_idx = rows.back().back();
        if (center_y[i] - center_y[last_idx] <= tolerance) {
            rows.back().push_back(i);
        } else {
            rows.push_back({i});
        }
    }

    for (auto& row : rows) {
        std::sort(row.begin(), row.end(), [&](size_t a, size_t b) {
            return rects[a].x < rects[b].x;
        });
    }

    std::vector<size_t> sorted_idx;
    sorted_idx.reserve(rects.size());
    for (const auto& row : rows) {
        sorted_idx.insert(sorted_idx.end(), row.begin(), row.end());
    }

    std::vector<rectangle> new_rects;
    std::vector<Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor>> new_mats;
    new_rects.reserve(rects.size());
    new_mats.reserve(digit_matrices.size());
    for (size_t i : sorted_idx) {
        new_rects.push_back(std::move(rects[i]));
        new_mats.push_back(std::move(digit_matrices[i]));
    }
    rects = std::move(new_rects);
    digit_matrices = std::move(new_mats);
}

void process_digits(
    const image_data& gray_image,
    std::vector<Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor>>&
        digit_matrices,
    std::vector<rectangle>& rects) {
    std::vector<image_data> digit_images;
    std::vector<rectangle> digit_rects;
    split_digits(gray_image, digit_images, digit_rects);
    digit_matrices.clear();
    digit_matrices.reserve(digit_images.size());
    rects = std::move(digit_rects);
    for (size_t i = 0; i < digit_images.size(); ++i) {
        const image_data& img = digit_images[i];
        Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            mat(img.h, img.w);
        std::memcpy(mat.data(), img.px.data(), img.px.size());
        digit_matrices.push_back(preprocess_to_mnist_size(mat));
    }
}

image_data process_from_file(
    const std::filesystem::path& filepath,
    std::vector<Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor>>&
        digit_matrices,
    std::vector<rectangle>& rects) {
    image_data img = load_image_from_file(filepath);
    img.px = preprocess_to_mnist_color(img.px, img.w, img.h);
    process_digits(img, digit_matrices, rects);
    return img;
}

image_data process_from_base64(
    const std::string& base64_str,
    std::vector<Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor>>&
        digit_matrices,
    std::vector<rectangle>& rects) {
    image_data img = load_image_from_base64(base64_str);
    img.px = preprocess_to_mnist_color(img.px, img.w, img.h);
    process_digits(img, digit_matrices, rects);
    return img;
}

}  // namespace image
}  // namespace mint