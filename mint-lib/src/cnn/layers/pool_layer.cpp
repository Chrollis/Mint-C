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

#include <cnn/layers/pool_layer.h>
#include <process/utils.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace mint {

pool_layer::pool_layer(
    io_shape in_shape,
    int kernel_size,
    int stride,
    pool_type p_type,
    bool training)
    : layer(
          in_shape,
          {in_shape.c, std::max(0, (in_shape.w - kernel_size) / stride + 1),
           std::max(0, (in_shape.h - kernel_size) / stride + 1)},
          layer_type::pool,
          training),
      p_type_(p_type),
      stride_(stride),
      kernel_size_(kernel_size) {}

std::vector<std::vector<Eigen::MatrixXd>> pool_layer::forward(
    const std::vector<std::vector<Eigen::MatrixXd>>& input) {
    if (input.empty() || input[0].empty()) return {};

    if (input[0].size() != static_cast<size_t>(input_shape_.c) ||
        input[0][0].cols() != input_shape_.w ||
        input[0][0].rows() != input_shape_.h) {
        throw std::invalid_argument(
            "pool_layer::forward: input shape mismatch, expected [c=" +
            std::to_string(input_shape_.c) +
            ", w=" + std::to_string(input_shape_.w) +
            ", h=" + std::to_string(input_shape_.h) + "]");
    }

    std::vector<std::vector<Eigen::MatrixXd>> output(input.size());

    if (training_ && p_type_ == pool_type::max) {
        max_indices_ = std::vector<std::vector<Eigen::MatrixXi>>(
            input.size(),
            std::vector<Eigen::MatrixXi>(
                input_shape_.c,
                Eigen::MatrixXi(output_shape_.h, output_shape_.w)));
    }

    for (size_t b = 0; b < input.size(); ++b) {
        output[b].resize(input_shape_.c);
        for (int c = 0; c < input_shape_.c; ++c) {
            const Eigen::MatrixXd& mat = input[b][c];
            Eigen::MatrixXd out(output_shape_.h, output_shape_.w);

            for (int y = 0; y < output_shape_.h; ++y) {
                int start_y = y * stride_;
                for (int x = 0; x < output_shape_.w; ++x) {
                    int start_x = x * stride_;
                    Eigen::MatrixXd block =
                        mat.block(start_y, start_x, kernel_size_, kernel_size_);

                    if (p_type_ == pool_type::max) {
                        int max_x, max_y;
                        double max_val = block.maxCoeff(&max_y, &max_x);
                        out(y, x) = max_val;
                        if (training_) {
                            max_indices_[b][c](y, x) =
                                max_y * kernel_size_ + max_x;
                        }
                    } else {
                        out(y, x) = block.mean();
                    }
                }
            }
            output[b][c] = std::move(out);
        }
    }
    return output;
}

std::vector<std::vector<Eigen::MatrixXd>> pool_layer::backward(
    const std::vector<std::vector<Eigen::MatrixXd>>& grad_output,
    double /*learning_rate*/) {
    if (!training_)
        throw std::invalid_argument(
            "pool_layer::backward: cannot backward when layer is not in "
            "training mode");

    if (p_type_ == pool_type::max && max_indices_.size() != grad_output.size())
        throw std::invalid_argument(
            "pool_layer::backward: max_indices batch size (" +
            std::to_string(max_indices_.size()) +
            ") != grad_output batch size (" +
            std::to_string(grad_output.size()) + ")");

    if (grad_output[0].size() != static_cast<size_t>(output_shape_.c) ||
        grad_output[0][0].cols() != output_shape_.w ||
        grad_output[0][0].rows() != output_shape_.h) {
        throw std::invalid_argument(
            "pool_layer::backward: grad_output shape mismatch, expected [c=" +
            std::to_string(output_shape_.c) +
            ", w=" + std::to_string(output_shape_.w) +
            ", h=" + std::to_string(output_shape_.h) + "]");
    }

    std::vector<std::vector<Eigen::MatrixXd>> grad_input(grad_output.size());
    int kernel_area = kernel_size_ * kernel_size_;

    for (size_t b = 0; b < grad_output.size(); ++b) {
        grad_input[b].resize(input_shape_.c);
        for (int c = 0; c < input_shape_.c; ++c) {
            Eigen::MatrixXd grad_in =
                Eigen::MatrixXd::Zero(input_shape_.h, input_shape_.w);
            const Eigen::MatrixXd& grad_out = grad_output[b][c];

            for (int y = 0; y < output_shape_.h; ++y) {
                int start_y = y * stride_;
                for (int x = 0; x < output_shape_.w; ++x) {
                    int start_x = x * stride_;
                    double grad_val = grad_out(y, x);

                    if (p_type_ == pool_type::max) {
                        int idx = max_indices_[b][c](y, x);
                        grad_in(
                            start_y + idx / kernel_size_,
                            start_x + idx % kernel_size_) += grad_val;
                    } else {
                        double avg_grad = grad_val / kernel_area;
                        grad_in
                            .block(start_y, start_x, kernel_size_, kernel_size_)
                            .array() += avg_grad;
                    }
                }
            }
            grad_input[b][c] = std::move(grad_in);
        }
    }
    return grad_input;
}

void pool_layer::save(std::ostream& os) const {
    save_header(os);

    uint32_t p_type = host_to_be32(static_cast<uint32_t>(p_type_));
    os.write(reinterpret_cast<const char*>(&p_type), 4);

    uint32_t stride = host_to_be32(static_cast<uint32_t>(stride_));
    os.write(reinterpret_cast<const char*>(&stride), 4);

    uint64_t ker_size = host_to_be64(static_cast<uint64_t>(kernel_size_));
    os.write(reinterpret_cast<const char*>(&ker_size), 8);
}

void pool_layer::load(std::istream& is) {
    load_header(is);

    uint32_t p_type;
    is.read(reinterpret_cast<char*>(&p_type), 4);
    p_type = be32_to_host(p_type);
    if (!is)
        throw std::runtime_error("pool_layer::load: failed to read pool type");
    p_type_ = static_cast<pool_type>(p_type);

    uint32_t stride;
    is.read(reinterpret_cast<char*>(&stride), 4);
    if (!is)
        throw std::runtime_error("pool_layer::load: failed to read stride");
    stride_ = static_cast<int>(be32_to_host(stride));

    uint64_t ker_size;
    is.read(reinterpret_cast<char*>(&ker_size), 8);
    if (!is)
        throw std::runtime_error(
            "pool_layer::load: failed to read kernel size");
    kernel_size_ = static_cast<int>(be64_to_host(ker_size));
}

}