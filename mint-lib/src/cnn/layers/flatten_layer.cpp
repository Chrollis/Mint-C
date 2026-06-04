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

#include <cnn/layers/flatten_layer.h>
#include <stdexcept>

namespace mint {

flatten_layer::flatten_layer(io_shape in_shape, bool training)
    : layer(
          in_shape,
          {1, in_shape.c * in_shape.w * in_shape.h, 1},
          layer_type::flatten,
          training) {}

std::vector<std::vector<Eigen::MatrixXd>> flatten_layer::forward(
    const std::vector<std::vector<Eigen::MatrixXd>>& input) {
    if (input.empty() || input[0].empty()) return {};

    if (input[0].size() != static_cast<size_t>(input_shape_.c) ||
        input[0][0].cols() != input_shape_.w ||
        input[0][0].rows() != input_shape_.h) {
        throw std::invalid_argument(
            "flatten_layer::forward: input shape mismatch, expected [c=" +
            std::to_string(input_shape_.c) +
            ", w=" + std::to_string(input_shape_.w) +
            ", h=" + std::to_string(input_shape_.h) + "]");
    }

    std::vector<std::vector<Eigen::MatrixXd>> output(input.size());
    int area = input_shape_.w * input_shape_.h;

    for (size_t b = 0; b < input.size(); ++b) {
        Eigen::MatrixXd flat(1, output_shape_.w);
        for (size_t c = 0; c < static_cast<size_t>(input_shape_.c); ++c) {
            flat.block(0, c * area, 1, area) =
                input[b][c].reshaped<Eigen::RowMajor>(1, area);
        }
        output[b] = {std::move(flat)};
    }
    return output;
}

std::vector<std::vector<Eigen::MatrixXd>> flatten_layer::backward(
    const std::vector<std::vector<Eigen::MatrixXd>>& grad_output,
    double /*learning_rate*/) {
    if (!training_)
        throw std::invalid_argument(
            "flatten_layer::backward: cannot backward when layer is not in "
            "training mode");

    if (grad_output.empty() || grad_output[0].empty()) return {};

    if (grad_output[0].size() != static_cast<size_t>(output_shape_.c) ||
        grad_output[0][0].cols() != output_shape_.w ||
        grad_output[0][0].rows() != output_shape_.h) {
        throw std::invalid_argument(
            "flatten_layer::backward: grad_output shape mismatch, expected "
            "[c=" +
            std::to_string(output_shape_.c) +
            ", w=" + std::to_string(output_shape_.w) +
            ", h=" + std::to_string(output_shape_.h) + "]");
    }

    std::vector<std::vector<Eigen::MatrixXd>> grad_input(grad_output.size());
    int area = input_shape_.w * input_shape_.h;

    for (size_t b = 0; b < grad_output.size(); ++b) {
        const Eigen::MatrixXd& flat = grad_output[b][0];
        std::vector<Eigen::MatrixXd> sample_grad(input_shape_.c);
        for (int c = 0; c < input_shape_.c; ++c) {
            sample_grad[c] =
                flat.block(0, c * area, 1, area)
                    .reshaped<Eigen::RowMajor>(input_shape_.h, input_shape_.w);
        }
        grad_input[b] = std::move(sample_grad);
    }
    return grad_input;
}

}  // namespace mint