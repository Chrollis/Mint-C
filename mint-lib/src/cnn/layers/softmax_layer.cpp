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

#include <cnn/layers/softmax_layer.h>

#include <cmath>
#include <stdexcept>

namespace mint {

softmax_layer::softmax_layer(io_shape in_shape, bool training)
    : layer(in_shape, in_shape, layer_type::softmax, training) {}

std::vector<std::vector<Eigen::MatrixXd>> softmax_layer::forward(
    const std::vector<std::vector<Eigen::MatrixXd>>& input) {
    if (input.empty() || input[0].empty()) return {};

    if (static_cast<int>(input[0].size()) != input_shape_.c ||
        input[0][0].cols() != input_shape_.w ||
        input[0][0].rows() != input_shape_.h) {
        throw std::invalid_argument(
            "softmax_layer::forward: input shape mismatch");
    }

    std::vector<std::vector<Eigen::MatrixXd>> output(input.size());
    if (training_) cached_prob_.resize(input.size());

    for (size_t b = 0; b < input.size(); ++b) {
        const Eigen::MatrixXd& x = input[b][0];

        double max_val = x.maxCoeff();
        Eigen::MatrixXd exp_x = (x.array() - max_val).exp();
        double sum_exp = exp_x.sum();
        Eigen::MatrixXd probs = exp_x / sum_exp;

        output[b] = {probs};
        if (training_) cached_prob_[b] = probs;
    }

    return output;
}

std::vector<std::vector<Eigen::MatrixXd>> softmax_layer::backward(
    const std::vector<std::vector<Eigen::MatrixXd>>& grad_output,
    double /*learning_rate*/) {
    if (!training_)
        throw std::invalid_argument(
            "softmax_layer::backward: cannot backward when layer is not in "
            "training mode");

    if (grad_output.empty() || grad_output[0].empty()) return {};

    if (cached_prob_.size() != grad_output.size())
        throw std::invalid_argument(
            "softmax_layer::backward: cached_prob batch size (" +
            std::to_string(cached_prob_.size()) +
            ") != grad_output batch size (" +
            std::to_string(grad_output.size()) + ")");

    if (grad_output[0].size() != static_cast<size_t>(output_shape_.c) ||
        grad_output[0][0].cols() != output_shape_.w ||
        grad_output[0][0].rows() != output_shape_.h) {
        throw std::invalid_argument(
            "softmax_layer::backward: grad_output shape mismatch, expected "
            "[c=" +
            std::to_string(output_shape_.c) +
            ", w=" + std::to_string(output_shape_.w) +
            ", h=" + std::to_string(output_shape_.h) + "]");
    }

    std::vector<std::vector<Eigen::MatrixXd>> grad_input(grad_output.size());

    for (size_t b = 0; b < grad_output.size(); ++b) {
        const Eigen::MatrixXd& prob = cached_prob_[b];
        const Eigen::MatrixXd& grad_y = grad_output[b][0];

        double s = (grad_y.array() * prob.array()).sum();
        Eigen::MatrixXd grad_x = prob.array() * (grad_y.array() - s);
        grad_input[b] = {grad_x};
    }
    return grad_input;
}

}  // namespace mint