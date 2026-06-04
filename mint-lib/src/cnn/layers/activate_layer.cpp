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

#include <cnn/layers/activate_layer.h>
#include <process/utils.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace mint {

double activate::forward(double x) const {
    switch (type_) {
        case func_type::sigmoid:
            return 1.0 / (1.0 + std::exp(-x));
        case func_type::tanh:
            return std::tanh(x);
        case func_type::relu:
            return x > 0 ? x : 0.0;
        case func_type::lrelu:
            return x > 0 ? x : 0.01 * x;
        case func_type::linear:
        default:
            return x;
    }
}

double activate::backward(double x) const {
    switch (type_) {
        case func_type::sigmoid: {
            double fx = 1.0 / (1.0 + std::exp(-x));
            return fx * (1.0 - fx);
        }
        case func_type::tanh: {
            double fx = std::tanh(x);
            return 1.0 - fx * fx;
        }
        case func_type::relu:
            return x > 0.0 ? 1.0 : 0.0;
        case func_type::lrelu:
            return x > 0.0 ? 1.0 : 0.01;
        case func_type::linear:
        default:
            return 1.0;
    }
}

Eigen::MatrixXd activate::forward(const Eigen::MatrixXd& x) const {
    return x.unaryExpr([this](double v) { return forward(v); });
}

Eigen::MatrixXd activate::backward(const Eigen::MatrixXd& x) const {
    return x.unaryExpr([this](double v) { return backward(v); });
}

activate_layer::activate_layer(
    io_shape in_shape, bool training, activate::func_type act_type)
    : layer(in_shape, in_shape, layer_type::activate, training),
      act_(act_type) {}

std::vector<std::vector<Eigen::MatrixXd>> activate_layer::forward(
    const std::vector<std::vector<Eigen::MatrixXd>>& input) {
    if (input.empty() || input[0].empty()) return {};

    if (input[0].size() != static_cast<size_t>(input_shape_.c) ||
        input[0][0].cols() != input_shape_.w ||
        input[0][0].rows() != input_shape_.h) {
        throw std::invalid_argument(
            "activate_layer::forward: input shape mismatch, expected [batch=" +
            std::to_string(input.size()) +
            ", c=" + std::to_string(input_shape_.c) +
            ", w=" + std::to_string(input_shape_.w) +
            ", h=" + std::to_string(input_shape_.h) + "]");
    }

    if (training_) cached_input_ = input;

    std::vector<std::vector<Eigen::MatrixXd>> output(input.size());
    for (size_t b = 0; b < input.size(); ++b) {
        output[b].resize(input_shape_.c);
        for (size_t c = 0; c < static_cast<size_t>(input_shape_.c); ++c) {
            output[b][c] = act_.forward(input[b][c]);
        }
    }
    return output;
}

std::vector<std::vector<Eigen::MatrixXd>> activate_layer::backward(
    const std::vector<std::vector<Eigen::MatrixXd>>& grad_output,
    double /*learning_rate*/) {
    if (!training_)
        throw std::invalid_argument(
            "activate_layer::backward: cannot backward when layer is not in "
            "training mode");

    if (cached_input_.size() != grad_output.size())
        throw std::invalid_argument(
            "activate_layer::backward: cached_input batch size (" +
            std::to_string(cached_input_.size()) +
            ") != grad_output batch size (" +
            std::to_string(grad_output.size()) + ")");

    if (grad_output[0].size() != static_cast<size_t>(output_shape_.c) ||
        grad_output[0][0].cols() != output_shape_.w ||
        grad_output[0][0].rows() != output_shape_.h)
        throw std::invalid_argument(
            "activate_layer::backward: grad_output shape mismatch, expected "
            "[c=" +
            std::to_string(output_shape_.c) +
            ", w=" + std::to_string(output_shape_.w) +
            ", h=" + std::to_string(output_shape_.h) + "]");

    std::vector<std::vector<Eigen::MatrixXd>> grad_input(cached_input_.size());
    for (size_t b = 0; b < cached_input_.size(); ++b) {
        grad_input[b].resize(input_shape_.c);
        for (size_t c = 0; c < static_cast<size_t>(input_shape_.c); ++c) {
            const auto& x = cached_input_[b][c];
            const auto& grad_y = grad_output[b][c];
            grad_input[b][c] = grad_y.cwiseProduct(act_.backward(x));
        }
    }
    return grad_input;
}

void activate_layer::save(std::ostream& os) const {
    save_header(os);
    uint64_t act_type = static_cast<uint64_t>(act_.type());
    act_type = host_to_be64(act_type);
    os.write(reinterpret_cast<const char*>(&act_type), 8);
}

void activate_layer::load(std::istream& is) {
    load_header(is);
    uint64_t act_type;
    is.read(reinterpret_cast<char*>(&act_type), 8);
    if (!is)
        throw std::runtime_error(
            "activate_layer::load: failed to read activation function type "
            "from stream");
    act_.set_type(static_cast<activate::func_type>(be64_to_host(act_type)));
}

}  // namespace mint