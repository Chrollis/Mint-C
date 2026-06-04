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

#include <cnn/layers/dropout_layer.h>
#include <process/utils.h>
#include <chrono>
#include <random>
#include <stdexcept>

namespace mint {

std::mt19937& dropout_layer::rng() {
    thread_local static std::mt19937 rng(
        static_cast<unsigned>(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    return rng;
}

dropout_layer::dropout_layer(io_shape in_shape, double keep_prob, bool training)
    : layer(in_shape, in_shape, layer_type::dropout, training),
      keep_prob_(keep_prob) {}

std::vector<std::vector<Eigen::MatrixXd>> dropout_layer::forward(
    const std::vector<std::vector<Eigen::MatrixXd>>& input) {
    if (input.empty() || input[0].empty()) return {};

    if (static_cast<int>(input[0].size()) != input_shape_.c ||
        input[0][0].cols() != input_shape_.w ||
        input[0][0].rows() != input_shape_.h) {
        throw std::invalid_argument(
            "dropout_layer::forward: input shape mismatch, expected [c=" +
            std::to_string(input_shape_.c) +
            ", w=" + std::to_string(input_shape_.w) +
            ", h=" + std::to_string(input_shape_.h) + "]");
    }

    std::vector<std::vector<Eigen::MatrixXd>> output = input;

    if (training_) {
        std::bernoulli_distribution dist(keep_prob_);
        masks_.resize(input.size());

        for (size_t b = 0; b < input.size(); ++b) {
            masks_[b].resize(input_shape_.c);
            for (int c = 0; c < input_shape_.c; ++c) {
                const auto& mat = input[b][c];
                Eigen::MatrixXd mask = Eigen::MatrixXd::NullaryExpr(
                    input_shape_.h, input_shape_.w,
                    [&]() { return dist(rng()) ? 1.0 : 0.0; });
                masks_[b][c] = mask;
                output[b][c] = mat.cwiseProduct(mask) / keep_prob_;
            }
        }
    }

    return output;
}

std::vector<std::vector<Eigen::MatrixXd>> dropout_layer::backward(
    const std::vector<std::vector<Eigen::MatrixXd>>& grad_output,
    double /*learning_rate*/) {
    if (!training_) {
        throw std::invalid_argument(
            "dropout_layer::backward: backward called in eval mode");
    }
    if (masks_.size() != grad_output.size()) {
        throw std::invalid_argument(
            "dropout_layer::backward: masks batch size (" +
            std::to_string(masks_.size()) + ") != grad_output batch size (" +
            std::to_string(grad_output.size()) + ")");
    }
    if (grad_output[0].size() != static_cast<size_t>(output_shape_.c) ||
        grad_output[0][0].cols() != output_shape_.w ||
        grad_output[0][0].rows() != output_shape_.h) {
        throw std::invalid_argument(
            "dropout_layer::backward: grad_output shape mismatch, expected "
            "[c=" +
            std::to_string(output_shape_.c) +
            ", w=" + std::to_string(output_shape_.w) +
            ", h=" + std::to_string(output_shape_.h) + "]");
    }

    std::vector<std::vector<Eigen::MatrixXd>> grad_input(grad_output.size());

    for (size_t b = 0; b < grad_output.size(); ++b) {
        grad_input[b].resize(grad_output[b].size());
        for (size_t c = 0; c < grad_output[b].size(); ++c) {
            grad_input[b][c] =
                grad_output[b][c].cwiseProduct(masks_[b][c]) / keep_prob_;
        }
    }

    return grad_input;
}

void dropout_layer::save(std::ostream& os) const {
    save_header(os);
    double kp = host_to_bedb(keep_prob_);
    os.write(reinterpret_cast<const char*>(&kp), 8);
}

void dropout_layer::load(std::istream& is) {
    load_header(is);
    double kp;
    is.read(reinterpret_cast<char*>(&kp), 8);
    if (!is)
        throw std::runtime_error(
            "dropout_layer::load: failed to read keep_prob");
    keep_prob_ = bedb_to_host(kp);
}

}