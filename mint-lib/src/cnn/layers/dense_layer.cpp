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

#include <cnn/layers/dense_layer.h>
#include <process/utils.h>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace mint {

std::mt19937& dense_layer::rng() {
    thread_local static std::mt19937 rng(
        static_cast<unsigned>(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    return rng;
}

dense_layer::dense_layer(
    io_shape in_shape, io_shape out_shape, init_type init, bool training)
    : layer(in_shape, out_shape, layer_type::dense, training),
      weights_(out_shape.w, in_shape.w),
      biases_(1, out_shape.w) {
    init_weights(init);
}

void dense_layer::init_weights(init_type type) {
    int in = input_shape_.w, out = output_shape_.w;
    switch (type) {
        case init_type::gaussian: {
            double stddev = std::sqrt(2.0 / (in + out));
            std::normal_distribution<double> dist(0.0, stddev);
            for (int i = 0; i < weights_.rows(); ++i)
                for (int j = 0; j < weights_.cols(); ++j)
                    weights_(i, j) = dist(rng());
            break;
        }
        case init_type::xavier: {
            double limit = std::sqrt(6.0 / (in + out));
            std::uniform_real_distribution<double> dist(-limit, limit);
            for (int i = 0; i < weights_.rows(); ++i)
                for (int j = 0; j < weights_.cols(); ++j)
                    weights_(i, j) = dist(rng());
            break;
        }
        case init_type::he: {
            double stddev = std::sqrt(2.0 / in);
            std::normal_distribution<double> dist(0.0, stddev);
            for (int i = 0; i < weights_.rows(); ++i)
                for (int j = 0; j < weights_.cols(); ++j)
                    weights_(i, j) = dist(rng());
            break;
        }
        case init_type::load:
        default:
            break;
    }
    biases_.setZero();
}

std::vector<std::vector<Eigen::MatrixXd>> dense_layer::forward(
    const std::vector<std::vector<Eigen::MatrixXd>>& input) {
    if (input.empty() || input[0].empty()) return {};

    if (static_cast<int>(input[0].size()) != input_shape_.c ||
        input[0][0].cols() != input_shape_.w ||
        input[0][0].rows() != input_shape_.h) {
        throw std::invalid_argument(
            "dense_layer::forward: input shape mismatch, expected [c=" +
            std::to_string(input_shape_.c) +
            ", w=" + std::to_string(input_shape_.w) +
            ", h=" + std::to_string(input_shape_.h) + "]");
    }

    if (training_) cached_input_ = input;

    std::vector<std::vector<Eigen::MatrixXd>> output(input.size());
    for (size_t b = 0; b < input.size(); ++b) {
        const Eigen::MatrixXd& x = input[b][0];
        Eigen::MatrixXd y = x * weights_.transpose() + biases_;
        output[b] = {y};
    }
    return output;
}

std::vector<std::vector<Eigen::MatrixXd>> dense_layer::backward(
    const std::vector<std::vector<Eigen::MatrixXd>>& grad_output,
    double learning_rate) {
    if (!training_)
        throw std::invalid_argument(
            "dense_layer::backward: cannot backward when layer is not in "
            "training mode");

    if (cached_input_.size() != grad_output.size())
        throw std::invalid_argument(
            "dense_layer::backward: cached_input batch size (" +
            std::to_string(cached_input_.size()) +
            ") != grad_output batch size (" +
            std::to_string(grad_output.size()) + ")");

    if (static_cast<int>(grad_output[0].size()) != output_shape_.c ||
        grad_output[0][0].cols() != output_shape_.w ||
        grad_output[0][0].rows() != output_shape_.h)
        throw std::invalid_argument(
            "dense_layer::backward: grad_output shape mismatch, expected [c=" +
            std::to_string(output_shape_.c) +
            ", w=" + std::to_string(output_shape_.w) +
            ", h=" + std::to_string(output_shape_.h) + "]");

    int in = input_shape_.w, out = output_shape_.w;
    Eigen::MatrixXd grad_w = Eigen::MatrixXd::Zero(out, in);
    Eigen::MatrixXd grad_b = Eigen::MatrixXd::Zero(1, out);

    std::vector<std::vector<Eigen::MatrixXd>> grad_input(grad_output.size());

    for (size_t b = 0; b < grad_output.size(); ++b) {
        const Eigen::MatrixXd& grad_y = grad_output[b][0];
        Eigen::MatrixXd grad_x = grad_y * weights_;
        grad_input[b] = {grad_x};

        const Eigen::MatrixXd& x = cached_input_[b][0];
        grad_w += grad_y.transpose() * x;
        grad_b += grad_y;
    }

    grad_w /= static_cast<double>(grad_output.size());
    grad_b /= static_cast<double>(grad_output.size());

    weights_ -= learning_rate * grad_w;
    biases_ -= learning_rate * grad_b;

    return grad_input;
}

void dense_layer::save(std::ostream& os) const {
    save_header(os);

    for (int i = 0; i < weights_.size(); ++i) {
        double x = host_to_bedb(weights_.data()[i]);
        os.write(reinterpret_cast<const char*>(&x), 8);
    }
    for (int i = 0; i < biases_.size(); ++i) {
        double x = host_to_bedb(biases_.data()[i]);
        os.write(reinterpret_cast<const char*>(&x), 8);
    }
}

void dense_layer::load(std::istream& is) {
    load_header(is);

    weights_.resize(output_shape_.w, input_shape_.w);
    biases_.resize(1, output_shape_.w);

    for (int i = 0; i < weights_.size(); ++i) {
        double x;
        is.read(reinterpret_cast<char*>(&x), 8);
        if (!is)
            throw std::runtime_error(
                "dense_layer::load: failed to read weights matrix");
        weights_.data()[i] = bedb_to_host(x);
    }
    for (int i = 0; i < biases_.size(); ++i) {
        double x;
        is.read(reinterpret_cast<char*>(&x), 8);
        if (!is)
            throw std::runtime_error(
                "dense_layer::load: failed to read biases vector");
        biases_.data()[i] = bedb_to_host(x);
    }
}

}  // namespace mint