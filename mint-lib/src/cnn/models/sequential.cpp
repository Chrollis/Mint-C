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
#include <cnn/layers/batchnorm_layer.h>
#include <cnn/layers/conv_layer.h>
#include <cnn/layers/dense_layer.h>
#include <cnn/layers/dropout_layer.h>
#include <cnn/layers/flatten_layer.h>
#include <cnn/layers/pool_layer.h>
#include <cnn/layers/softmax_layer.h>
#include <cnn/models/sequential.h>
#include <process/utils.h>
#include <cstdint>
#include <stdexcept>

namespace mint {

void sequential::add_layer(std::unique_ptr<layer> l) {
    layers_.push_back(std::move(l));
}

std::vector<std::vector<Eigen::MatrixXd>> sequential::forward(
    const std::vector<std::vector<Eigen::MatrixXd>>& input) {
    if (layers_.empty()) return input;

    std::vector<std::vector<Eigen::MatrixXd>> current = input;
    for (auto& l : layers_) {
        current = l->forward(current);
    }
    return current;
}

std::vector<std::vector<Eigen::MatrixXd>> sequential::backward(
    const std::vector<std::vector<Eigen::MatrixXd>>& grad_output,
    double learning_rate) {
    if (layers_.empty()) return grad_output;

    std::vector<std::vector<Eigen::MatrixXd>> current_grad = grad_output;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        current_grad = (*it)->backward(current_grad, learning_rate);
    }
    return current_grad;
}

void sequential::train() {
    for (auto& l : layers_) {
        l->train();
    }
}

void sequential::eval() {
    for (auto& l : layers_) {
        l->eval();
    }
}

void sequential::save(std::ostream& os) const {
    os.exceptions(std::ios::failbit | std::ios::badbit);
    uint64_t count = static_cast<uint64_t>(layers_.size());
    count = host_to_be64(count);
    os.write(reinterpret_cast<const char*>(&count), 8);
    for (const auto& l : layers_) {
        layer::save_type(os, l->type());
        l->save(os);
    }
}

void sequential::load(std::istream& is) {
    layers_.clear();
    uint64_t count;
    is.read(reinterpret_cast<char*>(&count), 8);
    if (!is)
        throw std::runtime_error(
            "sequential::load: failed to read layer count");
    count = be64_to_host(count);
    for (uint64_t i = 0; i < count; ++i) {
        layer::layer_type type = layer::load_type(is);
        std::unique_ptr<layer> l = make_layer(type);
        l->load(is);
        layers_.push_back(std::move(l));
    }
}

layer::io_shape sequential::input_shape() const {
    if (layers_.empty()) return {1, 0, 0};
    return layers_.front()->input_shape();
}

layer::io_shape sequential::output_shape() const {
    if (layers_.empty()) return {1, 0, 0};
    return layers_.back()->output_shape();
}

std::unique_ptr<layer> sequential::make_layer(layer::layer_type type) {
    switch (type) {
        case layer::layer_type::convolve:
            return std::make_unique<conv_layer>();
        case layer::layer_type::pool:
            return std::make_unique<pool_layer>();
        case layer::layer_type::flatten:
            return std::make_unique<flatten_layer>();
        case layer::layer_type::dense:
            return std::make_unique<dense_layer>();
        case layer::layer_type::dropout:
            return std::make_unique<dropout_layer>();
        case layer::layer_type::batchnorm:
            return std::make_unique<batchnorm_layer>();
        case layer::layer_type::activate:
            return std::make_unique<activate_layer>();
        case layer::layer_type::softmax:
            return std::make_unique<softmax_layer>();
        case layer::layer_type::base:
        default:
            throw std::invalid_argument(
                "sequential::make_layer: invalid layer type, val = " +
                std::to_string(static_cast<int>(type)));
    }
}

}  // namespace mint