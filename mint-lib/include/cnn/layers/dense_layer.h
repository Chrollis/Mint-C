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

#include <cnn/layers/layer.h>
#include <random>
#include <vector>

namespace mint {

class dense_layer : public layer {
public:
    enum class init_type {
        gaussian,
        xavier,
        he,
        load,
    };

    dense_layer(
        io_shape in_shape = {1, 0, 0},
        io_shape out_shape = {1, 0, 0},
        init_type init = init_type::he,
        bool training = true);

    std::vector<std::vector<Eigen::MatrixXd>> forward(
        const std::vector<std::vector<Eigen::MatrixXd>>& input) override;

    std::vector<std::vector<Eigen::MatrixXd>> backward(
        const std::vector<std::vector<Eigen::MatrixXd>>& grad_output,
        double learning_rate) override;

    void save(std::ostream& os) const override;
    void load(std::istream& is) override;

private:
    Eigen::MatrixXd weights_;
    Eigen::MatrixXd biases_;

    std::vector<std::vector<Eigen::MatrixXd>> cached_input_;

    static std::mt19937& rng();

    void init_weights(init_type type);
};

}  // namespace mint