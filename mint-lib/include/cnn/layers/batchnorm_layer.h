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
#include <vector>

namespace mint {

class batchnorm_layer : public layer {
public:
    batchnorm_layer(
        io_shape in_shape = {1, 0, 0},
        double eps = 1e-5,
        double momentum = 0.9,
        bool training = true);

    std::vector<std::vector<Eigen::MatrixXd>> forward(
        const std::vector<std::vector<Eigen::MatrixXd>>& input) override;

    std::vector<std::vector<Eigen::MatrixXd>> backward(
        const std::vector<std::vector<Eigen::MatrixXd>>& grad_output,
        double learning_rate) override;

    void save(std::ostream& os) const override;
    void load(std::istream& is) override;

    double eps() const { return eps_; }
    double momentum() const { return momentum_; }

private:
    double eps_;
    double momentum_;

    Eigen::VectorXd gamma_;
    Eigen::VectorXd beta_;

    Eigen::VectorXd running_mean_;
    Eigen::VectorXd running_var_;

    std::vector<std::vector<Eigen::MatrixXd>> cached_x_;
    Eigen::VectorXd cached_mean_;
    Eigen::VectorXd cached_var_;
    std::vector<std::vector<Eigen::MatrixXd>> cached_x_norm_;
};

}  // namespace mint