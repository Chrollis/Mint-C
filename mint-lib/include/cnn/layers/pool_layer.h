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
#include <cstdint>
#include <vector>

namespace mint {

class pool_layer : public layer {
public:
    enum class pool_type {
        max,
        avg,
    };

    pool_layer(
        io_shape in_shape = {1, 0, 0},
        int kernel_size = 3,
        int stride = 3,
        pool_type p_type = pool_type::max,
        bool training = true);

    std::vector<std::vector<Eigen::MatrixXd>> forward(
        const std::vector<std::vector<Eigen::MatrixXd>>& input) override;

    std::vector<std::vector<Eigen::MatrixXd>> backward(
        const std::vector<std::vector<Eigen::MatrixXd>>& grad_output,
        double learning_rate) override;

    void save(std::ostream& os) const override;
    void load(std::istream& is) override;

    pool_type p_type() const { return p_type_; }
    int stride() const { return stride_; }
    int kernel_size() const { return kernel_size_; }

private:
    pool_type p_type_;
    int stride_;
    int kernel_size_;
    std::vector<std::vector<Eigen::MatrixXi>> max_indices_;
};

}  // namespace mint