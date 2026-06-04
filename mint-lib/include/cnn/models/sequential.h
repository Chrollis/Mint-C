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
#include <istream>
#include <memory>
#include <ostream>
#include <vector>

namespace mint {

class sequential {
public:
    sequential() = default;
    ~sequential() = default;

    sequential(const sequential&) = delete;
    sequential& operator=(const sequential&) = delete;
    sequential(sequential&&) = default;
    sequential& operator=(sequential&&) = default;

    void add_layer(std::unique_ptr<layer> l);

    size_t size() const { return layers_.size(); }
    bool empty() const { return layers_.empty(); }

    std::vector<std::vector<Eigen::MatrixXd>> forward(
        const std::vector<std::vector<Eigen::MatrixXd>>& input);

    std::vector<std::vector<Eigen::MatrixXd>> backward(
        const std::vector<std::vector<Eigen::MatrixXd>>& grad_output,
        double learning_rate);

    void train();
    void eval();

    void save(std::ostream& os) const;
    void load(std::istream& is);

    layer::io_shape input_shape() const;
    layer::io_shape output_shape() const;
    const std::vector<std::unique_ptr<layer>>& layers() const {
        return layers_;
    }

private:
    std::vector<std::unique_ptr<layer>> layers_;

    static std::unique_ptr<layer> make_layer(layer::layer_type type);
};

}  // namespace mint