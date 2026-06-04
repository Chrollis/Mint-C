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

namespace mint {

class activate {
public:
    enum class func_type {
        sigmoid,
        tanh,
        relu,
        lrelu,
        linear,
    };

    explicit activate(func_type t) : type_(t) {}

    double forward(double x) const;
    double backward(double x) const;

    Eigen::MatrixXd forward(const Eigen::MatrixXd& x) const;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& x) const;

    void set_type(func_type t) { type_ = t; }
    func_type type() const { return type_; }

private:
    func_type type_;
};

class activate_layer : public layer {
public:
    explicit activate_layer(
        io_shape in_shape = {1, 0, 0},
        bool training = true,
        activate::func_type act_type = activate::func_type::relu);

    std::vector<std::vector<Eigen::MatrixXd>> forward(
        const std::vector<std::vector<Eigen::MatrixXd>>& input) override;

    std::vector<std::vector<Eigen::MatrixXd>> backward(
        const std::vector<std::vector<Eigen::MatrixXd>>& grad_output,
        double learning_rate) override;

    void save(std::ostream& os) const override;
    void load(std::istream& is) override;

    activate::func_type act_type() const { return act_.type(); }

private:
    activate act_;
    std::vector<std::vector<Eigen::MatrixXd>> cached_input_;
};

}  // namespace mint