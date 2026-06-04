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

#include <cnn/layers/batchnorm_layer.h>
#include <process/utils.h>
#include <cmath>
#include <stdexcept>

namespace mint {

batchnorm_layer::batchnorm_layer(
    io_shape in_shape, double eps, double momentum, bool training)
    : layer(in_shape, in_shape, layer_type::batchnorm, training),
      eps_(eps),
      momentum_(momentum),
      gamma_(Eigen::VectorXd::Ones(in_shape.c)),
      beta_(Eigen::VectorXd::Zero(in_shape.c)),
      running_mean_(Eigen::VectorXd::Zero(in_shape.c)),
      running_var_(Eigen::VectorXd::Ones(in_shape.c)) {}

std::vector<std::vector<Eigen::MatrixXd>> batchnorm_layer::forward(
    const std::vector<std::vector<Eigen::MatrixXd>>& input) {
    if (input.empty() || input[0].empty()) return {};

    if (static_cast<int>(input[0].size()) != input_shape_.c ||
        input[0][0].cols() != input_shape_.w ||
        input[0][0].rows() != input_shape_.h) {
        throw std::invalid_argument(
            "batchnorm_layer::forward: input shape mismatch, expected [c=" +
            std::to_string(input_shape_.c) +
            ", w=" + std::to_string(input_shape_.w) +
            ", h=" + std::to_string(input_shape_.h) + "]");
    }

    const int spatial_size = input_shape_.h * input_shape_.w;
    std::vector<std::vector<Eigen::MatrixXd>> output = input;

    if (training_) {
        cached_x_ = input;
        cached_x_norm_.resize(input.size());
        for (size_t b = 0; b < input.size(); ++b) {
            cached_x_norm_[b].resize(input_shape_.c);
        }

        Eigen::VectorXd mean(input_shape_.c);
        Eigen::VectorXd var(input_shape_.c);
        mean.setZero();
        var.setZero();

        for (size_t b = 0; b < input.size(); ++b) {
            for (int c = 0; c < input_shape_.c; ++c) {
                const auto& mat = input[b][c];
                double sum = mat.sum();
                double sum_sq = mat.array().square().sum();
                mean(c) += sum;
                var(c) += sum_sq;
            }
        }
        double total_elements =
            static_cast<double>(input.size() * spatial_size);
        mean /= total_elements;
        var /= total_elements;
        var = var.array() - mean.array().square();
        var = var.array() + eps_;

        cached_mean_ = mean;
        cached_var_ = var;

        running_mean_ = momentum_ * running_mean_ + (1.0 - momentum_) * mean;
        running_var_ = momentum_ * running_var_ + (1.0 - momentum_) * var;

        for (size_t b = 0; b < input.size(); ++b) {
            for (int c = 0; c < input_shape_.c; ++c) {
                const auto& x = input[b][c];
                double m = mean(c);
                double inv_std = 1.0 / std::sqrt(var(c));
                Eigen::MatrixXd x_norm = (x.array() - m) * inv_std;
                cached_x_norm_[b][c] = x_norm;
                output[b][c] = (x_norm * gamma_(c)).array() + beta_(c);
            }
        }
    } else {
        for (size_t b = 0; b < input.size(); ++b) {
            for (int c = 0; c < input_shape_.c; ++c) {
                const auto& x = input[b][c];
                double inv_std = 1.0 / std::sqrt(running_var_(c) + eps_);
                Eigen::MatrixXd x_norm =
                    (x.array() - running_mean_(c)) * inv_std;
                output[b][c] = (x_norm * gamma_(c)).array() + beta_(c);
            }
        }
    }
    return output;
}

std::vector<std::vector<Eigen::MatrixXd>> batchnorm_layer::backward(
    const std::vector<std::vector<Eigen::MatrixXd>>& grad_output,
    double learning_rate) {
    if (!training_) {
        throw std::invalid_argument(
            "batchnorm_layer::backward: cannot backward when layer is not in "
            "training mode");
    }
    if (cached_x_.size() != grad_output.size()) {
        throw std::invalid_argument(
            "batchnorm_layer::backward: cached_x batch size (" +
            std::to_string(cached_x_.size()) + ") != grad_output batch size (" +
            std::to_string(grad_output.size()) + ")");
    }

    const int spatial_size = input_shape_.h * input_shape_.w;
    const int N = static_cast<int>(grad_output.size()) * spatial_size;

    Eigen::VectorXd grad_gamma = Eigen::VectorXd::Zero(input_shape_.c);
    Eigen::VectorXd grad_beta = Eigen::VectorXd::Zero(input_shape_.c);

    std::vector<std::vector<Eigen::MatrixXd>> grad_input(grad_output.size());

    for (size_t b = 0; b < grad_output.size(); ++b) {
        grad_input[b].resize(input_shape_.c);
        for (int c = 0; c < input_shape_.c; ++c) {
            const auto& grad_y = grad_output[b][c];
            const auto& x_norm = cached_x_norm_[b][c];

            grad_gamma(c) += (grad_y.array() * x_norm.array()).sum();
            grad_beta(c) += grad_y.sum();

            Eigen::MatrixXd grad_x_norm = grad_y.array() * gamma_(c);

            double inv_std = 1.0 / std::sqrt(cached_var_(c));
            double inv_std_cube = inv_std * inv_std * inv_std;

            const auto& x = cached_x_[b][c];
            Eigen::MatrixXd grad_x = grad_x_norm * inv_std;

            double sum_grad_x_norm = grad_x_norm.sum();
            double sum_grad_x_norm_times_x_centered =
                (grad_x_norm.array() * (x.array() - cached_mean_(c))).sum();

            grad_x.array() -= (sum_grad_x_norm / N) * inv_std;
            grad_x.array() -=
                (x.array() - cached_mean_(c)) *
                (sum_grad_x_norm_times_x_centered * inv_std_cube / N);

            grad_input[b][c] = grad_x;
        }
    }

    grad_gamma /= N;
    grad_beta /= N;

    gamma_ -= learning_rate * grad_gamma;
    beta_ -= learning_rate * grad_beta;

    return grad_input;
}

void batchnorm_layer::save(std::ostream& os) const {
    save_header(os);

    double eps_be = host_to_bedb(eps_);
    os.write(reinterpret_cast<const char*>(&eps_be), 8);
    double mom_be = host_to_bedb(momentum_);
    os.write(reinterpret_cast<const char*>(&mom_be), 8);

    for (int i = 0; i < input_shape_.c; ++i) {
        double g_be = host_to_bedb(gamma_(i));
        os.write(reinterpret_cast<const char*>(&g_be), 8);
        double b_be = host_to_bedb(beta_(i));
        os.write(reinterpret_cast<const char*>(&b_be), 8);
    }

    for (int i = 0; i < input_shape_.c; ++i) {
        double rm_be = host_to_bedb(running_mean_(i));
        os.write(reinterpret_cast<const char*>(&rm_be), 8);
        double rv_be = host_to_bedb(running_var_(i));
        os.write(reinterpret_cast<const char*>(&rv_be), 8);
    }
}

void batchnorm_layer::load(std::istream& is) {
    load_header(is);

    double eps_be, mom_be;
    is.read(reinterpret_cast<char*>(&eps_be), 8);
    if (!is)
        throw std::runtime_error("batchnorm_layer::load: failed to read eps_");
    is.read(reinterpret_cast<char*>(&mom_be), 8);
    if (!is)
        throw std::runtime_error(
            "batchnorm_layer::load: failed to read momentum_");
    eps_ = bedb_to_host(eps_be);
    momentum_ = bedb_to_host(mom_be);

    gamma_.resize(input_shape_.c);
    beta_.resize(input_shape_.c);
    running_mean_.resize(input_shape_.c);
    running_var_.resize(input_shape_.c);

    for (int i = 0; i < input_shape_.c; ++i) {
        double g_be, b_be;
        is.read(reinterpret_cast<char*>(&g_be), 8);
        is.read(reinterpret_cast<char*>(&b_be), 8);
        if (!is)
            throw std::runtime_error(
                "batchnorm_layer::load: failed to read gamma/beta at index " +
                std::to_string(i));
        gamma_(i) = bedb_to_host(g_be);
        beta_(i) = bedb_to_host(b_be);
    }
    for (int i = 0; i < input_shape_.c; ++i) {
        double rm_be, rv_be;
        is.read(reinterpret_cast<char*>(&rm_be), 8);
        is.read(reinterpret_cast<char*>(&rv_be), 8);
        if (!is)
            throw std::runtime_error(
                "batchnorm_layer::load: failed to read running stats at "
                "index " +
                std::to_string(i));
        running_mean_(i) = bedb_to_host(rm_be);
        running_var_(i) = bedb_to_host(rv_be);
    }
}

}  // namespace mint