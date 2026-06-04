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

#include <cnn/layers/conv_layer.h>
#include <process/utils.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace mint {

std::mt19937& conv_layer::rng() {
    thread_local static std::mt19937 rng(
        static_cast<unsigned>(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    return rng;
}

conv_layer::conv_kernel::conv_kernel(
    init_type type, int in_channels, int kernel_size)
    : weights(in_channels, Eigen::MatrixXd::Zero(kernel_size, kernel_size)),
      bias(0.0) {
    int fan_in = in_channels * kernel_size * kernel_size;
    int fan_out = kernel_size * kernel_size;
    double stddev = 0.0, limit = 0.0;
    switch (type) {
        case init_type::gaussian: {
            stddev = std::sqrt(2.0 / (fan_in + fan_out));
            std::normal_distribution<double> dist(0.0, stddev);
            for (auto& w : weights) {
                for (int i = 0; i < w.size(); ++i) w.data()[i] = dist(rng());
            }
            break;
        }
        case init_type::xavier: {
            limit = std::sqrt(6.0 / (fan_in + fan_out));
            std::uniform_real_distribution<double> udist(-limit, limit);
            for (auto& w : weights) {
                for (int i = 0; i < w.size(); ++i) w.data()[i] = udist(rng());
            }
            break;
        }
        case init_type::he: {
            stddev = std::sqrt(2.0 / fan_in);
            std::normal_distribution<double> ndist(0.0, stddev);
            for (auto& w : weights) {
                for (int i = 0; i < w.size(); ++i) w.data()[i] = ndist(rng());
            }
            break;
        }
        case init_type::load:
        default:
            break;
    }
}

conv_layer::conv_layer(
    io_shape in_shape,
    int out_channels,
    int kernel_radius,
    init_type init,
    int stride,
    int padding,
    bool training)
    : layer(
          in_shape,
          {out_channels,
           (in_shape.w + 2 * padding - (2 * kernel_radius + 1)) / stride + 1,
           (in_shape.h + 2 * padding - (2 * kernel_radius + 1)) / stride + 1},
          layer_type::convolve,
          training),
      kernel_size_(2 * kernel_radius + 1),
      stride_(stride),
      padding_(padding) {
    kernels_ = std::vector<conv_kernel>(
        output_shape_.c, conv_kernel(init, input_shape_.c, kernel_size_));
}

void conv_layer::save(std::ostream& os) const {
    save_header(os);

    uint64_t ksize = host_to_be64(static_cast<uint64_t>(kernel_size_));
    uint32_t stride = host_to_be32(static_cast<uint32_t>(stride_));
    uint32_t pad = host_to_be32(static_cast<uint32_t>(padding_));
    os.write(reinterpret_cast<const char*>(&ksize), 8);
    os.write(reinterpret_cast<const char*>(&stride), 4);
    os.write(reinterpret_cast<const char*>(&pad), 4);

    for (const auto& kernel : kernels_) {
        for (const auto& w : kernel.weights) {
            for (int i = 0; i < w.size(); ++i) {
                double val = host_to_bedb(w.data()[i]);
                os.write(reinterpret_cast<const char*>(&val), 8);
            }
        }
        double bias_be = host_to_bedb(kernel.bias);
        os.write(reinterpret_cast<const char*>(&bias_be), 8);
    }
}

void conv_layer::load(std::istream& is) {
    load_header(is);

    uint64_t ksize;
    uint32_t stride, pad;
    is.read(reinterpret_cast<char*>(&ksize), 8);
    if (!is)
        throw std::runtime_error(
            "conv_layer::load: failed to read kernel size");
    kernel_size_ = static_cast<int>(be64_to_host(ksize));

    is.read(reinterpret_cast<char*>(&stride), 4);
    if (!is)
        throw std::runtime_error("conv_layer::load: failed to read stride");
    stride_ = static_cast<int>(be32_to_host(stride));

    is.read(reinterpret_cast<char*>(&pad), 4);
    if (!is)
        throw std::runtime_error("conv_layer::load: failed to read padding");
    padding_ = static_cast<int>(be32_to_host(pad));

    kernels_ = std::vector<conv_kernel>(
        output_shape_.c,
        conv_kernel(init_type::load, input_shape_.c, kernel_size_));

    for (auto& kernel : kernels_) {
        for (auto& w : kernel.weights) {
            for (int i = 0; i < w.size(); ++i) {
                double val;
                is.read(reinterpret_cast<char*>(&val), 8);
                if (!is)
                    throw std::runtime_error(
                        "conv_layer::load: failed to read weights");
                w.data()[i] = bedb_to_host(val);
            }
        }
        double bias_be;
        is.read(reinterpret_cast<char*>(&bias_be), 8);
        if (!is)
            throw std::runtime_error("conv_layer::load: failed to read bias");
        kernel.bias = bedb_to_host(bias_be);
    }
}

std::vector<Eigen::MatrixXd> conv_layer::convolve_general(
    const std::vector<Eigen::MatrixXd>& padded_input,
    const std::vector<conv_kernel>& kernels,
    int stride,
    int out_h,
    int out_w) {
    int in_channels = static_cast<int>(padded_input.size());
    int out_channels = static_cast<int>(kernels.size());
    int k = (kernels.empty() ? 0 : kernels[0].weights[0].rows());

    std::vector<Eigen::MatrixXd> output(out_channels);
    for (int oc = 0; oc < out_channels; ++oc)
        output[oc] = Eigen::MatrixXd::Zero(out_h, out_w);

    for (int y = 0; y < out_h; ++y) {
        int top = y * stride;
        for (int x = 0; x < out_w; ++x) {
            int left = x * stride;
            for (int oc = 0; oc < out_channels; ++oc) {
                double sum = kernels[oc].bias;
                for (int ic = 0; ic < in_channels; ++ic) {
                    const auto& in_block =
                        padded_input[ic].block(top, left, k, k);
                    const auto& weight = kernels[oc].weights[ic];
                    sum += (in_block.array() * weight.array()).sum();
                }
                output[oc](y, x) = sum;
            }
        }
    }
    return output;
}

std::vector<std::vector<Eigen::MatrixXd>> conv_layer::forward(
    const std::vector<std::vector<Eigen::MatrixXd>>& input) {
    if (input.empty() || input[0].empty()) return {};

    if (static_cast<int>(input[0].size()) != input_shape_.c ||
        input[0][0].rows() != input_shape_.h ||
        input[0][0].cols() != input_shape_.w) {
        throw std::invalid_argument(
            "conv_layer::forward: input shape mismatch, expected [c=" +
            std::to_string(input_shape_.c) +
            ", w=" + std::to_string(input_shape_.w) +
            ", h=" + std::to_string(input_shape_.h) + "]");
    }

    if (training_) cached_input_ = input;

    std::vector<std::vector<Eigen::MatrixXd>> output(input.size());

    for (size_t b = 0; b < input.size(); ++b) {
        std::vector<Eigen::MatrixXd> padded_input(input_shape_.c);
        for (int ic = 0; ic < input_shape_.c; ++ic) {
            const auto& mat = input[b][ic];
            if (padding_ == 0) {
                padded_input[ic] = mat;
            } else {
                padded_input[ic] = Eigen::MatrixXd::Zero(
                    input_shape_.h + 2 * padding_,
                    input_shape_.w + 2 * padding_);
                padded_input[ic].block(
                    padding_, padding_, input_shape_.h, input_shape_.w) = mat;
            }
        }
        output[b] = convolve_general(
            padded_input, kernels_, stride_, output_shape_.h, output_shape_.w);
    }
    return output;
}

std::vector<std::vector<Eigen::MatrixXd>> conv_layer::backward(
    const std::vector<std::vector<Eigen::MatrixXd>>& grad_output,
    double learning_rate) {
    if (!training_) {
        throw std::invalid_argument(
            "conv_layer::backward: cannot backward when layer is not in "
            "training mode");
    }
    if (cached_input_.size() != grad_output.size()) {
        throw std::invalid_argument(
            "conv_layer::backward: cached_input batch size (" +
            std::to_string(cached_input_.size()) +
            ") != grad_output batch size (" +
            std::to_string(grad_output.size()) + ")");
    }
    if (grad_output[0].size() != static_cast<size_t>(output_shape_.c) ||
        grad_output[0][0].cols() != output_shape_.w ||
        grad_output[0][0].rows() != output_shape_.h) {
        throw std::invalid_argument(
            "conv_layer::backward: grad_output shape mismatch, expected [c=" +
            std::to_string(output_shape_.c) +
            ", w=" + std::to_string(output_shape_.w) +
            ", h=" + std::to_string(output_shape_.h) + "]");
    }

    std::vector<conv_kernel> grad_kernels;
    grad_kernels.reserve(output_shape_.c);
    for (int oc = 0; oc < output_shape_.c; ++oc) {
        grad_kernels.emplace_back(
            init_type::load, input_shape_.c, kernel_size_);
        grad_kernels.back().bias = 0.0;
    }

    const int in_h_padded = input_shape_.h + 2 * padding_;
    const int in_w_padded = input_shape_.w + 2 * padding_;

    std::vector<std::vector<Eigen::MatrixXd>> grad_input_padded(
        grad_output.size());
    for (size_t b = 0; b < grad_output.size(); ++b) {
        grad_input_padded[b].resize(input_shape_.c);
        for (int ic = 0; ic < input_shape_.c; ++ic) {
            grad_input_padded[b][ic] =
                Eigen::MatrixXd::Zero(in_h_padded, in_w_padded);
        }
    }

    for (size_t b = 0; b < grad_output.size(); ++b) {
        std::vector<Eigen::MatrixXd> padded_input(input_shape_.c);
        for (int ic = 0; ic < input_shape_.c; ++ic) {
            const auto& mat = cached_input_[b][ic];
            if (padding_ == 0) {
                padded_input[ic] = mat;
            } else {
                padded_input[ic] =
                    Eigen::MatrixXd::Zero(in_h_padded, in_w_padded);
                padded_input[ic].block(
                    padding_, padding_, input_shape_.h, input_shape_.w) = mat;
            }
        }

        const auto& grad_out_b = grad_output[b];
        for (int y = 0; y < output_shape_.h; ++y) {
            int top = y * stride_;
            for (int x = 0; x < output_shape_.w; ++x) {
                int left = x * stride_;
                for (int oc = 0; oc < output_shape_.c; ++oc) {
                    double grad_val = grad_out_b[oc](y, x);
                    grad_kernels[oc].bias += grad_val;
                    for (int ic = 0; ic < input_shape_.c; ++ic) {
                        const auto& in_block = padded_input[ic].block(
                            top, left, kernel_size_, kernel_size_);
                        grad_kernels[oc].weights[ic] += grad_val * in_block;
                    }
                }
            }
        }
    }

    double inv_batch = 1.0 / grad_output.size();
    for (int oc = 0; oc < output_shape_.c; ++oc) {
        auto& kernel = kernels_[oc];
        const auto& gk = grad_kernels[oc];
        for (int ic = 0; ic < input_shape_.c; ++ic) {
            kernel.weights[ic] -= learning_rate * (gk.weights[ic] * inv_batch);
        }
        kernel.bias -= learning_rate * (gk.bias * inv_batch);
    }

    std::vector<conv_kernel> rot_kernels(input_shape_.c);
    for (int ic = 0; ic < input_shape_.c; ++ic) {
        rot_kernels[ic].weights.resize(output_shape_.c);
        rot_kernels[ic].bias = 0.0;
        for (int oc = 0; oc < output_shape_.c; ++oc) {
            rot_kernels[ic].weights[oc] = kernels_[oc].weights[ic].reverse();
        }
    }

    int up_h = (output_shape_.h - 1) * stride_ + 1;
    int up_w = (output_shape_.w - 1) * stride_ + 1;

    int pad_t_h = (input_shape_.h + kernel_size_ - 1 - up_h) / 2;
    int pad_t_w = (input_shape_.w + kernel_size_ - 1 - up_w) / 2;
    if (pad_t_h < 0 || pad_t_w < 0 ||
        (input_shape_.h + kernel_size_ - 1 - up_h) % 2 != 0 ||
        (input_shape_.w + kernel_size_ - 1 - up_w) % 2 != 0) {
        throw std::runtime_error(
            "conv_layer::backward: cannot compute transpose padding");
    }

    std::vector<std::vector<Eigen::MatrixXd>> grad_input(grad_output.size());
    for (size_t b = 0; b < grad_output.size(); ++b) {
        std::vector<Eigen::MatrixXd> up_grad(output_shape_.c);
        for (int oc = 0; oc < output_shape_.c; ++oc) {
            up_grad[oc] = Eigen::MatrixXd::Zero(up_h, up_w);
            const auto& src = grad_output[b][oc];
            for (int y = 0; y < output_shape_.h; ++y) {
                for (int x = 0; x < output_shape_.w; ++x) {
                    up_grad[oc](y * stride_, x * stride_) = src(y, x);
                }
            }
        }

        std::vector<Eigen::MatrixXd> padded_up_grad(output_shape_.c);
        for (int oc = 0; oc < output_shape_.c; ++oc) {
            padded_up_grad[oc] =
                Eigen::MatrixXd::Zero(up_h + 2 * pad_t_h, up_w + 2 * pad_t_w);
            padded_up_grad[oc].block(pad_t_h, pad_t_w, up_h, up_w) =
                up_grad[oc];
        }

        std::vector<Eigen::MatrixXd> grad_padded = convolve_general(
            padded_up_grad, rot_kernels, 1, input_shape_.h, input_shape_.w);

        for (size_t b = 0; b < grad_output.size(); ++b) {
            grad_input[b].resize(input_shape_.c);
            for (int ic = 0; ic < input_shape_.c; ++ic) {
                grad_input[b][ic] = grad_padded[ic];
            }
        }
    }

    return grad_input;
}

}  // namespace mint