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

#include <cnn/layers/layer.h>
#include <process/utils.h>
#include <cstdint>
#include <stdexcept>

namespace mint {

layer::layer(
    io_shape in_shape, io_shape out_shape, layer_type type, bool training)
    : input_shape_(in_shape),
      output_shape_(out_shape),
      type_(type),
      training_(training) {}

void layer::save_type(std::ostream& os, layer_type type) {
    uint32_t type_val = static_cast<uint32_t>(type);
    type_val = host_to_be32(type_val);
    os.write(reinterpret_cast<const char*>(&type_val), 4);
}

layer::layer_type layer::load_type(std::istream& is) {
    uint32_t type_val;
    is.read(reinterpret_cast<char*>(&type_val), 4);
    if (!is)
        throw std::runtime_error("layer::load_type: failed to read layer type");
    return static_cast<layer_type>(be32_to_host(type_val));
}

void layer::save_header(std::ostream& os) const {
    uint32_t training_flag = static_cast<uint32_t>(training_);
    training_flag = host_to_be32(training_flag);
    os.write(reinterpret_cast<const char*>(&training_flag), 4);

    uint32_t input_c = host_to_be32(static_cast<uint32_t>(input_shape_.c));
    os.write(reinterpret_cast<const char*>(&input_c), 4);
    uint32_t input_w = host_to_be32(static_cast<uint32_t>(input_shape_.w));
    os.write(reinterpret_cast<const char*>(&input_w), 4);
    uint32_t input_h = host_to_be32(static_cast<uint32_t>(input_shape_.h));
    os.write(reinterpret_cast<const char*>(&input_h), 4);

    uint32_t output_c = host_to_be32(static_cast<uint32_t>(output_shape_.c));
    os.write(reinterpret_cast<const char*>(&output_c), 4);
    uint32_t output_w = host_to_be32(static_cast<uint32_t>(output_shape_.w));
    os.write(reinterpret_cast<const char*>(&output_w), 4);
    uint32_t output_h = host_to_be32(static_cast<uint32_t>(output_shape_.h));
    os.write(reinterpret_cast<const char*>(&output_h), 4);

    os.flush();
}

void layer::load_header(std::istream& is) {
    uint32_t training_flag;
    is.read(reinterpret_cast<char*>(&training_flag), 4);
    training_ = static_cast<bool>(be32_to_host(training_flag));
    if (!is)
        throw std::runtime_error(
            "layer::load_header: failed to read training flag");

    uint32_t input_c;
    is.read(reinterpret_cast<char*>(&input_c), 4);
    input_shape_.c = static_cast<int>(be32_to_host(input_c));
    if (!is)
        throw std::runtime_error(
            "layer::load_header: failed to read input shape (input_c)");

    uint32_t input_w;
    is.read(reinterpret_cast<char*>(&input_w), 4);
    input_shape_.w = static_cast<int>(be32_to_host(input_w));
    if (!is)
        throw std::runtime_error(
            "layer::load_header: failed to read input shape (input_w)");

    uint32_t input_h;
    is.read(reinterpret_cast<char*>(&input_h), 4);
    input_shape_.h = static_cast<int>(be32_to_host(input_h));
    if (!is)
        throw std::runtime_error(
            "layer::load_header: failed to read input shape (input_h)");

    if (input_shape_.w <= 0 || input_shape_.h <= 0 || input_shape_.c < 0)
        throw std::invalid_argument(
            "layer::load_header: invalid input shape (w=" +
            std::to_string(input_shape_.w) +
            ", h=" + std::to_string(input_shape_.h) +
            ", c=" + std::to_string(input_shape_.c) + ")");

    uint32_t output_c;
    is.read(reinterpret_cast<char*>(&output_c), 4);
    output_shape_.c = static_cast<int>(be32_to_host(output_c));
    if (!is)
        throw std::runtime_error(
            "layer::load_header: failed to read output shape (output_c)");

    uint32_t output_w;
    is.read(reinterpret_cast<char*>(&output_w), 4);
    output_shape_.w = static_cast<int>(be32_to_host(output_w));
    if (!is)
        throw std::runtime_error(
            "layer::load_header: failed to read output shape (output_w)");

    uint32_t output_h;
    is.read(reinterpret_cast<char*>(&output_h), 4);
    output_shape_.h = static_cast<int>(be32_to_host(output_h));
    if (!is)
        throw std::runtime_error(
            "layer::load_header: failed to read output shape (output_h)");

    if (output_shape_.w <= 0 || output_shape_.h <= 0 || output_shape_.c < 0)
        throw std::invalid_argument(
            "layer::load_header: invalid output shape (w=" +
            std::to_string(output_shape_.w) +
            ", h=" + std::to_string(output_shape_.h) +
            ", c=" + std::to_string(output_shape_.c) + ")");
}

}  // namespace mint