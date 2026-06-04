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
#include <cnn/models/model.h>
#include <process/utils.h>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace mint {

model model::from_json(const nlohmann::json& config) {
    model m;
    layer::io_shape in_shape;

    if (!config.contains("input_shape") || !config["input_shape"].is_array()) {
        throw std::invalid_argument(
            "model::from_json: missing or invalid 'input_shape' field, "
            "expected array of 3 ints");
    }
    auto arr = config["input_shape"];
    if (arr.size() < 3) {
        throw std::invalid_argument(
            "model::from_json: 'input_shape' must have at least 3 elements, "
            "got " +
            std::to_string(arr.size()));
    }
    in_shape.c = arr[0].get<int>();
    in_shape.w = arr[1].get<int>();
    in_shape.h = arr[2].get<int>();

    if (!config.contains("num_classes")) {
        throw std::invalid_argument(
            "model::from_json: missing 'num_classes' field");
    }
    int num_classes = config["num_classes"].get<int>();
    if (num_classes <= 0) {
        throw std::invalid_argument(
            "model::from_json: 'num_classes' must be positive, got " +
            std::to_string(num_classes));
    }

    if (!config.contains("layers") || !config["layers"].is_array()) {
        throw std::invalid_argument(
            "model::from_json: missing or invalid 'layers' field, expected "
            "array");
    }

    layer::io_shape current_shape = in_shape;
    for (const auto& layer_json : config["layers"]) {
        if (!layer_json.contains("type")) {
            throw std::invalid_argument(
                "model::from_json: each layer must contain a 'type' field");
        }
        std::string type_str = layer_json["type"];
        layer::layer_type type;
        if (type_str == "convolve")
            type = layer::layer_type::convolve;
        else if (type_str == "pool")
            type = layer::layer_type::pool;
        else if (type_str == "flatten")
            type = layer::layer_type::flatten;
        else if (type_str == "dense")
            type = layer::layer_type::dense;
        else if (type_str == "dropout")
            type = layer::layer_type::dropout;
        else if (type_str == "batchnorm")
            type = layer::layer_type::batchnorm;
        else if (type_str == "activate")
            type = layer::layer_type::activate;
        else if (type_str == "softmax")
            type = layer::layer_type::softmax;
        else {
            throw std::invalid_argument(
                "model::from_json: unknown layer type '" + type_str + "'");
        }

        nlohmann::json params = layer_json;
        params.erase("type");

        auto l = make_layer(type, current_shape, params);
        current_shape = l->output_shape();
        m.net_.add_layer(std::move(l));
    }

    auto out_shape = m.net_.output_shape();
    if (out_shape.c != 1 || out_shape.w != num_classes || out_shape.h != 1) {
        throw std::runtime_error(
            "model::from_json: final output shape (" +
            std::to_string(out_shape.c) + "," + std::to_string(out_shape.w) +
            "," + std::to_string(out_shape.h) +
            ") does not match expected (1," + std::to_string(num_classes) +
            ",1)");
    }
    return m;
}

nlohmann::json model::to_json() const {
    nlohmann::json config;
    auto in_shape = net_.input_shape();
    config["input_shape"] = {in_shape.c, in_shape.w, in_shape.h};
    auto out_shape = net_.output_shape();
    config["num_classes"] = out_shape.w;

    nlohmann::json layers_array = nlohmann::json::array();
    for (const auto& l : net_.layers()) {
        nlohmann::json layer_json;
        switch (l->type()) {
            case layer::layer_type::convolve: {
                layer_json["type"] = "convolve";
                if (auto* conv = dynamic_cast<conv_layer*>(l.get())) {
                    layer_json["out_channels"] = conv->output_shape().c;
                    layer_json["kernel_radius"] = (conv->kernel_size() - 1) / 2;
                    layer_json["stride"] = conv->stride();
                    layer_json["padding"] = conv->padding();
                    layer_json["init"] = "load";
                }
                break;
            }
            case layer::layer_type::pool: {
                layer_json["type"] = "pool";
                if (auto* pool = dynamic_cast<pool_layer*>(l.get())) {
                    layer_json["kernel_size"] = pool->kernel_size();
                    layer_json["stride"] = pool->stride();
                    layer_json["pool_type"] =
                        (pool->p_type() == pool_layer::pool_type::max) ? "max"
                                                                       : "avg";
                }
                break;
            }
            case layer::layer_type::flatten:
                layer_json["type"] = "flatten";
                break;
            case layer::layer_type::dense: {
                layer_json["type"] = "dense";
                if (auto* dense = dynamic_cast<dense_layer*>(l.get())) {
                    layer_json["out_features"] = dense->output_shape().w;
                    layer_json["init"] = "load";
                }
                break;
            }
            case layer::layer_type::dropout: {
                layer_json["type"] = "dropout";
                if (auto* drop = dynamic_cast<dropout_layer*>(l.get())) {
                    layer_json["keep_prob"] = drop->keep_prob();
                }
                break;
            }
            case layer::layer_type::batchnorm: {
                layer_json["type"] = "batchnorm";
                if (auto* bn = dynamic_cast<batchnorm_layer*>(l.get())) {
                    layer_json["eps"] = bn->eps();
                    layer_json["momentum"] = bn->momentum();
                }
                break;
            }
            case layer::layer_type::activate: {
                layer_json["type"] = "activate";
                if (auto* act = dynamic_cast<activate_layer*>(l.get())) {
                    std::string act_name;
                    switch (act->act_type()) {
                        case activate::func_type::sigmoid:
                            act_name = "sigmoid";
                            break;
                        case activate::func_type::tanh:
                            act_name = "tanh";
                            break;
                        case activate::func_type::relu:
                            act_name = "relu";
                            break;
                        case activate::func_type::lrelu:
                            act_name = "lrelu";
                            break;
                        case activate::func_type::linear:
                            act_name = "linear";
                            break;
                    }
                    layer_json["activation"] = act_name;
                }
                break;
            }
            case layer::layer_type::softmax:
                layer_json["type"] = "softmax";
                break;
            default:
                continue;
        }
        layers_array.push_back(layer_json);
    }
    config["layers"] = layers_array;
    return config;
}

std::vector<std::pair<uint8_t, double>> model::predict(
    const std::vector<std::vector<Eigen::MatrixXd>>& input) {
    auto output = net_.forward(input);
    std::vector<std::pair<uint8_t, double>> results;
    results.reserve(output.size());

    for (size_t i = 0; i < output.size(); ++i) {
        if (output[i].empty()) {
            throw std::runtime_error(
                "model::predict: sample " + std::to_string(i) +
                " has empty output (no channels)");
        }
        const auto& prob_row = output[i][0];
        if (prob_row.rows() != 1 || prob_row.cols() == 0) {
            throw std::runtime_error(
                "model::predict: sample " + std::to_string(i) +
                " output shape is " + std::to_string(prob_row.rows()) + "x" +
                std::to_string(prob_row.cols()) + ", expected 1xN");
        }
        Eigen::Index max_idx;
        double max_val = prob_row.row(0).maxCoeff(&max_idx);
        results.emplace_back(static_cast<uint8_t>(max_idx), max_val);
    }
    return results;
}

nlohmann::json model::train(
    const std::vector<std::vector<Eigen::MatrixXd>>& input,
    std::vector<uint8_t> labels,
    double learning_rate) {
    if (input.size() != labels.size()) {
        throw std::invalid_argument(
            "model::train: input size (" + std::to_string(input.size()) +
            ") != labels size (" + std::to_string(labels.size()) + ")");
    }
    if (learning_rate <= 0.0) {
        throw std::invalid_argument(
            "model::train: learning_rate must be positive, got " +
            std::to_string(learning_rate));
    }

    net_.train();
    auto output = net_.forward(input);

    std::vector<Eigen::MatrixXd> probs;
    probs.reserve(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        if (output[i].empty()) {
            throw std::runtime_error(
                "model::train: sample " + std::to_string(i) +
                " has empty output");
        }
        probs.push_back(output[i][0]);
    }

    double loss = compute_loss(probs, labels);
    auto grad = loss_gradient(probs, labels);

    auto unused = net_.backward(grad, learning_rate);
    (void)unused;

    size_t correct = 0;
    for (size_t i = 0; i < probs.size(); ++i) {
        Eigen::Index max_idx;
        probs[i].row(0).maxCoeff(&max_idx);
        if (static_cast<uint8_t>(max_idx) == labels[i]) ++correct;
    }
    double accuracy = static_cast<double>(correct) / probs.size();

    nlohmann::json info;
    info["loss"] = loss;
    info["accuracy"] = accuracy;
    info["learning_rate"] = learning_rate;
    info["num_samples"] = probs.size();

    return info;
}

double model::evaluate(
    const std::vector<std::vector<Eigen::MatrixXd>>& input,
    std::vector<uint8_t> labels) {
    auto predictions = predict(input);
    if (predictions.size() != labels.size()) {
        throw std::invalid_argument(
            "model::evaluate: predictions size (" +
            std::to_string(predictions.size()) + ") != labels size (" +
            std::to_string(labels.size()) + ")");
    }
    size_t correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i].first == labels[i]) ++correct;
    }
    return static_cast<double>(correct) / predictions.size();
}

void model::save(std::filesystem::path filepath) const {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error(
            "model::save: cannot open file for writing: " + filepath.string());
    }
    ofs.exceptions(std::ios::failbit | std::ios::badbit);
    uint32_t magic = 0x4D4F444C;
    uint32_t version = 1;
    magic = host_to_be32(magic);
    version = host_to_be32(version);
    ofs.write(reinterpret_cast<const char*>(&magic), 4);
    ofs.write(reinterpret_cast<const char*>(&version), 4);
    net_.save(ofs);
    if (!ofs) {
        throw std::runtime_error(
            "model::save: write failed for " + filepath.string());
    }
}

void model::load(std::filesystem::path filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error(
            "model::load: cannot open file for reading: " + filepath.string());
    }
    uint32_t magic, version;
    ifs.read(reinterpret_cast<char*>(&magic), 4);
    ifs.read(reinterpret_cast<char*>(&version), 4);
    if (!ifs) {
        throw std::runtime_error(
            "model::load: failed to read magic/version from " +
            filepath.string());
    }
    magic = be32_to_host(magic);
    version = be32_to_host(version);
    if (magic != 0x4D4F444C) {
        throw std::runtime_error(
            "model::load: invalid magic number " + std::to_string(magic) +
            ", expected 1297040460 in file " + filepath.string());
    }
    if (version != 1) {
        throw std::runtime_error(
            "model::load: unsupported version " + std::to_string(version) +
            ", expected 1 in file " + filepath.string());
    }
    net_.load(ifs);
    if (!ifs) {
        throw std::runtime_error(
            "model::load: read failed for " + filepath.string());
    }
}

nlohmann::json model::get_parameter_range(
    layer::layer_type type, const layer::io_shape& in_shape) {
    nlohmann::json range;
    switch (type) {
        case layer::layer_type::convolve: {
            range["out_channels"] = {
                {"type", "int"}, {"default", 32}, {"min", 1}, {"max", 512}};
            int max_radius = std::min(in_shape.w, in_shape.h) / 2;
            if (max_radius < 1) max_radius = 1;
            range["kernel_radius"] = {
                {"type", "int"},
                {"default", 1},
                {"min", 1},
                {"max", max_radius}};
            range["stride"] = {
                {"type", "int"}, {"default", 1}, {"min", 1}, {"max", 8}};
            range["padding"] = {
                {"type", "int"}, {"default", 0}, {"min", 0}, {"max", 8}};
            range["init"] = {
                {"type", "string"},
                {"default", "he"},
                {"enum", {"gaussian", "xavier", "he"}}};
            break;
        }
        case layer::layer_type::pool: {
            int max_kernel = std::min(in_shape.w, in_shape.h);
            range["kernel_size"] = {
                {"type", "int"},
                {"default", 2},
                {"min", 1},
                {"max", max_kernel}};
            range["stride"] = {
                {"type", "int"},
                {"default", 2},
                {"min", 1},
                {"max", max_kernel}};
            range["pool_type"] = {
                {"type", "string"},
                {"default", "max"},
                {"enum", {"max", "avg"}}};
            break;
        }
        case layer::layer_type::dense: {
            int in_features = in_shape.w;
            range["out_features"] = {
                {"type", "int"},
                {"default", std::max(10, in_features / 2)},
                {"min", 1},
                {"max", 4096}};
            range["init"] = {
                {"type", "string"},
                {"default", "he"},
                {"enum", {"gaussian", "xavier", "he"}}};
            break;
        }
        case layer::layer_type::dropout:
            range["keep_prob"] = {
                {"type", "double"},
                {"default", 0.5},
                {"min", 0.0},
                {"max", 1.0}};
            break;
        case layer::layer_type::batchnorm:
            range["eps"] = {
                {"type", "double"},
                {"default", 1e-5},
                {"min", 1e-8},
                {"max", 1e-2}};
            range["momentum"] = {
                {"type", "double"},
                {"default", 0.9},
                {"min", 0.0},
                {"max", 1.0}};
            break;
        case layer::layer_type::activate:
            range["activation"] = {
                {"type", "string"},
                {"default", "relu"},
                {"enum", {"sigmoid", "tanh", "relu", "lrelu", "linear"}}};
            break;
        case layer::layer_type::softmax:
        default:
            break;
    }
    return range;
}

layer::io_shape model::compute_output_shape(
    layer::layer_type type,
    const layer::io_shape& in_shape,
    const nlohmann::json& params) {
    switch (type) {
        case layer::layer_type::convolve: {
            int out_channels = params.value("out_channels", 1);
            int kernel_radius = params.value("kernel_radius", 1);
            int stride = params.value("stride", 1);
            int padding = params.value("padding", 0);
            int kernel_size = 2 * kernel_radius + 1;
            int out_w = (in_shape.w + 2 * padding - kernel_size) / stride + 1;
            int out_h = (in_shape.h + 2 * padding - kernel_size) / stride + 1;
            if (out_w <= 0 || out_h <= 0) {
                throw std::invalid_argument(
                    "model::compute_output_shape: convolution yields invalid "
                    "spatial dimensions (out_w=" +
                    std::to_string(out_w) + ", out_h=" + std::to_string(out_h) +
                    ") for input (" + std::to_string(in_shape.w) + "," +
                    std::to_string(in_shape.h) +
                    "), kernel_radius=" + std::to_string(kernel_radius) +
                    ", stride=" + std::to_string(stride) +
                    ", padding=" + std::to_string(padding));
            }
            return {out_channels, out_w, out_h};
        }
        case layer::layer_type::pool: {
            int kernel_size = params.value("kernel_size", 2);
            int stride = params.value("stride", kernel_size);
            int out_w = (in_shape.w - kernel_size) / stride + 1;
            int out_h = (in_shape.h - kernel_size) / stride + 1;
            if (out_w <= 0 || out_h <= 0) {
                throw std::invalid_argument(
                    "model::compute_output_shape: pooling yields invalid "
                    "dimensions (out_w=" +
                    std::to_string(out_w) + ", out_h=" + std::to_string(out_h) +
                    ") for input (" + std::to_string(in_shape.w) + "," +
                    std::to_string(in_shape.h) +
                    "), kernel_size=" + std::to_string(kernel_size) +
                    ", stride=" + std::to_string(stride));
            }
            return {in_shape.c, out_w, out_h};
        }
        case layer::layer_type::flatten:
            return {1, in_shape.c * in_shape.w * in_shape.h, 1};
        case layer::layer_type::dense: {
            int out_features = params.value("out_features", 10);
            if (out_features <= 0) {
                throw std::invalid_argument(
                    "model::compute_output_shape: out_features must be "
                    "positive, got " +
                    std::to_string(out_features));
            }
            return {1, out_features, 1};
        }
        case layer::layer_type::dropout:
        case layer::layer_type::batchnorm:
        case layer::layer_type::activate:
        case layer::layer_type::softmax:
            return in_shape;
        default:
            throw std::invalid_argument(
                "model::compute_output_shape: unsupported layer type " +
                std::to_string(static_cast<int>(type)));
    }
}

double model::compute_loss(
    const std::vector<Eigen::MatrixXd>& probs,
    const std::vector<uint8_t>& labels) {
    if (probs.size() != labels.size()) {
        throw std::invalid_argument(
            "model::compute_loss: probs size (" + std::to_string(probs.size()) +
            ") != labels size (" + std::to_string(labels.size()) + ")");
    }
    double total_loss = 0.0;
    const double eps = 1e-12;
    for (size_t i = 0; i < probs.size(); ++i) {
        const auto& p = probs[i];
        if (p.rows() != 1) {
            throw std::runtime_error(
                "model::compute_loss: probability matrix for sample " +
                std::to_string(i) + " has rows=" + std::to_string(p.rows()) +
                ", expected 1");
        }
        int label = labels[i];
        if (label < 0 || label >= p.cols()) {
            throw std::out_of_range(
                "model::compute_loss: label " + std::to_string(label) +
                " out of range [0, " + std::to_string(p.cols() - 1) +
                "] for sample " + std::to_string(i));
        }
        double prob = p(0, label);
        prob = std::max(prob, eps);
        total_loss += -std::log(prob);
    }
    return total_loss / probs.size();
}

std::vector<std::vector<Eigen::MatrixXd>> model::loss_gradient(
    const std::vector<Eigen::MatrixXd>& probs,
    const std::vector<uint8_t>& labels) {
    if (probs.size() != labels.size()) {
        throw std::invalid_argument(
            "model::loss_gradient: probs size (" +
            std::to_string(probs.size()) + ") != labels size (" +
            std::to_string(labels.size()) + ")");
    }
    std::vector<std::vector<Eigen::MatrixXd>> grad;
    grad.reserve(probs.size());
    for (size_t i = 0; i < probs.size(); ++i) {
        const auto& p = probs[i];
        if (p.rows() != 1) {
            throw std::runtime_error(
                "model::loss_gradient: probability matrix for sample " +
                std::to_string(i) + " has rows=" + std::to_string(p.rows()) +
                ", expected 1");
        }
        int label = labels[i];
        if (label < 0 || label >= p.cols()) {
            throw std::out_of_range(
                "model::loss_gradient: label " + std::to_string(label) +
                " out of range [0, " + std::to_string(p.cols() - 1) +
                "] for sample " + std::to_string(i));
        }
        Eigen::MatrixXd grad_vec = p;
        grad_vec(0, label) -= 1.0;
        grad.push_back({std::move(grad_vec)});
    }
    return grad;
}

std::unique_ptr<layer> model::make_layer(
    layer::layer_type type,
    layer::io_shape in_shape,
    const nlohmann::json& params) {
    switch (type) {
        case layer::layer_type::convolve: {
            int out_channels = params.value("out_channels", 1);
            int kernel_radius = params.value("kernel_radius", 1);
            int stride = params.value("stride", 1);
            int padding = params.value("padding", 0);
            std::string init_str = params.value("init", "he");
            conv_layer::init_type init;
            if (init_str == "gaussian")
                init = conv_layer::init_type::gaussian;
            else if (init_str == "xavier")
                init = conv_layer::init_type::xavier;
            else if (init_str == "he")
                init = conv_layer::init_type::he;
            else
                init = conv_layer::init_type::load;
            return std::make_unique<conv_layer>(
                in_shape, out_channels, kernel_radius, init, stride, padding,
                true);
        }
        case layer::layer_type::pool: {
            int kernel_size = params.value("kernel_size", 2);
            int stride = params.value("stride", kernel_size);
            std::string pool_type_str = params.value("pool_type", "max");
            pool_layer::pool_type ptype = (pool_type_str == "max")
                                              ? pool_layer::pool_type::max
                                              : pool_layer::pool_type::avg;
            return std::make_unique<pool_layer>(
                in_shape, kernel_size, stride, ptype, true);
        }
        case layer::layer_type::flatten:
            return std::make_unique<flatten_layer>(in_shape, true);
        case layer::layer_type::dense: {
            int out_features = params.value("out_features", 10);
            std::string init_str = params.value("init", "he");
            dense_layer::init_type init;
            if (init_str == "gaussian")
                init = dense_layer::init_type::gaussian;
            else if (init_str == "xavier")
                init = dense_layer::init_type::xavier;
            else if (init_str == "he")
                init = dense_layer::init_type::he;
            else
                init = dense_layer::init_type::load;
            layer::io_shape out_shape{1, out_features, 1};
            return std::make_unique<dense_layer>(
                in_shape, out_shape, init, true);
        }
        case layer::layer_type::dropout: {
            double keep_prob = params.value("keep_prob", 0.5);
            if (keep_prob < 0.0 || keep_prob > 1.0) {
                throw std::invalid_argument(
                    "model::make_layer: dropout keep_prob=" +
                    std::to_string(keep_prob) + " out of range [0,1]");
            }
            return std::make_unique<dropout_layer>(in_shape, keep_prob, true);
        }
        case layer::layer_type::batchnorm: {
            double eps = params.value("eps", 1e-5);
            double momentum = params.value("momentum", 0.9);
            return std::make_unique<batchnorm_layer>(
                in_shape, eps, momentum, true);
        }
        case layer::layer_type::activate: {
            std::string act_name = params.value("activation", "relu");
            activate::func_type atype;
            if (act_name == "sigmoid")
                atype = activate::func_type::sigmoid;
            else if (act_name == "tanh")
                atype = activate::func_type::tanh;
            else if (act_name == "relu")
                atype = activate::func_type::relu;
            else if (act_name == "lrelu")
                atype = activate::func_type::lrelu;
            else if (act_name == "linear")
                atype = activate::func_type::linear;
            else {
                throw std::invalid_argument(
                    "model::make_layer: unknown activation '" + act_name + "'");
            }
            return std::make_unique<activate_layer>(in_shape, true, atype);
        }
        case layer::layer_type::softmax:
            return std::make_unique<softmax_layer>(in_shape, true);
        default:
            throw std::invalid_argument(
                "model::make_layer: unsupported layer type " +
                std::to_string(static_cast<int>(type)));
    }
}

}  // namespace mint