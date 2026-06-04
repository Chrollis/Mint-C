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

#include <api_handlers.h>
#include <core/config.h>
#include <core/evaluate_manager.h>
#include <core/logger.h>
#include <core/model_manager.h>
#include <core/train_manager.h>
#include <data/dataset_manager.h>
#include <data/image_processor.h>
#include <mint.h>

#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <sstream>
#include <string>

namespace server {

static std::string to_lower(const std::string& s) {
    std::string result = s;
    for (char& c : result)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return result;
}

static std::optional<log_level> string_to_log_level(const std::string& s) {
    std::string lower = to_lower(s);
    if (lower == "debug") return log_level::debug;
    if (lower == "info") return log_level::info;
    if (lower == "warning") return log_level::warning;
    if (lower == "error") return log_level::error;
    return std::nullopt;
}

static void send_error(
    httplib::Response& res, const std::string& msg, int status = 400) {
    nlohmann::json err;
    err["success"] = false;
    err["error"] = msg;
    res.set_content(err.dump(), "application/json");
    res.status = status;
}

static std::optional<mint::layer::layer_type> parse_layer_type(
    const std::string& type_str) {
    if (type_str == "convolve") return mint::layer::layer_type::convolve;
    if (type_str == "pool") return mint::layer::layer_type::pool;
    if (type_str == "flatten") return mint::layer::layer_type::flatten;
    if (type_str == "dense") return mint::layer::layer_type::dense;
    if (type_str == "dropout") return mint::layer::layer_type::dropout;
    if (type_str == "batchnorm") return mint::layer::layer_type::batchnorm;
    if (type_str == "activate") return mint::layer::layer_type::activate;
    if (type_str == "softmax") return mint::layer::layer_type::softmax;
    throw std::invalid_argument("Unknown layer type: " + type_str);
}

static std::vector<uint8_t> read_file_bytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open file: " + path.string());
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size))
        throw std::runtime_error("Cannot read file: " + path.string());
    return buffer;
}

static std::string generate_temp_filename(
    const std::string& prefix, const std::string& extension) {
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937_64 rng(static_cast<unsigned long long>(now));
    std::uniform_int_distribution<uint64_t> dist;
    uint64_t rand_part = dist(rng);

    std::ostringstream oss;
    oss << prefix << "_" << now << "_" << rand_part << extension;
    return oss.str();
}

void handle_health(const httplib::Request& req, httplib::Response& res) {
    nlohmann::json resp;
    resp["status"] = "ok";
    res.set_content(resp.dump(), "application/json");
    res.status = 200;
}

static auto start_time = std::chrono::steady_clock::now();

void handle_server_status(const httplib::Request& req, httplib::Response& res) {
    auto now = std::chrono::steady_clock::now();
    auto uptime =
        std::chrono::duration_cast<std::chrono::seconds>(now - start_time)
            .count();

    auto& cfg = config::instance();
    nlohmann::json resp;
    resp["uptime_seconds"] = uptime;
    res.set_content(resp.dump(), "application/json");
    res.status = 200;
}

void handle_get_logs(const httplib::Request& req, httplib::Response& res) {
    int count = 100;
    if (req.has_param("count")) {
        std::string count_str = req.get_param_value("count");
        try {
            count = std::stoi(count_str);
            if (count < 1) count = 1;
            if (count > 1000) count = 1000;
        } catch (...) {
        }
    }

    log_level min_level = log_level::info;
    if (req.has_param("level")) {
        std::string level_str = req.get_param_value("level");
        auto opt_level = string_to_log_level(level_str);
        if (opt_level.has_value()) {
            min_level = opt_level.value();
        } else {
            send_error(
                res,
                "Invalid level parameter. Allowed: debug, info, warning, "
                "error");
            return;
        }
    }

    auto& logger = logger::instance();
    std::vector<std::string> logs =
        logger.get_logs(static_cast<size_t>(count), min_level);

    nlohmann::json response;
    response["logs"] = logs;
    response["success"] = true;

    res.set_content(response.dump(), "application/json");
    res.status = 200;
}

void handle_layer_types(const httplib::Request& req, httplib::Response& res) {
    nlohmann::json result;
    result["types"] = {"convolve", "pool",      "flatten",  "dense",
                       "dropout",  "batchnorm", "activate", "softmax"};
    res.set_content(result.dump(), "application/json");
    res.status = 200;
}

void handle_parameter_range(
    const httplib::Request& req, httplib::Response& res) {
    try {
        auto body = nlohmann::json::parse(req.body);
        if (!body.contains("type") || !body["type"].is_string()) {
            send_error(res, "Missing or invalid 'type' field", 400);
            return;
        }
        if (!body.contains("in_shape") || !body["in_shape"].is_array() ||
            body["in_shape"].size() != 3) {
            send_error(
                res, "Missing or invalid 'in_shape' field (must be [c, w, h])",
                400);
            return;
        }

        std::string type_str = body["type"].get<std::string>();
        auto in_shape_arr = body["in_shape"];
        int c = in_shape_arr[0].get<int>();
        int w = in_shape_arr[1].get<int>();
        int h = in_shape_arr[2].get<int>();
        if (c <= 0 || w <= 0 || h <= 0) {
            send_error(
                res, "Invalid dimensions in 'in_shape' (must be positive)",
                400);
            return;
        }
        mint::layer::io_shape in_shape{c, w, h};

        auto ltype = parse_layer_type(type_str);
        if (!ltype.has_value()) {
            send_error(res, "Unknown layer type: " + type_str, 400);
            return;
        }

        nlohmann::json result =
            mint::model::get_parameter_range(ltype.value(), in_shape);
        res.set_content(result.dump(), "application/json");
        res.status = 200;
    } catch (const std::exception& e) {
        send_error(res, std::string("Error: ") + e.what(), 400);
    }
}

void handle_compute_output_shape(
    const httplib::Request& req, httplib::Response& res) {
    try {
        auto body = nlohmann::json::parse(req.body);
        if (!body.contains("type") || !body["type"].is_string()) {
            send_error(res, "Missing or invalid 'type' field", 400);
            return;
        }
        if (!body.contains("in_shape") || !body["in_shape"].is_array() ||
            body["in_shape"].size() != 3) {
            send_error(
                res, "Missing or invalid 'in_shape' field (must be [c, w, h])",
                400);
            return;
        }
        if (!body.contains("params") || !body["params"].is_object()) {
            send_error(
                res, "Missing or invalid 'params' field (must be an object)",
                400);
            return;
        }

        std::string type_str = body["type"].get<std::string>();
        auto in_shape_arr = body["in_shape"];
        int c = in_shape_arr[0].get<int>();
        int w = in_shape_arr[1].get<int>();
        int h = in_shape_arr[2].get<int>();
        if (c <= 0 || w <= 0 || h <= 0) {
            send_error(
                res, "Invalid dimensions in 'in_shape' (must be positive)",
                400);
            return;
        }

        mint::layer::io_shape in_shape{c, w, h};
        auto ltype = parse_layer_type(type_str);
        if (!ltype.has_value()) {
            send_error(res, "Unknown layer type: " + type_str, 400);
            return;
        }

        const nlohmann::json& params = body["params"];

        mint::layer::io_shape out_shape =
            mint::model::compute_output_shape(ltype.value(), in_shape, params);

        nlohmann::json result;
        result["out_shape"] = {out_shape.c, out_shape.w, out_shape.h};
        res.set_content(result.dump(), "application/json");
        res.status = 200;
    } catch (const std::invalid_argument& e) {
        send_error(res, e.what(), 400);
    } catch (const std::exception& e) {
        send_error(res, std::string("Error: ") + e.what(), 400);
    }
}

void handle_hyperparams_range(
    const httplib::Request& req, httplib::Response& res) {
    nlohmann::json result;
    result["learning_rate"] = {
        {"type", "double"}, {"default", 0.001}, {"min", 1e-6}, {"max", 0.1}};
    result["batch_size"] = {
        {"type", "int"}, {"default", 32}, {"min", 1}, {"max", 1024}};
    result["epochs"] = {
        {"type", "int"}, {"default", 10}, {"min", 1}, {"max", 1000}};
    res.set_content(result.dump(), "application/json");
    res.status = 200;
}

void handle_recognize(const httplib::Request& req, httplib::Response& res) {
    try {
        auto body = nlohmann::json::parse(req.body);
        if (!body.contains("base64") || !body["base64"].is_string()) {
            send_error(res, "Missing or invalid 'base64' field");
            return;
        }
        std::string base64_str = body.value("base64", "");
        if (base64_str.empty()) {
            send_error(res, "Empty base64 string");
            return;
        }

        nlohmann::json result =
            image_processor::instance().process_base64(base64_str);
        res.set_content(result.dump(), "application/json");
        res.status = result.value("success", false) ? 200 : 500;
    } catch (const std::exception& e) {
        send_error(res, std::string("JSON parse error: ") + e.what());
    }
}

void handle_get_config(const httplib::Request& req, httplib::Response& res) {
    auto& cfg = config::instance();
    nlohmann::json j;
    j["port"] = cfg.port();
    j["log_dir"] = cfg.log_dir().string();
    j["temp_dir"] = cfg.temp_dir().string();
    j["dataset_dir"] = cfg.dataset_dir().string();
    res.set_content(j.dump(), "application/json");
    res.status = 200;
}

void handle_update_config(const httplib::Request& req, httplib::Response& res) {
    try {
        nlohmann::json body = nlohmann::json::parse(req.body);
        auto& cfg = config::instance();

        if (body.contains("port") && body["port"].is_number_integer()) {
            cfg.set_port(body["port"].get<int>());
        }
        if (body.contains("log_dir") && body["log_dir"].is_string()) {
            cfg.set_log_dir(body["log_dir"].get<std::string>());
        }
        if (body.contains("temp_dir") && body["temp_dir"].is_string()) {
            cfg.set_temp_dir(body["temp_dir"].get<std::string>());
        }
        if (body.contains("dataset_dir") && body["dataset_dir"].is_string()) {
            cfg.set_dataset_dir(body["dataset_dir"].get<std::string>());
        }

        cfg.save();

        nlohmann::json resp;
        resp["success"] = true;
        res.set_content(resp.dump(), "application/json");
        res.status = 200;
    } catch (const std::exception& e) {
        send_error(res, std::string("Failed to update config: ") + e.what());
    }
}

void handle_list_datasets(const httplib::Request& req, httplib::Response& res) {
    auto& cfg = config::instance();
    std::filesystem::path dataset_root = cfg.dataset_dir();

    nlohmann::json result;
    result["datasets"] = nlohmann::json::array();

    if (!std::filesystem::exists(dataset_root) ||
        !std::filesystem::is_directory(dataset_root)) {
        send_error(
            res, "Dataset directory does not exist or is not a directory", 500);
        return;
    }

    for (const auto& entry :
         std::filesystem::directory_iterator(dataset_root)) {
        if (!entry.is_directory()) continue;

        std::string ds_name = entry.path().filename().string();
        std::filesystem::path images_path = entry.path() / "images.idx3-ubyte";
        std::filesystem::path labels_path = entry.path() / "labels.idx1-ubyte";

        nlohmann::json ds_info;
        ds_info["name"] = ds_name;

        if (!std::filesystem::exists(images_path) ||
            !std::filesystem::exists(labels_path)) {
            ds_info["sample_count"] = nullptr;
            ds_info["error"] = "Missing images.idx3-ubyte or labels.idx1-ubyte";
        } else {
            try {
                uint32_t count = dataset_manager::instance().get_sample_count(
                    images_path, labels_path);
                ds_info["sample_count"] = count;
                ds_info["error"] = nullptr;
            } catch (const std::exception& e) {
                ds_info["sample_count"] = nullptr;
                ds_info["error"] =
                    std::string("Failed to read dataset: ") + e.what();
            }
        }

        result["datasets"].push_back(ds_info);
    }

    res.set_content(result.dump(), "application/json");
    res.status = 200;
}

void handle_upload_dataset(
    const httplib::Request& req, httplib::Response& res) {
    struct TempFiles {
        std::filesystem::path images, labels;
        TempFiles(const std::filesystem::path& dir, const std::string& base)
            : images(dir / (base + "_images.tmp")),
              labels(dir / (base + "_labels.tmp")) {}
        ~TempFiles() {
            std::error_code ec;
            std::filesystem::remove(images, ec);
            std::filesystem::remove(labels, ec);
        }
    };

    try {
        if (!req.has_param("name")) {
            send_error(res, "Missing 'name' query parameter", 400);
            return;
        }
        std::string ds_name = req.get_param_value("name");
        if (ds_name.empty() || ds_name.find("..") != std::string::npos ||
            ds_name.find('/') != std::string::npos ||
            ds_name.find('\\') != std::string::npos) {
            send_error(res, "Invalid dataset name", 400);
            return;
        }

        auto& files = req.form.files;
        auto images_it = files.find("images_file");
        auto labels_it = files.find("labels_file");
        if (images_it == files.end() || labels_it == files.end()) {
            send_error(res, "Missing images_file or labels_file", 400);
            return;
        }
        const auto& images_file = images_it->second;
        const auto& labels_file = labels_it->second;

        auto& cfg = config::instance();
        std::filesystem::path temp_dir = cfg.temp_dir() / "dataset_upload_tmp";
        std::filesystem::create_directories(temp_dir);
        TempFiles tmp(temp_dir, ds_name);

        {
            std::ofstream ofs(tmp.images, std::ios::binary);
            ofs.write(images_file.content.data(), images_file.content.size());
        }
        {
            std::ofstream ofs(tmp.labels, std::ios::binary);
            ofs.write(labels_file.content.data(), labels_file.content.size());
        }

        uint32_t sample_count = 0;
        try {
            sample_count = dataset_manager::instance().get_sample_count(
                tmp.images, tmp.labels);
        } catch (const std::exception& e) {
            send_error(
                res, std::string("Dataset validation failed: ") + e.what(),
                400);
            return;
        }

        std::filesystem::path final_dir = cfg.dataset_dir() / ds_name;
        std::filesystem::create_directories(final_dir);
        std::filesystem::path final_images = final_dir / "images.idx3-ubyte";
        std::filesystem::path final_labels = final_dir / "labels.idx1-ubyte";

        std::error_code ec;
        std::filesystem::rename(tmp.images, final_images, ec);
        if (ec) {
            send_error(res, "Failed to save images file: " + ec.message(), 500);
            return;
        }
        std::filesystem::rename(tmp.labels, final_labels, ec);
        if (ec) {
            std::filesystem::remove(final_images, ec);
            send_error(res, "Failed to save labels file: " + ec.message(), 500);
            return;
        }

        nlohmann::json resp;
        resp["success"] = true;
        resp["dataset_name"] = ds_name;
        resp["sample_count"] = sample_count;
        res.set_content(resp.dump(), "application/json");
        res.status = 200;
    } catch (const std::exception& e) {
        send_error(res, std::string("Upload error: ") + e.what(), 500);
    }
}

void handle_download_dataset(
    const httplib::Request& req, httplib::Response& res) {
    if (!req.has_param("name")) {
        send_error(res, "Missing 'name' query parameter", 400);
        return;
    }
    std::string ds_name = req.get_param_value("name");
    if (ds_name.empty() || ds_name.find("..") != std::string::npos ||
        ds_name.find('/') != std::string::npos ||
        ds_name.find('\\') != std::string::npos) {
        send_error(res, "Invalid dataset name", 400);
        return;
    }

    auto& cfg = config::instance();
    std::filesystem::path ds_dir = cfg.dataset_dir() / ds_name;
    std::filesystem::path img_path = ds_dir / "images.idx3-ubyte";
    std::filesystem::path lbl_path = ds_dir / "labels.idx1-ubyte";

    if (!std::filesystem::exists(img_path) ||
        !std::filesystem::exists(lbl_path)) {
        send_error(res, "Dataset not found or missing files", 404);
        return;
    }

    try {
        auto img_data = read_file_bytes(img_path);
        auto lbl_data = read_file_bytes(lbl_path);

        uint32_t sample_count =
            dataset_manager::instance().get_sample_count(img_path, lbl_path);

        nlohmann::json result;
        result["name"] = ds_name;
        result["sample_count"] = sample_count;
        result["images_base64"] = base64_encode(img_data);
        result["labels_base64"] = base64_encode(lbl_data);

        res.set_content(result.dump(), "application/json");
        res.status = 200;
    } catch (const std::exception& e) {
        send_error(
            res, std::string("Failed to read dataset: ") + e.what(), 500);
    }
}

void handle_delete_dataset(
    const httplib::Request& req, httplib::Response& res) {
    try {
        auto body = nlohmann::json::parse(req.body);
        if (!body.contains("name") || !body["name"].is_string()) {
            send_error(res, "Missing or invalid 'name' field", 400);
            return;
        }
        std::string ds_name = body["name"].get<std::string>();
        if (ds_name.empty() || ds_name.find("..") != std::string::npos ||
            ds_name.find('/') != std::string::npos ||
            ds_name.find('\\') != std::string::npos) {
            send_error(res, "Invalid dataset name", 400);
            return;
        }

        auto& cfg = config::instance();
        std::filesystem::path ds_dir = cfg.dataset_dir() / ds_name;
        if (!std::filesystem::exists(ds_dir)) {
            send_error(res, "Dataset not found", 404);
            return;
        }
        if (!std::filesystem::is_directory(ds_dir)) {
            send_error(res, "Path is not a directory", 400);
            return;
        }

        std::error_code ec;
        std::filesystem::remove_all(ds_dir, ec);
        if (ec) {
            send_error(res, "Failed to delete dataset: " + ec.message(), 500);
            return;
        }

        nlohmann::json resp;
        resp["success"] = true;
        resp["message"] = "Dataset '" + ds_name + "' deleted";
        res.set_content(resp.dump(), "application/json");
        res.status = 200;
    } catch (const std::exception& e) {
        send_error(res, std::string("Error: ") + e.what(), 400);
    }
}

void handle_save_samples(const httplib::Request& req, httplib::Response& res) {
    try {
        auto body = nlohmann::json::parse(req.body);

        if (!body.contains("dataset_name") ||
            !body["dataset_name"].is_string() || !body.contains("samples") ||
            !body["samples"].is_array()) {
            send_error(res, "Missing required fields: dataset_name, samples");
            return;
        }

        std::string ds_name = body["dataset_name"].get<std::string>();
        if (ds_name.empty() || ds_name.find("..") != std::string::npos ||
            ds_name.find('/') != std::string::npos ||
            ds_name.find('\\') != std::string::npos) {
            send_error(res, "Invalid dataset name", 400);
            return;
        }

        bool append = body.value("append", false);

        auto& cfg = config::instance();
        std::filesystem::path ds_dir = cfg.dataset_dir() / ds_name;
        std::filesystem::create_directories(ds_dir);

        std::filesystem::path img_path = ds_dir / "images.idx3-ubyte";
        std::filesystem::path lbl_path = ds_dir / "labels.idx1-ubyte";

        std::vector<mint::mnist_sample> mnist_samples;
        for (const auto& item : body["samples"]) {
            if (!item.contains("image_base64") || !item.contains("label")) {
                send_error(
                    res, "Each sample must have 'image_base64' and 'label'");
                return;
            }
            std::string b64 = item["image_base64"].get<std::string>();
            int label = item["label"].get<int>();
            if (label < 0 || label > 9) {
                send_error(res, "Label must be between 0 and 9");
                return;
            }
            try {
                auto sample = mint::image::png_base64_to_mnist_sample(
                    b64, static_cast<uint8_t>(label));
                mnist_samples.push_back(sample);
            } catch (const std::exception& e) {
                send_error(
                    res, std::string("Failed to decode image: ") + e.what());
                return;
            }
        }

        auto result = dataset_manager::instance().save_samples(
            mnist_samples, img_path, lbl_path, append);

        res.set_content(result.dump(), "application/json");
        res.status = result.value("success", false) ? 200 : 500;
    } catch (const std::exception& e) {
        send_error(res, std::string("JSON parse error: ") + e.what());
    }
}

void handle_merge_datasets(
    const httplib::Request& req, httplib::Response& res) {
    try {
        auto body = nlohmann::json::parse(req.body);
        if (!body.contains("source_datasets") ||
            !body["source_datasets"].is_array() ||
            !body.contains("target_dataset") ||
            !body["target_dataset"].is_string()) {
            send_error(
                res,
                "Missing required fields: source_datasets (array), "
                "target_dataset (string)");
            return;
        }

        std::string target_name = body["target_dataset"].get<std::string>();
        if (target_name.empty() ||
            target_name.find("..") != std::string::npos ||
            target_name.find('/') != std::string::npos ||
            target_name.find('\\') != std::string::npos) {
            send_error(res, "Invalid target dataset name", 400);
            return;
        }

        auto& cfg = config::instance();
        std::filesystem::path target_dir = cfg.dataset_dir() / target_name;
        std::filesystem::path out_img = target_dir / "images.idx3-ubyte";
        std::filesystem::path out_lbl = target_dir / "labels.idx1-ubyte";

        std::vector<std::pair<std::filesystem::path, std::filesystem::path>>
            pairs;
        for (const auto& ds_name : body["source_datasets"]) {
            if (!ds_name.is_string()) {
                send_error(res, "Each source dataset name must be a string");
                return;
            }
            std::string name = ds_name.get<std::string>();
            if (name.empty() || name.find("..") != std::string::npos ||
                name.find('/') != std::string::npos ||
                name.find('\\') != std::string::npos) {
                send_error(res, "Invalid source dataset name: " + name, 400);
                return;
            }
            std::filesystem::path src_dir = cfg.dataset_dir() / name;
            std::filesystem::path img = src_dir / "images.idx3-ubyte";
            std::filesystem::path lbl = src_dir / "labels.idx1-ubyte";
            if (!std::filesystem::exists(img) ||
                !std::filesystem::exists(lbl)) {
                send_error(res, "Source dataset missing files: " + name, 400);
                return;
            }
            pairs.emplace_back(img, lbl);
        }

        if (pairs.empty()) {
            send_error(res, "No valid source datasets provided", 400);
            return;
        }

        std::filesystem::create_directories(target_dir);

        auto result =
            dataset_manager::instance().merge_datasets(pairs, out_img, out_lbl);
        res.set_content(result.dump(), "application/json");
        res.status = result.value("success", false) ? 200 : 500;
    } catch (const std::exception& e) {
        send_error(res, std::string("JSON parse error: ") + e.what());
    }
}

void handle_model_load_bin(
    const httplib::Request& req, httplib::Response& res) {
    try {
        auto& files = req.form.files;
        auto it = files.find("model_file");
        if (it == files.end()) {
            send_error(res, "Missing 'model_file' field", 400);
            return;
        }
        const auto& model_file = it->second;
        if (model_file.content.empty()) {
            send_error(res, "Empty model file", 400);
            return;
        }

        auto& cfg = config::instance();
        std::filesystem::path temp_dir = cfg.temp_dir();
        std::filesystem::create_directories(temp_dir);

        std::string temp_filename =
            generate_temp_filename("model_load", ".bin");
        std::filesystem::path temp_path = temp_dir / temp_filename;

        {
            std::ofstream ofs(temp_path, std::ios::binary);
            ofs.write(model_file.content.data(), model_file.content.size());
        }

        bool ok = model_manager::instance().load_from_bin(temp_path);

        std::error_code ec;
        std::filesystem::remove(temp_path, ec);

        if (ok) {
            nlohmann::json resp;
            resp["success"] = true;
            res.set_content(resp.dump(), "application/json");
            res.status = 200;
        } else {
            send_error(res, "Failed to load model from file", 500);
        }
    } catch (const std::exception& e) {
        send_error(res, std::string("Error: ") + e.what(), 500);
    }
}

void handle_model_init_json(
    const httplib::Request& req, httplib::Response& res) {
    try {
        nlohmann::json config = nlohmann::json::parse(req.body);
        if (!config.contains("input_shape") ||
            !config.contains("num_classes") || !config.contains("layers")) {
            send_error(
                res,
                "Invalid model config: missing required fields (input_shape, "
                "num_classes, layers)");
            return;
        }
        bool ok = model_manager::instance().init_from_json(config);
        if (ok) {
            nlohmann::json resp;
            resp["success"] = true;
            res.set_content(resp.dump(), "application/json");
            res.status = 200;
        } else {
            send_error(res, "Failed to initialize model from JSON", 500);
        }
    } catch (const std::exception& e) {
        send_error(res, std::string("JSON parse error: ") + e.what());
    }
}

void handle_model_save_bin(
    const httplib::Request& req, httplib::Response& res) {
    if (!model_manager::instance().is_ready()) {
        send_error(res, "Model not ready, cannot save", 400);
        return;
    }

    auto& cfg = config::instance();
    std::filesystem::path temp_dir = cfg.temp_dir();
    std::filesystem::create_directories(temp_dir);

    std::string temp_filename = generate_temp_filename("model_save", ".bin");
    std::filesystem::path temp_path = temp_dir / temp_filename;

    bool ok = model_manager::instance().save_to_bin(temp_path);
    if (!ok) {
        send_error(res, "Failed to save model to temporary file", 500);
        return;
    }

    std::vector<uint8_t> buffer;
    {
        std::ifstream ifs(temp_path, std::ios::binary | std::ios::ate);
        if (!ifs) {
            send_error(res, "Failed to read temporary model file", 500);
            std::error_code ec;
            std::filesystem::remove(temp_path, ec);
            return;
        }
        std::streamsize size = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        buffer.resize(static_cast<size_t>(size));
        if (!ifs.read(reinterpret_cast<char*>(buffer.data()), size)) {
            send_error(res, "Failed to read model data", 500);
            std::error_code ec;
            std::filesystem::remove(temp_path, ec);
            return;
        }
    }

    std::error_code ec;
    std::filesystem::remove(temp_path, ec);

    std::string content(
        reinterpret_cast<const char*>(buffer.data()), buffer.size());

    res.set_header("Content-Disposition", "attachment; filename=\"model.bin\"");
    res.set_content(content, "application/octet-stream");
    res.status = 200;
}

void handle_model_export_json(
    const httplib::Request& req, httplib::Response& res) {
    nlohmann::json config = model_manager::instance().export_to_json();
    res.set_content(config.dump(), "application/json");
    res.status = 200;
}

void handle_model_ready(const httplib::Request& req, httplib::Response& res) {
    bool ready = model_manager::instance().is_ready();
    nlohmann::json resp;
    resp["ready"] = ready;
    res.set_content(resp.dump(), "application/json");
    res.status = 200;
}

void handle_train_start(const httplib::Request& req, httplib::Response& res) {
    try {
        auto body = nlohmann::json::parse(req.body);

        if (!body.contains("dataset_name") ||
            !body["dataset_name"].is_string()) {
            send_error(res, "Missing 'dataset_name' field");
            return;
        }
        std::string ds_name = body["dataset_name"].get<std::string>();
        if (ds_name.empty() || ds_name.find("..") != std::string::npos ||
            ds_name.find('/') != std::string::npos ||
            ds_name.find('\\') != std::string::npos) {
            send_error(res, "Invalid dataset name", 400);
            return;
        }

        size_t offset = 0;
        if (body.contains("offset") && body["offset"].is_number_unsigned()) {
            offset = body["offset"].get<size_t>();
        }

        size_t count = 0;
        if (body.contains("count") && body["count"].is_number_unsigned()) {
            count = body["count"].get<size_t>();
        }

        nlohmann::json hyperparams;
        if (body.contains("hyperparams") && body["hyperparams"].is_object()) {
            hyperparams = body["hyperparams"];
        }

        auto& cfg = config::instance();
        std::filesystem::path ds_dir = cfg.dataset_dir() / ds_name;
        std::filesystem::path img_path = ds_dir / "images.idx3-ubyte";
        std::filesystem::path lbl_path = ds_dir / "labels.idx1-ubyte";

        if (!std::filesystem::exists(img_path) ||
            !std::filesystem::exists(lbl_path)) {
            send_error(res, "Dataset files not found for: " + ds_name);
            return;
        }

        uint32_t total_samples =
            dataset_manager::instance().get_sample_count(img_path, lbl_path);
        if (offset >= total_samples) {
            send_error(res, "Offset exceeds dataset size", 400);
            return;
        }
        if (count == 0 || offset + count > total_samples) {
            count = total_samples - offset;
        }

        auto samples = dataset_manager::instance().load_dataset(
            img_path, lbl_path, static_cast<uint32_t>(offset),
            static_cast<uint32_t>(count));
        if (samples.empty()) {
            send_error(res, "Failed to load dataset or dataset is empty");
            return;
        }

        bool ok =
            train_manager::instance().start_training(samples, hyperparams);
        if (ok) {
            nlohmann::json resp;
            resp["success"] = true;
            resp["message"] = "Training started with " +
                              std::to_string(samples.size()) +
                              " samples (offset " + std::to_string(offset) +
                              ", count " + std::to_string(count) + ")";
            res.set_content(resp.dump(), "application/json");
            res.status = 200;
        } else {
            send_error(
                res,
                "Failed to start training (maybe another training is already "
                "running)",
                409);
        }
    } catch (const std::exception& e) {
        send_error(res, std::string("Error: ") + e.what());
    }
}

void handle_train_cancel(const httplib::Request& req, httplib::Response& res) {
    train_manager::instance().cancel_training();
    nlohmann::json resp;
    resp["success"] = true;
    resp["message"] = "cancel requested";
    res.set_content(resp.dump(), "application/json");
    res.status = 200;
}

void handle_train_status(const httplib::Request& req, httplib::Response& res) {
    auto status = train_manager::instance().status();

    nlohmann::json resp;
    std::string state_str;
    switch (status.state) {
        case training_status::state_type::idle:
            state_str = "idle";
            break;
        case training_status::state_type::running:
            state_str = "running";
            break;
        case training_status::state_type::cancelling:
            state_str = "cancelling";
            break;
        case training_status::state_type::finished:
            state_str = "finished";
            break;
        case training_status::state_type::cancelled:
            state_str = "cancelled";
            break;
        case training_status::state_type::error:
            state_str = "error";
            break;
    }
    resp["state"] = state_str;
    resp["current_epoch"] = status.current_epoch;
    resp["total_epochs"] = status.total_epochs;
    resp["processed_samples"] = status.processed_samples;
    resp["total_samples"] = status.total_samples;
    resp["current_loss"] = status.current_loss;
    resp["current_accuracy"] = status.current_accuracy;
    resp["loss_history"] = status.loss_history;
    resp["accuracy_history"] = status.accuracy_history;
    resp["error_message"] = status.error_message;

    res.set_content(resp.dump(), "application/json");
    res.status = 200;
}

void handle_train_accept(const httplib::Request& req, httplib::Response& res) {
    try {
        auto body = nlohmann::json::parse(req.body);
        if (!body.contains("accept") || !body["accept"].is_boolean()) {
            send_error(
                res, "Missing or invalid 'accept' field (boolean required)");
            return;
        }
        bool accept = body["accept"].get<bool>();
        bool ok;
        if (accept) {
            ok = train_manager::instance().accept_trained_model();
        } else {
            ok = train_manager::instance().discard_trained_model();
        }

        nlohmann::json resp;
        if (ok) {
            resp["success"] = true;
            resp["message"] =
                accept ? "Trained model accepted and loaded into main model"
                       : "Trained model discarded";
            res.status = 200;
        } else {
            resp["success"] = false;
            resp["error"] = accept
                                ? "No trained model available or accept failed"
                                : "No trained model available to discard";
            res.status = 400;
        }
        res.set_content(resp.dump(), "application/json");
    } catch (const std::exception& e) {
        send_error(res, std::string("JSON parse error: ") + e.what());
    }
}

void handle_evaluate_start(
    const httplib::Request& req, httplib::Response& res) {
    try {
        auto body = nlohmann::json::parse(req.body);
        if (!body.contains("dataset_name") ||
            !body["dataset_name"].is_string()) {
            send_error(res, "Missing or invalid 'dataset_name' field");
            return;
        }
        std::string ds_name = body["dataset_name"].get<std::string>();
        if (ds_name.empty() || ds_name.find("..") != std::string::npos ||
            ds_name.find('/') != std::string::npos ||
            ds_name.find('\\') != std::string::npos) {
            send_error(res, "Invalid dataset name", 400);
            return;
        }

        size_t offset = 0;
        if (body.contains("offset") && body["offset"].is_number_unsigned()) {
            offset = body["offset"].get<size_t>();
        }
        size_t count = 0;
        if (body.contains("count") && body["count"].is_number_unsigned()) {
            count = body["count"].get<size_t>();
        }
        size_t batch_size = 32;
        if (body.contains("batch_size") &&
            body["batch_size"].is_number_unsigned()) {
            batch_size = body["batch_size"].get<size_t>();
            if (batch_size == 0) batch_size = 1;
        }

        bool ok = evaluate_manager::instance().start_evaluation(
            ds_name, offset, count, batch_size);
        if (ok) {
            nlohmann::json resp;
            resp["success"] = true;
            resp["message"] = "Evaluation started";
            res.set_content(resp.dump(), "application/json");
            res.status = 200;
        } else {
            send_error(
                res,
                "Failed to start evaluation (maybe another evaluation is "
                "already running)",
                409);
        }
    } catch (const std::exception& e) {
        send_error(res, std::string("JSON parse error: ") + e.what());
    }
}

void handle_evaluate_cancel(
    const httplib::Request& req, httplib::Response& res) {
    evaluate_manager::instance().cancel_evaluation();
    nlohmann::json resp;
    resp["success"] = true;
    resp["message"] = "cancel requested";
    res.set_content(resp.dump(), "application/json");
    res.status = 200;
}

void handle_evaluate_status(
    const httplib::Request& req, httplib::Response& res) {
    auto status = evaluate_manager::instance().status();
    nlohmann::json resp;

    std::string state_str;
    switch (status.state) {
        case evaluation_status::state_type::idle:
            state_str = "idle";
            break;
        case evaluation_status::state_type::running:
            state_str = "running";
            break;
        case evaluation_status::state_type::cancelling:
            state_str = "cancelling";
            break;
        case evaluation_status::state_type::finished:
            state_str = "finished";
            break;
        case evaluation_status::state_type::cancelled:
            state_str = "cancelled";
            break;
        case evaluation_status::state_type::error:
            state_str = "error";
            break;
        default:
            state_str = "unknown";
    }
    resp["state"] = state_str;
    resp["processed_samples"] = status.processed_samples;
    resp["total_samples"] = status.total_samples;
    resp["current_accuracy"] = status.current_accuracy;
    resp["error_message"] = status.error_message;

    if (status.state == evaluation_status::state_type::finished) {
        resp["final_accuracy"] = status.final_accuracy;
        resp["per_class_accuracy"] = status.per_class_accuracy;
    }

    res.set_content(resp.dump(), "application/json");
    res.status = 200;
}

void register_routes(httplib::Server& svr) {
    svr.Get("/api/health", handle_health);
    svr.Get("/api/status", handle_server_status);

    svr.Get("/api/config", handle_get_config);
    svr.Post("/api/config", handle_update_config);

    svr.Get("/api/datasets", handle_list_datasets);
    svr.Post("/api/datasets/upload", handle_upload_dataset);
    svr.Get("/api/datasets/download", handle_download_dataset);
    svr.Post("/api/datasets/delete", handle_delete_dataset);
    svr.Post("/api/datasets/save_samples", handle_save_samples);
    svr.Post("/api/datasets/merge", handle_merge_datasets);

    svr.Get("/api/logs", handle_get_logs);

    svr.Post("/api/recognize", handle_recognize);

    svr.Get("/api/layer/types", handle_layer_types);
    svr.Post("/api/layer/parameter_range", handle_parameter_range);
    svr.Post("/api/layer/compute_output_shape", handle_compute_output_shape);

    svr.Post("/api/model/load_bin", handle_model_load_bin);
    svr.Post("/api/model/init_json", handle_model_init_json);
    svr.Get("/api/model/save_bin", handle_model_save_bin);
    svr.Get("/api/model/export_json", handle_model_export_json);
    svr.Get("/api/model/ready", handle_model_ready);

    svr.Get("/api/train/hyperparams_range", handle_hyperparams_range);
    svr.Post("/api/train/start", handle_train_start);
    svr.Post("/api/train/cancel", handle_train_cancel);
    svr.Get("/api/train/status", handle_train_status);
    svr.Post("/api/train/accept", handle_train_accept);

    svr.Post("/api/evaluate/start", handle_evaluate_start);
    svr.Post("/api/evaluate/cancel", handle_evaluate_cancel);
    svr.Get("/api/evaluate/status", handle_evaluate_status);
}

}  // namespace server