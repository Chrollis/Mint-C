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

#include <core/config.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

namespace server {

config& config::instance() {
    static config cfg;
    return cfg;
}

void config::load(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        save(path);
        return;
    }
    try {
        nlohmann::json data = nlohmann::json::parse(f);
        if (data.contains("port")) port_ = data["port"].get<int>();
        if (data.contains("log_dir"))
            log_dir_ = data["log_dir"].get<std::string>();
        if (data.contains("temp_dir"))
            temp_dir_ = data["temp_dir"].get<std::string>();
        if (data.contains("dataset_dir"))
            dataset_dir_ = data["dataset_dir"].get<std::string>();
    } catch (const std::exception& e) {
        (void)e;
        save(path);
    }
}

void config::save(const std::filesystem::path& path) const {
    nlohmann::json data;
    data["port"] = port_;
    data["log_dir"] = log_dir_.string();
    data["temp_dir"] = temp_dir_.string();
    data["dataset_dir"] = dataset_dir_.string();

    std::filesystem::create_directories(path.parent_path());

    std::ofstream f(path);
    if (f.is_open()) {
        f << data.dump(4);
        f.close();
    }
}

}  // namespace server