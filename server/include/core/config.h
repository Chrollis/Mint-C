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

#include <filesystem>

namespace server {

class config {
public:
    static config& instance();

    void load(const std::filesystem::path& path = "./config.json");

    void save(const std::filesystem::path& path = "./config.json") const;

    int port() const { return port_; }
    void set_port(int port) { port_ = port; }

    const std::filesystem::path& log_dir() const { return log_dir_; }
    void set_log_dir(const std::filesystem::path& dir) { log_dir_ = dir; }

    const std::filesystem::path& temp_dir() const { return temp_dir_; }
    void set_temp_dir(const std::filesystem::path& dir) { temp_dir_ = dir; }

    const std::filesystem::path& dataset_dir() const { return dataset_dir_; }
    void set_dataset_dir(const std::filesystem::path& dir) {
        dataset_dir_ = dir;
    }

private:
    config() = default;

    int port_ = 3705;
    std::filesystem::path log_dir_ = "./logs";
    std::filesystem::path temp_dir_ = "./temp";
    std::filesystem::path dataset_dir_ = "./dataset";
};

}  // namespace server