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
#include <mutex>
#include <string>
#include <vector>

namespace server {

enum class log_level {
    debug,
    info,
    warning,
    error,
};

class logger {
public:
    static logger& instance();

    void set_level(log_level level);
    void set_log_dir(const std::filesystem::path& dir);

    void log(log_level level, const std::string& message);
    void debug(const std::string& msg) { log(log_level::debug, msg); }
    void info(const std::string& msg) { log(log_level::info, msg); }
    void warning(const std::string& msg) { log(log_level::warning, msg); }
    void error(const std::string& msg) { log(log_level::error, msg); }

    std::vector<std::string> get_logs(
        size_t count = 500, log_level min_level = log_level::debug) const;

private:
    logger();
    ~logger();

    static std::string level_to_string(log_level level);
    static std::string current_time();
    static std::string generate_filename();

    void flush_to_file(size_t from_count, size_t to_count);
    void flush_remaining();

    log_level level_ = log_level::info;
    std::filesystem::path log_dir_ = "./logs";

    mutable std::mutex mutex_;
    std::vector<std::string> ring_buffer_;
    size_t max_buffer_size_ = 500;
    size_t head_ = 0;
    size_t total_written_ = 0;
    size_t last_saved_count_ = 0;
};

}  // namespace server