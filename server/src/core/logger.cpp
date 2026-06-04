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

#include <core/logger.h>

#include <core/config.h>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

namespace server {

static std::string colorize(log_level level, const std::string& msg) {
    const char* color_code = "";
    switch (level) {
        case log_level::debug:
            color_code = "\033[90m";
            break;
        case log_level::info:
            color_code = "\033[32m";
            break;
        case log_level::warning:
            color_code = "\033[33m";
            break;
        case log_level::error:
            color_code = "\033[31m";
            break;
    }
    return std::string(color_code) + msg + "\033[0m";
}

logger& logger::instance() {
    static logger instance;
    return instance;
}

logger::logger() {
    ring_buffer_.resize(max_buffer_size_);
    auto& cfg = config::instance();
    log_dir_ = cfg.log_dir();
    std::error_code ec;
    std::filesystem::create_directories(log_dir_, ec);
}

logger::~logger() {
    flush_remaining();
}

void logger::set_level(log_level level) {
    std::lock_guard<std::mutex> lock(mutex_);
    level_ = level;
}

void logger::set_log_dir(const std::filesystem::path& dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    log_dir_ = dir;
    std::error_code ec;
    std::filesystem::create_directories(log_dir_, ec);
}

std::string logger::level_to_string(log_level level) {
    switch (level) {
        case log_level::debug:
            return "DEBUG";
        case log_level::info:
            return "INFO";
        case log_level::warning:
            return "WARNING";
        case log_level::error:
            return "ERROR";
    }
    return "UNKNOWN";
}

std::string logger::current_time() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) %
              1000;
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "." << std::setfill('0')
        << std::setw(3) << ms.count();
    return oss.str();
}

std::string logger::generate_filename() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) %
              1000;
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S") << "_" << std::setfill('0')
        << std::setw(3) << ms.count();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1000, 9999);
    oss << "_" << dist(gen) << ".log";

    return oss.str();
}

void logger::flush_to_file(size_t from_count, size_t to_count) {
    if (from_count >= to_count) return;

    std::string filename = generate_filename();
    std::filesystem::path filepath = log_dir_ / filename;
    std::ofstream file(filepath, std::ios::out | std::ios::app);
    if (!file.is_open()) return;

    for (size_t idx = from_count; idx < to_count; ++idx) {
        size_t pos = (head_ + idx - total_written_ + max_buffer_size_) %
                     max_buffer_size_;
        file << ring_buffer_[pos] << std::endl;
    }
}

void logger::flush_remaining() {
    if (total_written_ > last_saved_count_) {
        flush_to_file(last_saved_count_, total_written_);
        last_saved_count_ = total_written_;
    }
}

void logger::log(log_level level, const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (level < level_) return;

    std::string entry =
        current_time() + " [" + level_to_string(level) + "] " + message;

    if (total_written_ >= max_buffer_size_) {
        ring_buffer_[head_] = entry;
        head_ = (head_ + 1) % max_buffer_size_;
    } else {
        ring_buffer_[total_written_] = entry;
    }
    ++total_written_;

    if (total_written_ % max_buffer_size_ == 0 && total_written_ > 0) {
        flush_to_file(last_saved_count_, total_written_);
        last_saved_count_ = total_written_;
    }

    std::cout << colorize(level, entry) << std::endl;
}

std::vector<std::string> logger::get_logs(
    size_t count, log_level min_level) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> result;
    size_t available =
        total_written_ < max_buffer_size_ ? total_written_ : max_buffer_size_;
    size_t start = available > count ? available - count : 0;
    for (size_t i = start; i < available; ++i) {
        size_t pos = (head_ + i) % max_buffer_size_;
        const std::string& entry = ring_buffer_[pos];
        bool level_ok = true;
        if (min_level != log_level::debug) {
            if (entry.find("[DEBUG]") != std::string::npos &&
                min_level > log_level::debug)
                level_ok = false;
            else if (
                entry.find("[INFO]") != std::string::npos &&
                min_level > log_level::info)
                level_ok = false;
            else if (
                entry.find("[WARNING]") != std::string::npos &&
                min_level > log_level::warning)
                level_ok = false;
            else if (
                entry.find("[ERROR]") != std::string::npos &&
                min_level > log_level::error)
                level_ok = false;
        }
        if (level_ok) result.push_back(entry);
    }
    return result;
}

}  // namespace server