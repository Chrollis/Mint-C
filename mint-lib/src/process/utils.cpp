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

#include <process/utils.h>
#include <array>
#include <cctype>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace mint {

static bool is_little_endian() noexcept {
    static const bool is_little = [] {
        static const uint32_t test = 1;
        return reinterpret_cast<const char*>(&test)[0] == 1;
    }();
    return is_little;
}

uint32_t be32_to_host(uint32_t x) noexcept {
    if (!is_little_endian()) return x;
    return ((x & 0x000000FFu) << 24) | ((x & 0x0000FF00u) << 8) |
           ((x & 0x00FF0000u) >> 8) | ((x & 0xFF000000u) >> 24);
}

uint32_t host_to_be32(uint32_t x) noexcept {
    return be32_to_host(x);
}

uint64_t be64_to_host(uint64_t x) noexcept {
    if (!is_little_endian()) return x;
    return ((x & 0x00000000000000FFull) << 56) |
           ((x & 0x000000000000FF00ull) << 40) |
           ((x & 0x0000000000FF0000ull) << 24) |
           ((x & 0x00000000FF000000ull) << 8) |
           ((x & 0x000000FF00000000ull) >> 8) |
           ((x & 0x0000FF0000000000ull) >> 24) |
           ((x & 0x00FF000000000000ull) >> 40) |
           ((x & 0xFF00000000000000ull) >> 56);
}

uint64_t host_to_be64(uint64_t x) noexcept {
    return be64_to_host(x);
}

double bedb_to_host(double x) noexcept {
    uint64_t tmp;
    memcpy(&tmp, &x, 8);
    tmp = be64_to_host(tmp);
    double res;
    memcpy(&res, &tmp, 8);
    return res;
}

double host_to_bedb(double x) noexcept {
    return bedb_to_host(x);
}

std::vector<uint8_t> base64_decode(const std::string& base64_str) {
    static std::array<int, 256> dec_table = [] {
        std::array<int, 256> dec_table;
        dec_table.fill(-1);
        const char* base64_chars =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789+/";
        for (int i = 0; i < 64; ++i) {
            dec_table[static_cast<uint8_t>(base64_chars[i])] = i;
        }
        return dec_table;
    }();

    std::string_view b64_sv(base64_str);
    const std::string_view prefix = "base64,";
    auto prefix_pos = b64_sv.find(prefix);
    if (prefix_pos != std::string_view::npos) {
        b64_sv.remove_prefix(prefix_pos + prefix.size());
    }

    std::vector<uint8_t> result;
    result.reserve(b64_sv.size() * 3 / 4);
    int val = 0, valb = -8;

    for (size_t i = 0; i < b64_sv.size(); i++) {
        uint8_t c = static_cast<uint8_t>(b64_sv[i]);
        if (isspace(c)) continue;
        if (c == '=') break;
        if (dec_table[c] == -1) {
            std::ostringstream oss;
            oss << "Invalid base64 character: ";
            if (isprint(static_cast<unsigned char>(c))) {
                oss << static_cast<char>(c);
            } else {
                oss << "\\" << std::oct << std::setw(3) << std::setfill('0')
                    << static_cast<unsigned int>(c);
            }
            throw std::invalid_argument("base64_decode: " + oss.str());
        }
        val = (val << 6) + dec_table[c];
        valb += 6;
        if (valb >= 0) {
            result.push_back(static_cast<uint8_t>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return result;
}

int union_find::add() {
    parent.push_back(static_cast<int>(parent.size()));
    rank.push_back(0);
    return static_cast<int>(parent.size() - 1);
}

int union_find::find(int x) {
    while (parent[x] != x) {
        x = parent[x] = parent[parent[x]];
    }
    return x;
}

void union_find::unite(int x, int y) {
    int rx = find(x), ry = find(y);
    if (rx == ry) return;
    if (rank[rx] < rank[ry])
        parent[rx] = ry;
    else if (rank[rx] > rank[ry])
        parent[ry] = rx;
    else {
        parent[ry] = rx;
        rank[rx]++;
    }
}

Eigen::MatrixXd reconstruct_to_matrixxd(
    const Eigen::Matrix<uint8_t, 28, 28, Eigen::RowMajor>& src) {
    Eigen::MatrixXd dst(28, 28);
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            dst(y, x) = src(y, x) < 128 ? 0.0 : 1.0;
        }
    }
    return dst;
}

}  // namespace mint