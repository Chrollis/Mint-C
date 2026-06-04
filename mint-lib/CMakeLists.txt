# Mint-C - Web-based digit recognition tool with C++ CNN backend
# Copyright (C) 2026 Chrollis
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 4.0.0)
project(mint-lib LANGUAGES CXX)

file(GLOB_RECURSE MINT_SOURCES "src/*.cpp")

add_library(mint-lib STATIC ${MINT_SOURCES})

target_include_directories(mint-lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${stb_SOURCE_DIR}
)

target_link_libraries(mint-lib
    PUBLIC
        eigen
        nlohmann_json::nlohmann_json
    PRIVATE
)
