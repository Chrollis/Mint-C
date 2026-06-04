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
project(mint-server LANGUAGES CXX)

file(GLOB_RECURSE SERVER_SOURCES "src/*.cpp")

add_executable(mint-server ${SERVER_SOURCES} app_icon.rc)

target_link_libraries(mint-server
    PRIVATE
        mint-lib                      
        httplib::httplib          
        OpenMP::OpenMP_CXX        
)

target_include_directories(mint-server
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${stb_SOURCE_DIR}
)

add_custom_command(TARGET mint-server POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory 
        $<TARGET_FILE_DIR:mint-server>/web
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
        ${CMAKE_CURRENT_SOURCE_DIR}/web 
        $<TARGET_FILE_DIR:mint-server>/web
    COMMAND ${CMAKE_COMMAND} -E remove_directory 
        $<TARGET_FILE_DIR:mint-server>/dataset
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
        ${CMAKE_CURRENT_SOURCE_DIR}/dataset 
        $<TARGET_FILE_DIR:mint-server>/dataset
)

set_target_properties(mint-server PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)