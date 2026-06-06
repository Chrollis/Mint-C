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

if(WIN32)
    if(MSVC)
        target_link_options(mint-server PRIVATE 
            /SUBSYSTEM:WINDOWS 
            /ENTRY:mainCRTStartup
        )
    elseif(MINGW)
        target_link_options(mint-server PRIVATE -mwindows)
    endif()
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GTK REQUIRED gtk+-3.0 gdk-pixbuf-2.0)
    target_link_libraries(mint-server PRIVATE ${GTK_LIBRARIES})
    target_include_directories(mint-server PRIVATE ${GTK_INCLUDE_DIRS})
endif()

target_link_libraries(mint-server
    PRIVATE
        mint-lib                      
        httplib::httplib
        webview::core          
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

    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
        ${CMAKE_CURRENT_SOURCE_DIR}/chrollapp.ico 
        $<TARGET_FILE_DIR:mint-server>/chrollapp.ico
    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
        ${CMAKE_CURRENT_SOURCE_DIR}/chrollapp.png 
        $<TARGET_FILE_DIR:mint-server>/chrollapp.png
)

set_target_properties(mint-server PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)