#!/bin/bash
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

set -e

echo "========================================"
echo " MINT-C Build Script (Linux)"
echo "========================================"

echo ""
echo "[1/3] Building frontend..."
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi
npm run build

echo ""
echo "[2/3] Configuring CMake..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

echo ""
echo "[3/3] Compiling..."
cmake --build .

echo ""
echo "========================================"
echo " Build successful!"
echo " The executable and web files are ready."
echo "========================================"
cd ..

echo ""
read -p "Do you want to copy the Release build output to the current directory? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -d "build/bin/Release" ]; then
        echo "Copying build/bin/Release/ to ./Release/ ..."
        cp -r build/bin/Release ./Release
        echo "Done. Output is in ./Release/"
    else
        echo "Warning: build/bin/Release/ not found. Check your build output location."
    fi
else
    echo "Skipping copy."
fi

echo ""
read -p "Press any key to exit..." -n 1
echo