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

check_dependencies() {
    echo ""
    echo "[0/3] Checking system dependencies..."

    MISSING=0
    PKGS="gtk+-3.0 webkit2gtk-4.1 gdk-pixbuf-2.0"

    for pkg in $PKGS; do
        if ! pkg-config --exists $pkg; then
            echo "Missing development package: $pkg"
            MISSING=1
        else
            echo "Found $pkg"
        fi
    done

    if [ $MISSING -eq 1 ]; then
        echo ""
        echo "=========================================="
        echo " Missing required dependencies!"
        echo " Please install them using your distribution's package manager:"
        echo ""
        echo "   Debian/Ubuntu:"
        echo "     sudo apt install libgtk-3-dev libwebkit2gtk-4.1-dev libgdk-pixbuf2.0-dev"
        echo ""
        echo "   Fedora/RHEL:"
        echo "     sudo dnf install gtk3-devel webkit2gtk4.1-devel gdk-pixbuf2-devel"
        echo ""
        echo "   Arch Linux:"
        echo "     sudo pacman -S gtk3 webkit2gtk gdk-pixbuf2"
        echo "=========================================="
        exit 1
    fi

    echo "All dependencies satisfied."
}

check_dependencies

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
    if [ -d "build/bin/Release" ] && [ -f "build/bin/Release/mint-server" ]; then
        SRC_DIR="build/bin/Release"
    elif [ -d "build/bin" ] && [ -f "build/bin/mint-server" ]; then
        SRC_DIR="build/bin"
    else
        echo "Error: Could not find mint-server executable in build/bin/ or build/bin/Release/"
        exit 1
    fi

    echo "Copying $SRC_DIR/ to ./Mint-C/ ..."
    rm -rf ./Mint-C
    mkdir -p ./Mint-C
    cp -r "$SRC_DIR"/* ./Mint-C/
    echo "Done. Output is in ./Mint-C/"
else
    echo "Skipping copy."
fi

echo ""
read -p "Press any key to exit..." -n 1
echo