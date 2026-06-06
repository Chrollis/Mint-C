@echo off
REM Mint-C - Web-based digit recognition tool with C++ CNN backend
REM Copyright (C) 2026 Chrollis
REM
REM This program is free software: you can redistribute it and/or modify
REM it under the terms of the GNU General Public License as published by
REM the Free Software Foundation, either version 3 of the License, or
REM (at your option) any later version.
REM
REM This program is distributed in the hope that it will be useful,
REM but WITHOUT ANY WARRANTY; without even the implied warranty of
REM MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
REM GNU General Public License for more details.
REM
REM You should have received a copy of the GNU General Public License
REM along with this program.  If not, see <https://www.gnu.org/licenses/>.

setlocal

echo ========================================
echo  MINT-C Build Script (Windows)
echo ========================================

echo.
echo [0/3] Checking WebView2 Runtime...

reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" >nul 2>&1
if %errorlevel% equ 0 (
    echo WebView2 Runtime is installed.
) else (
    echo.
    echo ==========================================
    echo  WARNING: WebView2 Runtime not found!
    echo  The application will not run without it.
    echo  Please download and install from:
    echo  https://developer.microsoft.com/en-us/microsoft-edge/webview2/
    echo ==========================================
    echo.
    set /p choice="Do you want to continue building anyway? (y/N): "
    if /i not "!choice!"=="y" (
        echo Build cancelled.
        exit /b 1
    )
)

echo.
echo [1/3] Building frontend...
if not exist "node_modules" (
    echo Installing npm dependencies...
    call npm install
)
call npm run build
if %ERRORLEVEL% NEQ 0 (
    echo Frontend build failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [2/3] Configuring CMake...
if not exist "build" mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [3/3] Compiling...
cmake --build . --config Release
if %ERRORLEVEL% NEQ 0 (
    echo Compilation failed!
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo  Build successful!
echo  The executable and web files are ready.
echo ========================================
cd ..

echo.
set /p choice="Do you want to copy the Release build output to the current directory? (y/N): "
if /i "%choice%"=="y" (
    if exist "build\bin\Release" (
        echo Copying build\bin\Release\ to .\Release\ ...
        if exist "Release" (
            echo Removing existing .\Release\ folder...
            rmdir /s /q Release 2>nul
        )
        xcopy /E /I /H /Y build\bin\Release Release
        echo Done. Output is in .\Release\
    ) else (
        echo Warning: build\bin\Release\ not found. Check your build output location.
    )
) else (
    echo Skipping copy.
)

echo.
echo Press any key to exit...
pause > nul

endlocal