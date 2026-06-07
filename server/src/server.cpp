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

#include <httplib.h>
#include <omp.h>
#include <webview/webview.h>
#include <Eigen/Dense>

#include <atomic>
#include <csignal>
#include <filesystem>
#include <string>
#include <thread>

#include <api_handlers.h>
#include <core/config.h>
#include <core/logger.h>
#include <core/model_manager.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <resource.h>
#include <shellapi.h>
#include <windows.h>
#else
#include <gdk/gdk.h>
#include <gtk/gtk.h>
#endif

static std::atomic<httplib::Server*> g_backend_server{nullptr};
static std::atomic<bool> g_should_exit{false};
static std::mutex g_ready_mutex;
static std::condition_variable g_ready_cv;
static bool g_backend_ready = false;

void signal_handler(int sig);

void run_backend_server(int port, const std::string& web_root);

#ifdef _WIN32
struct FullscreenState {
    WINDOWPLACEMENT old_placement = {sizeof(WINDOWPLACEMENT)};
    DWORD old_style = 0;
    bool is_fullscreen = false;
};
static FullscreenState g_fs_state;
void ToggleFullscreenWindows(HWND hwnd, FullscreenState& state);
#else
static bool g_fs_is_fullscreen = false;
void ToggleFullscreenLinux(GtkWidget* window, bool& is_fullscreen);
#endif
void OnToggleFullscreen(
    const std::string& seq, const std::string& req, void* arg);

int main() {
    static_assert(
        sizeof(double) == 8, "Platform must have 64-bit double (8 bytes)");
    Eigen::setNbThreads(omp_get_max_threads());

    std::cout << "\033[34m" <<
        R"(
__/\\\\____________/\\\\________________________________________________________/\\\\\\\\\_        
 _\/\\\\\\________/\\\\\\_____________________________________________________/\\\////////__       
  _\/\\\//\\\____/\\\//\\\__/\\\___________________/\\\______________________/\\\/___________      
   _\/\\\\///\\\/\\\/_\/\\\_\///___/\\/\\\\\\____/\\\\\\\\\\\__/\\\\\\\\\\\__/\\\_____________     
    _\/\\\__\///\\\/___\/\\\__/\\\_\/\\\////\\\__\////\\\////__\///////////__\/\\\_____________    
     _\/\\\____\///_____\/\\\_\/\\\_\/\\\__\//\\\____\/\\\____________________\//\\\____________   
      _\/\\\_____________\/\\\_\/\\\_\/\\\___\/\\\____\/\\\_/\\_________________\///\\\__________  
       _\/\\\_____________\/\\\_\/\\\_\/\\\___\/\\\____\//\\\\\____________________\////\\\\\\\\\_ 
        _\///______________\///__\///__\///____\///______\/////________________________\/////////__
    )" << "\033[0m"
              << std::endl;

    auto& cfg = server::config::instance();
    cfg.load();
    int port = cfg.port();
    std::string url = "http://127.0.0.1:" + std::to_string(cfg.port());
    std::string web_root = "./web";

    auto& lgr = server::logger::instance();
    lgr.info("MINT-C Starting... (backend URL: " + url + ")");

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::thread backend_thread(run_backend_server, port, web_root);
    {
        std::unique_lock<std::mutex> lock(g_ready_mutex);
        g_ready_cv.wait(lock, [] { return g_backend_ready; });
    }
    lgr.info("Backend is ready, now creating WebView window...");

    webview::webview w(true, nullptr);
    w.set_title("Mint-C - Digit Recognition");
    w.set_size(1200, 675, WEBVIEW_HINT_NONE);
    w.bind("toggle_fullscreen", OnToggleFullscreen, &w);

#ifdef _DEBUG
    w.navigate("http://localhost:5173");
#else
    w.navigate(url);
#endif

#ifdef _WIN32
    auto hwnd_result = w.window();
    if (hwnd_result.has_value()) {
        HWND hwnd = static_cast<HWND>(hwnd_result.value());
        if (hwnd) {
            HICON hIcon =
                LoadIcon(GetModuleHandle(NULL), MAKEINTRESOURCE(IDI_ICON1));
            if (hIcon) {
                SendMessage(hwnd, WM_SETICON, ICON_SMALL, (LPARAM)hIcon);
                SendMessage(hwnd, WM_SETICON, ICON_BIG, (LPARAM)hIcon);
            } else {
                hIcon = (HICON)LoadImage(
                    NULL, "./chrollapp.ico", IMAGE_ICON, 0, 0,
                    LR_LOADFROMFILE | LR_DEFAULTSIZE);
                if (hIcon) {
                    SendMessage(hwnd, WM_SETICON, ICON_SMALL, (LPARAM)hIcon);
                }
            }
        }
    }
#else
    auto window_result = w.window();
    if (window_result) {
        GtkWidget* window = static_cast<GtkWidget*>(window_result.value());
        if (window && GTK_IS_WINDOW(window)) {
            GdkPixbuf* icon = gdk_pixbuf_new_from_file("./chrollapp.png", NULL);
            if (icon) {
                gtk_window_set_icon(GTK_WINDOW(window), icon);
                g_object_unref(icon);
            }
        }
    }
#endif

    w.run();
    lgr.info("WebView window closed, stopping backend...");
    auto* svr = g_backend_server.load(std::memory_order_acquire);
    if (svr) {
        svr->stop();
    }

    if (backend_thread.joinable()) {
        backend_thread.join();
    }

    lgr.info("MINT-C has terminated gracefully.");
    return 0;
}

void signal_handler(int sig) {
    (void)sig;
    auto* svr = g_backend_server.load(std::memory_order_acquire);
    if (svr) {
        svr->stop();
    }
    g_should_exit.store(true, std::memory_order_release);
}

void run_backend_server(int port, const std::string& web_root) {
    httplib::Server svr;
    g_backend_server.store(&svr, std::memory_order_release);

    svr.set_mount_point("/", web_root);
    server::register_routes(svr);

    auto& lgr = server::logger::instance();
    lgr.info(
        "HTTP backend listening on http://127.0.0.1:" + std::to_string(port));

    {
        std::lock_guard<std::mutex> lock(g_ready_mutex);
        g_backend_ready = true;
        g_ready_cv.notify_one();
    }

    if (!svr.listen("127.0.0.1", port)) {
        lgr.error(
            "Failed to start HTTP server on port " + std::to_string(port));
    }

    lgr.info("HTTP backend has stopped.");
    g_backend_server.store(nullptr, std::memory_order_release);
}

#ifdef _WIN32
void ToggleFullscreenWindows(HWND hwnd, FullscreenState& state) {
    if (!state.is_fullscreen) {
        state.old_style = GetWindowLong(hwnd, GWL_STYLE);
        GetWindowPlacement(hwnd, &state.old_placement);
        SetWindowLong(
            hwnd, GWL_STYLE, state.old_style & ~(WS_CAPTION | WS_THICKFRAME));

        HMONITOR monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTOPRIMARY);
        MONITORINFO mi = {sizeof(mi)};
        GetMonitorInfo(monitor, &mi);
        SetWindowPos(
            hwnd, HWND_TOPMOST, mi.rcMonitor.left, mi.rcMonitor.top,
            mi.rcMonitor.right - mi.rcMonitor.left,
            mi.rcMonitor.bottom - mi.rcMonitor.top,
            SWP_NOZORDER | SWP_FRAMECHANGED);
        state.is_fullscreen = true;
        ITaskbarList2* pTaskbarList = nullptr;
        HRESULT hr = CoCreateInstance(
            CLSID_TaskbarList, NULL, CLSCTX_INPROC_SERVER, IID_ITaskbarList2,
            (void**)&pTaskbarList);
        if (SUCCEEDED(hr) && pTaskbarList) {
            pTaskbarList->MarkFullscreenWindow(hwnd, TRUE);
            pTaskbarList->Release();
        }
    } else {
        SetWindowLong(hwnd, GWL_STYLE, state.old_style);
        SetWindowPlacement(hwnd, &state.old_placement);
        SetWindowPos(
            hwnd, NULL, 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
        state.is_fullscreen = false;
        ITaskbarList2* pTaskbarList = nullptr;
        HRESULT hr = CoCreateInstance(
            CLSID_TaskbarList, NULL, CLSCTX_INPROC_SERVER, IID_ITaskbarList2,
            (void**)&pTaskbarList);
        if (SUCCEEDED(hr) && pTaskbarList) {
            pTaskbarList->MarkFullscreenWindow(hwnd, FALSE);
            pTaskbarList->Release();
        }
    }
}
void OnToggleFullscreen(
    const std::string& seq, const std::string& req, void* arg) {
    webview::webview* w = static_cast<webview::webview*>(arg);
    auto hwnd_res = w->window();
    if (hwnd_res.has_value()) {
        HWND hwnd = static_cast<HWND>(hwnd_res.value());
        ToggleFullscreenWindows(hwnd, g_fs_state);
        w->resolve(seq, 0, "");
    } else {
        w->resolve(seq, 1, "Failed to get window handle");
    }
}
#else
void ToggleFullscreenLinux(GtkWidget* window, bool& is_fullscreen) {
    if (is_fullscreen) {
        gtk_window_unfullscreen(GTK_WINDOW(window));
    } else {
        gtk_window_fullscreen(GTK_WINDOW(window));
    }
    is_fullscreen = !is_fullscreen;
}
void OnToggleFullscreen(
    const std::string& seq, const std::string& req, void* arg) {
    webview::webview* w = static_cast<webview::webview*>(arg);
    auto window_res = w->window();
    if (window_res) {
        GtkWidget* window = static_cast<GtkWidget*>(window_res.value());
        if (window && GTK_IS_WINDOW(window)) {
            ToggleFullscreenLinux(window, g_fs_is_fullscreen);
        }
        w->resolve(seq, 0, "");
    } else {
        w->resolve(seq, 1, "Failed to get window handle");
    }
}
#endif