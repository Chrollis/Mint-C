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
#include <Eigen/Dense>

#include <csignal>
#include <filesystem>
#include <string>

#include <api_handlers.h>
#include <core/config.h>
#include <core/logger.h>
#include <core/model_manager.h>

static httplib::Server* g_svr = nullptr;

void signal_handler(int sig) {
    (void)sig;
    if (g_svr) {
        g_svr->stop();
    }
}

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
    std::string url = "http://127.0.0.1:" + std::to_string(cfg.port());

    auto& lgr = server::logger::instance();
    lgr.info("MINT Server starting... on " + url);

    httplib::Server svr;
    svr.set_mount_point("/", "./web");
    server::register_routes(svr);

    g_svr = &svr;
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

#ifdef _WIN32
    std::string cmd = "start " + std::string(url);
#else
    std::string cmd = "xdg-open " + std::string(url);
#endif
    system(cmd.c_str());

    if (!svr.listen("127.0.0.1", cfg.port())) {
        lgr.error(
            "Failed to start server on port " + std::to_string(cfg.port()));
        return 1;
    }

    lgr.info("MINT Server stopped.");
    return 0;
}