# <span id="english">Mint-C</span>

[![C++23](https://img.shields.io/badge/C++-23-blue.svg)](https://en.cppreference.com/w/cpp/23)
[![CMake](https://img.shields.io/badge/CMake-3.14+-green.svg)](https://cmake.org)
[![Vue.js](https://img.shields.io/badge/Vue-3.x-4FC08D.svg)](https://vuejs.org)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

<span><a href="#english">English</a> | <a href="#chinese">中文</a> </span>

> A full-featured digit recognition tool based on Web. Backend uses C++ for image processing, CNN inference and training; frontend provides visual dashboard, dataset management, model training and real-time recognition.

## ✨ Features

- 🖥️ **Pure Web Operation** – No desktop framework required, ready to use via browser
- ⚡ **High-Performance Inference** – C++ backend supports multi-digit recognition and automatic segmentation
- 🧠 **Complete Training Pipeline** – Supports MNIST format datasets, real-time loss/accuracy monitoring
- 📦 **Dataset Management** – Upload/download/merge IDX format datasets, save online drawn samples
- 🔧 **Model Management** – Build models from JSON architecture, export/import binary weights
- 📊 **Visual Dashboard** – Training curves, log viewer, system status monitoring
- 🌍 **Internationalization** – Built-in English/Chinese UI

## 🚀 Quick Start

### Requirements

| Component    | Version / Notes                                  |
| ------------ | ------------------------------------------------ |
| OS           | Windows / Linux (supports 8-byte double)         |
| C++ Compiler | C++23 or later (GCC 13+, Clang 16+, MSVC 19.35+) |
| CMake        | ≥ 4.0.0                                          |
| Git          | Any version                                      |
| Node.js      | ≥ 24.0.0 (for building frontend)                 |

### Build & Run

1. **Clone the repository** `git clone https://github.com/Chrollis/Mint-C.git`, `cd Mint-C`

2. **One‑click build**
   - **Windows**: `./build_windows.bat`
   - **Linux**: `chmod +x build_linux.sh`, `./build_linux.sh`
   - All C++ dependencies (Eigen5, cpp-httplib, nlohmann/json, stb) are automatically downloaded via CMake `FetchContent`.

3. **Start the server** `./build/bin/Release/mint-server` (Windows: `mint-server.exe`)
4. **Open your browser** and go to [http://127.0.0.1:3705](http://127.0.0.1:3705)

## 🧱 Technical Architecture

### Backend (C++23)

- **HTTP Server**: cpp-httplib
- **Numerical Computing**: Eigen5
- **Image Processing**: stb_image (PNG decoding)
- **Serialization**: Binary stream + nlohmann/json
- **Parallelism**: OpenMP (training acceleration)

### Frontend (Vue 3)

- **UI Framework**: Vuetify 3 + custom SCSS
- **State Management**: Pinia
- **Charts**: ApexCharts (training curves)
- **HTTP Client**: Axios
- **Internationalization**: vue-i18n
- **Build Tool**: Vite

## 📁 Project Structure

```bash
Mint-C/
├── mint-lib/        # C++ CNN core library
├── server/          # HTTP server
├── webui/           # Vue 3 frontend source
├── build_linux.sh   # Linux build script
├── build_windows.bat# Windows build script
└── LICENSE          # GPLv3 license
```

## 🧪 Usage Examples

### Recognize handwritten digits

1. Go to the “Recognition” page in the WebUI.
2. Upload an image (PNG/JPG) containing one or more handwritten digits.
3. The system automatically segments each digit region, preprocesses to 28×28 and recognizes them one by one.
4. The predicted value and confidence for each digit are returned, with the positions marked on the original image.

### Train a custom model

1. Prepare a dataset in MNIST format (or collect samples via the drawing board on the “Save Sample” page).
2. Upload or merge datasets on the “Dataset” page.
3. Go to the “Model” page and define the network architecture using interactive cards (e.g., convolutional layers, pooling layers, etc.).
4. On the “Training” page, select the dataset, hyperparameters (learning rate, batch size, epochs), and start training.
5. Monitor the loss/accuracy curves in real time. After training, you can accept the new model or discard it.

## ⚙️ Configuration

The configuration file is located at `config.json` in the server’s working directory (automatically created on first run). Main options:

```json
{
  "port": 3705,
  "log_dir": "./logs",
  "temp_dir": "./temp",
  "dataset_dir": "./datasets"
}
```

You can modify it via the “Settings” page in the frontend, or edit the file directly and restart the server.

## 🤝 Contributing

Issues and Pull Requests are welcome.

- **Code style**: Follow `.clang-format` (C++) and ESLint (Vue).
- **Commit messages**: Use semantic format (`feat:`, `fix:`, `docs:`, etc.).

## 📧 Contact

Author: [Chrollis](https://github.com/Chrollis)  
For questions or suggestions, please reach out via GitHub Issues.

## 🙏 Acknowledgements

- [Eigen](http://eigen.tuxfamily.org) – linear algebra library, used with OpenMP
- [cpp-httplib](https://github.com/yhirose/cpp-httplib) – HTTP framework
- [stb](https://github.com/nothings/stb) – single-header image library for image format decoding
- [Vuetify](https://vuetifyjs.com) – Vue component library
- [ApexCharts](https://apexcharts.com) – interactive charts
- [DeepSeek](https://www.deepseek.com) – README creation, internationalization support

## 📜 License

This project is open-sourced under the [GNU General Public License v3.0](https://github.com/Chrollis/Mint-C/?tab=GPL-3.0-1-ov-file).

---

---

# <span id="chinese">Mint-C</span>

[![C++23](https://img.shields.io/badge/C++-23-blue.svg)](https://en.cppreference.com/w/cpp/23)
[![CMake](https://img.shields.io/badge/CMake-3.14+-green.svg)](https://cmake.org)
[![Vue.js](https://img.shields.io/badge/Vue-3.x-4FC08D.svg)](https://vuejs.org)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

<span><a href="#english">English</a> | <a href="#chinese">中文</a> </span>

> 基于 Web 的全功能数字识别工具，后端使用 C++ 实现图像处理与 CNN 推理与训练，前端提供可视化仪表盘、数据集管理、模型训练与实时识别。

## ✨ 特性

- 🖥️ **纯 Web 操作** – 无需安装桌面框架，浏览器访问即用
- ⚡ **高性能推理** – 后端 C++ 实现，支持多数字同时识别与自动分割
- 🧠 **完整训练流水线** – 支持 MNIST 格式数据集，实时监控损失/准确率
- 📦 **数据集管理** – 上传/下载/合并 IDX 格式数据集，支持在线绘制样本保存
- 🔧 **模型管理** – 从 JSON 架构构建模型，导出/导入二进制权重
- 📊 **可视化仪表盘** – 训练曲线、日志查看、系统状态监控
- 🌍 **国际化** – 内置中英文界面

## 🚀 快速开始

### 环境要求

| 组件       | 版本/要求                                      |
| ---------- | ---------------------------------------------- |
| 操作系统   | Windows / Linux (支持 8 字节 double)           |
| C++ 编译器 | C++23 及以上 (GCC 13+, Clang 16+, MSVC 19.35+) |
| CMake      | ≥ 4.0.0                                        |
| Git        | 任意版本                                       |
| Node.js    | ≥ 24.0.0 (用于构建前端)                        |

### 构建与运行

1. **克隆仓库** `git clone https://github.com/Chrollis/Mint-C.git`, `cd Mint-C`

2. **一键构建**
   - **Windows**: `./build_windows.bat`
   - **Linux**: `chmod +x build_linux.sh`, `./build_linux.sh`
   - 所有 C++ 依赖 (Eigen5, cpp-httplib, nlohmann/json, stb) 会通过 CMake `FetchContent` 自动下载。

3. **启动服务** `./build/bin/Release/mint-server` (Windows 下为 `mint-server.exe`)

4. **打开浏览器** 访问 [http://127.0.0.1:3705](http://127.0.0.1:3705)

## 🧱 技术架构

### 后端 (C++23)

- **HTTP 服务**: cpp-httplib
- **数值计算**: Eigen5
- **图像处理**: stb_image (PNG 解码)
- **序列化**: 二进制流 + nlohmann/json
- **并行**: OpenMP (训练加速)

### 前端 (Vue 3)

- **UI 框架**: Vuetify 3 + 自定义 SCSS
- **状态管理**: Pinia
- **图表**: ApexCharts (训练曲线)
- **HTTP 客户端**: Axios
- **国际化**: vue-i18n
- **构建工具**: Vite

## 📁 项目结构

```bash
Mint-C/
├── mint-lib/         # C++ CNN 核心库
├── server/           # HTTP 服务端
├── webui/            # Vue 3 前端源码
├── build_linux.sh    # Linux 构建脚本
├── build_windows.bat # Windows 构建脚本
└── LICENSE           # GPLv3 许可证
```

## 🧪 使用示例

### 识别手写数字

1. 在 WebUI 中点击“识别”页面。
2. 上传包含一个或多个手写数字的图片（支持 PNG/JPG）。
3. 系统自动分割每个数字区域、预处理为 28×28 并逐个识别。
4. 返回每个数字的预测值及置信度，并在原图上标注位置。

### 训练自定义模型

1. 准备 MNIST 格式的数据集（或通过“保存样本”从手写板收集）。
2. 在“数据集”页面上传或合并数据集。
3. 进入“模型”页面，通过交互卡片定义网络结构（如卷积层、池化层等）。
4. 在“训练”页面选择数据集、超参数（学习率、批次大小、轮数），点击开始训练。
5. 实时观察损失/准确率曲线，训练完成后可选择接受新模型或丢弃。

## ⚙️ 配置

配置文件位于服务端运行目录下的 `config.json`（首次运行自动生成），主要选项：

```json
{
  "port": 3705,
  "log_dir": "./logs",
  "temp_dir": "./temp",
  "dataset_dir": "./datasets"
}
```

可通过前端“设置”页面修改或直接编辑文件后重启服务。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request。

- 代码风格: 遵循 `.clang-format` (C++) 和 ESLint (Vue)。
- 提交信息: 使用语义化格式 (`feat:`, `fix:`, `docs:` 等)。

## 📧 联系方式

项目作者: [Chrollis](https://github.com/Chrollis)  
如有问题或建议，请通过 GitHub Issues 联系。

## 🙏 致谢

- [Eigen](http://eigen.tuxfamily.org) – 线性代数库，用于启用OpenMP
- [cpp-httplib](https://github.com/yhirose/cpp-httplib) – HTTP 框架
- [stb](https://github.com/nothings/stb) – 单头文件图像库，用于解析图像格式
- [Vuetify](https://vuetifyjs.com) – Vue 组件库
- [ApexCharts](https://apexcharts.com) – 交互式图表
- [DeepSeek](https://www.deepseek.com) - README制作、国际化支持

## 📜 许可证

本项目使用 [GNU General Public License v3.0](https://github.com/Chrollis/Mint-C/?tab=GPL-3.0-1-ov-file) 开源。
