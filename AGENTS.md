# Pikafish 项目上下文

## 项目概述

**Pikafish** 是一个免费且强大的**中国象棋 UCI 引擎**，基于 Stockfish 衍生开发。

**核心功能**:
- 分析象棋局面
- 计算最优着法
- 支持 NNUE（高效神经网络更新）评估
- 支持多线程搜索
- 支持 Fishtest 分布式测试

**项目信息**:
- **许可证**: GNU General Public License v3.0
- **官方网站**: https://pikafish.org
- **GitHub**: https://github.com/official-pikafish/Pikafish
- **Discord 社区**: https://discord.com/invite/uSb3RXb7cY

---

## 技术栈

| 类别 | 技术 |
|------|------|
| **编程语言** | C++17 |
| **构建系统** | GNU Make (Makefile) |
| **支持的编译器** | GCC 9.3+、Clang 10.0+、Intel oneAPI、MinGW |
| **神经网络** | NNUE (高效神经网络更新) |
| **压缩库** | Zstandard (内置) |
| **测试框架** | 自定义 Python 测试框架 (`tests/testing.py`) |
| **持续集成** | GitHub Actions |

---

## 核心架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        UCI Interface                             │
│                         (uci.cpp)                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Engine Core                              │
│                        (engine.cpp)                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  Position   │  │   Search    │  │      ThreadPool         │ │
│  │  (position) │  │   (search)  │  │       (thread)          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Transposition│ │  Evaluate   │  │      NUMA Config        │ │
│  │  Table (tt) │  │  (evaluate) │  │       (numa.h)          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      NNUE Network Layer                          │
│                    (nnue/network.cpp)                            │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │ Feature Transformer│  │   Affine Layers │  │  Activation   │ │
│  │  (HalfKAv2_hm)    │  │  (fc_0, fc_1, fc_2)│  │  (ReLU, Sqr)  │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 目录结构

```
D:\2026.2.6\Pikafish\
├── src/                          # 源代码目录
│   ├── main.cpp                  # 程序入口
│   ├── engine.cpp/h              # 引擎核心类
│   ├── uci.cpp/h                 # UCI 协议接口
│   ├── ucioption.cpp/h           # UCI 选项处理
│   ├── position.cpp/h            # 棋盘局面表示
│   ├── types.h                   # 类型定义（棋子、颜色、格子）
│   ├── bitboard.cpp/h            # 位棋盘操作
│   ├── movegen.cpp/h             # 着法生成
│   ├── movepick.cpp/h            # 着法选择
│   ├── search.cpp/h              # 主搜索算法
│   ├── thread.cpp/h              # 线程管理
│   ├── timeman.cpp/h             # 时间管理
│   ├── evaluate.cpp/h            # 局面评估
│   ├── score.cpp/h               # 分数处理
│   ├── tt.cpp/h                  # 置换表（哈希表）
│   ├── history.h                 # 历史启发
│   ├── magics.h                  # 魔法位棋盘
│   ├── memory.cpp/h              # 内存管理
│   ├── misc.cpp/h                # 杂项工具
│   ├── tune.cpp/h                # 参数调优
│   ├── numa.h                    # NUMA 支持
│   ├── Makefile                  # 构建脚本
│   ├── nnue/                     # NNUE 神经网络
│   │   ├── network.cpp/h         # 网络类
│   │   ├── nnue_architecture.h   # 网络架构定义
│   │   ├── nnue_common.h         # NNUE 常量
│   │   ├── nnue_accumulator.cpp/h # 累加器栈
│   │   ├── nnue_feature_transformer.h # 特征变换器
│   │   ├── simd.h                # SIMD 优化
│   │   ├── features/             # 特征提取
│   │   │   ├── half_ka_v2_hm.cpp/h  # 主特征集
│   │   │   └── full_threats.cpp/h   # 威胁特征
│   │   └── layers/               # 网络层
│   │       ├── affine_transform.h
│   │       ├── affine_transform_sparse_input.h
│   │       ├── clipped_relu.h
│   │       └── sqr_clipped_relu.h
│   └── external/                 # Zstandard 压缩库
│       ├── zstd.h
│       ├── zstd_errors.h
│       ├── common/               # 通用工具
│       └── decompress/           # 解压缩核心
├── tests/                        # 测试目录
│   ├── testing.py                # Python 测试框架
│   ├── instrumented.py           # 测试执行器
│   ├── perft.sh                  # Perft 测试
│   ├── reprosearch.sh            # 可重复性测试
│   └── signature.sh              # 签名验证
├── scripts/                      # 脚本目录
│   ├── get_native_properties.sh  # CPU 特性检测
│   └── net.sh                    # NNUE 网络下载
├── .github/workflows/            # CI/CD 配置
│   ├── pikafish.yml              # 主构建流程
│   ├── analyzer.yml              # 代码分析
│   └── tuning.yml                # 调优流程
├── AGENTS.md                     # 本文件
├── README.md                     # 项目说明
├── CONTRIBUTING.md               # 贡献指南
├── AUTHORS                       # 贡献者名单
└── Copying.txt                   # GPL 许可证
```

---

## 构建和运行

### 快速开始

```powershell
# 进入源代码目录
cd src

# 标准构建（带性能引导优化）
make -j profile-build

# 或者跳过性能引导优化
make -j build

# 下载 NNUE 网络文件
make net
```

### 架构选择

使用 `native` 自动检测当前 CPU，或手动指定：

```powershell
# Intel Ice Lake / AMD Zen 4 (最新)
make -j profile-build ARCH=x86-64-avx512icl

# BMI2 支持（排除 AMD Zen1/2）
make -j profile-build ARCH=x86-64-bmi2

# AVX2 支持
make -j profile-build ARCH=x86-64-avx2

# ARM 64 位
make -j profile-build ARCH=armv8

# Apple Silicon M 系列
make -j profile-build ARCH=apple-silicon
```

### 编译器选择

```powershell
# 使用 GCC（默认）
make -j profile-build COMP=gcc

# 使用 Clang
make -j profile-build COMP=clang

# 使用 Intel oneAPI
make -j profile-build COMP=icx

# Windows 交叉编译
make -j profile-build COMP=mingw

# Android 编译（需要 NDK r27c+）
make -j profile-build COMP=ndk
```

### profile-build 四步流程

1. **构建性能分析版本**（带 `-fprofile-generate`）
2. **运行基准测试**（PGOBENCH）
3. **构建优化版本**（带 `-fprofile-use`）
4. **删除性能分析数据**

### 其他构建目标

```powershell
make help              # 显示帮助信息
make net               # 下载 NNUE 网络文件
make strip             # 剥离可执行文件符号
make install           # 安装到系统目录
make clean             # 清理构建产物
make format            # 格式化代码（clang-format-20）
make analyze           # 运行 IWYU 分析
```

### 运行引擎

```powershell
# 交互式 UCI 模式
.\pikafish.exe

# 基准测试
.\pikafish.exe bench

# 加载网络文件
.\pikafish.exe setoption name EvalFile value pikafish.nnue
```

---

## 测试方法

### Perft 测试（着法生成验证）

```powershell
cd tests
bash perft.sh
```

验证标准位置的着法生成节点数，包括：
- 起始位置（depth 7: 3,195,901,860 节点）
- 多个标准 FEN 位置
- Chess960 变体位置

### Python 测试套件

```powershell
# 运行完整测试套件
python tests/instrumented.py

# Valgrind 内存检查
python tests/instrumented.py --valgrind

# 线程 sanitizer 检查
python tests/instrumented.py --sanitizer-thread

# 未定义行为检查
python tests/instrumented.py --sanitizer-undefined
```

### 可重复性测试

```powershell
cd tests
bash reprosearch.sh
```

验证相同节点数的搜索产生相同结果（运行 20 轮）。

### 签名验证

```powershell
cd tests
bash signature.sh
```

运行基准测试并验证节点数签名。

---

## 开发规范

### 代码风格

- 遵循 `.clang-format` 定义的风格
- 使用 `make format` 格式化代码
- 需要 clang-format 版本 20

### 提交规范

遵循 **Conventional Commits** 规范：

```
<类型>(<范围>): <简述>

[正文]

[尾注]
```

**类型推荐值**:
| 类型 | 说明 |
|------|------|
| `init` | 项目初始化 |
| `feat` | 新增功能 |
| `fix` | 错误修复 |
| `docs` | 文档变更 |
| `style` | 代码格式化 |
| `refactor` | 代码重构 |
| `perf` | 性能优化 |
| `test` | 测试相关 |
| `build` | 构建系统变更 |
| `ci` | CI 配置变更 |
| `chore` | 辅助工具变动 |
| `revert` | 撤销提交 |

### 贡献流程

1. 功能改进需在 Fishtest 测试（https://test.pikafish.org）
2. 非功能修改无需测试（除非影响性能）
3. 首次贡献者需将名字添加到 `AUTHORS` 文件
4. 遵循原子提交原则（单一目的）

### 重要限制

- 项目不专注于添加新功能
- 功能相关的 PR 可能不经讨论直接关闭
- Bug 报告需包含详细信息（复现步骤、环境信息）

---

## CI/CD 配置

### pikafish.yml（主构建流程）

**触发条件**: 推送到 `master` 分支

**构建矩阵**:
- **Linux** (Ubuntu 22.04): x86-64 (7 种变体)
- **Windows** (MSYS2): x86-64 (7 种变体)
- **macOS**: Apple Silicon
- **Android**: armv8-dotprod, armv8

**支持的 x86-64 架构变体**:
```
-avx512icl    → Intel Ice Lake / AMD Zen 4
-vnni512      → AVX-512 VNNI
-avx512       → 基础 AVX-512
-avxvnni      → AVX VNNI 256-bit
-bmi2         → BMI2 指令集
-avx2         → AVX2 指令集
-sse41-popcnt → SSE4.1 + POPCNT
```

### analyzer.yml（代码分析）

**触发条件**: 每周一 18:17 UTC

**分析任务**:
1. **CodeQL 安全扫描** - 语言：C++
2. **IWYU 头文件检查** - include-what-you-use 0.25
3. **Clang-Format 风格检查** - clang-format 21

### tuning.yml（调优流程）

**触发条件**: 推送到 `tune` 分支

**流程**:
1. 编译两个版本（bmi2 + 默认架构）
2. 上传到 fishtest 测试服务器
3. 打印变量表

---

## 重要注意事项

### NNUE 网络文件依赖

- **来源**: GitHub Releases
- **URL**: https://github.com/official-pikafish/Networks/releases/download/master-net/pikafish.nnue
- **位置**: `src/pikafish.nnue`
- **下载命令**: `make net`
- **超时**: 300 秒

### 跨平台构建要求

| 平台 | 特殊依赖 |
|------|----------|
| **Windows** | MSYS2, mingw-w64-clang, make |
| **Android** | NDK r27c+, llvm-strip |
| **macOS** | Xcode Command Line Tools, xcrun |
| **Linux** | libatomic (部分架构) |

### 性能分析构建要求

**LLVM 工具链**:
- `llvm-profdata` - 性能数据合并
- 版本必须与编译器匹配

**profile-build 额外依赖**:
- GCC: `libgcov`
- Clang: `llvm-profdata`
- Intel: `llvm-profdata` (oneAPI)

### 架构特定注意事项

| 架构 | 特殊标志 |
|------|----------|
| ARM/Android | `-latomic`, `-fPIE` |
| RISC-V | `-latomic` |
| LoongArch | `-latomic` |
| Windows | `-static` |
| macOS | `-mmacosx-version-min=10.15` |

### 已知问题

- **AMD Zen1/2 排除**: `bmi2` 架构会排除这些 CPU（已知设计）
- **Windows LTO**: 静态链接需要 GCC 10.1+
- **网络下载超时**: 5 分钟超时可能导致大文件下载失败

---

## 关键数据结构

### 棋子类型定义

```cpp
enum PieceType : std::uint8_t {
    NO_PIECE_TYPE, ROOK, ADVISOR, CANNON, PAWN, KNIGHT, BISHOP, KING
};

enum Piece : std::uint8_t {
    NO_PIECE,
    W_ROOK, W_ADVISOR, W_CANNON, W_PAWN, W_KNIGHT, W_BISHOP, W_KING,
    B_ROOK, B_ADVISOR, B_CANNON, B_PAWN, B_KNIGHT, B_BISHOP, B_KING
};
```

### 棋子价值

```cpp
constexpr Value RookValue    = 1305;
constexpr Value AdvisorValue = 219;
constexpr Value CannonValue  = 773;
constexpr Value PawnValue    = 144;
constexpr Value KnightValue  = 720;
constexpr Value BishopValue  = 187;
```

### NNUE 网络架构

```
输入特征：HalfKAv2_hm (王的位置 + 棋子位置) + Full_Threats
         ↓
特征变换器：1024 维
         ↓
隐藏层 1: 15 输出 (SqrClippedReLU)
         ↓
隐藏层 2: 32 输出 (ClippedReLU)
         ↓
输出层：1 (局面评分)
```

---

## 核心类和接口参考

| 类名 | 文件位置 | 职责 |
|------|----------|------|
| `Engine` | `src/engine.h` | 引擎主协调器 |
| `UCIEngine` | `src/uci.h` | UCI 协议解析器 |
| `Position` | `src/position.h` | 棋盘局面表示 |
| `Search::Worker` | `src/search.h` | 搜索线程工作单元 |
| `Eval::NNUE::Network` | `src/nnue/network.h` | 神经网络评估器 |
| `TranspositionTable` | `src/tt.h` | 置换表（哈希表） |
| `Bitboards` | `src/bitboard.h` | 位板操作工具集 |
| `ThreadPool` | `src/thread.h` | 线程池管理器 |

---

*最后更新：2026-02-22*
*Git HEAD: beeca5cc785ceb2026c5f9248f6e057573b12906*
