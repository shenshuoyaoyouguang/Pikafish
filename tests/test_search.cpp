// ===========================================================================
// search.cpp LMR θ₀ 等价性测试
// ===========================================================================
// 回归保护测试：验证 LMR_Continuous=true + 默认 θ₀ 参数时，bench 节点数 = 1,484,403
// （与 legacy 硬编码路径逐位等价）。
//
// 实现方式：通过子进程运行完整 pikafish（bench runner），喂入 "bench\nquit\n"，
// 解析 stderr 中的 "Nodes searched  : N" 行，验证 N == 1484403。
//
// 参考：
//   - src/search.cpp 的 Search::Worker::reduction_lmr()
//   - src/engine.cpp 的 UCI 选项 LMR_Continuous / LMR_Theta（默认 θ₀）
//   - src/uci.cpp 的 UCIEngine::bench()（输出 "Nodes searched  : " 到 stderr）
// ===========================================================================

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

// bench runner 路径由 CMake 通过 configure_file 生成的头文件提供
#include "bench_runner_path.h"

namespace {

// 运行 bench runner 子进程，返回合并的 stdout+stderr 输出。
// 通过临时文件传递 stdin 和捕获输出，兼容 Windows/Unix。
std::string run_bench_runner() {
    // 写入 bench 命令到输入文件
    {
        std::ofstream in("bench_input.txt");
        in << "bench\nquit\n";
    }

    // 构造 bench runner 路径：Windows 上 cmd.exe 不识别正斜杠，需转换为反斜杠
    std::string runner_path = BENCH_RUNNER_PATH;
#ifdef _WIN32
    for (char& c : runner_path)
        if (c == '/') c = '\\';
#endif

    // 构造命令：重定向 stdin/stdout，合并 stderr
    std::string cmd = std::string("\"") + runner_path + "\""
                      + " < bench_input.txt > bench_output.txt 2>&1";

    int rc = std::system(cmd.c_str());
    (void) rc;  // bench runner 正常退出码非 0 也继续解析输出

    // 读取输出
    std::ifstream out("bench_output.txt");
    return std::string((std::istreambuf_iterator<char>(out)), std::istreambuf_iterator<char>());
}

// 从 bench 输出中解析 "Nodes searched  : N" 行的节点数。
// bench 输出格式（见 src/uci.cpp UCIEngine::bench）：
//   "\nNodes searched  : " << nodes    （注意有两个空格）
// 使用 rfind 找最后一个匹配（bench 末尾的总节点数）。
long long parse_bench_nodes(const std::string& output) {
    const std::string key = "Nodes searched  : ";
    auto              pos = output.rfind(key);
    if (pos == std::string::npos)
        return -1;

    // 跳过 key，解析数字
    std::string rest = output.substr(pos + key.size());
    try
    {
        return std::stoll(rest);
    }
    catch (...)
    {
        return -1;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// LMR θ₀ 等价性测试
//   预期：LMR_Continuous=true（默认）+ 默认 LMR_Theta（θ₀）时，
//   bench 节点数 = 1,484,403（与 legacy 硬编码路径一致）。
//   这是回归保护测试，防止 LMR 参数化引入 bug。
// ---------------------------------------------------------------------------
TEST(LmrTheta0, BenchNodesEqualsLegacy) {
    std::string  output = run_bench_runner();
    long long    nodes  = parse_bench_nodes(output);

    // 验证能解析到节点数
    ASSERT_NE(nodes, -1)
        << "Failed to parse bench nodes from output.\n"
        << "=== bench runner output ===\n"
        << output << "\n============================";

    // 验证节点数与 legacy 一致
    EXPECT_EQ(nodes, 1484403LL)
        << "LMR θ₀ equivalence broken: expected 1484403 nodes but got " << nodes
        << ".\nThis indicates LMR_Continuous=true + default θ₀ diverges from legacy path.\n"
        << "=== bench runner output ===\n"
        << output << "\n============================";
}
