// ===========================================================================
// movegen.cpp perft 单元测试
// ===========================================================================
// perft（性能测试，递归计数走法数）节点数验证。
// 使用中国象棋初始局面的已知 perft 基准数据。
//
// 参考：src/movegen.cpp（走法生成）、src/perft.h（perft 模板）
// 基准数据来源：https://www.chessprogramming.org/Chinese_Chess_Perft_Results
// ===========================================================================

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "attacks.h"
#include "movegen.h"
#include "position.h"

using namespace Stockfish;

namespace {

// 引擎全局状态初始化
struct EngineInit {
    EngineInit() {
        Attacks::init();
        Position::init();
    }
};
EngineInit& engine_init() {
    static EngineInit init;
    return init;
}

// 安静的 perft 实现（不输出每个走法的计数）。
//
// 注意：src/perft.h 中的 Benchmark::perft<true> 会在 Root 层输出每个走法的计数
// 到 stdout（sync_cout），不适合单元测试。perft<false> 在 depth<=1 时会无限递归
// （因为 `Root && depth <= 1` 短路分支不生效）。因此这里自行实现一个简洁的 perft。
//
// 逻辑：depth==0 返回 1；否则枚举所有合法走法，递归计数。
u64 perft_quiet(Position& pos, int depth) {
    if (depth == 0)
        return 1;

    StateInfo st;
    u64        nodes = 0;
    for (const auto& m : MoveList<LEGAL>(pos))
    {
        pos.do_move(m, st);
        nodes += (depth == 1) ? 1 : perft_quiet(pos, depth - 1);
        pos.undo_move(m);
    }
    return nodes;
}

// 从 FEN 设置局面并运行 perft
u64 perft_from_fen(const std::string& fen, int depth) {
    engine_init();
    StateInfo st;
    Position  p;
    if (auto err = p.set(fen, &st))
        return 0;  // FEN 无效
    return perft_quiet(p, depth);
}

}  // namespace

// ---------------------------------------------------------------------------
// 中国象棋初始局面 FEN
// ---------------------------------------------------------------------------
constexpr const char* kStartFen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";

// ---------------------------------------------------------------------------
// perft 基准数据（初始局面）
//   来源：chessprogramming wiki / Pikafish 文档
//   perft(1) = 44         （初始局面合法走法数）
//   perft(2) = 1,920
//   perft(3) = 79,666
//   perft(4) = 3,290,240
//   perft(5) = 133,314,736  （太大，不在默认测试中）
// ---------------------------------------------------------------------------
struct PerftCase {
    int depth;
    uint64_t expected;
};

const std::vector<PerftCase> kStartPerftCases = {
    {1, 44ULL},
    {2, 1920ULL},
    {3, 79666ULL},
    {4, 3290240ULL},
};

// ---------------------------------------------------------------------------
// 测试用例
// ---------------------------------------------------------------------------

// 初始局面 perft 各深度
TEST(MovegenPerft, InitialPosition) {
    for (const auto& c : kStartPerftCases) {
        uint64_t got = perft_from_fen(kStartFen, c.depth);
        EXPECT_EQ(got, c.expected)
          << "perft(" << c.depth << ") mismatch for initial position";
    }
}

// 初始局面 perft(1) 单独验证（44 个合法走法）
TEST(MovegenPerft, InitialPositionDepth1) {
    uint64_t got = perft_from_fen(kStartFen, 1);
    EXPECT_EQ(got, 44ULL);
}

// 初始局面 perft(2) 单独验证
TEST(MovegenPerft, InitialPositionDepth2) {
    uint64_t got = perft_from_fen(kStartFen, 2);
    EXPECT_EQ(got, 1920ULL);
}

// 初始局面 perft(3) 单独验证
TEST(MovegenPerft, InitialPositionDepth3) {
    uint64_t got = perft_from_fen(kStartFen, 3);
    EXPECT_EQ(got, 79666ULL);
}

// 验证非初始局面也能正确生成走法（perft(1) 应为合法走法数 > 0）
TEST(MovegenPerft, MiddleGameHasMoves) {
    // 中局局面（来自 src/benchmark.cpp Defaults）
    std::string fen = "r1ba1a3/4kn3/2n1b4/pNp1p1p1p/4c4/6P2/P1P2R2P/1CcC5/9/2BAKAB2 w - - 0 1";
    uint64_t    got = perft_from_fen(fen, 1);
    EXPECT_GT(got, 0ULL) << "Middle game position should have legal moves";
}

// 验证黑方走棋局面 perft
TEST(MovegenPerft, BlackToMove) {
    std::string fen = "2bak4/9/3a5/p2Np3p/3n1P3/3pc3P/P4r1c1/B2CC2R1/4A4/3AK1B2 b - - 0 1";
    uint64_t    got = perft_from_fen(fen, 1);
    EXPECT_GT(got, 0ULL) << "Black-to-move position should have legal moves";
}