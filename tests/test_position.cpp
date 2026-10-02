// ===========================================================================
// position.cpp FEN round-trip 单元测试
// ===========================================================================
// 验证 Position::set(fen) → Position::fen() 的往返一致性。
// 覆盖：初始局面、中局、残局、含炮/马/象/士/将的中国象棋特有局面。
//
// 参考：src/position.cpp 的 Position::set() 与 Position::fen()
// ===========================================================================

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "attacks.h"
#include "position.h"

using namespace Stockfish;

namespace {

// 引擎全局状态初始化（Attacks 攻击表 + Position Zobrist 哈希）
// 对应 src/main.cpp 中的 Attacks::init(); Position::init();
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

// FEN 二次 round-trip 验证：set(fen) → fen1 → set(fen1) → fen2，比较 fen1 == fen2。
// 使用二次 round-trip 是因为输入 FEN 的第 3/4 字段（中国象棋无王车易位/吃过路兵）
// 会被 fen() 规范化为 "- -"，一次 round-trip 后的 fen1 已是规范形式。
struct RoundTripResult {
    bool   set_ok;     // 第一次 set 是否成功
    bool   roundtrip;  // fen1 == fen2
    std::string fen1;  // 第一次 fen() 输出
    std::string fen2;  // 第二次 fen() 输出
    std::string error; // 错误信息
};

RoundTripResult fen_roundtrip(const std::string& fen) {
    engine_init();
    RoundTripResult r{};

    StateInfo st1, st2;
    Position  p1, p2;

    if (auto err = p1.set(fen, &st1)) {
        r.set_ok = false;
        r.error = err->what();
        return r;
    }
    r.set_ok  = true;
    r.fen1    = p1.fen();

    if (auto err = p2.set(r.fen1, &st2)) {
        r.roundtrip = false;
        r.error     = err->what();
        return r;
    }
    r.fen2      = p2.fen();
    r.roundtrip = (r.fen1 == r.fen2);
    return r;
}

}  // namespace

// ---------------------------------------------------------------------------
// 测试数据：中国象棋各类局面（FEN 字段：棋子布局 走棋方 - - rule60 fullmove）
// ---------------------------------------------------------------------------
const std::vector<std::string> kTestFens = {
    // 初始局面
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",

    // 中局（含炮/马/象/士/将，覆盖中国象棋特有棋子）
    "r1ba1a3/4kn3/2n1b4/pNp1p1p1p/4c4/6P2/P1P2R2P/1CcC5/9/2BAKAB2 w - - 0 1",
    "1cbak4/9/n2a5/2p1p3p/5cp2/2n2N3/6PCP/3AB4/2C6/3A1K1N1 w - - 0 1",
    "2b1ka2r/3na2c1/4b3n/8R/8C/4C1P2/P1P1P3P/4B1N2/1r2A4/2BAK4 w - - 0 1",
    "2bakab2/9/2n1c1R1c/3r4p/4N4/r8/6P1P/6C1C/4A4/1RBAK1B2 w - - 0 1",

    // 残局
    "5a3/3k5/3aR4/9/5r3/5n3/9/3A1A3/5K3/2BC2B2 w - - 0 1",
    "4ka3/3Pa4/r6R1/2C4C1/9/9/8n/9/4p3r/3K3R1 w - - 0 1",
    "3ak4/3Pa4/4b3b/5r3/1R3N3/9/9/B8/2p1A4/2B1KA3 w - - 0 1",

    // 黑方走棋
    "2bak4/9/3a5/p2Np3p/3n1P3/3pc3P/P4r1c1/B2CC2R1/4A4/3AK1B2 b - - 0 1",
    "1r1akabr1/1c7/2n1b1n2/p1p1p3p/6p2/PN3R3/1cP1P1P1P/2C1C1N2/1R7/2BAKAB2 b - - 0 1",

    // 仅将仕（最小合法局面，两将不同列避免飞将）
    "3ak4/9/9/9/9/9/9/9/9/3K1A3 w - - 0 1",

    // 仅双将（不同列，合法）
    "3k5/9/9/9/9/9/9/9/9/4K4 w - - 0 1",

    // 含炮的简单局面（炮挡在两将之间，合法）
    "4k4/9/9/9/9/9/9/9/4C4/4K4 w - - 0 1",

    // 含马的简单局面（马挡在两将之间，合法）
    "4k4/9/9/9/9/9/9/9/4N4/4K4 w - - 0 1",

    // 含象的简单局面（象在合法点位 (0,2)，两将不同列避免飞将）
    "3k5/9/9/9/9/9/9/9/9/2B1K4 w - - 0 1",
};

// ---------------------------------------------------------------------------
// 测试用例
// ---------------------------------------------------------------------------

// 初始局面 round-trip
TEST(PositionFen, StartPositionRoundTrip) {
    std::string startfen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";
    auto        r        = fen_roundtrip(startfen);
    ASSERT_TRUE(r.set_ok) << "set() failed: " << r.error;
    EXPECT_TRUE(r.roundtrip) << "fen1 != fen2\nfen1: " << r.fen1 << "\nfen2: " << r.fen2;
}

// 所有测试局面 round-trip
TEST(PositionFen, RoundTripAllPositions) {
    for (const auto& fen : kTestFens) {
        auto r = fen_roundtrip(fen);
        ASSERT_TRUE(r.set_ok) << "set() failed for: " << fen << "\nerror: " << r.error;
        EXPECT_TRUE(r.roundtrip)
          << "FEN round-trip failed for: " << fen << "\nfen1: " << r.fen1 << "\nfen2: " << r.fen2;
    }
}

// 验证初始局面的 FEN 输出格式正确
TEST(PositionFen, StartFenOutputFormat) {
    engine_init();
    std::string  startfen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";
    StateInfo    st;
    Position     p;
    auto         err = p.set(startfen, &st);
    ASSERT_FALSE(err.has_value()) << err->what();
    std::string  out = p.fen();
    EXPECT_EQ(out, startfen) << "Initial position FEN mismatch";
}

// 验证非法 FEN 被正确拒绝
TEST(PositionFen, RejectInvalidFen) {
    engine_init();
    StateInfo st;
    Position  p;

    // 双将缺失
    auto err = p.set("9/9/9/9/9/9/9/9/9/9 w - - 0 1", &st);
    EXPECT_TRUE(err.has_value()) << "Should reject FEN with no kings";

    // 非法棋子字符
    err = p.set("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR z - - 0 1", &st);
    EXPECT_TRUE(err.has_value()) << "Should reject FEN with invalid side to move";

    // 飞将（两将同列无遮挡，中国象棋非法局面）
    err = p.set("4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1", &st);
    EXPECT_TRUE(err.has_value()) << "Should reject FEN with kings facing each other (flying general)";
}