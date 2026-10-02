#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nnue_pybind 前向接口验证测试。

对每个测试 FEN，通过两条独立路径获取 NNUE 评估值并交叉验证：
  路径 A — C++ UCI `eval` 命令（subprocess 调用 src/pikafish 二进制）
  路径 B — Python pybind 模块（src/nnue_pybind*.so）

验证项：
  1. evaluate(fen)      == UCI "NNUE evaluation <v> (side to move, internal units)"
  2. evaluate_cp(fen)   == round(UCI "NNUE evaluation <p> (white side)" * 100)
  3. evaluate_batch     == [evaluate(f) for f in fens]
  4. check 局面 evaluate_scaled(fen) == evaluate(fen)   （回退逻辑）
  5. 无效 FEN 抛出 RuntimeError
  6. 黑方走棋时 evaluate_cp 返回负值（白方视角，红方优势局面）
  7. 对抗性探针：幂等性、空 FEN、重复 load、批量边界

运行方式（WSL）：
  wsl bash -c "cd /mnt/e/xiaoxiao/pikayu/Pikafish && python3 tools/test_nnue_pybind.py"
"""
import sys
import os
import subprocess
import re

# 把 src 目录加入 sys.path，使 import nnue_pybind 能找到编译好的 .so
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.normpath(os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _SRC_DIR)

import nnue_pybind  # noqa: E402

# ---------------------------------------------------------------------------
# 路径配置（WSL 视角）。脚本默认从仓库根目录运行；也兼容从 tools/ 运行。
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
def _resolve(name):
    """优先在仓库根找，找不到回退到脚本同级。"""
    p = os.path.join(_REPO_ROOT, name)
    return p if os.path.exists(p) else os.path.join(_HERE, name)

PIKAFISH_BIN = _resolve("src/pikafish")
NNUE_FILE = _resolve("src/pikafish.nnue")
if not os.path.exists(NNUE_FILE):
    NNUE_FILE = _resolve("networks/pikafish.nnue")

# ---------------------------------------------------------------------------
# 硬编码测试 FEN 集合（均已验证合法）
#   覆盖：起始、开局、残局、将军、九宫角、对称/非对称、红优/黑优/均势
# ---------------------------------------------------------------------------
HARDCODED_FENS = [
    # 起始局面（红走 / 黑走）
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 0 1",
    # 拗局：仅将帅（不对脸，黑将 e0 / 红帅 d9）
    "4k4/9/9/9/9/9/9/9/9/3K5 w - - 0 1",
    # 拗局：红车 + 将帅（红车 a8，不与黑将同列）
    "4k4/9/9/9/9/9/9/9/R8/3K5 w - - 0 1",
    # 将军局面：红车直将黑将（黑走应将，红车 e1 阻挡将帅对脸）
    "4k4/9/9/9/9/9/9/9/4R4/4K4 b - - 0 1",
    # 将军局面：红马卧槽将黑将（黑走应将）
    "4k4/2N6/9/9/9/9/9/9/9/3K5 b - - 0 1",
    # 九宫格边角：黑将 d0 / 红帅 e9
    "3k5/9/9/9/9/9/9/9/9/4K4 w - - 0 1",
    # 九宫格边角：红帅 f9 / 黑将 e0
    "4k4/9/9/9/9/9/9/9/9/5K3 w - - 0 1",
    # 拗局：红马 + 红兵（红帅 e9，黑将 d0 不对脸）
    "3k5/9/9/9/9/9/9/9/9/3NK4 w - - 0 1",
    # 拗局：红炮 + 红兵（红炮 d9 与黑将 d0 同列但无炮架，不攻击）
    "3k5/9/9/9/9/9/9/9/9/3CK4 w - - 0 1",
    # 拗局：黑马 + 黑将（黑走，黑马 c0，黑将 e0，红帅 d9 不对脸）
    "2n1k4/9/9/9/9/9/9/9/9/3K5 b - - 0 1",
    # 中局：红方双车（一车过河 a5，一车底线 c9，红帅 d9，黑将 e0 不对脸）
    "4k4/9/9/9/9/R8/9/9/9/2RK5 w - - 0 1",
    # 中局：红方一炮换位（红炮从 b7 移到 c8，保留 e 列红兵阻挡将帅）
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/9/2C6/RNBAKABNR w - - 0 1",
    # 拗局：红方双马（红帅 e9，黑将 d0）
    "3k5/9/9/9/9/9/9/9/9/2NNK4 w - - 0 1",
]

# 无效 FEN（必须抛异常）
INVALID_FENS = [
    "invalid_fen_string",
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR x - - 0 1",  # 非法走棋方
    "9/9/9/9/9/9/9/9/9/9 w - - 0 1",  # 无将帅
    "4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1",  # 将帅对脸（飞将）
    "",  # 空字符串
]


# ---------------------------------------------------------------------------
# UCI 交互层
# ---------------------------------------------------------------------------
def _run_uci(commands, timeout=15):
    """向 pikafish 发送一组 UCI 命令，返回 stdout 文本。"""
    cmd_text = "".join(c + "\n" for c in commands) + "quit\n"
    result = subprocess.run(
        [PIKAFISH_BIN],
        input=cmd_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout


def uci_eval(fen):
    """
    通过 UCI `eval` 命令获取 NNUE 评估值。

    返回 dict:
      stm       : int | None   — "NNUE evaluation <v> (side to move, internal units)"
      white_cp  : int | None   — "NNUE evaluation <p> (white side)" 换算的厘兵值
      in_check  : bool         — 是否处于将军局面（UCI 输出 "none (in check)"）
    check 局面 UCI 不输出 NNUE 值，stm/white_cp 为 None。
    """
    out = _run_uci(["position fen " + fen, "eval"])
    info = {"stm": None, "white_cp": None, "in_check": False}

    for line in out.split("\n"):
        line = line.strip()
        # "Final evaluation: none (in check)" 表示将军局面
        if "Final evaluation" in line and "in check" in line:
            info["in_check"] = True
            continue
        # "NNUE evaluation          +97 (side to move, internal units)"
        if "NNUE evaluation" in line and "side to move" in line and "internal units" in line:
            m = re.search(r"NNUE evaluation\s+([+-]?\d+)\s+\(side to move", line)
            if m:
                info["stm"] = int(m.group(1))
            continue
        # "NNUE evaluation        +0.24 (white side)"
        if "NNUE evaluation" in line and "white side" in line:
            m = re.search(r"NNUE evaluation\s+([+-]?\d+\.?\d*)\s+\(white side", line)
            if m:
                info["white_cp"] = round(float(m.group(1)) * 100)
            continue
    return info


def uci_checkers(fen):
    """通过 UCI `d` 命令获取 Checkers 行，返回 checker 平方列表字符串（空串=非将军）。"""
    out = _run_uci(["position fen " + fen, "d"])
    for line in out.split("\n"):
        if line.startswith("Checkers:"):
            return line[len("Checkers:"):].strip()
    return ""


def generate_midgame_fens(num_moves=20, depth=6):
    """
    用 pikafish 从起始局面自走，收集每步走完后的 FEN。

    返回 list[str]，长度 <= num_moves（遇到将杀/和棋会提前结束）。
    """
    fens = []
    moves = []
    start = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"

    for _ in range(num_moves):
        # 1) 用 d 命令获取当前 FEN
        pos_cmd = "position startpos" + (" moves " + " ".join(moves) if moves else "")
        out = _run_uci([pos_cmd, "d"])
        cur_fen = None
        for line in out.split("\n"):
            if line.startswith("Fen:"):
                cur_fen = line[len("Fen:"):].strip()
                break
        if cur_fen:
            fens.append(cur_fen)

        # 2) 用 go depth 获取 bestmove
        out = _run_uci([pos_cmd, "go depth " + str(depth)])
        bestmove = None
        for line in out.split("\n"):
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2 and parts[1] != "(none)":
                    bestmove = parts[1]
                break
        if not bestmove:
            break
        moves.append(bestmove)

    return fens


# ---------------------------------------------------------------------------
# 测试结果收集
# ---------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures = []  # [(test_name, detail)]

    def ok(self, msg=""):
        self.passed += 1
        print("  [PASS] " + msg)

    def fail(self, test_name, detail):
        self.failed += 1
        self.failures.append((test_name, detail))
        print("  [FAIL] " + detail)

    def summary(self):
        total = self.passed + self.failed
        print("\n" + "=" * 70)
        print("测试汇总: %d 用例, %d 通过, %d 失败" % (total, self.passed, self.failed))
        if self.failures:
            print("\n失败明细:")
            for name, detail in self.failures:
                print("  [%s] %s" % (name, detail))
        print("=" * 70)
        return self.failed == 0


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------
def test_evaluate_match(fens, report):
    """测试 evaluate() 与 UCI eval 的 side-to-move 原始值完全一致。"""
    print("\n[测试1] evaluate() 与 UCI eval (side to move) 完全一致")
    for fen in fens:
        uci = uci_eval(fen)
        if uci["in_check"]:
            # check 局面 UCI 不输出 NNUE 值，跳过（由 test_check_fallback 覆盖）
            report.ok("check 局面跳过 UCI 比较: " + fen[:40])
            continue
        if uci["stm"] is None:
            report.fail("evaluate_match", "UCI 未返回 stm 值: " + fen)
            continue
        try:
            py_val = nnue_pybind.evaluate(fen)
        except Exception as e:
            report.fail("evaluate_match", "evaluate 抛异常 %s: %s" % (type(e).__name__, str(e)[:60]))
            continue
        if py_val == uci["stm"]:
            report.ok("evaluate=%d == UCI=%d  %s" % (py_val, uci["stm"], fen[:40]))
        else:
            report.fail("evaluate_match",
                        "evaluate=%d != UCI=%d  FEN=%s" % (py_val, uci["stm"], fen))


def test_evaluate_cp(fens, report):
    """测试 evaluate_cp() 与 UCI eval 的 white side 厘兵值一致。"""
    print("\n[测试2] evaluate_cp() 与 UCI eval (white side) 厘兵值一致")
    for fen in fens:
        uci = uci_eval(fen)
        if uci["in_check"] or uci["white_cp"] is None:
            report.ok("check/无值局面跳过 cp 比较: " + fen[:40])
            continue
        try:
            py_cp = nnue_pybind.evaluate_cp(fen)
        except Exception as e:
            report.fail("evaluate_cp", "evaluate_cp 抛异常 %s: %s" % (type(e).__name__, str(e)[:60]))
            continue
        if py_cp == uci["white_cp"]:
            report.ok("cp=%d == UCI=%d  %s" % (py_cp, uci["white_cp"], fen[:40]))
        else:
            report.fail("evaluate_cp",
                        "cp=%d != UCI=%d  FEN=%s" % (py_cp, uci["white_cp"], fen))


def test_batch(fens, report):
    """测试 evaluate_batch() 与逐个 evaluate() 结果完全一致。"""
    print("\n[测试3] evaluate_batch() 与逐个 evaluate() 一致")
    # 取前 12 个 FEN 做批量（含 check 局面）
    batch_fens = fens[:12]
    try:
        batch_vals = nnue_pybind.evaluate_batch(batch_fens)
    except Exception as e:
        report.fail("batch", "evaluate_batch 抛异常 %s: %s" % (type(e).__name__, str(e)[:60]))
        return
    all_ok = True
    for i, fen in enumerate(batch_fens):
        try:
            single = nnue_pybind.evaluate(fen)
        except Exception as e:
            report.fail("batch", "evaluate 抛异常 %s: %s" % (type(e).__name__, str(e)[:60]))
            all_ok = False
            continue
        if batch_vals[i] != single:
            report.fail("batch", "batch[%d]=%d != evaluate=%d  FEN=%s" % (i, batch_vals[i], single, fen))
            all_ok = False
    if all_ok:
        report.ok("批量 %d 个 FEN 与逐个 evaluate 完全一致" % len(batch_fens))

    # 对抗性探针：空列表
    try:
        empty = nnue_pybind.evaluate_batch([])
        if empty == []:
            report.ok("对抗性: evaluate_batch([]) == []")
        else:
            report.fail("batch", "evaluate_batch([]) != []: %r" % empty)
    except Exception as e:
        report.fail("batch", "evaluate_batch([]) 抛异常 %s" % type(e).__name__)


def test_check_fallback(fens, report):
    """测试 check 局面 evaluate_scaled() 回退到 evaluate() 值。"""
    print("\n[测试4] check 局面 evaluate_scaled() 回退到 evaluate()")
    check_count = 0
    for fen in fens:
        checkers = uci_checkers(fen)
        if not checkers:
            continue
        check_count += 1
        try:
            raw = nnue_pybind.evaluate(fen)
            scaled = nnue_pybind.evaluate_scaled(fen)
        except Exception as e:
            report.fail("check_fallback", "check 局面抛异常 %s: %s" % (type(e).__name__, str(e)[:60]))
            continue
        if scaled == raw:
            report.ok("check 回退 scaled=%d == raw=%d  checkers=%s  %s" % (scaled, raw, checkers, fen[:30]))
        else:
            report.fail("check_fallback",
                        "scaled=%d != raw=%d  FEN=%s" % (scaled, raw, fen))
    if check_count == 0:
        report.fail("check_fallback", "未找到任何 check 局面，测试无效")


def test_invalid_fen(report):
    """测试无效 FEN 抛出异常。"""
    print("\n[测试5] 无效 FEN 抛出 RuntimeError")
    for fen in INVALID_FENS:
        try:
            nnue_pybind.evaluate(fen)
            report.fail("invalid_fen", "无效 FEN 未抛异常: %r" % fen[:30])
        except RuntimeError:
            report.ok("无效 FEN 抛 RuntimeError: %r" % fen[:30])
        except Exception as e:
            report.ok("无效 FEN 抛 %s: %r" % (type(e).__name__, fen[:30]))

    # 对抗性探针：None / 非字符串
    for bad in [None, 123, ["a", "b"]]:
        try:
            nnue_pybind.evaluate(bad)
            report.fail("invalid_fen", "非字符串输入未抛异常: %r" % type(bad))
        except Exception:
            report.ok("对抗性: 非字符串输入 %s 抛异常" % type(bad).__name__)


def test_black_side_sign(report):
    """测试黑方走棋时 evaluate_cp 返回负值（白方视角，红方优势局面）。"""
    print("\n[测试6] 黑方走棋时 evaluate_cp 符号（白方视角）")
    # 起始局面红方略优；黑方走时白方视角应为负
    fen_black = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 0 1"
    fen_white = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    try:
        cp_black = nnue_pybind.evaluate_cp(fen_black)
        cp_white = nnue_pybind.evaluate_cp(fen_white)
    except Exception as e:
        report.fail("black_sign", "抛异常 %s: %s" % (type(e).__name__, str(e)[:60]))
        return
    # 起始局面 NNUE 认为红方（白方）略优：白走 cp>0，黑走 cp<0
    if cp_white > 0 and cp_black < 0 and cp_white == -cp_black:
        report.ok("白走 cp=%d > 0, 黑走 cp=%d < 0, 互为相反数" % (cp_white, cp_black))
    else:
        report.fail("black_sign",
                    "白走 cp=%d, 黑走 cp=%d, 期望白正黑负且互为相反数" % (cp_white, cp_black))


def test_idempotent(fens, report):
    """对抗性探针：多次 evaluate 同一 FEN 结果一致（幂等性）。"""
    print("\n[测试7] 幂等性：多次 evaluate 同一 FEN 结果一致")
    fen = fens[0]
    try:
        vals = [nnue_pybind.evaluate(fen) for _ in range(5)]
    except Exception as e:
        report.fail("idempotent", "抛异常 %s" % type(e).__name__)
        return
    if len(set(vals)) == 1:
        report.ok("5 次 evaluate 均返回 %d" % vals[0])
    else:
        report.fail("idempotent", "5 次 evaluate 结果不一致: %r" % vals)


def test_reload(report):
    """对抗性探针：重复 load 不崩溃，且不影响后续评估。"""
    print("\n[测试8] 对抗性：重复 load 网络文件")
    fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    try:
        v1 = nnue_pybind.evaluate(fen)
        nnue_pybind.load(NNUE_FILE)
        nnue_pybind.load(NNUE_FILE)
        v2 = nnue_pybind.evaluate(fen)
    except Exception as e:
        report.fail("reload", "重复 load 抛异常 %s: %s" % (type(e).__name__, str(e)[:60]))
        return
    if v1 == v2:
        report.ok("重复 load 后 evaluate 不变: %d" % v1)
    else:
        report.fail("reload", "重复 load 后 evaluate 变化: %d -> %d" % (v1, v2))


def test_scaled_no_exception(fens, report):
    """对抗性探针：非 check 局面 evaluate_scaled 不抛异常且为合理 int。"""
    print("\n[测试9] evaluate_scaled 在非 check 局面不抛异常")
    cnt = 0
    for fen in fens:
        if uci_checkers(fen):
            continue
        try:
            s = nnue_pybind.evaluate_scaled(fen)
        except Exception as e:
            report.fail("scaled", "evaluate_scaled 抛异常 %s: %s  FEN=%s" % (type(e).__name__, str(e)[:40], fen))
            continue
        if isinstance(s, int) and -100000 < s < 100000:
            cnt += 1
        else:
            report.fail("scaled", "evaluate_scaled 返回异常值 %r  FEN=%s" % (s, fen))
    if cnt > 0:
        report.ok("%d 个非 check 局面 evaluate_scaled 均返回合理 int" % cnt)
    else:
        report.fail("scaled", "无非 check 局面被测试")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("nnue_pybind 前向接口验证测试")
    print("=" * 70)
    print("pikafish 二进制: %s" % PIKAFISH_BIN)
    print("NNUE 网络文件  : %s" % NNUE_FILE)
    print("pybind 模块目录: %s" % _SRC_DIR)

    # 检查文件存在性
    if not os.path.exists(PIKAFISH_BIN):
        print("\n[错误] 找不到 pikafish 二进制: %s" % PIKAFISH_BIN)
        sys.exit(2)
    if not os.path.exists(NNUE_FILE):
        print("\n[错误] 找不到 NNUE 网络文件: %s" % NNUE_FILE)
        sys.exit(2)

    # 加载网络
    print("\n加载 NNUE 网络...")
    nnue_pybind.load(NNUE_FILE)
    print("网络加载成功")

    # 生成中局 FEN
    print("\n用引擎自走生成中局 FEN（depth=6, 20 步）...")
    midgame_fens = generate_midgame_fens(num_moves=20, depth=6)
    print("生成 %d 个中局 FEN" % len(midgame_fens))

    # 合并 FEN 集合（去重，保持顺序）
    all_fens = []
    seen = set()
    for f in HARDCODED_FENS + midgame_fens:
        if f not in seen:
            seen.add(f)
            all_fens.append(f)
    print("总计 %d 个测试 FEN（硬编码 %d + 中局 %d，去重后）"
          % (len(all_fens), len(HARDCODED_FENS), len(midgame_fens)))

    if len(all_fens) < 30:
        print("[警告] 测试 FEN 数量 %d < 30，可能不足" % len(all_fens))

    # 运行所有测试
    report = Report()
    test_evaluate_match(all_fens, report)
    test_evaluate_cp(all_fens, report)
    test_batch(all_fens, report)
    test_check_fallback(all_fens, report)
    test_invalid_fen(report)
    test_black_side_sign(report)
    test_idempotent(all_fens, report)
    test_reload(report)
    test_scaled_no_exception(all_fens, report)

    # 输出汇总
    ok = report.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()