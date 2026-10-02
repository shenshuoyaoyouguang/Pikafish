#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elo 对弈验证脚本 — Pikafish NNUE 训练后网络 vs 原始网络

用同一份 Pikafish 引擎二进制分别加载训练后 NNUE (Engine A) 与原始 NNUE (Engine B)，
自对弈若干局，每局固定节点数搜索，交替先后手，多起始局面。
使用 SPRT (Sequential Probability Ratio Test) 进行早停判定，
并输出 Elo 估计、95% 置信区间与 SPRT 判定结果。

用法:
    python3 tools/elo_match.py
    MATCH_GAMES=5 python3 tools/elo_match.py          # 5 局快速验证
    MATCH_GAMES=500 MATCH_NODES=10000 python3 tools/elo_match.py

环境变量配置:
    MATCH_GAMES   最大局数        (默认 500)
    MATCH_NODES   每局每方节点数  (默认 10000)
    MATCH_ELO0    H0 的 Elo       (默认 0.0)
    MATCH_ELO1    H1 的 Elo       (默认 10.0)
    MATCH_ALPHA   第一类错误概率  (默认 0.05)
    MATCH_BETA    第二类错误概率  (默认 0.05)
"""

import os
import sys
import time
import json
import math
import select
import shutil
import subprocess
from datetime import datetime

# ============================================================================
# 路径与常量配置
# ============================================================================
ENGINE_PATH = "/mnt/e/xiaoxiao/pikayu/Pikafish/src/pikafish"
ENGINE_CWD = "/mnt/e/xiaoxiao/pikayu/Pikafish/src/"
EVAL_FILE_A = "pikafish_trained.nnue"   # 训练后网络 (fc_2)
EVAL_FILE_B = "pikafish.nnue"           # 原始网络
RESULT_PATH = "/mnt/e/xiaoxiao/pikayu/Pikafish/tools/elo_match_result.txt"

MAX_MOVES = 200        # 单局最大走子数，超过判和
READ_TIMEOUT = 5.0     # 读取 bestmove 的超时秒数
INIT_TIMEOUT = 15.0    # 引擎初始化 (uci/readyok) 超时秒数

# ============================================================================
# 起始局面集 (前 5 个已验证合法；脚本启动时会自动过滤非法 FEN)
# ============================================================================
FENS = [
    # 1. 标准起始局面
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w",
    # 2. 中盘局面 (红方有马踏卒)
    "r1ba1a3/4kn3/2n1b4/pNp1p1p1p/4c4/6P2/P1P2R2P/1CcC5/9/2BAKAB2 w",
    # 3. 中盘局面 (复杂对抗)
    "1cbak4/9/n2a5/2p1p3p/5cp2/2n2N3/6PCP/3AB4/2C6/3A1K1N1 w",
    # 4. 残局局面 (车对车马)
    "5a3/3k5/3aR4/9/5r3/5n3/9/3A1A3/5K3/2BC2B2 w",
    # 5. 中盘局面 (黑方走子)
    "2bak4/9/3a5/p2Np3p/3n1P3/3pc3P/P4r1c1/B2CC2R1/4A4/3AK1B2 b",
    # 6. 起始局面 (重复，用于平衡先后手轮次)
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w",
    # 7. 开局变化: 双正马
    "2bakabnr/9/1c4n2/p1p1p1p1p/9/9/P1P1P1P1P/2N1C4/9/R1BAKAB2 w",
    # 8. 开局变化: 巡河炮 (可能非法，会被自动过滤)
    "rnbakabnr/9/4c4/p1pc1p1p1/9/9/P1PC1P1P1/4C4/9/RNBAKABNR w",
]


# ============================================================================
# 配置读取
# ============================================================================
def get_config():
    """从环境变量读取配置，带默认值与类型转换。"""
    def _env(name, default, cast):
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return default
        try:
            return cast(raw)
        except (ValueError, TypeError):
            print(f"警告: 环境变量 {name}='{raw}' 无效，使用默认值 {default}")
            return default

    return {
        "games":  _env("MATCH_GAMES",  500,    int),
        "nodes":  _env("MATCH_NODES",  10000,  int),
        "elo0":   _env("MATCH_ELO0",   0.0,    float),
        "elo1":   _env("MATCH_ELO1",   10.0,   float),
        "alpha":  _env("MATCH_ALPHA",  0.05,   float),
        "beta":   _env("MATCH_BETA",   0.05,   float),
    }


# ============================================================================
# 引擎封装
# ============================================================================
class PikafishEngine:
    """Pikafish UCI 引擎封装，通过 subprocess 管道交互。"""

    def __init__(self, path, cwd, eval_file, name):
        self.path = path
        self.cwd = cwd
        self.eval_file = eval_file
        self.name = name
        self.proc = None

    def start(self):
        """启动引擎进程并发送 UCI 初始化握手。"""
        if not os.path.isfile(self.path):
            raise RuntimeError(
                f"引擎二进制不存在: {self.path}\n"
                f"请先在 WSL 中编译 Pikafish (make -C src pikafish)。"
            )

        # 使用 stdbuf 强制行缓冲，确保 UCI 输出及时刷新
        if shutil.which("stdbuf"):
            cmd = ["stdbuf", "-oL", self.path]
        else:
            cmd = [self.path]
            print(f"  提示: 未找到 stdbuf，直接启动引擎 (输出可能有缓冲延迟)")

        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=self.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,   # 合并 stderr 到 stdout 便于调试
                bufsize=0,                   # 无缓冲，配合 select
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"无法启动引擎 {self.name}: {e}")

        # 检查进程是否立即退出
        time.sleep(0.1)
        if self.proc.poll() is not None:
            out = self.proc.stdout.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"引擎 {self.name} 启动后立即退出 (code={self.proc.returncode})\n"
                f"输出:\n{out}"
            )

        # UCI 握手
        self._send("uci")
        if not self._wait_for("uciok", timeout=INIT_TIMEOUT):
            raise RuntimeError(f"引擎 {self.name} 未返回 uciok (超时 {INIT_TIMEOUT}s)")

        # 设置 EvalFile
        if self.eval_file:
            self._send(f"setoption name EvalFile value {self.eval_file}")

        # 等待就绪 (NNUE 加载可能较慢)
        self._send("isready")
        if not self._wait_for("readyok", timeout=INIT_TIMEOUT * 2):
            raise RuntimeError(f"引擎 {self.name} 未返回 readyok (超时 {INIT_TIMEOUT*2}s)")

    def _send(self, command):
        """向引擎 stdin 发送一行命令。"""
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError(f"引擎 {self.name} 进程已终止，无法发送: {command}")
        try:
            self.proc.stdin.write((command + "\n").encode("utf-8"))
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise RuntimeError(f"引擎 {self.name} 管道断开，无法发送: {command}")

    def _readline(self, timeout):
        """带超时地读取一行输出，超时返回 None，EOF 返回 None。"""
        if self.proc is None:
            return None
        fd = self.proc.stdout.fileno()
        try:
            ready, _, _ = select.select([fd], [], [], timeout)
        except (OSError, ValueError):
            return None
        if not ready:
            return None
        line = self.proc.stdout.readline()
        if not line:
            return None  # EOF
        return line.decode("utf-8", errors="replace").rstrip("\r\n")

    def _wait_for(self, token, timeout):
        """等待包含 token 的输出行，返回是否成功。"""
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return False
            line = self._readline(remaining)
            if line is None:
                # 超时或 EOF
                if self.proc and self.proc.poll() is not None:
                    return False  # 进程已退出
                continue
            if line.startswith(token):
                return True

    def search(self, fen, moves, nodes):
        """
        在给定局面下搜索，返回 bestmove (字符串)。
        - fen:   起始 FEN
        - moves: 已走步列表 (UCI 格式，如 ['a3a4', 'b7b6'])
        - nodes: 搜索节点数
        返回 None 表示超时或异常。
        """
        # 构造 position 命令
        if moves:
            pos_cmd = f"position fen {fen} moves {' '.join(moves)}"
        else:
            pos_cmd = f"position fen {fen}"
        self._send(pos_cmd)
        self._send(f"go nodes {nodes}")

        # 读取输出直到 bestmove
        bestmove = None
        deadline = time.time() + READ_TIMEOUT
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None  # 超时
            line = self._readline(remaining)
            if line is None:
                if self.proc and self.proc.poll() is not None:
                    return None  # 进程退出
                continue
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    bestmove = parts[1]
                else:
                    bestmove = "0000"
                break
        return bestmove

    def close(self):
        """安全关闭引擎: quit → wait → kill。"""
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                try:
                    self._send("quit")
                except RuntimeError:
                    pass
                self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        finally:
            for stream in (self.proc.stdin, self.proc.stdout):
                if stream:
                    try:
                        stream.close()
                    except OSError:
                        pass
            self.proc = None


# ============================================================================
# 对弈逻辑
# ============================================================================
def play_game(eng_a, eng_b, fen, a_is_white, nodes):
    """
    在给定 FEN 下让 Engine A 与 Engine B 对弈一局。
    - a_is_white: True 表示 A 执红 (先手)，False 表示 A 执黑 (后手)
    返回 'A_WIN', 'B_WIN' 或 'DRAW'。
    """
    moves = []
    # 解析 FEN 中的走子方 ('w' 或 'b')
    fen_tokens = fen.split()
    fen_side = fen_tokens[1] if len(fen_tokens) >= 2 else "w"

    for ply in range(MAX_MOVES):
        # 当前走子方
        if ply % 2 == 0:
            current_side = fen_side
        else:
            current_side = "b" if fen_side == "w" else "w"

        # 判断当前是否 Engine A 走
        # A 执白: current_side=='w' → A 走; A 执黑: current_side=='b' → A 走
        if a_is_white:
            is_a_turn = (current_side == "w")
        else:
            is_a_turn = (current_side == "b")

        eng = eng_a if is_a_turn else eng_b

        # 搜索
        bestmove = eng.search(fen, moves, nodes)

        # bestmove 为空/0000/(none) → 当前方无法走棋 → 对方胜
        if bestmove is None or bestmove in ("0000", "(none)", "none", ""):
            return "B_WIN" if is_a_turn else "A_WIN"

        moves.append(bestmove)

    # 超过最大走子数 → 和棋
    return "DRAW"


# ============================================================================
# SPRT 早停
# ============================================================================
class SPRT:
    """
    Sequential Probability Ratio Test (序贯概率比检验)。
    H0: Elo = elo0 (训练后网络不更好)
    H1: Elo = elo1 (训练后网络好 elo1 Elo)
    每局结果 s: 1.0=胜, 0.5=和, 0.0=负 (从 Engine A 视角)
    """

    def __init__(self, elo0, elo1, alpha, beta):
        self.elo0 = elo0
        self.elo1 = elo1
        self.alpha = alpha
        self.beta = beta

        # 在 H0/H1 下的预期胜率
        self.p0 = 1.0 / (1.0 + 10.0 ** (-elo0 / 400.0))
        self.p1 = 1.0 / (1.0 + 10.0 ** (-elo1 / 400.0))

        # 对数似然比的单步增量系数 (预计算)
        self._ln_p1_p0 = math.log(self.p1 / self.p0)
        self._ln_1mp1_1mp0 = math.log((1.0 - self.p1) / (1.0 - self.p0))

        # 决策边界
        self.upper = math.log((1.0 - beta) / alpha)   # 接受 H1 (PASS)
        self.lower = math.log(beta / (1.0 - alpha))    # 接受 H0 (FAIL)

        self.llr = 0.0
        self.games = 0

    def update(self, score):
        """用一局结果更新 LLR。score: 1.0/0.5/0.0。"""
        self.llr += score * self._ln_p1_p0 + (1.0 - score) * self._ln_1mp1_1mp0
        self.games += 1

    @property
    def status(self):
        """返回 'PASS', 'FAIL' 或 'CONTINUE'。"""
        if self.llr >= self.upper:
            return "PASS"
        if self.llr <= self.lower:
            return "FAIL"
        return "CONTINUE"


# ============================================================================
# Elo 计算
# ============================================================================
def compute_elo(wins, losses, draws):
    """
    从 Engine A 视角计算 Elo 估计与 95% 置信区间半宽。
    返回 (elo, ci_half_width)。
    """
    total = wins + losses + draws
    if total == 0:
        return 0.0, 0.0

    win_rate = (wins + 0.5 * draws) / total

    # 边界情况: 全胜或全负
    if win_rate <= 0.0:
        return float("-inf"), float("inf")
    if win_rate >= 1.0:
        return float("inf"), float("inf")

    # Elo = -400 * log10(1/win_rate - 1)
    elo = -400.0 * math.log10(1.0 / win_rate - 1.0)

    # 正态近似的置信区间
    # std_dev = sqrt(n * p * (1-p))
    # d_elo = 400 / (ln(10) * p * (1-p))   (Elo 对 win_rate 的导数)
    # ci = 1.96 * d_elo * std_dev / n
    std_dev = math.sqrt(total * win_rate * (1.0 - win_rate))
    d_elo = 400.0 / (math.log(10.0) * win_rate * (1.0 - win_rate))
    ci = 1.96 * d_elo * std_dev / total

    return elo, ci


# ============================================================================
# FEN 合法性验证
# ============================================================================
def validate_fens(path, cwd, eval_file, fens, nodes=100):
    """
    启动一个独立的临时引擎验证每个 FEN 是否合法。
    重要: 非法 FEN 会触发 Pikafish CRITICAL ERROR 并导致引擎退出，
    因此必须用独立引擎验证，验证完即关闭，不影响后续对弈引擎。
    返回合法 FEN 列表。
    """
    valid = []
    for i, fen in enumerate(fens):
        # 每个 FEN 用一个全新的引擎实例，避免非法 FEN 崩溃影响后续验证
        probe = PikafishEngine(path, cwd, eval_file, f"probe#{i+1}")
        try:
            probe.start()
            bestmove = probe.search(fen, [], nodes)
            if bestmove and bestmove not in ("0000", "(none)", "none", ""):
                valid.append(fen)
            else:
                print(f"  跳过非法或终局 FEN #{i+1}: {fen}")
        except RuntimeError as e:
            print(f"  跳过 FEN #{i+1} (引擎错误): {fen}")
            print(f"    原因: {e}")
        finally:
            probe.close()
    return valid


# ============================================================================
# 主函数
# ============================================================================
def main():
    config = get_config()

    print("=" * 72)
    print("Pikafish Elo 对弈验证")
    print("=" * 72)
    print(f"Engine A: {EVAL_FILE_A} (trained fc_2)")
    print(f"Engine B: {EVAL_FILE_B} (original)")
    print(f"最大局数: {config['games']}  每局节点: {config['nodes']}")
    print(f"SPRT: elo0={config['elo0']}  elo1={config['elo1']}  "
          f"α={config['alpha']}  β={config['beta']}")
    print("-" * 72)

    # 启动引擎
    eng_a = PikafishEngine(ENGINE_PATH, ENGINE_CWD, EVAL_FILE_A, "A")
    eng_b = PikafishEngine(ENGINE_PATH, ENGINE_CWD, EVAL_FILE_B, "B")

    try:
        # 先用独立临时引擎验证 FEN 合法性 (非法 FEN 会导致引擎崩溃，
        # 必须在对弈引擎启动前完成验证)
        print("[1/4] 验证起始局面集 (独立引擎)...")
        valid_fens = validate_fens(ENGINE_PATH, ENGINE_CWD, EVAL_FILE_B, FENS)
        if not valid_fens:
            raise RuntimeError("没有合法的起始局面，无法对弈。")
        print(f"  合法局面数: {len(valid_fens)} / {len(FENS)}")

        print("[2/4] 启动 Engine A (trained)...")
        eng_a.start()
        print("[3/4] 启动 Engine B (original)...")
        eng_b.start()
        print("[4/4] 开始对弈")
        print("-" * 72)

        # 初始化 SPRT
        sprt = SPRT(config["elo0"], config["elo1"],
                    config["alpha"], config["beta"])
        print(f"SPRT 边界: 下界={sprt.lower:+.4f}  上界={sprt.upper:+.4f}")
        print(f"  p0={sprt.p0:.6f}  p1={sprt.p1:.6f}")
        print("-" * 72)

        # 对弈主循环
        wins = 0
        losses = 0
        draws = 0
        last_result = "---"
        early_stopped = False
        stop_reason = ""

        # 小样本时每局打印，大样本每 10 局打印
        print_interval = 1 if config["games"] <= 20 else 10

        t_start = time.time()

        for i in range(config["games"]):
            # 选择起始局面 (循环复用)
            fen = valid_fens[i % len(valid_fens)]
            # 交替先后手: 偶数局 A 先, 奇数局 B 先
            a_is_white = (i % 2 == 0)

            # 对弈
            result = play_game(eng_a, eng_b, fen, a_is_white, config["nodes"])
            last_result = result

            # 统计
            if result == "A_WIN":
                wins += 1
                score = 1.0
            elif result == "B_WIN":
                losses += 1
                score = 0.0
            else:
                draws += 1
                score = 0.5

            sprt.update(score)

            # 进度输出
            game_num = i + 1
            if game_num % print_interval == 0 or game_num == config["games"]:
                elo, ci = compute_elo(wins, losses, draws)
                elapsed = time.time() - t_start
                if config["games"] <= 20:
                    # 小样本: 详细每局输出
                    side = "A先" if a_is_white else "B先"
                    fen_idx = i % len(valid_fens)
                    print(f"Game {game_num:3d}: {result:5s} | {side} fen#{fen_idx+1} "
                          f"| W:{wins} L:{losses} D:{draws} "
                          f"| Elo: {elo:+.1f} ± {ci:.1f} "
                          f"| LLR: {sprt.llr:+.3f} "
                          f"| {elapsed:.1f}s")
                else:
                    print(f"Game {game_num:4d}: {result:5s}  "
                          f"| W:{wins} L:{losses} D:{draws} "
                          f"| Elo: {elo:+.1f} ± {ci:.1f} "
                          f"| LLR: {sprt.llr:+.3f} "
                          f"| {elapsed:.0f}s")

            # SPRT 早停检查
            status = sprt.status
            if status != "CONTINUE":
                early_stopped = True
                stop_reason = f"SPRT {status} at game {game_num}"
                break

        games_played = wins + losses + draws
        elapsed_total = time.time() - t_start

    finally:
        # 确保引擎被关闭
        eng_a.close()
        eng_b.close()

    # ========================================================================
    # 结果计算与输出
    # ========================================================================
    total = games_played
    win_rate = (wins + 0.5 * draws) / total if total > 0 else 0.0
    elo, ci = compute_elo(wins, losses, draws)
    sprt_status = sprt.status

    # Verdict 判定
    if sprt_status == "PASS":
        verdict = f"PASS (H1 accepted: Elo > {config['elo0']})"
    elif sprt_status == "FAIL":
        verdict = f"FAIL (H0 accepted: Elo <= {config['elo0']})"
    else:
        # CONTINUE: 基于 Elo 符号给出倾向性判定
        if elo > 0:
            verdict = "PASS (Elo > 0)"
        elif elo < 0:
            verdict = "FAIL (Elo < 0)"
        else:
            verdict = "INCONCLUSIVE (Elo = 0)"

    # 人类可读最终结果
    print()
    print("=" * 72)
    print("=== Elo Match Results ===")
    print(f"Engine A: {EVAL_FILE_A} (trained fc_2)")
    print(f"Engine B: {EVAL_FILE_B} (original)")
    if early_stopped:
        print(f"Games: {total} (SPRT early stop at game {total})")
    else:
        print(f"Games: {total} (reached max {config['games']})")
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
    print(f"Win Rate: {win_rate*100:.2f}%")
    if math.isfinite(elo):
        print(f"Elo: {elo:+.1f} ± {ci:.1f} (95% CI)")
    else:
        print(f"Elo: {elo} (边界情况)")
    print(f"LLR: {sprt.llr:+.4f}  [下界 {sprt.lower:+.4f}, 上界 {sprt.upper:+.4f}]")
    if sprt_status == "CONTINUE":
        print(f"SPRT: CONTINUE (need more games)")
    else:
        print(f"SPRT: {sprt_status}")
    print(f"Verdict: {verdict}")
    print(f"Time: {elapsed_total:.1f}s  ({elapsed_total/max(total,1):.2f}s/game)")
    print("=" * 72)

    # JSON 结果文件 (便于后续解析)
    result_json = {
        "engine_a": EVAL_FILE_A,
        "engine_b": EVAL_FILE_B,
        "engine_path": ENGINE_PATH,
        "games_played": total,
        "max_games": config["games"],
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "elo": elo if math.isfinite(elo) else None,
        "elo_ci_half_width": ci if math.isfinite(ci) else None,
        "llr": sprt.llr,
        "llr_upper": sprt.upper,
        "llr_lower": sprt.lower,
        "sprt_status": sprt_status,
        "verdict": verdict,
        "early_stopped": early_stopped,
        "stop_reason": stop_reason,
        "nodes_per_game": config["nodes"],
        "elapsed_seconds": round(elapsed_total, 2),
        "fens_used": len(valid_fens),
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)
        print(f"\nJSON 结果已写入: {RESULT_PATH}")
    except OSError as e:
        print(f"\n警告: 无法写入结果文件 {RESULT_PATH}: {e}")

    # 退出码: PASS → 0, FAIL → 1, CONTINUE → 2
    return {"PASS": 0, "FAIL": 1, "CONTINUE": 2}.get(sprt_status, 0)


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n用户中断，退出。")
        sys.exit(130)
    except RuntimeError as e:
        print(f"\n错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n未预期的错误: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)