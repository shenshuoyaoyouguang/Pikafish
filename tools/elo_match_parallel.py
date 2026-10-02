#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发 Elo 对弈验证 — 多进程版本

用 N_WORKERS 个进程并发对弈，每个进程启动两个引擎实例，
独立对弈 GAMES_PER_WORKER 局，最后汇总结果计算 Elo/CI/SPRT。

用法:
    python3 tools/elo_match_parallel.py
    N_WORKERS=10 GAMES_PER_WORKER=500 python3 tools/elo_match_parallel.py

环境变量:
    N_WORKERS          worker 进程数     (默认 10)
    GAMES_PER_WORKER   每 worker 局数    (默认 500)
    MATCH_NODES        每局节点数        (默认 10000)
    MATCH_ELO0         H0 的 Elo         (默认 0.0)
    MATCH_ELO1         H1 的 Elo         (默认 10.0)
"""

import os
import sys
import time
import json
import math
import select
import shutil
import subprocess
import multiprocessing
from datetime import datetime

# ============================================================================
# 路径与常量
# ============================================================================
ENGINE_PATH = "/mnt/e/xiaoxiao/pikayu/Pikafish/src/pikafish"
ENGINE_CWD  = "/mnt/e/xiaoxiao/pikayu/Pikafish/src/"
EVAL_FILE_A = "pikafish_trained.nnue"
EVAL_FILE_B = "pikafish.nnue"
RESULT_PATH = "/mnt/e/xiaoxiao/pikayu/Pikafish/tools/elo_match_parallel_result.txt"

MAX_MOVES    = 200
READ_TIMEOUT = 5.0
INIT_TIMEOUT = 15.0

FENS = [
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w",
    "r1ba1a3/4kn3/2n1b4/pNp1p1p1p/4c4/6P2/P1P2R2P/1CcC5/9/2BAKAB2 w",
    "1cbak4/9/n2a5/2p1p3p/5cp2/2n2N3/6PCP/3AB4/2C6/3A1K1N1 w",
    "5a3/3k5/3aR4/9/5r3/5n3/9/3A1A3/5K3/2BC2B2 w",
    "2bak4/9/3a5/p2Np3p/3n1P3/3pc3P/P4r1c1/B2CC2R1/4A4/3AK1B2 b",
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w",
    "2bakabnr/9/1c4n2/p1p1p1p1p/9/9/P1P1P1P1P/2N1C4/9/R1BAKAB2 w",
]

# ============================================================================
# 引擎封装 (与 elo_match.py 一致)
# ============================================================================
class PikafishEngine:
    def __init__(self, path, cwd, eval_file, name):
        self.path = path
        self.cwd = cwd
        self.eval_file = eval_file
        self.name = name
        self.proc = None

    def start(self):
        if shutil.which("stdbuf"):
            cmd = ["stdbuf", "-oL", self.path]
        else:
            cmd = [self.path]
        self.proc = subprocess.Popen(
            cmd, cwd=self.cwd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=0,
        )
        time.sleep(0.1)
        if self.proc.poll() is not None:
            raise RuntimeError(f"引擎 {self.name} 启动后立即退出")
        self._send("uci")
        if not self._wait_for("uciok", INIT_TIMEOUT):
            raise RuntimeError(f"引擎 {self.name} uciok 超时")
        if self.eval_file:
            self._send(f"setoption name EvalFile value {self.eval_file}")
        self._send("isready")
        if not self._wait_for("readyok", INIT_TIMEOUT * 2):
            raise RuntimeError(f"引擎 {self.name} readyok 超时")

    def _send(self, cmd):
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError(f"引擎 {self.name} 已终止")
        self.proc.stdin.write((cmd + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def _readline(self, timeout):
        fd = self.proc.stdout.fileno()
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            return None
        line = self.proc.stdout.readline()
        if not line:
            return None
        return line.decode("utf-8", errors="replace").rstrip("\r\n")

    def _wait_for(self, token, timeout):
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return False
            line = self._readline(remaining)
            if line is None:
                if self.proc and self.proc.poll() is not None:
                    return False
                continue
            if line.startswith(token):
                return True

    def search(self, fen, moves, nodes):
        if moves:
            self._send(f"position fen {fen} moves {' '.join(moves)}")
        else:
            self._send(f"position fen {fen}")
        self._send(f"go nodes {nodes}")
        deadline = time.time() + READ_TIMEOUT
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            line = self._readline(remaining)
            if line is None:
                if self.proc and self.proc.poll() is not None:
                    return None
                continue
            if line.startswith("bestmove"):
                parts = line.split()
                return parts[1] if len(parts) >= 2 else "0000"

    def close(self):
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
            for s in (self.proc.stdin, self.proc.stdout):
                if s:
                    try:
                        s.close()
                    except OSError:
                        pass
            self.proc = None


# ============================================================================
# 对弈逻辑
# ============================================================================
def play_game(eng_a, eng_b, fen, a_is_white, nodes):
    moves = []
    fen_side = fen.split()[1] if len(fen.split()) >= 2 else "w"
    for ply in range(MAX_MOVES):
        white_to_move = (fen_side == "w") == (ply % 2 == 0)
        eng = eng_a if (white_to_move == a_is_white) else eng_b
        bestmove = eng.search(fen, moves, nodes)
        if bestmove is None or bestmove in ("0000", "(none)"):
            return 'B_WIN' if eng is eng_a else 'A_WIN'
        moves.append(bestmove)
    return 'DRAW'


# ============================================================================
# Worker 函数 (每个进程独立运行)
# ============================================================================
def worker_fn(args):
    worker_id, games, fens, nodes = args
    try:
        eng_a = PikafishEngine(ENGINE_PATH, ENGINE_CWD, EVAL_FILE_A, f"A-{worker_id}")
        eng_b = PikafishEngine(ENGINE_PATH, ENGINE_CWD, EVAL_FILE_B, f"B-{worker_id}")
        eng_a.start()
        eng_b.start()
    except Exception as e:
        return {"worker": worker_id, "error": str(e), "wins": 0, "losses": 0, "draws": 0}

    wins, losses, draws = 0, 0, 0
    t0 = time.time()
    for i in range(games):
        fen = fens[(worker_id * 37 + i) % len(fens)]  # 错位分配局面
        a_is_white = (i % 2 == 0)
        result = play_game(eng_a, eng_b, fen, a_is_white, nodes)
        if result == 'A_WIN':
            wins += 1
        elif result == 'B_WIN':
            losses += 1
        else:
            draws += 1
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            sys.stderr.write(f"  Worker {worker_id}: {i+1}/{games} | W:{wins} L:{losses} D:{draws} | {elapsed:.0f}s\n")
            sys.stderr.flush()

    eng_a.close()
    eng_b.close()
    elapsed = time.time() - t0
    return {"worker": worker_id, "wins": wins, "losses": losses, "draws": draws, "time": elapsed}


# ============================================================================
# SPRT / Elo 计算
# ============================================================================
def compute_elo(wins, losses, draws):
    total = wins + losses + draws
    if total == 0:
        return 0.0, 0.0
    win_rate = (wins + 0.5 * draws) / total
    if win_rate <= 0.0 or win_rate >= 1.0:
        return (999.0 if win_rate >= 1.0 else -999.0), 0.0
    elo = -400.0 * math.log10(1.0 / win_rate - 1.0)
    std_dev = math.sqrt(total * win_rate * (1 - win_rate))
    d_elo = 400.0 / (math.log(10) * win_rate * (1 - win_rate))
    ci = 1.96 * d_elo * std_dev / total
    return elo, ci

def compute_llr(wins, losses, draws, elo0, elo1):
    total = wins + losses + draws
    if total == 0:
        return 0.0
    p0 = 1.0 / (1.0 + 10.0 ** (-elo0 / 400.0))
    p1 = 1.0 / (1.0 + 10.0 ** (-elo1 / 400.0))
    s = (wins + 0.5 * draws) / total
    return s * math.log(p1 / p0) + (1 - s) * math.log((1 - p1) / (1 - p0))


# ============================================================================
# 主函数
# ============================================================================
def main():
    n_workers        = int(os.environ.get("N_WORKERS", "10"))
    games_per_worker = int(os.environ.get("GAMES_PER_WORKER", "500"))
    nodes            = int(os.environ.get("MATCH_NODES", "10000"))
    elo0             = float(os.environ.get("MATCH_ELO0", "0.0"))
    elo1             = float(os.environ.get("MATCH_ELO1", "10.0"))
    alpha            = 0.05
    beta             = 0.05

    total_games = n_workers * games_per_worker
    print(f"=== 并发 Elo 对弈验证 ===")
    print(f"Workers: {n_workers} × {games_per_worker} 局 = {total_games} 局")
    print(f"Engine A: {EVAL_FILE_A} (trained)")
    print(f"Engine B: {EVAL_FILE_B} (original)")
    print(f"Nodes: {nodes}/局 | SPRT: elo0={elo0}, elo1={elo1}, α=β={alpha}")
    print(f"起始局面: {len(FENS)} 个")
    print()

    # 启动并发对弈
    t0 = time.time()
    tasks = [(i, games_per_worker, FENS, nodes) for i in range(n_workers)]

    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(worker_fn, tasks)

    elapsed = time.time() - t0

    # 汇总结果
    total_wins   = sum(r["wins"]   for r in results)
    total_losses = sum(r["losses"] for r in results)
    total_draws  = sum(r["draws"]  for r in results)
    total = total_wins + total_losses + total_draws

    # 检查错误
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"⚠️ {len(errors)} 个 worker 出错:")
        for e in errors:
            print(f"  Worker {e['worker']}: {e['error']}")

    # 计算 Elo / CI / SPRT
    elo, ci = compute_elo(total_wins, total_losses, total_draws)
    llr = compute_llr(total_wins, total_losses, total_draws, elo0, elo1)
    llr_upper = math.log((1 - beta) / alpha)
    llr_lower = math.log(beta / (1 - alpha))

    if llr >= llr_upper:
        sprt_status = "PASS"
        verdict = f"PASS (H1 accepted: Elo >= {elo1})"
    elif llr <= llr_lower:
        sprt_status = "FAIL"
        verdict = f"FAIL (H0 accepted: Elo <= {elo0})"
    else:
        sprt_status = "CONTINUE"
        verdict = "INCONCLUSIVE (need more games)"

    win_rate = (total_wins + 0.5 * total_draws) / total if total > 0 else 0

    # 输出每个 worker 的结果
    print(f"=== Worker 结果 ===")
    for r in sorted(results, key=lambda x: x["worker"]):
        if "error" in r:
            print(f"  Worker {r['worker']}: ERROR - {r['error']}")
        else:
            w, l, d = r["wins"], r["losses"], r["draws"]
            wr = (w + 0.5 * d) / (w + l + d) * 100 if (w + l + d) > 0 else 0
            print(f"  Worker {r['worker']:2d}: W:{w:3d} L:{l:3d} D:{d:3d} | 胜率:{wr:5.1f}% | {r.get('time',0):.0f}s")
    print()

    # 输出汇总结果
    print(f"========================================================================")
    print(f"=== 并发 Elo 对弈结果 ===")
    print(f"Engine A: {EVAL_FILE_A} (trained)")
    print(f"Engine B: {EVAL_FILE_B} (original)")
    print(f"Workers: {n_workers} | Games: {total} ({n_workers}×{games_per_worker})")
    print(f"Wins: {total_wins}, Losses: {total_losses}, Draws: {total_draws}")
    print(f"Win Rate: {win_rate*100:.2f}%")
    print(f"Elo: {elo:+.1f} ± {ci:.1f} (95% CI: [{elo-ci:.1f}, {elo+ci:.1f}])")
    print(f"LLR: {llr:.4f}  [下界 {llr_lower:.4f}, 上界 {llr_upper:.4f}]")
    print(f"SPRT: {sprt_status}")
    print(f"Verdict: {verdict}")
    print(f"Time: {elapsed:.1f}s  ({elapsed/total:.2f}s/game, {n_workers}x 并发)")
    print(f"========================================================================")

    # 统计显著性判定
    ci_lower = elo - ci
    ci_upper = elo + ci
    if ci_lower > 0:
        print(f"✅ 统计显著: v2 网络强于原始 (Elo > 0, CI 下界 {ci_lower:.1f} > 0)")
    elif ci_upper < 0:
        print(f"❌ 统计显著: v2 网络弱于原始 (Elo < 0, CI 上界 {ci_upper:.1f} < 0)")
    else:
        print(f"⚠️ 统计不显著: CI [{ci_lower:.1f}, {ci_upper:.1f}] 包含 0，需要更多局数")

    # 与 500 局对比
    print(f"\n===GAMES_PER_WORKER={games_per_worker} N_WORKERS={n_workers} | 之前 500 局: Elo=+3.5±30.5")

    # 写入 JSON 结果
    result_json = {
        "engine_a": EVAL_FILE_A,
        "engine_b": EVAL_FILE_B,
        "n_workers": n_workers,
        "games_per_worker": games_per_worker,
        "games_played": total,
        "wins": total_wins,
        "losses": total_losses,
        "draws": total_draws,
        "win_rate": win_rate,
        "elo": elo,
        "elo_ci_half_width": ci,
        "elo_ci_lower": ci_lower,
        "elo_ci_upper": ci_upper,
        "llr": llr,
        "llr_upper": llr_upper,
        "llr_lower": llr_lower,
        "sprt_status": sprt_status,
        "verdict": verdict,
        "nodes_per_game": nodes,
        "elapsed_seconds": elapsed,
        "config": {"elo0": elo0, "elo1": elo1, "alpha": alpha, "beta": beta},
        "worker_results": results,
        "timestamp": datetime.now().isoformat(),
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    print(f"\nJSON 结果已写入: {RESULT_PATH}")


if __name__ == "__main__":
    main()