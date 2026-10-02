#!/usr/bin/env python3
"""并行自对弈训练数据生成

用 Pikafish 自对弈生成带标签的训练数据，用于 Texel Tuning。
每行格式：FEN|result
result 从走棋方（STM）视角：1.0=胜，0.5=和，0.0=负

用法：
    python3 tools/generate_training_data.py [-n 1000] [-o training_data.txt]

在 WSL 中运行：
    wsl bash -c "cd /mnt/e/xiaoxiao/pikayu/Pikafish && python3 tools/generate_training_data.py"
"""
import subprocess
import multiprocessing
import os
import sys
import time
import random
import re
import argparse
import threading

# ── 配置 ──────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
PIKAFISH_PATH = os.path.join(HERE, '..', 'src', 'pikafish')
OUTPUT_FILE = os.path.join(HERE, '..', 'training_data.txt')
N_GAMES = 1000
MOVETIME = 100          # 每步搜索时间 (ms)
SKIP_OPENING = 8        # 跳过前 8 步开局
MAX_PLIES = 300         # 每局最大步数
N_PROCESSES = min(8, os.cpu_count() or 4)
EVAL_THRESHOLD = 5.0    # pawn 单位阈值，对应 500 厘兵
MULTIPV = 3             # 多 PV 数量，用于增加走法多样性
GAME_TIMEOUT = 120      # 单局总超时 (秒)，超时杀引擎进程


class PikafishGame:
    """Pikafish UCI 协议交互封装，持久进程方式。"""

    def __init__(self, pikafish_path):
        self.proc = subprocess.Popen(
            [pikafish_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # UCI 握手
        self._send('uci')
        self._wait_for('uciok')
        # 设置 MultiPV 以获取多个候选走法，增加自对弈多样性
        self._send(f'setoption name MultiPV value {MULTIPV}')
        self._send('isready')
        self._wait_for('readyok')

    def _send(self, cmd):
        """发送一条 UCI 命令。"""
        self.proc.stdin.write(cmd + '\n')
        self.proc.stdin.flush()

    def _readline(self):
        """读取一行，引擎退出时返回空字符串。"""
        line = self.proc.stdout.readline()
        return line

    def _wait_for(self, token):
        """读取输出直到出现包含 token 的行，返回该行。"""
        while True:
            line = self._readline()
            if not line:
                raise RuntimeError('引擎意外退出')
            line = line.strip()
            if token in line:
                return line

    def get_position_info(self):
        """发送 d 命令，解析返回 (fen, checkers)。

        d 命令输出格式：
            <棋盘图多行>
            Fen: <fen>
            Key: <key>
            Checkers: <checkers>
        """
        self._send('d')
        fen = None
        checkers = None
        for _ in range(50):  # d 输出最多约 20 行，50 行足够安全
            line = self._readline()
            if not line:
                raise RuntimeError('引擎意外退出')
            line = line.strip()
            if line.startswith('Fen:'):
                fen = line[4:].strip()
            elif line.startswith('Checkers:'):
                checkers = line[len('Checkers:'):].strip()
                break
        if fen is None:
            raise RuntimeError('d 命令未输出 Fen 行')
        return fen, checkers or ''

    def search_multipv(self, movetime=100):
        """搜索多个候选走法，返回走法列表（按 MultiPV 排序）。

        若引擎返回 bestmove (none) 则返回空列表（游戏结束）。
        """
        self._send(f'go movetime {movetime}')
        pvs = {}
        bestmove = None
        while True:
            line = self._readline()
            if not line:
                raise RuntimeError('引擎意外退出')
            line = line.strip()
            if line.startswith('info') and 'multipv' in line and ' pv ' in line:
                parts = line.split()
                try:
                    mpv = int(parts[parts.index('multipv') + 1])
                    pv_idx = parts.index('pv')
                    if pv_idx + 1 < len(parts):
                        pvs[mpv] = parts[pv_idx + 1]
                except (ValueError, IndexError):
                    pass
            elif line.startswith('bestmove'):
                parts = line.split()
                bestmove = parts[1] if len(parts) > 1 else '(none)'
                break
        # 优先返回 MultiPV 收集到的走法
        candidates = [pvs[i] for i in sorted(pvs.keys())
                      if pvs[i] and pvs[i] != '(none)']
        if not candidates and bestmove and bestmove != '(none)':
            candidates = [bestmove]
        return candidates

    def position(self, moves=None):
        """设置局面为 startpos 加上走法序列。"""
        if moves:
            self._send(f'position startpos moves {" ".join(moves)}')
        else:
            self._send('position startpos')

    def eval_white(self):
        """获取评估值（pawn 单位，白方视角）。

        解析 eval 命令输出中的：
            Final evaluation      +0.29 (white side) [with scaled NNUE, ...]
        超时或解析失败时返回 0.0（视为和棋）。
        """
        self._send('eval')
        for _ in range(100):  # eval 输出约 30 行，100 行足够安全
            line = self._readline()
            if not line:
                return 0.0  # 引擎退出，返回默认值
            line = line.strip()
            if 'Final evaluation' in line and 'white side' in line:
                match = re.search(r'([+-]?\d+\.?\d*)\s*\(white side\)', line)
                if match:
                    return float(match.group(1))
                return 0.0
        return 0.0  # 超过最大行数，返回默认值

    def kill(self):
        """强制杀死引擎进程（用于超时处理）。"""
        try:
            self.proc.kill()
        except Exception:
            pass

    def quit(self):
        """关闭引擎进程。"""
        try:
            self._send('quit')
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def play_one_game(args):
    """玩一局自对弈，返回 [(fen, result), ...]。

    result 从走棋方视角：1.0=胜，0.5=和，0.0=负。
    内部 game_result 统一为白方结果（1.0=白胜, 0.5=和, 0.0=黑胜），
    最后按每个局面的 stm 转换。

    使用 threading.Timer 实现单局超时：超时后杀掉引擎进程，
    readline 返回空字符串触发异常，已收集的数据被保留。
    """
    game_idx, pikafish_path, movetime, skip_opening, max_plies = args
    rng = random.Random(game_idx * 7919 + 1)  # 可复现的随机源

    engine = PikafishGame(pikafish_path)
    positions = []  # [(fen, stm), ...]
    moves = []
    game_result = None  # 白方结果：1.0=白胜, 0.5=和, 0.0=黑胜
    timed_out = False

    # 超时定时器：超时后杀引擎进程，使 readline 立即返回空
    def _on_timeout():
        nonlocal timed_out
        timed_out = True
        engine.kill()

    timer = threading.Timer(GAME_TIMEOUT, _on_timeout)
    timer.daemon = True
    timer.start()

    try:
        for ply in range(max_plies):
            if timed_out:
                break

            engine.position(moves if moves else None)

            # 获取当前局面信息（FEN + Checkers）
            fen, checkers = engine.get_position_info()
            stm = fen.split()[1]  # 'w' or 'b'

            # 记录局面（跳过前 skip_opening 步开局）
            if ply >= skip_opening:
                positions.append((fen, stm))

            # 搜索候选走法
            candidates = engine.search_multipv(movetime)

            if not candidates:
                # 游戏结束：bestmove (none)
                if checkers.strip():
                    # 被将杀 → 当前走棋方负
                    game_result = 0.0 if stm == 'w' else 1.0
                else:
                    # 困毙（无棋可走但未被将军）→ 和棋
                    game_result = 0.5
                break

            # 按概率选择走法，增加自对弈多样性
            # 70% 选最佳，20% 选次佳，10% 选第三
            r = rng.random()
            if r < 0.70 or len(candidates) == 1:
                bestmove = candidates[0]
            elif r < 0.90 and len(candidates) >= 2:
                bestmove = candidates[1]
            elif len(candidates) >= 3:
                bestmove = candidates[2]
            else:
                bestmove = candidates[0]

            moves.append(bestmove)
        else:
            # 达到最大步数，用 eval 判定结果
            if not timed_out:
                engine.position(moves)
                eval_pawn = engine.eval_white()
                if eval_pawn > EVAL_THRESHOLD:
                    game_result = 1.0   # 白方胜
                elif eval_pawn < -EVAL_THRESHOLD:
                    game_result = 0.0   # 黑方胜
                else:
                    game_result = 0.5   # 和棋
    except Exception as e:
        if not timed_out:
            sys.stderr.write(f'[游戏 {game_idx}] 异常: {e}\n')
            sys.stderr.flush()
    finally:
        timer.cancel()
        engine.quit()

    # 超时时，用已收集的局面对局，标记为和棋
    if game_result is None:
        if timed_out and positions:
            game_result = 0.5  # 超时视为和棋
        else:
            return []

    # 将白方结果转换为每个局面的走棋方视角结果
    results = []
    for fen, stm in positions:
        r = game_result if stm == 'w' else 1.0 - game_result
        results.append((fen, r))

    return results


def main():
    parser = argparse.ArgumentParser(description='并行自对弈训练数据生成')
    parser.add_argument('-n', '--games', type=int, default=N_GAMES,
                        help=f'游戏局数（默认 {N_GAMES}）')
    parser.add_argument('-o', '--output', type=str, default=OUTPUT_FILE,
                        help='输出文件路径')
    parser.add_argument('-t', '--movetime', type=int, default=MOVETIME,
                        help=f'每步搜索时间 ms（默认 {MOVETIME}）')
    parser.add_argument('-p', '--processes', type=int, default=N_PROCESSES,
                        help=f'进程数（默认 {N_PROCESSES}）')
    args = parser.parse_args()

    n_games = args.games
    output_file = args.output
    movetime = args.movetime
    n_proc = args.processes

    # 确保输出实时刷新（行缓冲）
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print(f'=== 并行自对弈训练数据生成 ===')
    print(f'局数：{n_games}，进程：{n_proc}，每步：{movetime}ms')
    print(f'跳过开局：{SKIP_OPENING} 步，最大步数：{MAX_PLIES}')
    print(f'单局超时：{GAME_TIMEOUT}s')
    print(f'引擎：{PIKAFISH_PATH}')
    print(f'输出：{output_file}')

    if not os.path.exists(PIKAFISH_PATH):
        print(f'错误：引擎不存在：{PIKAFISH_PATH}')
        sys.exit(1)

    # 准备每局参数
    args_list = [
        (i, PIKAFISH_PATH, movetime, SKIP_OPENING, MAX_PLIES)
        for i in range(n_games)
    ]

    start_time = time.time()

    # 并行执行
    all_results = []
    completed = 0
    game_lengths = []

    with multiprocessing.Pool(n_proc) as pool:
        for result in pool.imap_unordered(play_one_game, args_list):
            all_results.extend(result)
            game_lengths.append(len(result))
            completed += 1
            if completed % 10 == 0 or completed == n_games:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (n_games - completed) / rate if rate > 0 else 0
                print(f'  完成 {completed}/{n_games} 局，'
                      f'{len(all_results)} 局面，'
                      f'用时 {elapsed:.0f}s，ETA {eta:.0f}s')

    elapsed = time.time() - start_time

    # 写入文件
    with open(output_file, 'w') as f:
        for fen, result in all_results:
            f.write(f'{fen}|{result}\n')

    # 统计
    n_total = len(all_results)
    n_win = sum(1 for _, r in all_results if r == 1.0)
    n_draw = sum(1 for _, r in all_results if r == 0.5)
    n_loss = sum(1 for _, r in all_results if r == 0.0)

    print(f'\n=== 完成 ===')
    print(f'总局面数：{n_total}')
    if n_total > 0:
        print(f'胜（1.0）：{n_win} ({n_win / n_total * 100:.1f}%)')
        print(f'和（0.5）：{n_draw} ({n_draw / n_total * 100:.1f}%)')
        print(f'负（0.0）：{n_loss} ({n_loss / n_total * 100:.1f}%)')
    if game_lengths:
        avg_len = sum(game_lengths) / len(game_lengths)
        max_len = max(game_lengths)
        min_len = min(game_lengths)
        print(f'平均游戏长度：{avg_len:.1f} 局面/局')
        print(f'游戏长度范围：{min_len} ~ {max_len} 局面')
    print(f'完成局数：{completed}')
    print(f'总用时：{elapsed:.1f}s ({elapsed / 60:.1f} 分钟)')
    print(f'输出文件：{output_file}')

    # 验收检查
    if n_total >= 50000:
        print(f'\n✓ 验收通过：局面数 {n_total} ≥ 50000')
    else:
        print(f'\n✗ 验收警告：局面数 {n_total} < 50000')


if __name__ == '__main__':
    main()