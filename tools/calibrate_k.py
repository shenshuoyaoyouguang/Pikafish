#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K 标定 + 基线交叉熵计算

对 NNUE 评估值做 Texel 风格的 K 标定：
    p_i = sigmoid(K * v_i)
    L(K) = (1/N) * sum_i [ r_i * softplus(-K*v_i) + (1-r_i) * softplus(K*v_i) ]

通过两阶段网格搜索（粗 + 细）找到使 L(K) 最小的 K*，并输出基线交叉熵 L(θ₀)。

用法（在 WSL 中）：
    cd /mnt/e/xiaoxiao/pikayu/Pikafish
    python3 tools/calibrate_k.py [--data training_data.txt] [--network src/pikafish.nnue] [--output tools/k_calibration_results.json]

输出：
    1. 打印最优 K* 和基线交叉熵 L(θ₀)
    2. 保存 K-L 曲线数据到 tools/k_calibration_results.json
    3. 保存最优 K 到 tools/optimal_k.txt（供后续 validate_gradient.py 使用）
"""
import sys
import os
import json
import time
import argparse
import numpy as np

# 把 src 目录加入 sys.path，使 import nnue_pybind 能找到编译好的 .so
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.normpath(os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _SRC_DIR)

import nnue_pybind  # noqa: E402

# tqdm 可选导入：存在则显示进度条，否则降级为无进度
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ── 默认路径配置 ─────────────────────────────────────────────────────
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))


def _resolve(name):
    """优先在仓库根找，找不到回退到脚本同级。"""
    p = os.path.join(_REPO_ROOT, name)
    return p if os.path.exists(p) else os.path.join(_HERE, name)


DEFAULT_DATA_FILE = _resolve("training_data.txt")
DEFAULT_NETWORK_FILE = _resolve("src/pikafish.nnue")
DEFAULT_OUTPUT_JSON = os.path.join(_HERE, "k_calibration_results.json")
OPTIMAL_K_FILE = os.path.join(_HERE, "optimal_k.txt")

# ── 网格搜索参数 ─────────────────────────────────────────────────────
K_COARSE_LOW = 0.0001
K_COARSE_HIGH = 0.01
K_COARSE_POINTS = 1000
K_FINE_HALF_WIDTH = 0.0005
K_FINE_POINTS = 1000
BATCH_SIZE = 5000  # 批量评估每批大小，避免内存问题


# ── 数值稳定的 softplus ──────────────────────────────────────────────
def softplus(x):
    """数值稳定的 softplus(x) = log(1 + exp(x))。

    - x > 20:  softplus(x) ≈ x
    - x < -20: softplus(x) ≈ exp(x)
    - else:    softplus(x) = log(1 + exp(x))

    输入为 numpy 数组，返回同形状 float 数组。
    """
    # 用 float64 保证精度
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    # 大于阈值：直接取 x（exp 项可忽略）
    mask_big = x > 20.0
    # 小于阈值：取 exp(x)（log(1+eps) ≈ eps）
    mask_small = x < -20.0
    # 中间区间：精确计算
    mask_mid = ~(mask_big | mask_small)

    out[mask_big] = x[mask_big]
    out[mask_small] = np.exp(x[mask_small])
    out[mask_mid] = np.log1p(np.exp(x[mask_mid]))
    return out


def cross_entropy_loss(K, v, r):
    """计算给定 K 下的平均交叉熵 L(K)。

    L_i = r_i * softplus(-K*v_i) + (1 - r_i) * softplus(K*v_i)
    L(K) = mean(L_i)

    参数：
        K : float — 标定常数
        v : np.ndarray[int] — NNUE 内部单位值（STM 视角）
        r : np.ndarray[float] — 真实结果（STM 视角，1.0/0.5/0.0）

    返回：
        float — 平均交叉熵
    """
    kv = K * v.astype(np.float64)
    # L_i = r * softplus(-kv) + (1-r) * softplus(kv)
    loss = r * softplus(-kv) + (1.0 - r) * softplus(kv)
    return float(np.mean(loss))


# ── 数据加载 ─────────────────────────────────────────────────────────
def load_training_data(path):
    """读取训练数据文件，返回 (fens, results)。

    文件格式：每行 `FEN|result`，result ∈ {1.0, 0.5, 0.0}（STM 视角）。

    返回：
        fens : list[str]      — FEN 字符串列表
        results : np.ndarray  — float64 结果数组
    """
    fens = []
    results = []
    n_skip = 0
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" not in line:
                n_skip += 1
                continue
            fen, _, result_str = line.rpartition("|")
            fen = fen.strip()
            try:
                result = float(result_str.strip())
            except ValueError:
                print("警告: 第 %d 行 result 解析失败，跳过: %s" % (lineno, line[:60]))
                n_skip += 1
                continue
            if not fen:
                n_skip += 1
                continue
            fens.append(fen)
            results.append(result)
    if n_skip > 0:
        print("警告: 跳过 %d 行无效数据" % n_skip)
    if not fens:
        raise RuntimeError("训练数据为空: %s" % path)
    results = np.asarray(results, dtype=np.float64)
    return fens, results


# ── 批量评估 ─────────────────────────────────────────────────────────
def batch_evaluate(fens, batch_size=BATCH_SIZE):
    """分批调用 nnue_pybind.evaluate_batch() 评估所有 FEN。

    参数：
        fens : list[str]       — FEN 字符串列表
        batch_size : int       — 每批大小

    返回：
        np.ndarray[int64] — NNUE 内部单位值数组（STM 视角）
    """
    n = len(fens)
    values = np.empty(n, dtype=np.int64)
    n_batches = (n + batch_size - 1) // batch_size

    iterator = range(n_batches)
    if _HAS_TQDM:
        iterator = tqdm(iterator, desc="批量评估", unit="batch")

    print("开始批量评估 %d 个局面（%d 批，每批 %d）..." % (n, n_batches, batch_size))
    t0 = time.time()
    for bi in iterator:
        lo = bi * batch_size
        hi = min(lo + batch_size, n)
        batch = fens[lo:hi]
        batch_vals = nnue_pybind.evaluate_batch(batch)
        values[lo:hi] = batch_vals
    elapsed = time.time() - t0
    print("批量评估完成，耗时 %.2f 秒（%.0f 局面/秒）" % (elapsed, n / max(elapsed, 1e-9)))
    return values


# ── 网格搜索 ─────────────────────────────────────────────────────────
def grid_search(v, r, k_low, k_high, n_points, desc="网格搜索"):
    """在 [k_low, k_high] 上等距取 n_points 个 K，计算 L(K)。

    返回：
        ks : np.ndarray[float] — K 数组
        losses : np.ndarray[float] — 对应的 L(K) 数组
        best_k : float — 最优 K（最小 L）
        best_loss : float — 最小 L
    """
    ks = np.linspace(k_low, k_high, n_points)
    losses = np.empty(n_points, dtype=np.float64)

    iterator = range(n_points)
    if _HAS_TQDM:
        iterator = tqdm(iterator, desc=desc, unit="K")

    for i in iterator:
        losses[i] = cross_entropy_loss(ks[i], v, r)

    best_idx = int(np.argmin(losses))
    return ks, losses, float(ks[best_idx]), float(losses[best_idx])


# ── 结果统计 ─────────────────────────────────────────────────────────
def count_results(r):
    """统计胜/和/负数量（基于 STM 视角 result）。"""
    n_win = int(np.sum(r == 1.0))
    n_draw = int(np.sum(r == 0.5))
    n_loss = int(np.sum(r == 0.0))
    n_other = int(len(r) - n_win - n_draw - n_loss)
    return n_win, n_draw, n_loss, n_other


# ── 主流程 ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="K 标定 + 基线交叉熵计算（Texel Tuning 第一步）"
    )
    parser.add_argument(
        "--data", default=DEFAULT_DATA_FILE,
        help="训练数据文件路径（默认 %(default)s）"
    )
    parser.add_argument(
        "--network", default=DEFAULT_NETWORK_FILE,
        help="NNUE 网络文件路径（默认 %(default)s）"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_JSON,
        help="K-L 曲线 JSON 输出路径（默认 %(default)s）"
    )
    args = parser.parse_args()

    # 1. 加载网络
    print("=" * 70)
    print("K 标定 + 基线交叉熵计算")
    print("=" * 70)
    print("网络文件: %s" % args.network)
    print("训练数据: %s" % args.data)
    print("输出 JSON: %s" % args.output)
    print("最优 K 文件: %s" % OPTIMAL_K_FILE)
    print("-" * 70)

    if not os.path.exists(args.network):
        print("错误: 网络文件不存在: %s" % args.network)
        sys.exit(1)
    if not os.path.exists(args.data):
        print("错误: 训练数据文件不存在: %s" % args.data)
        sys.exit(1)

    print("加载 NNUE 网络...")
    nnue_pybind.load(args.network)
    print("网络加载完成。")

    # 2. 加载训练数据
    print("-" * 70)
    fens, results = load_training_data(args.data)
    n = len(fens)
    n_win, n_draw, n_loss, n_other = count_results(results)
    print("训练数据加载完成: %d 个局面" % n)
    print("  胜（result=1.0）: %d  (%.2f%%)" % (n_win, 100.0 * n_win / n))
    print("  和（result=0.5）: %d  (%.2f%%)" % (n_draw, 100.0 * n_draw / n))
    print("  负（result=0.0）: %d  (%.2f%%)" % (n_loss, 100.0 * n_loss / n))
    if n_other > 0:
        print("  其他: %d" % n_other)

    # 3. 批量评估所有 FEN
    print("-" * 70)
    v = batch_evaluate(fens, batch_size=BATCH_SIZE)

    # 输出 NNUE 评估值分布
    v_min, v_max = int(v.min()), int(v.max())
    v_mean, v_std = float(np.mean(v)), float(np.std(v))
    print("NNUE 评估值统计: min=%d, max=%d, mean=%.2f, std=%.2f"
          % (v_min, v_max, v_mean, v_std))

    # 4. 粗搜索
    print("-" * 70)
    print("粗搜索: K ∈ [%.6f, %.6f], %d 个点" % (K_COARSE_LOW, K_COARSE_HIGH, K_COARSE_POINTS))
    ks_c, losses_c, best_k_c, best_loss_c = grid_search(
        v, results, K_COARSE_LOW, K_COARSE_HIGH, K_COARSE_POINTS, desc="粗搜索"
    )
    print("粗搜索最优: K* = %.6f, L(K*) = %.6f" % (best_k_c, best_loss_c))

    # 5. 细搜索
    fine_low = max(K_COARSE_LOW, best_k_c - K_FINE_HALF_WIDTH)
    fine_high = min(K_COARSE_HIGH, best_k_c + K_FINE_HALF_WIDTH)
    print("-" * 70)
    print("细搜索: K ∈ [%.6f, %.6f], %d 个点" % (fine_low, fine_high, K_FINE_POINTS))
    ks_f, losses_f, best_k_f, best_loss_f = grid_search(
        v, results, fine_low, fine_high, K_FINE_POINTS, desc="细搜索"
    )
    print("细搜索最优: K* = %.6f, L(K*) = %.6f" % (best_k_f, best_loss_f))

    # 6. 输出结果
    print("=" * 70)
    print("最终结果")
    print("=" * 70)
    print("最优 K* = %.6f" % best_k_f)
    print("基线交叉熵 L(θ₀) = %.6f" % best_loss_f)
    print("  （参考: 随机猜测 ln(2) = %.6f）" % float(np.log(2.0)))
    print("训练数据统计: 胜=%d, 和=%d, 负=%d (共 %d)" % (n_win, n_draw, n_loss, n))

    # 7. 保存 K-L 曲线 JSON
    result_data = {
        "optimal_k": best_k_f,
        "baseline_cross_entropy": best_loss_f,
        "random_guess_ce": float(np.log(2.0)),
        "data_file": os.path.abspath(args.data),
        "network_file": os.path.abspath(args.network),
        "n_samples": n,
        "result_counts": {
            "win": n_win,
            "draw": n_draw,
            "loss": n_loss,
            "other": n_other,
        },
        "nnue_value_stats": {
            "min": v_min,
            "max": v_max,
            "mean": v_mean,
            "std": v_std,
        },
        "coarse_search": {
            "k_range": [K_COARSE_LOW, K_COARSE_HIGH],
            "n_points": K_COARSE_POINTS,
            "best_k": best_k_c,
            "best_loss": best_loss_c,
            "ks": [float(x) for x in ks_c],
            "losses": [float(x) for x in losses_c],
        },
        "fine_search": {
            "k_range": [fine_low, fine_high],
            "n_points": K_FINE_POINTS,
            "best_k": best_k_f,
            "best_loss": best_loss_f,
            "ks": [float(x) for x in ks_f],
            "losses": [float(x) for x in losses_f],
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print("K-L 曲线数据已保存: %s" % args.output)

    # 8. 保存最优 K 到文件（供后续脚本使用）
    with open(OPTIMAL_K_FILE, "w", encoding="utf-8") as f:
        f.write("%.10f\n" % best_k_f)
    print("最优 K 已保存: %s" % OPTIMAL_K_FILE)

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())