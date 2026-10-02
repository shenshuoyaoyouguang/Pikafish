#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K 级梯度验证

用法（在 WSL 中）：
    cd /mnt/e/xiaoxiao/pikayu/Pikafish
    python3 tools/validate_gradient.py [--data training_data.txt] [--network src/pikafish.nnue] [--k-file tools/optimal_k.txt]

验证内容：
    1. 解析梯度 vs 有限差分梯度（相对误差 < 1e-4）
    2. 单步梯度下降：L(K_new) < L(K)
    3. 多步梯度下降（10步）：L 单调下降
    4. 学习率敏感性测试：不同 α 下的下降率

数学模型
--------
交叉熵目标函数（K 为唯一可调标量）：
    L(K) = (1/N) Σ_i [ r_i · softplus(-K·v_i) + (1-r_i) · softplus(K·v_i) ]

解析梯度：
    dL/dK = (1/N) Σ_i v_i · (σ(K·v_i) - r_i)

解析 Hessian（二阶导数，用于自适应学习率）：
    d²L/dK² = (1/N) Σ_i v_i² · σ(K·v_i) · (1 - σ(K·v_i))

有限差分梯度：
    二点中心差分：dL/dK ≈ (L(K+ε) - L(K-ε)) / (2ε)        误差 O(ε²)
    五点差分    ：dL/dK ≈ [-f(K+2ε)+8f(K+ε)-8f(K-ε)+f(K-2ε)] / (12ε)  误差 O(ε⁴)

注：由于 NNUE 评估值 v 量级约 1700，三阶导数 d³L/dK³ 量级约 1e8，
    二点差分在 ε=1e-6 下截断误差约 5e-3（相对），不满足 1e-4 阈值。
    五点差分误差 O(ε⁴)，在 ε=1e-6 下截断误差约 1e-9（相对），满足要求。
    因此验证1 主判据采用五点差分，同时报告二点差分结果作为对比。

学习率选择：
    L(K) 在 K* 附近近似二次型，Hessian H = d²L/dK² ≈ 1.56e5。
    梯度下降收敛条件：α < 2/H ≈ 1.29e-5。
    最优学习率：α* = 1/H ≈ 6.4e-6。
    题目建议的 α=0.01 远超稳定上界，会过冲；脚本自动用 α_rec=1/H 做验证2/3，
    并在验证4 中测试对数空间学习率范围，展示有效区间。
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
DEFAULT_K_FILE = os.path.join(_HERE, "optimal_k.txt")
DEFAULT_TRACE_FILE = os.path.join(_HERE, "gradient_descent_trace.json")

# ── 默认超参数 ───────────────────────────────────────────────────────
BATCH_SIZE = 5000          # 批量评估每批大小
DEFAULT_EPSILON = 1e-6     # 有限差分步长
DEFAULT_LEARNING_RATE = 0.01  # 默认学习率 α（用户接口；实际验证用自适应 1/H）
GRAD_REL_TOL = 1e-4        # 解析梯度 vs 有限差分梯度的相对误差阈值
N_MULTISTEP = 10           # 多步梯度下降步数
# 学习率敏感性测试：对数空间从 1e-7 到 1e-1，覆盖有效与过冲区间
LEARNING_RATES = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


# ── 数值稳定的 softplus / sigmoid ────────────────────────────────────
def softplus(x):
    """数值稳定的 softplus(x) = log(1 + exp(x))。

    - x > 20:  softplus(x) ≈ x
    - x < -20: softplus(x) ≈ exp(x)
    - else:    softplus(x) = log(1 + exp(x))

    输入为 numpy 数组，返回同形状 float64 数组。
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask_big = x > 20.0
    mask_small = x < -20.0
    mask_mid = ~(mask_big | mask_small)

    out[mask_big] = x[mask_big]
    out[mask_small] = np.exp(x[mask_small])
    out[mask_mid] = np.log1p(np.exp(x[mask_mid]))
    return out


def sigmoid(x):
    """数值稳定的 sigmoid σ(x) = 1/(1+exp(-x))。

    - x > 20:  σ(x) ≈ 1.0
    - x < -20: σ(x) ≈ exp(x)
    - else:    σ(x) = 1.0/(1.0+exp(-x))

    输入为 numpy 数组，返回同形状 float64 数组。
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask_big = x > 20.0
    mask_small = x < -20.0
    mask_mid = ~(mask_big | mask_small)

    out[mask_big] = 1.0
    out[mask_small] = np.exp(x[mask_small])
    out[mask_mid] = 1.0 / (1.0 + np.exp(-x[mask_mid]))
    return out


# ── 损失函数、梯度与 Hessian ─────────────────────────────────────────
def cross_entropy_loss(K, v, r):
    """计算给定 K 下的平均交叉熵 L(K)。

    L_i = r_i * softplus(-K*v_i) + (1 - r_i) * softplus(K*v_i)
    L(K) = mean(L_i)

    参数：
        K : float — 标定常数
        v : np.ndarray — NNUE 内部单位值（STM 视角，int 或 float）
        r : np.ndarray[float64] — 真实结果（STM 视角，1.0/0.5/0.0）

    返回：
        float — 平均交叉熵
    """
    kv = K * v.astype(np.float64)
    loss = r * softplus(-kv) + (1.0 - r) * softplus(kv)
    return float(np.mean(loss))


def analytic_gradient(K, v, r):
    """计算解析梯度 dL/dK。

    dL/dK = (1/N) Σ_i v_i · (σ(K·v_i) - r_i)

    返回：
        float — dL/dK
    """
    kv = K * v.astype(np.float64)
    sig = sigmoid(kv)
    grad = v.astype(np.float64) * (sig - r)
    return float(np.mean(grad))


def analytic_hessian(K, v, r):
    """计算解析 Hessian d²L/dK²（用于自适应学习率）。

    d²L/dK² = (1/N) Σ_i v_i² · σ(K·v_i) · (1 - σ(K·v_i))

    推导：d/dK [v_i·(σ(Kv_i) - r_i)] = v_i² · σ'(Kv_i) = v_i² · σ(Kv_i)·(1-σ(Kv_i))

    返回：
        float — d²L/dK²（恒 > 0，因 σ(1-σ) > 0）
    """
    kv = K * v.astype(np.float64)
    sig = sigmoid(kv)
    vf = v.astype(np.float64)
    hess = vf * vf * sig * (1.0 - sig)
    return float(np.mean(hess))


def finite_diff_gradient(K, v, r, epsilon=DEFAULT_EPSILON):
    """二点中心差分梯度：(L(K+ε) - L(K-ε)) / (2ε)。误差 O(ε²)。

    对于本问题 v 量级 ~1700，三阶导数大，ε=1e-6 截断误差约 5e-3（相对）。
    """
    L_plus = cross_entropy_loss(K + epsilon, v, r)
    L_minus = cross_entropy_loss(K - epsilon, v, r)
    return (L_plus - L_minus) / (2.0 * epsilon)


def finite_diff_gradient_5point(K, v, r, epsilon=DEFAULT_EPSILON):
    """五点差分梯度（Richardson 外推）：误差 O(ε⁴)。

    f'(x) ≈ [-f(x+2ε) + 8f(x+ε) - 8f(x-ε) + f(x-2ε)] / (12ε)

    在 ε=1e-6 下截断误差约 1e-9（相对），远优于二点差分。
    """
    L_p2 = cross_entropy_loss(K + 2 * epsilon, v, r)
    L_p1 = cross_entropy_loss(K + epsilon, v, r)
    L_m1 = cross_entropy_loss(K - epsilon, v, r)
    L_m2 = cross_entropy_loss(K - 2 * epsilon, v, r)
    return (-L_p2 + 8.0 * L_p1 - 8.0 * L_m1 + L_m2) / (12.0 * epsilon)


# ── 数据加载 ─────────────────────────────────────────────────────────
def load_training_data(path):
    """读取训练数据文件，返回 (fens, results)。

    文件格式：每行 `FEN|result`，result ∈ {1.0, 0.5, 0.0}（STM 视角）。
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


def load_optimal_k(path):
    """从文件读取最优 K* 值。"""
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
    return float(line)


# ── 批量评估 ─────────────────────────────────────────────────────────
def batch_evaluate(fens, batch_size=BATCH_SIZE):
    """分批调用 nnue_pybind.evaluate_batch() 评估所有 FEN。

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


# ── 梯度下降工具 ─────────────────────────────────────────────────────
def gradient_descent(K0, v, r, alpha, n_steps):
    """执行 n_steps 步梯度下降：K_{t+1} = K_t - α · dL/dK(K_t)。

    参数：
        K0 : float — 初始 K
        v, r : np.ndarray — 评估值与结果
        alpha : float — 学习率
        n_steps : int — 步数

    返回：
        list[dict] — 每步记录 {step, K, loss, grad}
    """
    trace = []
    K = float(K0)
    for step in range(n_steps + 1):  # 包含初始点 step=0
        loss = cross_entropy_loss(K, v, r)
        grad = analytic_gradient(K, v, r)
        trace.append({
            "step": step,
            "K": K,
            "loss": loss,
            "grad": grad,
        })
        if step == n_steps:
            break
        K = K - alpha * grad
    return trace


# ── 验证项 ───────────────────────────────────────────────────────────
def verify_gradient_vs_finite_diff(K, v, r, epsilon):
    """验证1：解析梯度 vs 有限差分梯度。

    主判据用五点差分（O(ε⁴) 误差），同时报告二点差分（O(ε²) 误差）对比。
    """
    print("\n[验证1] 解析梯度 vs 有限差分梯度")
    print("  K = %.10f, ε = %.2e" % (K, epsilon))

    g_analytic = analytic_gradient(K, v, r)
    g_finite_2 = finite_diff_gradient(K, v, r, epsilon)
    g_finite_5 = finite_diff_gradient_5point(K, v, r, epsilon)

    rel_err_2 = abs(g_analytic - g_finite_2) / max(abs(g_analytic), 1e-10)
    rel_err_5 = abs(g_analytic - g_finite_5) / max(abs(g_analytic), 1e-10)

    print("  解析梯度        dL/dK = %.10e" % g_analytic)
    print("  二点差分 (O(ε²)) dL/dK = %.10e  相对误差 = %.3e" % (g_finite_2, rel_err_2))
    print("  五点差分 (O(ε⁴)) dL/dK = %.10e  相对误差 = %.3e" % (g_finite_5, rel_err_5))
    print("  阈值 = %.6e" % GRAD_REL_TOL)
    print("  说明: v 量级 ~1700, 三阶导数大, 二点差分 ε=1e-6 截断误差偏大;")
    print("        五点差分误差 O(ε⁴), 精度远优于二点, 作为主判据。")

    passed = rel_err_5 < GRAD_REL_TOL
    status = "PASS ✓" if passed else "FAIL ✗"
    print("  结果: %s  (五点差分相对误差 %.3e < %.0e)" % (status, rel_err_5, GRAD_REL_TOL))

    return {
        "name": "analytic_vs_finite_diff",
        "passed": passed,
        "K": K,
        "epsilon": epsilon,
        "analytic_grad": g_analytic,
        "finite_diff_2point": g_finite_2,
        "finite_diff_5point": g_finite_5,
        "rel_error_2point": rel_err_2,
        "rel_error_5point": rel_err_5,
        "threshold": GRAD_REL_TOL,
    }


def verify_single_step_descent(K, v, r, alpha_user, alpha_rec):
    """验证2：单步梯度下降 L(K_new) < L(K)。

    用自适应学习率 α_rec = 1/H 做主判据（保证不过冲），
    同时报告用户指定 α_user 的结果作为对比。
    """
    print("\n[验证2] 单步梯度下降")
    print("  K = %.10f" % K)
    print("  自适应学习率 α_rec = 1/H = %.6e" % alpha_rec)
    print("  用户学习率   α_user = %.6e" % alpha_user)

    L_before = cross_entropy_loss(K, v, r)
    grad = analytic_gradient(K, v, r)

    # 自适应学习率单步
    K_new_rec = K - alpha_rec * grad
    L_after_rec = cross_entropy_loss(K_new_rec, v, r)
    delta_L_rec = L_before - L_after_rec

    # 用户学习率单步（对比）
    K_new_user = K - alpha_user * grad
    L_after_user = cross_entropy_loss(K_new_user, v, r)
    delta_L_user = L_before - L_after_user

    print("  L(K)           = %.12f" % L_before)
    print("  dL/dK          = %.10e" % grad)
    print("  ── 自适应 α_rec ──")
    print("  K_new          = %.12f" % K_new_rec)
    print("  L(K_new)       = %.12f" % L_after_rec)
    print("  ΔL = L(K)-L(K_new) = %.6e" % delta_L_rec)
    print("  ── 用户 α_user (对比) ──")
    print("  K_new          = %.12f" % K_new_user)
    print("  L(K_new)       = %.12f" % L_after_user)
    print("  ΔL = L(K)-L(K_new) = %.6e  %s" % (delta_L_user, "(过冲)" if delta_L_user < 0 else ""))

    passed = L_after_rec < L_before
    status = "PASS ✓" if passed else "FAIL ✗"
    print("  结果: %s  (自适应 α_rec 下 L(K_new) < L(K))" % status)

    return {
        "name": "single_step_descent",
        "passed": passed,
        "K": K,
        "alpha_rec": alpha_rec,
        "alpha_user": alpha_user,
        "grad": grad,
        "K_new_rec": K_new_rec,
        "K_new_user": K_new_user,
        "loss_before": L_before,
        "loss_after_rec": L_after_rec,
        "loss_after_user": L_after_user,
        "delta_L_rec": delta_L_rec,
        "delta_L_user": delta_L_user,
    }


def verify_multistep_descent(K0, v, r, alpha, n_steps):
    """验证3：多步梯度下降，L 单调下降。"""
    print("\n[验证3] 多步梯度下降（%d 步）" % n_steps)
    print("  K0 = %.10f, α = %.6e (自适应)" % (K0, alpha))

    trace = gradient_descent(K0, v, r, alpha, n_steps)

    print("  %-6s %-24s %-16s %-14s" % ("step", "K", "L(K)", "dL/dK"))
    print("  " + "-" * 62)
    for rec in trace:
        print("  %-6d %-24.14f %.12f  %.6e" % (rec["step"], rec["K"], rec["loss"], rec["grad"]))

    # 检查单调下降：允许数值噪声导致的微小上升（< 1e-15）
    losses = [rec["loss"] for rec in trace]
    n_violations = 0
    max_rise = 0.0
    for i in range(1, len(losses)):
        rise = losses[i] - losses[i - 1]
        if rise > 1e-15:
            n_violations += 1
            max_rise = max(max_rise, rise)

    total_drop = losses[0] - losses[-1]
    print("  总下降量 L(K0) - L(K_final) = %.6e" % total_drop)
    if n_violations == 0:
        print("  单调性: 完全单调下降")
        passed = True
    else:
        print("  单调性: 有 %d 步上升（最大上升 %.6e）" % (n_violations, max_rise))
        passed = max_rise < 1e-12

    status = "PASS ✓" if passed else "FAIL ✗"
    print("  结果: %s  (L 单调下降)" % status)

    return {
        "name": "multistep_descent",
        "passed": passed,
        "K0": K0,
        "alpha": alpha,
        "n_steps": n_steps,
        "trace": trace,
        "total_drop": total_drop,
        "n_violations": n_violations,
        "max_rise": max_rise,
    }


def verify_learning_rate_sensitivity(K0, v, r, n_steps, alpha_rec):
    """验证4：学习率敏感性测试。

    测试对数空间 α ∈ {1e-7, ..., 1e-1}，覆盖有效与过冲区间。
    """
    print("\n[验证4] 学习率敏感性测试")
    print("  K0 = %.10f, 测试 α ∈ %s, 各 %d 步" % (K0, LEARNING_RATES, n_steps))
    print("  自适应推荐 α_rec = 1/H = %.6e" % alpha_rec)

    L0 = cross_entropy_loss(K0, v, r)
    print("  基准 L(K0) = %.12f" % L0)
    print("  %-10s %-18s %-16s %-12s %s" % ("alpha", "L_final", "下降量", "下降率(%)", "状态"))
    print("  " + "-" * 72)

    results = []
    best_alpha = None
    best_drop = -np.inf

    for alpha in LEARNING_RATES:
        trace = gradient_descent(K0, v, r, alpha, n_steps)
        L_final = trace[-1]["loss"]
        drop = L0 - L_final
        drop_pct = 100.0 * drop / L0 if L0 > 0 else 0.0
        # 判定该学习率是否有效（最终 L < 初始 L，即下降量 > 0）
        is_effective = drop > 0
        tag = "有效" if is_effective else "过冲"
        marker = " ←推荐" if abs(alpha - alpha_rec) / alpha_rec < 0.5 else ""
        print("  %-10.0e %-18.12f %-16.6e %-12.4f %s%s"
              % (alpha, L_final, drop, drop_pct, tag, marker))
        results.append({
            "alpha": alpha,
            "L_final": L_final,
            "drop": drop,
            "drop_pct": drop_pct,
            "effective": is_effective,
            "trace": trace,
        })
        if drop > best_drop:
            best_drop = drop
            best_alpha = alpha

    n_effective = sum(1 for x in results if x["effective"])
    print("  有效学习率数量: %d / %d" % (n_effective, len(results)))
    print("  最佳学习率: α = %.0e（下降量 %.6e, 下降率 %.4f%%）"
          % (best_alpha, best_drop, 100.0 * best_drop / L0))
    print("  自适应推荐: α_rec = %.6e（理论最优 1/H）" % alpha_rec)

    # PASS 条件：至少有一个学习率能下降（drop > 0）
    passed = best_drop > 0
    status = "PASS ✓" if passed else "FAIL ✗"
    print("  结果: %s  (存在有效学习率使 L 下降)" % status)

    return {
        "name": "learning_rate_sensitivity",
        "passed": passed,
        "K0": K0,
        "L0": L0,
        "n_steps": n_steps,
        "alpha_rec": alpha_rec,
        "rates": results,
        "best_alpha": best_alpha,
        "best_drop": best_drop,
        "n_effective": n_effective,
    }


# ── 主流程 ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="K 级梯度验证（Texel Tuning 梯度下降有效性验证）"
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
        "--k-file", default=DEFAULT_K_FILE,
        help="最优 K 文件路径（默认 %(default)s）"
    )
    parser.add_argument(
        "--epsilon", type=float, default=DEFAULT_EPSILON,
        help="有限差分步长（默认 %(default).2e）"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
        help="用户学习率 α，用于对比（默认 %(default)s；验证实际用自适应 1/H）"
    )
    parser.add_argument(
        "--output", default=DEFAULT_TRACE_FILE,
        help="梯度下降轨迹 JSON 输出路径（默认 %(default)s）"
    )
    args = parser.parse_args()

    # ── 头部信息 ──
    print("=" * 72)
    print("K 级梯度验证")
    print("=" * 72)
    print("网络文件   : %s" % args.network)
    print("训练数据   : %s" % args.data)
    print("最优 K 文件: %s" % args.k_file)
    print("轨迹输出   : %s" % args.output)
    print("超参数     : ε = %.2e, 用户 α = %.4f, 相对误差阈值 = %.2e"
          % (args.epsilon, args.learning_rate, GRAD_REL_TOL))
    print("-" * 72)

    # ── 文件存在性检查 ──
    if not os.path.exists(args.network):
        print("错误: 网络文件不存在: %s" % args.network)
        sys.exit(1)
    if not os.path.exists(args.data):
        print("错误: 训练数据文件不存在: %s" % args.data)
        sys.exit(1)
    if not os.path.exists(args.k_file):
        print("错误: 最优 K 文件不存在: %s" % args.k_file)
        sys.exit(1)

    # ── 1. 加载网络 ──
    print("加载 NNUE 网络...")
    nnue_pybind.load(args.network)
    print("网络加载完成。")

    # ── 2. 加载最优 K ──
    K_star = load_optimal_k(args.k_file)
    print("最优 K* = %.10f" % K_star)

    # ── 3. 加载训练数据 ──
    print("-" * 72)
    fens, results = load_training_data(args.data)
    n = len(fens)
    print("训练数据加载完成: %d 个局面" % n)

    # ── 4. 批量评估所有 FEN ──
    print("-" * 72)
    v = batch_evaluate(fens, batch_size=BATCH_SIZE)

    v_min, v_max = int(v.min()), int(v.max())
    v_mean, v_std = float(np.mean(v)), float(np.std(v))
    print("NNUE 评估值统计: min=%d, max=%d, mean=%.2f, std=%.2f"
          % (v_min, v_max, v_mean, v_std))

    # ── 5. 基线交叉熵与曲率分析 ──
    print("-" * 72)
    L_star = cross_entropy_loss(K_star, v, results)
    g_star = analytic_gradient(K_star, v, results)
    H_star = analytic_hessian(K_star, v, results)
    alpha_rec = 1.0 / H_star  # 自适应学习率（二次型最优）
    alpha_stable = 2.0 / H_star  # 稳定上界
    K_true_opt = K_star - g_star / H_star  # 牛顿一步估计真正最优

    print("基线交叉熵 L(K*)       = %.12f" % L_star)
    print("  （参考: 随机猜测 ln(2) = %.12f）" % float(np.log(2.0)))
    print("残余梯度 g(K*) = dL/dK  = %.10e" % g_star)
    print("Hessian  H(K*) = d²L/dK² = %.6e" % H_star)
    print("自适应学习率 α_rec = 1/H = %.6e" % alpha_rec)
    print("稳定上界     2/H       = %.6e" % alpha_stable)
    print("牛顿一步估计真正最优 K  ≈ %.12f" % K_true_opt)
    print("  （K* 偏移 g/H = %.3e，网格搜索残余误差）" % (g_star / H_star))
    if args.learning_rate > alpha_stable:
        print("  ⚠ 用户 α=%.4e 超过稳定上界 2/H=%.4e，会过冲；验证2/3 改用 α_rec。"
              % (args.learning_rate, alpha_stable))

    # ── 6. 验证 1：解析梯度 vs 有限差分 ──
    print("=" * 72)
    print("开始梯度验证")
    print("=" * 72)
    r1 = verify_gradient_vs_finite_diff(K_star, v, results, args.epsilon)

    # ── 7. 验证 2：单步梯度下降（自适应学习率）──
    r2 = verify_single_step_descent(K_star, v, results, args.learning_rate, alpha_rec)

    # ── 8. 验证 3：多步梯度下降（自适应学习率）──
    r3 = verify_multistep_descent(K_star, v, results, alpha_rec, N_MULTISTEP)

    # ── 9. 验证 4：学习率敏感性 ──
    r4 = verify_learning_rate_sensitivity(K_star, v, results, N_MULTISTEP, alpha_rec)

    # ── 10. 汇总 ──
    print("\n" + "=" * 72)
    print("验证汇总")
    print("=" * 72)
    all_results = [r1, r2, r3, r4]
    all_passed = True
    for r in all_results:
        status = "PASS ✓" if r["passed"] else "FAIL ✗"
        print("  [%s] %s" % (status, r["name"]))
        if not r["passed"]:
            all_passed = False

    print("-" * 72)
    overall = "PASS ✓" if all_passed else "FAIL ✗"
    print("总体结果: %s  (%d/%d 通过)"
          % (overall, sum(1 for r in all_results if r["passed"]), len(all_results)))

    # ── 11. 保存轨迹 JSON ──
    trace_data = {
        "optimal_k": K_star,
        "baseline_cross_entropy": L_star,
        "residual_gradient": g_star,
        "hessian": H_star,
        "alpha_recommended": alpha_rec,
        "alpha_stable_upper": alpha_stable,
        "k_newton_estimate": K_true_opt,
        "random_guess_ce": float(np.log(2.0)),
        "epsilon": args.epsilon,
        "learning_rate_user": args.learning_rate,
        "n_samples": n,
        "nnue_value_stats": {
            "min": v_min,
            "max": v_max,
            "mean": v_mean,
            "std": v_std,
        },
        "verifications": {
            "gradient_vs_finite_diff": {
                "passed": r1["passed"],
                "analytic_grad": r1["analytic_grad"],
                "finite_diff_2point": r1["finite_diff_2point"],
                "finite_diff_5point": r1["finite_diff_5point"],
                "rel_error_2point": r1["rel_error_2point"],
                "rel_error_5point": r1["rel_error_5point"],
                "threshold": r1["threshold"],
            },
            "single_step_descent": {
                "passed": r2["passed"],
                "K": r2["K"],
                "alpha_rec": r2["alpha_rec"],
                "alpha_user": r2["alpha_user"],
                "grad": r2["grad"],
                "K_new_rec": r2["K_new_rec"],
                "K_new_user": r2["K_new_user"],
                "loss_before": r2["loss_before"],
                "loss_after_rec": r2["loss_after_rec"],
                "loss_after_user": r2["loss_after_user"],
                "delta_L_rec": r2["delta_L_rec"],
                "delta_L_user": r2["delta_L_user"],
            },
            "multistep_descent": {
                "passed": r3["passed"],
                "alpha": r3["alpha"],
                "n_steps": r3["n_steps"],
                "total_drop": r3["total_drop"],
                "n_violations": r3["n_violations"],
                "max_rise": r3["max_rise"],
                "trace": r3["trace"],
            },
            "learning_rate_sensitivity": {
                "passed": r4["passed"],
                "L0": r4["L0"],
                "alpha_rec": r4["alpha_rec"],
                "best_alpha": r4["best_alpha"],
                "best_drop": r4["best_drop"],
                "n_effective": r4["n_effective"],
                "rates": [
                    {
                        "alpha": x["alpha"],
                        "L_final": x["L_final"],
                        "drop": x["drop"],
                        "drop_pct": x["drop_pct"],
                        "effective": x["effective"],
                        "trace": x["trace"],
                    } for x in r4["rates"]
                ],
            },
        },
        "overall_passed": all_passed,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, indent=2, ensure_ascii=False)
    print("梯度下降轨迹已保存: %s" % args.output)

    print("=" * 72)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())