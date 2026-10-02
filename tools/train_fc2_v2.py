#!/usr/bin/env python3
"""
fc_2 输出层 Texel Tuning 训练 v2 — 改进版

用法（在 WSL 中）：
    cd /mnt/e/xiaoxiao/pikayu/Pikafish
    python3 tools/train_fc2_v2.py \
        --data training_data.txt \
        --network src/pikafish.nnue \
        --lr 1e-3 \
        --weight-decay 1e-4 \
        --epochs 100 \
        --batch-size 512 \
        --eval-interval 10 \
        --output src/pikafish_trained_v2.nnue

相对 v1 (tools/train_fc2.py) 的改进：
    1. L2 weight decay 正则化 (默认 λ=1e-4) — 抑制过拟合，限制权重变化幅度
    2. 降低默认学习率到 1e-3 (v1 默认 1e-3 但实际 P4 用 1e-2 导致权重变化过大)
    3. Cosine annealing 学习率调度 — 平滑衰减，末期细调
    4. 评估尺度监控 — 训练前后对比 evaluate 值分布，检测尺度偏移
    5. 训练中 Elo 快速验证 — 每 N epoch 跑快速对弈，Elo < 阈值则早停
    6. 完整命令行参数支持

数学模型（与 v1 一致）：
    v = psqt/OutputScale + (fc_2(concat) + skip_0) * effective_scale
    effective_scale = 9600 / (16384 * 16) = 0.03662109375

    交叉熵损失：
        L = (1/N) Σ [r_i * softplus(-K*v_i) + (1-r_i) * softplus(K*v_i)]

    L2 正则化（仅作用于权重 W，不作用于偏置 B）：
        L_total = L + (λ/2) * Σ W[b,j]^2
        ∂L_total/∂W[b,j] = ∂L/∂W[b,j] + λ * W[b,j]

    Cosine annealing 学习率：
        lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# 常量（与 v1 一致）
# ---------------------------------------------------------------------------
OUTPUT_SCALE = 16
HIDDEN_ONE_VAL = 128
WEIGHT_SCALE_BITS = 6
MULTIPLIER = 600 * OUTPUT_SCALE              # 9600
DENOMINATOR = HIDDEN_ONE_VAL * (1 << WEIGHT_SCALE_BITS) * 2  # 16384
EFFECTIVE_SCALE = MULTIPLIER / (DENOMINATOR * OUTPUT_SCALE)  # 0.03662109375

NUM_BUCKETS = 16
FC2_INPUTS = 128   # fc_2 输入维度（concat_buffer 大小）
NUM_WEIGHTS = NUM_BUCKETS * FC2_INPUTS + NUM_BUCKETS  # 2064

# Adam 默认参数
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

# 早停耐心（验证集 loss）
EARLY_STOP_PATIENCE = 10

# 尺度监控阈值
SCALE_MEAN_SHIFT_WARN = 50.0     # |mean_shift| > 50 发出警告
SCALE_STD_RATIO_WARN = 0.1       # |std_ratio - 1| > 0.1 发出警告
SCALE_MONITOR_SAMPLE = 500       # 尺度监控采样局面数

# Elo 快速验证默认参数
ELO_EVAL_GAMES = 20
ELO_EVAL_NODES = 5000
ELO_EARLY_STOP_THRESHOLD = -50.0  # Elo < -50 早停训练


# ---------------------------------------------------------------------------
# Scramble 置换（与 C++ AffineTransform::get_weight_index_scrambled 一致）
# ---------------------------------------------------------------------------
def build_scramble_perm(input_dims=FC2_INPUTS, output_dims=1):
    """构建 scramble 置换数组 perm[i] = get_weight_index_scrambled(i)。"""
    padded_in = input_dims  # 128 已是 32 的倍数
    perm = np.empty(input_dims, dtype=np.int64)
    for i in range(input_dims):
        input_index = i % padded_in
        block = input_index // 32
        chunk = (input_index % 32) // 4
        input_index = block * 32 + ((chunk % 2) * 4 + chunk // 2) * 4 + input_index % 4
        perm[i] = (input_index // 4) * output_dims * 4 + (i // padded_in) * 4 + input_index % 4
    assert sorted(perm.tolist()) == list(range(input_dims)), "scramble 不是有效置换"
    return perm


# ---------------------------------------------------------------------------
# 数值稳定函数
# ---------------------------------------------------------------------------
def softplus(x):
    """数值稳定的 softplus: log(1+exp(x))"""
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def sigmoid_stable(x):
    """数值稳定的 sigmoid"""
    return np.where(x > 20, 1.0, np.where(x < -20, np.exp(x), 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))))


def cross_entropy_loss(v, r, k):
    """交叉熵损失（向量化）。
        L = mean(r * softplus(-K*v) + (1-r) * softplus(K*v))
    """
    kv = k * v
    return float(np.mean(r * softplus(-kv) + (1.0 - r) * softplus(kv)))


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_training_data(path):
    """读取 training_data.txt，返回 (fens, results)。
    每行格式: FEN|result，result 是 STM 视角: 1.0/0.5/0.0
    """
    fens = []
    results = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit('|', 1)
            if len(parts) != 2:
                continue
            fen = parts[0].strip()
            try:
                r = float(parts[1].strip())
            except ValueError:
                continue
            fens.append(fen)
            results.append(r)
    return fens, np.array(results, dtype=np.float32)


# ---------------------------------------------------------------------------
# 批量前向追踪
# ---------------------------------------------------------------------------
def batch_forward_trace(fens, nnue, batch_size=5000, desc="trace"):
    """对所有 FEN 调用 evaluate_with_trace，缓存关键数据。"""
    n = len(fens)
    buckets = np.empty(n, dtype=np.int32)
    psqts = np.empty(n, dtype=np.int32)
    skip_0s = np.empty(n, dtype=np.int32)
    concats = np.empty((n, FC2_INPUTS), dtype=np.float32)
    values = np.empty(n, dtype=np.int32)

    perm = build_scramble_perm()

    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        for i in range(start, end):
            r = nnue.evaluate_with_trace(fens[i])
            b = int(r['bucket'])
            buckets[i] = b
            psqts[i] = int(r['psqt'])
            skip_0s[i] = int(r['skip_0'])
            values[i] = int(r['value'])
            cb = r['concat_buffer']
            concats[i, :] = [cb[perm[j]] for j in range(FC2_INPUTS)]
        elapsed = time.time() - t0
        rate = end / elapsed if elapsed > 0 else 0
        print(f"  [{desc}] {end}/{n}  ({rate:.0f} pos/s)", flush=True)

    return {
        'buckets': buckets,
        'psqts': psqts,
        'skip_0s': skip_0s,
        'concats': concats,
        'values': values,
    }


# ---------------------------------------------------------------------------
# 读取/写回 fc_2 权重
# ---------------------------------------------------------------------------
def read_fc2_weights_float(nnue):
    """读取 fc_2 权重为 float32 数组。返回 (W [16,128], B [16])。"""
    W = np.zeros((NUM_BUCKETS, FC2_INPUTS), dtype=np.float32)
    B = np.zeros(NUM_BUCKETS, dtype=np.float32)
    for b in range(NUM_BUCKETS):
        for j in range(FC2_INPUTS):
            W[b, j] = float(nnue.get_fc2_weight(b, j))
        B[b] = float(nnue.get_fc2_bias(b))
    return W, B


def write_fc2_weights_int(nnue, W, B):
    """将 float32 权重量化回 int8/int32 并写回网络。"""
    for b in range(NUM_BUCKETS):
        for j in range(FC2_INPUTS):
            v = int(round(float(W[b, j])))
            v = max(-128, min(127, v))  # clip int8
            nnue.set_fc2_weight(b, j, v)
        nnue.set_fc2_bias(b, int(round(float(B[b]))))


# ---------------------------------------------------------------------------
# 前向计算（Python 侧，用缓存数据 + float32 权重）
# ---------------------------------------------------------------------------
def forward_predict(W, B, trace):
    """用 float32 权重计算所有样本的 v_pred（浮点，用于训练）。"""
    buckets = trace['buckets']
    fc2_outs = np.sum(W[buckets] * trace['concats'], axis=1) + B[buckets]
    v_pred = trace['psqts'] / 16.0 + (fc2_outs + trace['skip_0s']) * EFFECTIVE_SCALE
    return v_pred


def _trunc_int(x):
    """向零取整（与 C++ static_cast<int> 一致）"""
    return np.where(x >= 0, np.floor(x), np.ceil(x)).astype(np.int64)


def forward_predict_int(W, B, trace):
    """用整数取整模拟 C++ 的精确计算，用于一致性验证。"""
    buckets = trace['buckets']
    fc2_outs = np.sum(W[buckets] * trace['concats'], axis=1) + B[buckets]
    fwdOut = fc2_outs + trace['skip_0s']
    outputValue = _trunc_int(fwdOut * MULTIPLIER / DENOMINATOR)
    value = _trunc_int(trace['psqts'] / OUTPUT_SCALE) + _trunc_int(outputValue / OUTPUT_SCALE)
    return value


# ---------------------------------------------------------------------------
# 梯度计算（mini-batch，含 L2 正则化）
# ---------------------------------------------------------------------------
def compute_gradients(W, B, trace, results, indices, k, weight_decay=0.0):
    """对一个 mini-batch 计算梯度，可选 L2 正则化。

    L2 正则化仅作用于权重 W（不作用于偏置 B），这是常见做法：
        grad_W += weight_decay * W
    返回 (grad_W [16,128], grad_B [16])，已除以 batch_size。
    """
    batch_buckets = trace['buckets'][indices]
    batch_concats = trace['concats'][indices]          # [B, 128]
    batch_psqts = trace['psqts'][indices]
    batch_skip = trace['skip_0s'][indices]
    batch_r = results[indices]

    # 前向
    fc2_outs = np.sum(W[batch_buckets] * batch_concats, axis=1) + B[batch_buckets]
    v_pred = batch_psqts / 16.0 + (fc2_outs + batch_skip) * EFFECTIVE_SCALE

    # error_i = K * (σ(K*v_pred) - r_i)
    sig = sigmoid_stable(k * v_pred)
    errors = k * (sig - batch_r)    # [B]

    # 梯度
    grad_W = np.zeros_like(W)
    grad_B = np.zeros_like(B)
    for b in range(NUM_BUCKETS):
        mask = batch_buckets == b
        if not np.any(mask):
            continue
        e = errors[mask]                       # [m]
        c = batch_concats[mask]                # [m, 128]
        grad_W[b] = EFFECTIVE_SCALE * np.sum(e[:, None] * c, axis=0)
        grad_B[b] = EFFECTIVE_SCALE * np.sum(e)

    grad_W /= len(indices)
    grad_B /= len(indices)

    # L2 正则化：∂(λ/2 * Σ W^2)/∂W = λ * W（仅权重，不偏置）
    if weight_decay > 0.0:
        grad_W += weight_decay * W

    return grad_W, grad_B


# ---------------------------------------------------------------------------
# Adam 优化器（支持动态学习率）
# ---------------------------------------------------------------------------
class AdamOptimizer:
    def __init__(self, shape, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.t = 0

    def step(self, params, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad * grad
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return params


# ---------------------------------------------------------------------------
# Cosine annealing 学习率调度
# ---------------------------------------------------------------------------
def cosine_annealing_lr(epoch, max_epoch, lr_max, lr_min_ratio=0.01):
    """Cosine annealing 学习率。

    lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
    其中 lr_min = lr_max * lr_min_ratio。

    epoch=0 时返回 lr_max，epoch=max_epoch 时返回 lr_min。
    """
    lr_min = lr_max * lr_min_ratio
    if max_epoch <= 0:
        return lr_max
    cos_val = math.cos(math.pi * epoch / max_epoch)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + cos_val)


# ---------------------------------------------------------------------------
# 评估尺度监控
# ---------------------------------------------------------------------------
def evaluate_sample_fens(nnue, sample_fens):
    """对一组 FEN 调用 evaluate，返回 int32 值数组。"""
    values = np.empty(len(sample_fens), dtype=np.int32)
    for i, fen in enumerate(sample_fens):
        values[i] = int(nnue.evaluate(fen))
    return values


def monitor_scale(eval_before, eval_after):
    """对比训练前后的评估值分布，返回统计字典。"""
    before = eval_before.astype(np.float64)
    after = eval_after.astype(np.float64)
    mean_before = float(np.mean(before))
    mean_after = float(np.mean(after))
    std_before = float(np.std(before))
    std_after = float(np.std(after))
    mean_shift = mean_after - mean_before
    std_ratio = std_after / std_before if std_before > 0 else float('inf')

    return {
        'mean_before': mean_before,
        'mean_after': mean_after,
        'std_before': std_before,
        'std_after': std_after,
        'mean_shift': mean_shift,
        'std_ratio': std_ratio,
        'mean_shift_acceptable': abs(mean_shift) <= SCALE_MEAN_SHIFT_WARN,
        'std_ratio_acceptable': abs(std_ratio - 1.0) <= SCALE_STD_RATIO_WARN,
    }


# ---------------------------------------------------------------------------
# 训练中 Elo 快速验证
# ---------------------------------------------------------------------------
def run_elo_quick_check(trained_network_path, games=ELO_EVAL_GAMES,
                        nodes=ELO_EVAL_NODES, timeout=300):
    """运行 elo_match.py 做快速 Elo 验证。

    需要 trained_network_path 对应的文件名是 'pikafish_trained.nnue'，
    因为 elo_match.py 的 Engine A 固定加载该文件名。
    我们通过复制/重命名到 src/pikafish_trained.nnue 来实现。

    返回 (elo, verdict_str) 或 (None, error_str)。
    """
    src_dir = "src"
    target_eval_file = os.path.join(src_dir, "pikafish_trained.nnue")

    # 复制训练后网络到 elo_match.py 期望的位置
    try:
        import shutil
        shutil.copy2(trained_network_path, target_eval_file)
    except Exception as e:
        return None, f"复制网络文件失败: {e}"

    # 运行 elo_match.py
    env = os.environ.copy()
    env["MATCH_GAMES"] = str(games)
    env["MATCH_NODES"] = str(nodes)
    # 快速验证用宽松的 SPRT 边界，避免过早停止影响 Elo 估计
    env.setdefault("MATCH_ELO0", "0.0")
    env.setdefault("MATCH_ELO1", "10.0")

    cmd = ["python3", "tools/elo_match.py"]
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=timeout, cwd="."
        )
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return None, f"elo_match.py 超时 (>{timeout}s)"
    except Exception as e:
        return None, f"运行 elo_match.py 失败: {e}"

    # 解析 JSON 结果文件
    result_path = "tools/elo_match_result.txt"
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        elo = result.get('elo')
        verdict = result.get('verdict', 'UNKNOWN')
        wins = result.get('wins', 0)
        losses = result.get('losses', 0)
        draws = result.get('draws', 0)
        return elo, f"{verdict} (W:{wins} L:{losses} D:{draws})"
    except Exception:
        # JSON 解析失败，尝试从输出中提取 Elo
        for line in output.split('\n'):
            if line.startswith("Elo:"):
                try:
                    # "Elo: +12.3 ± 5.6 (95% CI)"
                    parts = line.split()
                    elo = float(parts[1])
                    return elo, "parsed from stdout"
                except (IndexError, ValueError):
                    pass
        return None, f"无法解析 Elo 结果\n输出尾部:\n{output[-500:]}"


# ---------------------------------------------------------------------------
# 主训练流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="fc_2 输出层 Texel Tuning 训练 v2 (改进版)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data', default='training_data.txt', help='训练数据文件')
    parser.add_argument('--network', default='src/pikafish.nnue', help='输入网络文件')
    parser.add_argument('--k-file', default='tools/optimal_k.txt', help='最优 K 文件')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=512, help='mini-batch 大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率 (cosine annealing 最大值)')
    parser.add_argument('--lr-min-ratio', type=float, default=0.01,
                        help='cosine annealing 最小学习率比例 (lr_min = lr * ratio)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='L2 weight decay 正则化系数 (仅作用于权重 W)')
    parser.add_argument('--output', default='src/pikafish_trained_v2.nnue', help='输出网络文件')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='验证集比例')
    # 尺度监控
    parser.add_argument('--scale-monitor', action='store_true', default=True,
                        help='启用评估尺度监控')
    parser.add_argument('--no-scale-monitor', dest='scale_monitor', action='store_false',
                        help='禁用评估尺度监控')
    # Elo 快速验证
    parser.add_argument('--eval-interval', type=int, default=10,
                        help='每 N epoch 跑一次 Elo 快速验证 (0=禁用)')
    parser.add_argument('--eval-games', type=int, default=ELO_EVAL_GAMES,
                        help='Elo 快速验证局数')
    parser.add_argument('--eval-nodes', type=int, default=ELO_EVAL_NODES,
                        help='Elo 快速验证每局节点数')
    parser.add_argument('--elo-early-stop', type=float, default=ELO_EARLY_STOP_THRESHOLD,
                        help='Elo 早停阈值 (Elo < 此值则停止训练)')
    parser.add_argument('--early-stop-patience', type=int, default=EARLY_STOP_PATIENCE,
                        help='验证集 loss 早停耐心 (epoch)')
    args = parser.parse_args()

    print("=" * 72)
    print("fc_2 输出层 Texel Tuning 训练 v2 (改进版)")
    print("=" * 72)
    print(f"effective_scale = {EFFECTIVE_SCALE}")
    print(f"参数量: {NUM_BUCKETS}×{FC2_INPUTS} + {NUM_BUCKETS} = {NUM_WEIGHTS}")
    print(f"配置: epochs={args.epochs}, batch_size={args.batch_size}, "
          f"lr={args.lr}, lr_min_ratio={args.lr_min_ratio}, "
          f"weight_decay={args.weight_decay}, seed={args.seed}")
    print(f"改进: L2 正则化 + Cosine annealing LR + 尺度监控 + Elo 快速验证")
    print()

    # ------------------------------------------------------------------
    # 1. 加载 pybind 模块和网络
    # ------------------------------------------------------------------
    sys.path.insert(0, 'src')
    import nnue_pybind
    nnue_pybind.load(args.network)
    print(f"[OK] 已加载网络: {args.network}")

    # 读取 K
    with open(args.k_file) as f:
        K = float(f.read().strip())
    print(f"[OK] K = {K}")

    # ------------------------------------------------------------------
    # 2. 加载训练数据
    # ------------------------------------------------------------------
    print(f"\n[1/7] 加载训练数据: {args.data}")
    fens, results = load_training_data(args.data)
    n_total = len(fens)
    print(f"  共 {n_total} 个局面")

    # 90% 训练 + 10% 验证（随机分割）
    rng = np.random.RandomState(args.seed)
    perm_idx = rng.permutation(n_total)
    n_val = int(n_total * args.val_ratio)
    val_indices = perm_idx[:n_val]
    train_indices = perm_idx[n_val:]
    print(f"  训练集: {len(train_indices)}，验证集: {n_val}")

    # ------------------------------------------------------------------
    # 3. 批量前向追踪
    # ------------------------------------------------------------------
    print(f"\n[2/7] 批量前向追踪 (evaluate_with_trace)")
    print("  --- 训练集 ---")
    train_fens = [fens[i] for i in train_indices]
    train_results = results[train_indices]
    train_trace = batch_forward_trace(train_fens, nnue_pybind, desc="train")

    print("  --- 验证集 ---")
    val_fens = [fens[i] for i in val_indices]
    val_results = results[val_indices]
    val_trace = batch_forward_trace(val_fens, nnue_pybind, desc="val")

    # ------------------------------------------------------------------
    # 4. 读取 fc_2 权重
    # ------------------------------------------------------------------
    print(f"\n[3/7] 读取 fc_2 权重")
    W, B = read_fc2_weights_float(nnue_pybind)
    print(f"  W shape: {W.shape}, B shape: {B.shape}")
    print(f"  W 范围: [{W.min():.1f}, {W.max():.1f}], B 范围: [{B.min():.1f}, {B.max():.1f}]")
    # L2 范数（用于监控正则化效果）
    w_l2_norm = float(np.sqrt(np.sum(W ** 2)))
    print(f"  初始 W L2 范数: {w_l2_norm:.4f}")

    # ------------------------------------------------------------------
    # 5. 验证前向一致性
    # ------------------------------------------------------------------
    print(f"\n[4/7] 验证前向一致性 (Python v_pred vs trace value)")
    train_v_int = forward_predict_int(W, B, train_trace)
    int_match = np.array_equal(train_v_int, train_trace['values'])
    int_mismatch_count = int(np.sum(train_v_int != train_trace['values']))
    print(f"  整数前向: 精确匹配 = {'✓' if int_match else '✗'}"
          f"{'  (不匹配数: %d)' % int_mismatch_count if not int_match else ''}")

    train_v_pred = forward_predict(W, B, train_trace)
    train_diff = train_v_pred - train_trace['values']
    max_diff = float(np.max(np.abs(train_diff)))
    mean_diff = float(np.mean(np.abs(train_diff)))
    consistent = int_match and max_diff < 2.5
    print(f"  浮点前向: max|v_pred - value| = {max_diff:.6f}, mean = {mean_diff:.6f} (固有取整误差, <2.5 正常)")
    print(f"  一致性: {'✓ 通过' if consistent else '✗ 失败'}")

    val_v_pred = forward_predict(W, B, val_trace)
    val_v_int = forward_predict_int(W, B, val_trace)
    val_int_match = np.array_equal(val_v_int, val_trace['values'])
    val_max_diff = float(np.max(np.abs(val_v_pred - val_trace['values'])))
    print(f"  验证集整数: 精确匹配 = {'✓' if val_int_match else '✗'}, 浮点 max|diff| = {val_max_diff:.6f}")

    if not consistent:
        print("\n[警告] 前向一致性验证失败！公式可能不正确。继续训练但结果可能有问题。")

    # ------------------------------------------------------------------
    # 5.5 评估尺度监控 — 训练前采样
    # ------------------------------------------------------------------
    scale_monitor_result = None
    sample_fens_for_scale = []
    eval_before = None
    if args.scale_monitor:
        print(f"\n[4.5/7] 评估尺度监控 — 训练前采样")
        # 从全部 FEN 中均匀采样
        n_sample = min(SCALE_MONITOR_SAMPLE, n_total)
        sample_idx = np.linspace(0, n_total - 1, n_sample, dtype=np.int64)
        sample_fens_for_scale = [fens[i] for i in sample_idx]
        eval_before = evaluate_sample_fens(nnue_pybind, sample_fens_for_scale)
        print(f"  采样 {n_sample} 个局面")
        print(f"  训练前: mean = {np.mean(eval_before):.2f}, std = {np.std(eval_before):.2f}, "
              f"range = [{eval_before.min()}, {eval_before.max()}]")

    # ------------------------------------------------------------------
    # 6. 训练循环
    # ------------------------------------------------------------------
    print(f"\n[5/7] 开始训练")
    print("-" * 72)

    # 初始 loss
    init_train_loss = cross_entropy_loss(train_v_pred, train_results, K)
    init_val_loss = cross_entropy_loss(val_v_pred, val_results, K)
    print(f"  初始: train_loss = {init_train_loss:.6f}, val_loss = {init_val_loss:.6f}")

    # Adam 优化器
    opt_W = AdamOptimizer(W.shape, lr=args.lr)
    opt_B = AdamOptimizer(B.shape, lr=args.lr)

    best_val_loss = init_val_loss
    best_W = W.copy()
    best_B = B.copy()
    best_epoch = 0

    no_improve_count = 0
    elo_early_stopped = False
    elo_history = []

    trace_log = {
        'version': 2,
        'init_train_loss': init_train_loss,
        'init_val_loss': init_val_loss,
        'init_w_l2_norm': w_l2_norm,
        'epochs': [],
        'elo_checks': [],
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'lr_min_ratio': args.lr_min_ratio,
            'weight_decay': args.weight_decay,
            'K': K,
            'effective_scale': EFFECTIVE_SCALE,
            'seed': args.seed,
            'eval_interval': args.eval_interval,
            'eval_games': args.eval_games,
            'eval_nodes': args.eval_nodes,
        },
    }

    n_train = len(train_indices)
    train_idx_arr = np.arange(n_train)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # 打乱训练集
        rng.shuffle(train_idx_arr)

        # Cosine annealing 学习率
        current_lr = cosine_annealing_lr(
            epoch - 1, args.epochs, args.lr, args.lr_min_ratio
        )
        opt_W.lr = current_lr
        opt_B.lr = current_lr

        # mini-batch 训练
        for batch_start in range(0, n_train, args.batch_size):
            batch_idx = train_idx_arr[batch_start:batch_start + args.batch_size]
            grad_W, grad_B = compute_gradients(
                W, B, train_trace, train_results, batch_idx, K,
                weight_decay=args.weight_decay,
            )
            W = opt_W.step(W, grad_W)
            B = opt_B.step(B, grad_B)

        # 评估
        train_v_pred = forward_predict(W, B, train_trace)
        val_v_pred = forward_predict(W, B, val_trace)
        train_loss = cross_entropy_loss(train_v_pred, train_results, K)
        val_loss = cross_entropy_loss(val_v_pred, val_results, K)

        elapsed = time.time() - t0
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_W = W.copy()
            best_B = B.copy()
            best_epoch = epoch
            no_improve_count = 0
            improved = " *"
        else:
            no_improve_count += 1

        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss: {train_loss:.4f} | "
              f"val_loss: {val_loss:.4f} | lr: {current_lr:.2e} | "
              f"wd: {args.weight_decay:.2e} | {elapsed:.1f}s{improved}")

        trace_log['epochs'].append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'lr': current_lr,
            'time': round(elapsed, 2),
        })

        # Elo 快速验证
        if args.eval_interval > 0 and epoch % args.eval_interval == 0 and epoch < args.epochs:
            print(f"\n  --- Elo 快速验证 (epoch {epoch}) ---")
            # 保存当前权重到临时网络
            write_fc2_weights_int(nnue_pybind, W, B)
            temp_net = "src/pikafish_trained_v2_temp.nnue"
            nnue_pybind.save_network(temp_net)

            t_elo = time.time()
            elo, verdict = run_elo_quick_check(
                temp_net, games=args.eval_games, nodes=args.eval_nodes
            )
            elo_elapsed = time.time() - t_elo

            if elo is not None:
                print(f"  Elo: {elo:+.1f} | {verdict} | {elo_elapsed:.1f}s")
                elo_history.append({'epoch': epoch, 'elo': elo, 'verdict': verdict})
                trace_log['elo_checks'].append({
                    'epoch': epoch,
                    'elo': elo,
                    'verdict': verdict,
                    'time': round(elo_elapsed, 2),
                })
                # Elo 早停
                if elo < args.elo_early_stop:
                    print(f"  [早停] Elo {elo:+.1f} < {args.elo_early_stop}，"
                          f"训练后网络明显变弱，停止训练")
                    elo_early_stopped = True
            else:
                print(f"  Elo 验证失败: {verdict}")
                elo_history.append({'epoch': epoch, 'elo': None, 'verdict': verdict})
                trace_log['elo_checks'].append({
                    'epoch': epoch,
                    'elo': None,
                    'verdict': verdict,
                    'time': round(elo_elapsed, 2),
                })

            # 清理临时文件
            try:
                os.remove(temp_net)
            except OSError:
                pass
            print(f"  --- 继续训练 ---\n")

            if elo_early_stopped:
                break

        # 验证集 loss 早停
        if no_improve_count >= args.early_stop_patience:
            print(f"\n  早停: 验证集 loss 连续 {args.early_stop_patience} epoch 未下降")
            break

    print("-" * 72)

    # 使用最佳权重
    W = best_W
    B = best_B
    final_train_v_pred = forward_predict(W, B, train_trace)
    final_val_v_pred = forward_predict(W, B, val_trace)
    final_train_loss = cross_entropy_loss(final_train_v_pred, train_results, K)
    final_val_loss = cross_entropy_loss(final_val_v_pred, val_results, K)

    train_drop = (init_train_loss - final_train_loss) / init_train_loss * 100
    val_drop = (init_val_loss - final_val_loss) / init_val_loss * 100
    final_w_l2_norm = float(np.sqrt(np.sum(W ** 2)))

    trace_log['final_train_loss'] = final_train_loss
    trace_log['final_val_loss'] = final_val_loss
    trace_log['best_epoch'] = best_epoch
    trace_log['train_drop_pct'] = train_drop
    trace_log['val_drop_pct'] = val_drop
    trace_log['final_w_l2_norm'] = final_w_l2_norm
    trace_log['elo_early_stopped'] = elo_early_stopped

    # ------------------------------------------------------------------
    # 7. 量化回 int8 并保存网络
    # ------------------------------------------------------------------
    print(f"\n[6/7] 量化回 int8 并保存网络")
    write_fc2_weights_int(nnue_pybind, W, B)
    nnue_pybind.save_network(args.output)
    print(f"  已保存: {args.output}")

    # 验证保存后的网络
    nnue_pybind.load(args.output)
    verify_v = nnue_pybind.evaluate_with_trace(fens[0])['value']
    print(f"  验证: 保存后 evaluate(FEN[0]) = {verify_v}")

    # ------------------------------------------------------------------
    # 8. 评估尺度监控 — 训练后对比
    # ------------------------------------------------------------------
    if args.scale_monitor and eval_before is not None:
        print(f"\n[7/7] 评估尺度监控 — 训练后对比")
        # 重新加载训练后的网络以获取训练后的评估值
        eval_after = evaluate_sample_fens(nnue_pybind, sample_fens_for_scale)
        print(f"  训练后: mean = {np.mean(eval_after):.2f}, std = {np.std(eval_after):.2f}, "
              f"range = [{eval_after.min()}, {eval_after.max()}]")

        scale_monitor_result = monitor_scale(eval_before, eval_after)

        print()
        print("=== Scale Monitor ===")
        print(f"  训练前 mean: {scale_monitor_result['mean_before']:+.2f}  "
              f"std: {scale_monitor_result['std_before']:.2f}")
        print(f"  训练后 mean: {scale_monitor_result['mean_after']:+.2f}  "
              f"std: {scale_monitor_result['std_after']:.2f}")
        mean_shift = scale_monitor_result['mean_shift']
        std_ratio = scale_monitor_result['std_ratio']
        mean_status = "acceptable" if scale_monitor_result['mean_shift_acceptable'] else "⚠ WARNING"
        std_status = "acceptable" if scale_monitor_result['std_ratio_acceptable'] else "⚠ WARNING"
        print(f"  Mean shift: {mean_shift:+.2f} ({mean_status})")
        print(f"  Std ratio:  {std_ratio:.4f} ({std_status})")

        if not scale_monitor_result['mean_shift_acceptable']:
            print(f"  ⚠ 警告: 评估值均值偏移 {abs(mean_shift):.2f} > {SCALE_MEAN_SHIFT_WARN}，"
                  f"可能存在尺度偏移！")
        if not scale_monitor_result['std_ratio_acceptable']:
            print(f"  ⚠ 警告: 评估值标准差变化 |{std_ratio:.4f} - 1| > {SCALE_STD_RATIO_WARN}，"
                  f"输出尺度可能改变！")

        trace_log['scale_monitor'] = scale_monitor_result
    else:
        print(f"\n[7/7] 评估尺度监控已禁用")

    # 保存训练轨迹
    trace_path = 'tools/fc2_training_trace_v2.json'
    with open(trace_path, 'w') as f:
        json.dump(trace_log, f, indent=2)
    print(f"\n  训练轨迹: {trace_path}")

    # ------------------------------------------------------------------
    # 最终报告
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("=== Training Complete ===")
    print("=" * 72)
    print(f"  Best epoch: {best_epoch} (val_loss: {best_val_loss:.6f})")
    print(f"  Cross-entropy improvement: {val_drop:.2f}%")
    print(f"  训练集: {init_train_loss:.6f} -> {final_train_loss:.6f}  ({train_drop:+.2f}%)")
    print(f"  验证集: {init_val_loss:.6f} -> {final_val_loss:.6f}  ({val_drop:+.2f}%)")
    print(f"  W L2 范数: {w_l2_norm:.4f} -> {final_w_l2_norm:.4f}  "
          f"(weight_decay={args.weight_decay})")
    print(f"  Network saved to: {args.output}")
    if elo_history:
        print(f"  Elo 验证历史:")
        for h in elo_history:
            if h['elo'] is not None:
                print(f"    epoch {h['epoch']:3d}: Elo {h['elo']:+.1f} ({h['verdict']})")
            else:
                print(f"    epoch {h['epoch']:3d}: 失败 ({h['verdict']})")
    if elo_early_stopped:
        print(f"  ⚠ 训练因 Elo 早停")
    print(f"  训练轨迹: {trace_path}")


if __name__ == '__main__':
    main()