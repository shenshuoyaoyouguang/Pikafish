#!/usr/bin/env python3
"""
fc_2 输出层 Texel Tuning 训练

用法（在 WSL 中）：
    cd /mnt/e/xiaoxiao/pikayu/Pikafish
    python3 tools/train_fc2.py [--data training_data.txt] [--network src/pikafish.nnue]
                                [--k-file tools/optimal_k.txt] [--epochs 50] [--batch-size 256]
                                [--lr 1e-3] [--output src/pikafish_trained.nnue]

训练流程：
    1. 加载训练数据，90% 训练 + 10% 验证
    2. 对所有局面调用 evaluate_with_trace 获取 v, bucket, concat_buffer
    3. 读取 fc_2 权重为 float32 数组（16×128 weights + 16 biases = 2064 params）
    4. Mini-batch Adam 优化
    5. 每 epoch 评估验证集交叉熵
    6. 训练完成后量化回 int8，写回网络，保存

数学模型：
    v = psqt/OutputScale + (fc_2(concat) + skip_0) * effective_scale
    其中 effective_scale = 600*OutputScale / (HiddenOneVal*(1<<WeightScaleBits)*2 * OutputScale)
                         = 9600 / (16384 * 16) = 0.03662109375

    交叉熵损失：
        L = (1/N) Σ [r_i * softplus(-K*v_i) + (1-r_i) * softplus(K*v_i)]

    fc_2 解析梯度（对 bucket b 的权重 w[b][j] 和偏置 bias[b]）：
        ∂L/∂w[b][j]   = effective_scale * (K/N) Σ_{i: bucket_i=b} (σ(K*v_i) - r_i) * concat_i[j]
        ∂L/∂bias[b]   = effective_scale * (K/N) Σ_{i: bucket_i=b} (σ(K*v_i) - r_i)

注意：get_fc2_weight 返回的是逻辑权重，需要配合 concat[scramble(i)] 使用。
      本脚本在缓存阶段预计算 scramble 后的 concat，使训练时可直接用逻辑权重点积。
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# 常量
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

# 早停耐心
EARLY_STOP_PATIENCE = 5


# ---------------------------------------------------------------------------
# Scramble 置换（与 C++ AffineTransform::get_weight_index_scrambled 一致）
# ---------------------------------------------------------------------------
def build_scramble_perm(input_dims=FC2_INPUTS, output_dims=1):
    """构建 scramble 置换数组 perm[i] = get_weight_index_scrambled(i)。

    USE_PAIR_ACTIVATIONS + 非 AVX512 分支：
        block  = inputIndex / 32
        chunk  = (inputIndex % 32) / 4
        inputIndex = block*32 + ((chunk%2)*4 + chunk/2)*4 + inputIndex%4
        return inputIndex/4 * OutDims*4 + i/PaddedIn*4 + inputIndex%4
    """
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
    # 对大 x 直接返回 x；对小 x 用 exp
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
    """对所有 FEN 调用 evaluate_with_trace，缓存关键数据。

    返回 dict:
        buckets  : int32 [N]
        psqts    : int32 [N]
        skip_0s  : int32 [N]
        concats  : float32 [N, 128]  (已 scramble，可直接与逻辑权重点积)
        values   : int32 [N]         (trace 中的 value，用于一致性验证)
    """
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
            # concat_buffer 是物理顺序，用 perm 重排为逻辑顺序
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
    """用 float32 权重计算所有样本的 v_pred（浮点，用于训练）。

    v_pred = psqt/16 + (dot(W[bucket], concat) + bias[bucket] + skip_0) * effective_scale
    """
    buckets = trace['buckets']
    # dot(W[bucket], concat) 对每个样本
    # W[buckets] -> [N, 128], concats -> [N, 128]
    fc2_outs = np.sum(W[buckets] * trace['concats'], axis=1) + B[buckets]
    v_pred = trace['psqts'] / 16.0 + (fc2_outs + trace['skip_0s']) * EFFECTIVE_SCALE
    return v_pred


def _trunc_int(x):
    """向零取整（与 C++ static_cast<int> 一致）"""
    return np.where(x >= 0, np.floor(x), np.ceil(x)).astype(np.int64)


def forward_predict_int(W, B, trace):
    """用整数取整模拟 C++ 的精确计算，用于一致性验证。

    C++ 流程：
        outputFile = trunc(fwdOut * 9600 / 16384)      # 整数
        value = trunc(psqt / 16) + trunc(outputFile / 16)  # 整数
    """
    buckets = trace['buckets']
    fc2_outs = np.sum(W[buckets] * trace['concats'], axis=1) + B[buckets]
    fwdOut = fc2_outs + trace['skip_0s']
    outputValue = _trunc_int(fwdOut * MULTIPLIER / DENOMINATOR)
    value = _trunc_int(trace['psqts'] / OUTPUT_SCALE) + _trunc_int(outputValue / OUTPUT_SCALE)
    return value


# ---------------------------------------------------------------------------
# 梯度计算（mini-batch）
# ---------------------------------------------------------------------------
def compute_gradients(W, B, trace, results, indices, k):
    """对一个 mini-batch 计算梯度。

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
    return grad_W, grad_B


# ---------------------------------------------------------------------------
# Adam 优化器
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
# 主训练流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="fc_2 输出层 Texel Tuning 训练")
    parser.add_argument('--data', default='training_data.txt', help='训练数据文件')
    parser.add_argument('--network', default='src/pikafish.nnue', help='输入网络文件')
    parser.add_argument('--k-file', default='tools/optimal_k.txt', help='最优 K 文件')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=256, help='mini-batch 大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--output', default='src/pikafish_trained.nnue', help='输出网络文件')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--lr-decay', type=float, default=0.5, help='学习率衰减因子')
    parser.add_argument('--lr-patience', type=int, default=3, help='学习率衰减耐心')
    args = parser.parse_args()

    print("=" * 70)
    print("fc_2 输出层 Texel Tuning 训练")
    print("=" * 70)
    print(f"effective_scale = {EFFECTIVE_SCALE}")
    print(f"参数量: {NUM_BUCKETS}×{FC2_INPUTS} + {NUM_BUCKETS} = {NUM_WEIGHTS}")
    print(f"配置: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, seed={args.seed}")
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
    print(f"\n[1/6] 加载训练数据: {args.data}")
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
    print(f"\n[2/6] 批量前向追踪 (evaluate_with_trace)")
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
    print(f"\n[3/6] 读取 fc_2 权重")
    W, B = read_fc2_weights_float(nnue_pybind)
    print(f"  W shape: {W.shape}, B shape: {B.shape}")
    print(f"  W 范围: [{W.min():.1f}, {W.max():.1f}], B 范围: [{B.min():.1f}, {B.max():.1f}]")

    # ------------------------------------------------------------------
    # 5. 验证前向一致性
    # ------------------------------------------------------------------
    print(f"\n[4/6] 验证前向一致性 (Python v_pred vs trace value)")
    # 整数前向验证（模拟 C++ 整数取整，应精确匹配）
    train_v_int = forward_predict_int(W, B, train_trace)
    int_match = np.array_equal(train_v_int, train_trace['values'])
    int_mismatch_count = int(np.sum(train_v_int != train_trace['values']))
    print(f"  整数前向: 精确匹配 = {'✓' if int_match else '✗'}"
          f"{'  (不匹配数: %d)' % int_mismatch_count if not int_match else ''}")

    # 浮点前向（训练用，与整数有 < 2.0 的固有取整误差）
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
    # 6. 训练循环
    # ------------------------------------------------------------------
    print(f"\n[5/6] 开始训练")
    print("-" * 70)

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
    lr_no_improve_count = 0
    current_lr = args.lr

    trace_log = {
        'init_train_loss': init_train_loss,
        'init_val_loss': init_val_loss,
        'epochs': [],
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'K': K,
            'effective_scale': EFFECTIVE_SCALE,
            'seed': args.seed,
        },
    }

    n_train = len(train_indices)
    train_idx_arr = np.arange(n_train)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # 打乱训练集
        rng.shuffle(train_idx_arr)

        # mini-batch 训练
        for batch_start in range(0, n_train, args.batch_size):
            batch_idx = train_idx_arr[batch_start:batch_start + args.batch_size]
            grad_W, grad_B = compute_gradients(W, B, train_trace, train_results, batch_idx, K)
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
            lr_no_improve_count = 0
            improved = " *"
        else:
            no_improve_count += 1
            lr_no_improve_count += 1

        # 学习率调度
        if lr_no_improve_count >= args.lr_patience:
            current_lr *= args.lr_decay
            opt_W.lr = current_lr
            opt_B.lr = current_lr
            lr_no_improve_count = 0
            improved += f" [lr->{current_lr:.2e}]"

        print(f"  Epoch {epoch:3d}/{args.epochs}: train={train_loss:.6f}  "
              f"val={val_loss:.6f}  best={best_val_loss:.6f}  "
              f"lr={current_lr:.2e}  {elapsed:.1f}s{improved}")

        trace_log['epochs'].append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'lr': current_lr,
            'time': round(elapsed, 2),
        })

        # 早停
        if no_improve_count >= EARLY_STOP_PATIENCE:
            print(f"\n  早停: 验证集 loss 连续 {EARLY_STOP_PATIENCE} epoch 未下降")
            break

    print("-" * 70)

    # 使用最佳权重
    W = best_W
    B = best_B
    final_train_v_pred = forward_predict(W, B, train_trace)
    final_val_v_pred = forward_predict(W, B, val_trace)
    final_train_loss = cross_entropy_loss(final_train_v_pred, train_results, K)
    final_val_loss = cross_entropy_loss(final_val_v_pred, val_results, K)

    train_drop = (init_train_loss - final_train_loss) / init_train_loss * 100
    val_drop = (init_val_loss - final_val_loss) / init_val_loss * 100

    print(f"\n  训练前: train_loss = {init_train_loss:.6f}, val_loss = {init_val_loss:.6f}")
    print(f"  训练后: train_loss = {final_train_loss:.6f}, val_loss = {final_val_loss:.6f}")
    print(f"  下降:   train = {train_drop:.2f}%, val = {val_drop:.2f}%")
    print(f"  最佳 epoch: {best_epoch}")

    trace_log['final_train_loss'] = final_train_loss
    trace_log['final_val_loss'] = final_val_loss
    trace_log['best_epoch'] = best_epoch
    trace_log['train_drop_pct'] = train_drop
    trace_log['val_drop_pct'] = val_drop

    # ------------------------------------------------------------------
    # 7. 量化回 int8 并保存网络
    # ------------------------------------------------------------------
    print(f"\n[6/6] 量化回 int8 并保存网络")
    write_fc2_weights_int(nnue_pybind, W, B)
    nnue_pybind.save_network(args.output)
    print(f"  已保存: {args.output}")

    # 验证保存后的网络
    nnue_pybind.load(args.output)
    verify_v = nnue_pybind.evaluate_with_trace(fens[0])['value']
    print(f"  验证: 保存后 evaluate(FEN[0]) = {verify_v} (原 {train_trace['values'][0] if train_indices[0] == 0 else 'N/A'})")

    # 保存训练轨迹
    trace_path = 'tools/fc2_training_trace.json'
    with open(trace_path, 'w') as f:
        json.dump(trace_log, f, indent=2)
    print(f"  训练轨迹: {trace_path}")

    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"  前向一致性: max|v_pred - value| = {max_diff:.6f} {'✓' if consistent else '✗'}")
    print(f"  训练集交叉熵: {init_train_loss:.6f} -> {final_train_loss:.6f}  ({train_drop:+.2f}%)")
    print(f"  验证集交叉熵: {init_val_loss:.6f} -> {final_val_loss:.6f}  ({val_drop:+.2f}%)")
    print(f"  最佳 epoch: {best_epoch}")
    print(f"  输出网络: {args.output}")
    print(f"  训练轨迹: {trace_path}")


if __name__ == '__main__':
    main()