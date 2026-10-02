import sys
sys.path.insert(0, '/mnt/e/xiaoxiao/pikayu/Pikafish/src')
import nnue_pybind

# 加载网络
nnue_pybind.load('/mnt/e/xiaoxiao/pikayu/Pikafish/src/pikafish.nnue')

# 测试 evaluate_with_trace
fen = "rnbakabr1/9/1c4n1c/p3p1p1p/2p6/2P3P2/P3P3P/1C2C1N2/9/RNBAKAB1R w - - 8 5"
result = nnue_pybind.evaluate_with_trace(fen)
print(f"value: {result['value']}")
print(f"bucket: {result['bucket']}")
print(f"concat_buffer len: {len(result['concat_buffer'])}")  # 应为 128
print(f"fc_0_out len: {len(result['fc_0_out'])}")  # 应为 32
print(f"fc_1_out len: {len(result['fc_1_out'])}")  # 应为 32

# 验证 value 与 evaluate 一致
v_plain = nnue_pybind.evaluate(fen)
assert result['value'] == v_plain, f"Mismatch: {result['value']} vs {v_plain}"
print(f"OK evaluate_with_trace value matches evaluate: {v_plain}")

# 测试 fc_2 权重访问
bucket = result['bucket']
w0 = nnue_pybind.get_fc2_weight(bucket, 0)
print(f"fc_2 weight[bucket={bucket}, idx=0]: {w0}")

# 测试 set + get round-trip
nnue_pybind.set_fc2_weight(bucket, 0, 42)
assert nnue_pybind.get_fc2_weight(bucket, 0) == 42
print("OK set/get fc_2 weight round-trip")

# 恢复原始值
nnue_pybind.set_fc2_weight(bucket, 0, w0)

# 测试 bias 访问
b0 = nnue_pybind.get_fc2_bias(bucket)
print(f"fc_2 bias[bucket={bucket}]: {b0}")

# 测试 bias set + get round-trip
nnue_pybind.set_fc2_bias(bucket, 12345)
assert nnue_pybind.get_fc2_bias(bucket) == 12345
print("OK set/get fc_2 bias round-trip")
nnue_pybind.set_fc2_bias(bucket, b0)

# 测试 save_network
nnue_pybind.save_network('/tmp/test_save.nnue')
print("OK save_network succeeded")

# 验证保存的网络可以重新加载
nnue_pybind.load('/tmp/test_save.nnue')
v_after = nnue_pybind.evaluate(fen)
assert v_after == v_plain, f"After reload mismatch: {v_after} vs {v_plain}"
print(f"OK Saved network reloads and evaluates identically: {v_after}")

# 测试 weight clamping (value > 127 should clamp to 127)
nnue_pybind.set_fc2_weight(bucket, 0, 200)
assert nnue_pybind.get_fc2_weight(bucket, 0) == 127, "clamp to 127 failed"
print("OK weight clamping to 127")
nnue_pybind.set_fc2_weight(bucket, 0, -200)
assert nnue_pybind.get_fc2_weight(bucket, 0) == -128, "clamp to -128 failed"
print("OK weight clamping to -128")
nnue_pybind.set_fc2_weight(bucket, 0, w0)

# 测试 idx 越界
try:
    nnue_pybind.get_fc2_weight(bucket, 128)
    print("FAIL: expected exception for idx=128")
except Exception as e:
    print(f"OK idx=128 raises: {e}")

# 测试 bucket 越界
try:
    nnue_pybind.get_fc2_weight(16, 0)
    print("FAIL: expected exception for bucket=16")
except Exception as e:
    print(f"OK bucket=16 raises: {e}")

print("\n=== All verifications PASSED ===")