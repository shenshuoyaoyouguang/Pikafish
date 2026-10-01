# Changelog

## [Unreleased] - 2026-10-01

### Added
- LMR 连续参数化：reduction() 函数支持 26 维 θ 参数向量驱动（LMR_Continuous + LMR_Theta UCI 选项）
- 联合搜索参数化：null-move/futility/singular extension 支持 41 维参数化（Joint_Continuous + Joint_Theta UCI 选项）
- export_theta UCI 命令：输出当前 θ 参数向量
- LMR_Sample UCI 选项：控制采样模式，收集 (φ(x), r) 训练数据
- SPSA 优化框架：spsa_optimize.py + spsa_joint_optimize.py
- 对弈验证工具：match_test.py + joint_validate.py
- 基线拟合工具：fit_theta.py
- SDD 文档：lmr-reduction-tuning 和 joint-search-tuning 两套 spec/design/tasks

### Changed
- search.h: 新增 LMR_THETA_SIZE=26, JOINT_THETA_SIZE=41, LmrFeatures 结构体, jointTheta 数组
- search.cpp: reduction_lmr() 实现, init_lmr_theta()/init_joint_theta() 初始化, Step 9/10/16 参数化
- engine.cpp: 新增 LMR_Continuous/LMR_Theta/LMR_Sample/Joint_Continuous/Joint_Theta UCI 选项注册
- uci.cpp: 新增 export_theta 命令处理

### Known Issues
- Windows/MinGW 下深度 9 段错误崩溃（预存 bug，WSL 下正常）
- 搜索参数调优 Elo 天花板约 ±5，未达到 +15 目标
- 41 维 SPSA + 200 局/评估噪声过大，优化未收敛

### Performance
- LMR 26 维 SPSA 优化：500 局/10K 节点 Elo = -2.8 ± 30.5
- 联合 41 维 SPSA 优化：500 局/10K 节点 Elo = -4.2 ± 30.5
- bench 节点数 = 1,484,403（与 legacy 一致）