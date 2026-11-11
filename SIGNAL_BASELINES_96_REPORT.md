# 一维信号八方法、六工况、两次重复实验

正式批次：`logs/signal_baselines_96_20261027`。后台启动时间为北京时间 2026-10-27 00:03:02。

全部 96/96 项正式训练与测试已于北京时间 2026-10-27 23:12:18 完成，两张 GPU 队列退出码均为 0。完成后复核了 96 份最佳模型文件和 96 组原始测试预测/统一指标，准确率重算一致。

## 实验安排

方法顺序为 ACR、BaCon、CoSSL、CPE、SimPro、FixMatch、UA-MAN、SAT-iFixMatch。每张卡依次执行以下六组配置，每组包含全部八种方法。

| 配置 | 有标签头类数量 | 有标签比例 | 无标签数量参数 | 无标签比例/类型 |
|---|---:|---:|---:|---|
| run_1_pic_lb100_50_ulb250_50_exp | 100 | 50 | 250 | 50 / exp |
| run_2_pic_lb100_100_ulb250_100_exp | 100 | 100 | 250 | 100 / exp |
| run_3_pic_lb100_50_ulb315_1.0_exp | 100 | 50 | 315 | 1.0 / exp |
| run_4_pic_lb100_100_ulb315_1.0_exp | 100 | 100 | 315 | 1.0 / exp |
| run_5_pic_lb100_50_ulb250_-50_pxe | 100 | 50 | 250 | -50 / pxe |
| run_6_pic_lb100_100_ulb250_-100_pxe | 100 | 100 | 250 | -100 / pxe |

GPU 0：第 1 次重复，seed=2，48 次训练。GPU 1：第 2 次重复，seed=3，48 次训练。两张卡同时运行，每张卡同一时间只执行本批次的一项训练。保留机器上其他用户已有的训练进程。

沿用当前项目/参考项目的主要训练参数：100 epochs、20,000 次参数更新、AdamW、lr=0.0005、weight_decay=0.05；有标签 batch=64，无标签 batch=128。UA-MAN 也执行每 epoch 200 次监督参数更新，额外执行其原生 200 步无监督专家聚合。

## 数据与一维模型

所有方法读取同一份复制后固定的数据 `data/raw_signal_dataset_G_1024.npz`，输入为 `[B, 1, 1024]`。

数据 SHA256：`d14701484e48eaac68f3f8337f904bb053e2031c0026506c313815145836af47`。

统一使用 `get_raw_signal`，按 seed 分层划分 70%/10%/20% 的 train/validation/test。同一 seed、同一配置的八方法采用相同样本划分、类别数和归一化；两个 seed 的划分独立。归一化均值、标准差只由完整训练划分计算。12 组 seed/配置的类别数、数据指纹及归一化参数见批次 `split_audit.json`。

原有数据约定保持一致：不平衡时 100/250 是头类数量；无标签比例为 1.0 时，315 是全部无标签数量，即 9 类各 35 个。`pxe` 是类别数量反向排列。

ACR、BaCon、CoSSL、CPE、SimPro、FixMatch 使用新增的原生一维 WRN-28-2；残差结构复用导入代码，各卷积、归一化和池化改为一维。BaCon 保留辅助分类器和 32 维投影；CPE 保留 H/M/T 三组归一化与三个分类器；ACR、CoSSL 保留其包装分类器。UA-MAN 保留原生一维多专家网络；SAT-iFixMatch 保留原生 ResNet1D。不同方法的架构和损失遵循各自实现，因此本实验不声称参数量相同。

UA-MAN 原生主损失是监督多专家损失，其无标签集参与后续无监督专家聚合，本次使用各配置自己的 train_ulb 进行聚合。SAT-iFixMatch 保留原生增强、动态阈值和一致性/OOD 损失。其拒识样本在统一准确率中计为错误。

## 已发现并修复的问题

| 问题 | 修复 |
|---|---|
| 图片骨干需要 `[B,3,H,W]`，各算法要求的输出头不一致 | 新增 `semilearn/nets/signal_wrn.py` 三种一维骨干变体，保持各自的输出键与特征接口 |
| UA-MAN / SAT-iFixMatch 的 PIC 入口仍读取图片 | 增加 raw_signal 入口，统一使用信号划分及单通道输入 |
| BaCon 对比阶段写死在 262144/3 步之后，20,000 步预算完全不会启动 | 改为当前训练预算的 1/3；温度进度也使用当前总步数 |
| BaCon 某类无特征时计算空均值会生成 NaN；无负样本时损失不稳定 | 无数据时保留该类中心，缺少正/负样本时跳过该项 |
| CoSSL 对标量类别编号执行 argmax，始终使用第 0 类的 mixup 概率 | 直接使用整数类别编号 |
| CoSSL 教师分类器同时出现在主优化器和专用优化器中 | 从主优化器排除教师参数，避免一次梯度被更新两次 |
| CoSSL 设置不存在于 BasicDataset 中的 transform 属性 | 使用实际生效的 weak_transform 属性 |
| CoSSL 分类器预热的特征 squeeze 会在 batch=1 时丢掉样本维度 | 使用 flatten(1) |
| CoSSL EMA 对整数状态 float 转换后更新临时张量，计数器未同步 | 原位更新浮点张量，直接复制整数计数器，初始化由模型复制到 EMA |
| CPE 的 CLI cut1/cut2 是 float，不能作张量切片边界 | 在算法初始化时转换为 int |
| SAT 检查点仅保存最终阈值，不能精确还原 local/global 状态 | 保存并重载 local、global_value、initialized，兼容旧字段 |
| 方法间评估输出头与拒识指标口径不同 | 使用各方法正确输出头；保存原始预测和统一 Accuracy / Macro-F1 等指标 |

修改前文件备份：`backups/signal_baselines_20261026_235605`。

## 已通过的验证

- 八方法双 GPU 冒烟验证，全部通过：训练、最佳模型保存、最佳模型重载、测试评估；目录 `logs/signal_baselines_smoke_20261027`。
- 短训练跨越 ACR 自适应阶段、BaCon 对比分支和 CoSSL 教师预热阶段。另用已选中样本验证 BaCon 的非零对比损失反向传播及缺类边界。
- 三种骨干全程使用一维运算、输出形状正确，各输出头梯度存在且有限。
- CoSSL EMA 浮点权重和整数 BN 计数器更新正确。
- 全部方法短训练的测试标签顺序一致，统一指标与保存预测重新计算的结果一致。
- 正式 96 配置无重复名字，两个 GPU 各 48 项，六配置顺序、seed、样本比例和训练预算均正确。
- 审计 12 组数据划分，检查 `[1,1024]` 输入、目标类别数及 train/validation/test 互斥。

## 后台状态、结果和异常记录

查看状态：

```bash
/home/kv/.conda/envs/cw/bin/python status_signal_baselines.py
```

正式批次保留 `code/` 代码快照、`source_sha256.json` 源码哈希、`manifest.json` 全部命令及 `configs/` 96 份配置。训练使用快照，不受工作区后续修改影响。

- `queue.pid`：监督进程 PID；`gpu0/worker.pid`、`gpu1/worker.pid`：队列进程 PID。
- `state/*.json`：各任务 pending/running/completed/failed 状态、实际训练 PID 及日志路径。
- `gpu0/`、`gpu1/`：每方法、配置、重复的训练/评估日志。
- `models/<方法_配置_重复_seed>/`：最佳模型、统一 `test_metrics.json`、`test_predictions.npz`、原始计数混淆矩阵及图片。
- `summary/runs.csv`、`summary/comparison.md`、`summary/aggregate.json`：每 15 秒更新的进度与对比汇总；同方法同配置两次都完成后计算均值、样本标准差。
- `issues_gpu0.jsonl`、`issues_gpu1.jsonl`：如有失败，记录退出码、阶段、日志路径和错误栈末尾。失败自动从头重试一次，原始失败产物移入 `failed_attempts/`；第二次仍失败则标记 failed 并继续其他任务。这里没有自动改写未知代码的机制。

队列可在原批次上重新启动；会跳过标记 completed 的任务。未完成任务会归档并从头重训，不宣称恢复了算法内存中的自适应状态。不要在已有监督进程运行时启动第二份队列，文件锁会阻止重复运行。

```bash
/home/kv/.conda/envs/cw/bin/python logs/signal_baselines_96_20261027/code/run_signal_baselines.py logs/signal_baselines_96_20261027
```

全部正式实验已结束；最终正式分数见 `summary/comparison.md`、`summary/runs.csv` 和 `summary/aggregate.json`。
