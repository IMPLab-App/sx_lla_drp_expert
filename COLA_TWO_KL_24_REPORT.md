最终结果更新（2026-10-26）：24 项正式实验全部完成，失败 0，最后完成时间 19:33。Full 六组平均 Top-1 Accuracy=95.02%、Macro-F1=95.09%。详细对比见 [最终结果汇总](results/cola_two_kl_24/final_summary.md)。以下保留实施和调度历史。

双 GPU 更新（2026-10-26）：已按用户要求分配为两个各 12 项的串行分片。GPU 0：full 六组 + no_mff 六组；GPU 1：no_dea_aggregator 六组 + no_ddde_kl 六组。每张卡同时只运行一个本项目训练任务，数据配置、随机种子和超参数不变。

GPU 0 第一项已完成；已开始的 Full 第二项保留原进度。后台交接监督器会在该任务训练、评估、checkpoint 验证和汇总全部完成后替换空闲的旧调度器，新的 GPU 0 分片会校验并跳过已完成项。GPU 1 的首项已进入训练并正常保存 checkpoint。

实时状态改看 `logs/cola_two_kl_24/gpu_0/current.json`、`gpu_1/current.json`，各卡队列状态在同目录 `queue.json`。GPU 0 交接期间由监督器同步旧队列状态，标记 `handoff_pending=true`。交接进度见 `handoff_supervisor.json`、`handoff.log`。分配清单见 `dual_gpu_plan.json`。任务日志和共享 status.tsv 路径保持，新增每卡队列日志 `gpu_0/queue.log`、`gpu_1/queue.log`。

新增每卡锁、分片锁、任务锁、共享状态写入锁和汇总锁。一个分片失败时记录全局停止请求，另一个分片在当前任务完成后不再启动新项。原任务与旧结果均保留。调度相关源码备份见 `backups/cola_dual_queue_20261026_104416/`，本次差异见 `logs/cola_two_kl_24/setup/dual_queue_changes.patch`。

双卡数值及调度测试共 15 项通过，包括真实轻量进程的安全交接测试、并发状态写入、分片完整性、任务互斥；结果见 `logs/cola_two_kl_24/setup/dual_queue_tests.log`。两卡 dry-run 均为 12 项。只改调度代码，没有改训练源码或 24 份配置。

后续手动启动对应分片的命令如下；已经运行的分片受互斥锁保护，不要重复启动：

```bash
GPU_ID=0 nohup ./run_cola_two_kl_24.sh --worker 0 >> logs/cola_two_kl_24/gpu_0/nohup.log 2>&1 &
GPU_ID=1 nohup ./run_cola_two_kl_24.sh --worker 1 >> logs/cola_two_kl_24/gpu_1/nohup.log 2>&1 &
```

以下保留首次单卡交付和启动记录，当前调度方式以上述双卡更新为准。

正式实验已于 2026-10-26T10:17:23+08:00 在物理 GPU 0 启动，当前按 24 项串行队列运行。队列 PID：1270362；首项训练 PID：1270401。实时状态以 `logs/cola_two_kl_24/current.json` 和 `status.tsv` 为准。

已按附件指令完成实际源码修改、配置生成和验证。附件中的 `codex exec --sandbox workspace-write -` 已实际执行，但独立 CLI 返回 HTTP 401（缺少认证），未执行代码修改；本次工作由当前已认证会话直接完成。首次交付时正式 24 项训练未启动，后续启动状态见本文开头。

原始 CLI 日志：[codex_exec.log](logs/cola_two_kl_24/setup/codex_exec.log)。问题与修复记录：[issues.md](logs/cola_two_kl_24/setup/issues.md)。命令语法与 [OpenAI 官方自动化示例](https://developers.openai.com/cookbook/examples/codex/build_iterative_repair_loops_with_codex)一致；认证问题不能通过修改训练源码解决。

**源码保护**

修改前备份：`/media/docs_1/mete-expert/backups/cola_two_kl_20261026_012737`。其中保留了涉及源码、全部原 custom_sweep YAML、原 Git 状态和原有差异。未执行 reset/clean，未更改原始 CoLA 补充代码，未覆盖原 saved_models、logs 或 data。所有测试索引/模型均写入 `/tmp/cola_*` 隔离目录。原 custom_sweep 配置逐字节比较通过。

当前 `dataset: pic` 使用项目已有 `raw_signal_dataset_G_1024.npz` 一维信号加载器。保留原信号增强、采样、类别划分、随机种子和 EMA 行为。

**修改文件与核心差异**

| 文件 | 修改 |
|---|---|
| `semilearn/imb_algorithms/metaexpert/cola_ddde.py`（新增） | 稳健 Effective Rank、DDDE、pi_u EMA、双路对称 KL、下一 epoch τ 更新；全部状态使用 buffers |
| `semilearn/imb_algorithms/metaexpert/metaexpert.py` | Tail 专用动态 τ、真实消融路径、checkpoint 状态、历史最佳模型保留、分布历史日志；删除旧的分布统计占位代码 |
| `semilearn/nets/vit/vit_metaexpert.py` | Head/E1 与 Tail/E3 的独立 Linear 直接消费相同 `[B,D]` 最终共享特征 |
| `semilearn/nets/wrn/wrn_metaexpert.py` | 同上；`feat` 改为 B 个共享样本，避免把 2B 专家 BN 特征用于 DDDE |
| `semilearn/core/algorithmbase.py` | 优化器参数 id 覆盖断言；epoch 中途恢复跳过已完成批次 |
| `train.py` | GPU/DDP 包装后重建优化器和调度器；断点恢复失败直接报错 |
| `eval_confusion_matrix_new.py` | 隔离训练/评估 argparse；严格加载 EMA；导出三种评估切分的五项指标和混淆矩阵 |
| `prepare_cola_two_kl_24.py`（新增） | 根据六份准确原始 YAML 生成 24 份配置并检查受保护参数 |
| `cola_two_kl_queue.py`、`run_cola_two_kl_24.sh`（新增） | 严格串行训练和评估、互斥锁、状态表、日志、失败停止、完成结果指纹校验 |
| `summarize_cola_two_kl_24.py`（新增） | 真实结果汇总和 pending/failed 标记 |
| `verify_cola_two_kl_24.py`（新增） | 四模式真实 GPU/优化器/checkpoint Smoke Test |
| `tests/test_cola_two_kl.py`、`tests/test_cola_queue.py`（新增） | 数值和队列运行行为测试 |

本次修改相对会话开始时源码的完整差异：[changes.patch](logs/cola_two_kl_24/setup/changes.patch)。文件清单：[changed_files.json](logs/cola_two_kl_24/setup/changed_files.json)。该差异以备份为基准，包含用户原修改之外的本次变更。

三种输入 `x_l / x_u_w / x_u_s` 共同拼接后经过 Backbone，Head 与 Tail 都处理全部样本。共享最终特征供两套独立分类参数使用；保留原专家 BN 辅助无监督路径、MFF、DEA、Aggregator 和原有监督/无监督目标。伪标签分别由 Head/Tail 的原始弱增强 logits 生成，原置信门槛和强增强监督结构保持。

DDDE 仅取 Tail 原始弱增强 logits 及对应共享特征，以 softmax 最大概率 >=0.95 筛选，再按伪标签分组、中心化、SVD、计算 exp(奇异值归一化后的熵)。各类 Effective Rank 归一化得到 pi_DDDE，以 ema_u=0.9 更新 pi_u。空类为零贡献，单样本 rank=1；零奇异值、SVD 失败、无高置信样本和 NaN/Inf 均有保护。无有效统计时不更新 pi_u，也不修改上一轮 τ。

**双路映射**

```python
pi_uni = uniform
pi_rev = flip(pi_l)
D_uni = 0.5 * (KL(pi_u || pi_uni) + KL(pi_uni || pi_u))
D_rev = 0.5 * (KL(pi_u || pi_rev) + KL(pi_rev || pi_u))
weights = softmax(stack([-D_uni, -D_rev]), dim=0)
tau_dyn = tau_max * weights[1]  # tau_max = 4.0
z_T_adjusted = z_T_raw + tau_dyn * log(pi_l)
```

这是定制的双路映射，与 CoLA 原始三路映射不同。实现没有 KL_con 计算、累积或日志；保留 pi_l 用于 Logit Adjustment，也未引入可学习 atau 阶段。

epoch 采用零基编号：0–17 预热 τ=4；18 开始累计有效 batch 的 KL；19 开始依据上一 epoch 的有效 KL 平均值更新 τ。τ 整个 epoch 固定，范围 [0,4]。Head 始终保留原固定 tau_H。网络融合路径中的 Tail τ buffer 与监督损失同步。pi_u、pi_DDDE、KL 累积/计数、上一均值、窗口 epoch、动态 τ 和配置均支持 checkpoint 保存/恢复。

原融合损失的预热条件（epoch <= est_epoch+1）保留，因此其训练目标从 epoch 20 开始生效。这与 τ 的统计起始时刻是两个原有/指定条件。

**消融模式**

| 模式 | MFF | DEA/Aggregator | DDDE/KL | 实际行为 |
|---|---|---|---|---|
| full | 开 | 开 | 开 | 完整模型 |
| no_mff | 关 | 开 | 开 | 使用 `[feat4, feat4]` 拼接替代 |
| no_dea_aggregator | 配置保留开 | 关 | 开 | 跳过 predict/MFF，移除融合训练目标；训练和评估使用 0.5/0.5 原始 logits 平均 |
| no_ddde_kl | 开 | 开 | 关 | 不创建 DDDE 状态、不执行估计/EMA/KL；Tail τ=4 |

`no_dea_aggregator` 移除聚合路径后，MFF 不参与最终预测，不能用该实验单独证明 MFF 的独立效果。对应无梯度参数均明确冻结。原代码已没有残差连接的 MFF.shortcut、旧的中间分类器也被标记为不训练；信号输入不使用 ViT 的图像投影分支。保留其结构参数便于加载模型和兼容图像输入。其他消融的关键模块梯度按启用情况验证。

**六组数据配置与 24 份 YAML**

| 顺序 | 有标签不平衡比 | 无标签数量参数 | 无标签不平衡比 | 无标签类型 |
|---|---:|---:|---:|---|
| 1 | 50 | 250 | 50 | exp |
| 2 | 100 | 250 | 100 | exp |
| 3 | 50 | 315 | 1.0 | exp |
| 4 | 100 | 315 | 1.0 | exp |
| 5 | 50 | 250 | -50 | pxe |
| 6 | 100 | 250 | -100 | pxe |

全部保留原 `num_labels=100`、seed=2、batch_size=64、epoch=100、num_train_iter=20000、lr=0.0005、num_classes=9、p_cutoff=0.95 及其他原训练/数据参数。只新增模式开关、独立输出路径、禁止覆盖/恢复设置和单卡逻辑 GPU=0。原 YAML 中的 1.0、-50、-100 值保持不变。

任务清单和配置摘要：[manifest.json](config/cola_two_kl_24/manifest.json)。24 份完整路径如下：

```text
config/cola_two_kl_24/full/run_1_pic_lb100_50_ulb250_50_exp.yaml
config/cola_two_kl_24/full/run_2_pic_lb100_100_ulb250_100_exp.yaml
config/cola_two_kl_24/full/run_3_pic_lb100_50_ulb315_1.0_exp.yaml
config/cola_two_kl_24/full/run_4_pic_lb100_100_ulb315_1.0_exp.yaml
config/cola_two_kl_24/full/run_5_pic_lb100_50_ulb250_-50_pxe.yaml
config/cola_two_kl_24/full/run_6_pic_lb100_100_ulb250_-100_pxe.yaml
config/cola_two_kl_24/no_mff/run_1_pic_lb100_50_ulb250_50_exp.yaml
config/cola_two_kl_24/no_mff/run_2_pic_lb100_100_ulb250_100_exp.yaml
config/cola_two_kl_24/no_mff/run_3_pic_lb100_50_ulb315_1.0_exp.yaml
config/cola_two_kl_24/no_mff/run_4_pic_lb100_100_ulb315_1.0_exp.yaml
config/cola_two_kl_24/no_mff/run_5_pic_lb100_50_ulb250_-50_pxe.yaml
config/cola_two_kl_24/no_mff/run_6_pic_lb100_100_ulb250_-100_pxe.yaml
config/cola_two_kl_24/no_dea_aggregator/run_1_pic_lb100_50_ulb250_50_exp.yaml
config/cola_two_kl_24/no_dea_aggregator/run_2_pic_lb100_100_ulb250_100_exp.yaml
config/cola_two_kl_24/no_dea_aggregator/run_3_pic_lb100_50_ulb315_1.0_exp.yaml
config/cola_two_kl_24/no_dea_aggregator/run_4_pic_lb100_100_ulb315_1.0_exp.yaml
config/cola_two_kl_24/no_dea_aggregator/run_5_pic_lb100_50_ulb250_-50_pxe.yaml
config/cola_two_kl_24/no_dea_aggregator/run_6_pic_lb100_100_ulb250_-100_pxe.yaml
config/cola_two_kl_24/no_ddde_kl/run_1_pic_lb100_50_ulb250_50_exp.yaml
config/cola_two_kl_24/no_ddde_kl/run_2_pic_lb100_100_ulb250_100_exp.yaml
config/cola_two_kl_24/no_ddde_kl/run_3_pic_lb100_50_ulb315_1.0_exp.yaml
config/cola_two_kl_24/no_ddde_kl/run_4_pic_lb100_100_ulb315_1.0_exp.yaml
config/cola_two_kl_24/no_ddde_kl/run_5_pic_lb100_50_ulb250_-50_pxe.yaml
config/cola_two_kl_24/no_ddde_kl/run_6_pic_lb100_100_ulb250_-100_pxe.yaml
```

配置使用 `saved_models_cola_two_kl/<variant>/<run_name>__<variant>/`，不与原基线输出重叠。

**验证结果**

| 检查 | 结果 |
|---|---|
| Python compileall / Bash -n | 通过 |
| 数值与队列单元测试 | 9/9 通过 |
| DDDE/KL 数值、双路公式、范围/单调性、无置信样本、空类/单样本/SVD 失败/零值/NaN/Inf | 通过 |
| 四种模式 GPU forward/backward/optimizer.step | 全部通过 |
| 实际 CUDA SVD 与 FP16 特征输入、FP32 状态 | 通过 |
| DDDE 输入确为 Tail 弱增强 logits 和对齐的共享特征 | 通过 |
| ViT 共享特征形状 | `[4,192]`，两类 logits 均 `[4,9]` |
| WRN 共享特征形状与 backward | `[4,64]`，两类 logits 均 `[4,9]`，通过 |
| checkpoint 网络/优化器/EMA/DDDE 状态保存恢复 | 通过 |
| 实际端到端训练和独立评估 | 隔离目录 6 步训练通过，最终 it=6 / epoch=3；heldout/balanced/imbalanced 产物齐全 |
| epoch 中途恢复 | 从 it=3、epoch=1 恢复到 it=6、epoch=3，通过 |
| 已完成 checkpoint 恢复到新目录 | 历史最佳模型保留，通过 |
| 六组真实数据采样 | 全部通过，原 50/100 和正/均匀/反向配置正确 |
| 后台队列 dry-run | 严格 4 模式 × 6 组，24 项顺序正确 |
| 重复队列、训练失败、已有日志、损坏指纹、未完成 checkpoint | 拒绝重复/覆盖；按预期停止 |

优化器按参数 id 校验，无遗漏、无失效参数引用、无重复：

| 模式 | 可训练参数张量数 | 非零有限梯度与一步参数更新 |
|---|---:|---|
| full | 190 | backbone, head, tail, mff, routing：通过 |
| no_mff | 166 | backbone, head, tail, routing：通过 |
| no_dea_aggregator | 160 | backbone, head, tail：通过 |
| no_ddde_kl | 190 | backbone, head, tail, mff, routing：通过 |

Full 日志包含 `KL_uni / KL_rev / tau_dyn / DDDE estimated distribution`。No-DDDE/KL 测试通过禁止调用钩子验证未执行 DDDE，Tail τ=4。Smoke Test 的短程模型指标仅用于运行验证，没有写入正式结果 CSV。

验证日志位于 `logs/cola_two_kl_24/setup/`：`unit_tests.log`、`gpu_smoke.log/json`、`compileall.log`、`e2e_train.log`、`e2e_eval.log`、`e2e_result.log`、`resume_smoke.log`、`completed_resume.log`、`six_data_preflight.log`、`queue_dry_run.log`、`environment.json`。

**后台队列使用**

在当前 Conda 环境执行：

```bash
cd /media/docs_1/mete-expert
./run_cola_two_kl_24.sh --dry-run
GPU_ID=0 nohup ./run_cola_two_kl_24.sh >> logs/cola_two_kl_24/nohup.log 2>&1 &
```

首次交付时未执行 nohup；用户后续已授权并启动正式后台队列，见本文开头。可使用 GPU_ID=1 选择另一张物理卡；脚本通过 CUDA_VISIBLE_DEVICES 仅暴露该卡，配置内 gpu=0。Python 从启动时当前 Conda 环境解析，可用 PYTHON_BIN 显式指定。

队列从 full 1→6、no_mff 1→6、no_dea_aggregator 1→6、no_ddde_kl 1→6 严格串行。`flock` 拒绝第二个队列；每次训练成功后调用仓库现有评估工具。训练/评估失败即停止，保存 PID、时间、阶段和退出码，不重试、不调整超参数。

每任务日志使用独占新建模式；已有未完成输出或日志会阻止继续，需人工检查、保留并处理后重启。自动跳过仅接受已完成标记、完整最终/最佳 checkpoint、有效评估产物及匹配的 SHA256 指纹。脚本不自动恢复未完成训练。

运行记录：

- `logs/cola_two_kl_24/queue.log`：队列总日志；`nohup.log`：后台输出。
- `logs/cola_two_kl_24/status.tsv`：追加式任务状态表。
- `logs/cola_two_kl_24/current.json`：运行时当前任务/PID/退出码。
- `logs/cola_two_kl_24/queue.json`：运行时队列 PID/环境/状态。
- `logs/cola_two_kl_24/<variant>__<run_name>.train.log` 和 `.eval.log`：任务日志。
- `saved_models_cola_two_kl/<variant>/<run_name>__<variant>/cola_history.jsonl`：完整 τ、双 KL、DDDE/EMA 分布变化。
- 模型目录下 `evaluation/`：EMA 最佳模型的 heldout、balanced、imbalanced 指标 JSON 和混淆矩阵 CSV/PNG。

**结果汇总与当前状态**

```bash
python summarize_cola_two_kl_24.py
```

输出 [results.csv](results/cola_two_kl_24/results.csv) 和 [ablation_summary.csv](results/cola_two_kl_24/ablation_summary.csv)。每行标记 variant、run_name、seed、状态；仅已完成并校验通过的实验填充准确率、平衡准确率、Macro Precision/Recall/F1、最佳 epoch/iteration、最终 τ、KL/分布起止值及完整历史路径。主指标采用完整 heldout 测试集，另外两种切分的结果保留在 evaluation.json。

六种数据配置不同，汇总均值和标准差仅是跨配置描述，不代表重复随机种子实验的不确定性。

首次交付时：0/24 正式实验完成，24/24 为 pending，指标全部留空。尚无正式实验准确率结果。

独立 Codex CLI 的 HTTP 401 认证问题仍存在，需要在该 CLI 所在环境完成登录；后台训练/评估队列直接调用 Python，不依赖该 CLI。项目实现与上述已执行测试没有剩余失败项。正式 24 项的完整收敛与最终准确率须待后续后台运行得到。
