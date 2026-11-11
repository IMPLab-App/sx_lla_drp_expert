# BACN 11组：完整模型和两组消融

用户已确认：在验证其余实现完全一致后，复用`logs/bacn_11_20261105`的11组作为无DDDE/自适应τ消融；新增完整模型11组及无DEA11组，共22个正式训练任务。新增任务等待原11组训练、最佳模型重载和900样本测试全部完成后启动。

| 版本 | DDDE/双KL自适应τ | 动态聚合 | 融合损失 | 预测 |
|---|---|---|---|---|
| no_ddde_kl | 关闭，Tail τ固定4 | 保留 | 保留 | 原动态融合 |
| full | 启用 | 保留 | 保留 | 原动态融合 |
| no_dea | 启用 | 整条移除 | 专用融合损失关闭 | 双专家原始logits固定0.5/0.5平均 |

完整模型保留原11组的ViT、线性层多尺度融合、路由网络、专家分类器、增强、损失、优化器及初始化，仅新增DDDE/自适应τ。没有切换到历史SegFormer融合结构。无DEA版本从同样初始化得到的模型中移除聚合路径，保留骨干和专家分支及其原监督/无监督损失。

新增算法为`metaexpert_bacn`，实现位于`semilearn/imb_algorithms/metaexpert/bacn_ablation.py`。`ablation_no_ddde_kl=true`且DEA保留时直接调用原MetaExpert.train_step，不创建DDDE、不注册额外前向钩子，不改变随机数使用。原`metaexpert.py`保持不变。原批次代码快照与所有历史结果保留。

DDDE使用Tail弱增强原始logits及同批共享特征，经置信度0.95筛选、按伪标签计算Effective Rank，归一化后以EMA=0.9更新分布。分别与均匀分布、反向有标签先验计算对称KL，经双路softmax得到τ∈[0,4]。沿用现有TwoKLDDDE数值实现：epoch 0–17预热，18开始收集统计，19起使用上一epoch统计更新τ，epoch内τ固定。原融合损失的epoch>19启用条件保持。

动态τ同时用于Tail监督Logit Adjustment和融合网络Tail路径。checkpoint保存DDDE分布、KL窗口、τ及模块设置，训练内最佳模型重载和独立评估均恢复这些状态，避免评估错误回退到τ=4。

所有正式参数和原11组一致：100 epochs、20,000步、seed=2、batch=64、uratio=2、AdamW lr=0.0005、weight_decay=0.05、est_epoch=18。有标签、无标签、验证和测试均按原NPZ行索引逐项比较；测试集固定900样本。继续采用用户确认的记录内随机350/50/100划分。

GPU0：完整模型奇数任务1/3/5/7/9/11，再运行无DEA偶数任务2/4/6/8/10。GPU1：完整模型偶数任务2/4/6/8/10，再运行无DEA奇数任务1/3/5/7/9/11。每卡11个新增任务，两卡并行，各卡内部顺序执行。

正式批次为`logs/bacn_ablation_22_20261105`，路径记录在`logs/bacn_ablation.latest`。批次内`manifest.json`记录22项任务和复用结果来源，`verification.json`记录一致性验证与短训练验证，`configs/`为配置，另导出至`config/custom_sweep_ablation/bacn_11/full/`和`no_dea/`。`gpu0/`和`gpu1/`为日志，`models/`为新模型结果，`comparison/comparison.md`和`comparison/three_variants.csv`汇总33项比较。

接续进程状态在`successor.json`和`status.txt`。原批次发生失败、调度器异常退出或产物不完整时，接续进程记录错误且不会启动新训练。等待期间新增任务均保持pending。通过所有验证后再启动接续监督器，独立会话退出不影响等待或后续训练。

已完成验证：11组原模型与关闭DDDE的新模型在完全相同数据上，初始化、前向、epoch 0/20的真实训练损失、梯度、优化器参数分组与更新逐项比特一致，且DDDE构造与调用均被禁止；Full/No-DEA的实际动态τ更新、弱增强特征对齐、关键参数梯度和checkpoint/评估恢复均通过；12项采样、DDDE数值和接续队列单元测试通过。额外真实GPU测试确认No-DEA保留原专家监督/无监督目标，覆盖高置信伪标签分支，损失、骨干梯度及优化器更新比特一致。

全部22项短训练、最佳模型重载和900样本独立评估通过，无失败或重试。短训练批次`logs/bacn_ablation_smoke_20261105`只验证流程，不作为正式实验结果。正式批次验证记录已引用此结果。

北京时间2026-11-05 01:47:14已启动独立后台接续监督器，PID为3535322。启动记录`launch.json`，日志`launcher.log`。启动时状态为`waiting_for_baseline`，22项正式任务均为pending，未创建正式GPU训练队列；待原11项全部成功完成后自动转为running。已实测重复启动返回退出码2，保留现有监督器及其状态。

查看新增队列进度：

```bash
/home/kv/.conda/envs/cw/bin/python status_signal_baselines.py logs/bacn_ablation_22_20261105
```

三版本指标汇总查看`logs/bacn_ablation_22_20261105/comparison/comparison.md`。`bash run_bacn_ablation.sh`使用同一等待和互斥机制，已运行时无需重复启动。
