# 指定13场景：完整模型及两种消融，共39项

场景顺序按用户要求：r2、r4、r5、r7、r9、CD2、CD3，再依次六个L/U分布任务。每个场景排完整模型、无DDDE/自适应τ、无DEA三个版本，共39项。

使用当前本地模型：官方Meta-Expert原有lin1–lin8融合算子、ViT双专家、DDDE及双KL自适应τ、DEA。无DDDE版本使用原metaexpert且τ固定4，保留融合和DEA；无DEA版本保留DDDE/τ，整条动态聚合路径和专用融合损失关闭，双专家原始logits按0.5/0.5平均。

39项不能整数平分，两张GPU分配20/19项。按场景顺序将三个版本依次排入全局队列，再交替分到GPU0/1；各卡内部顺序执行，跨卡并行，不设每个场景完成后的同步屏障。

| 本次编号 | 场景 | 原任务名 | L/U数量 |
|---|---|---|---:|
| 1 | r2 | `run_2_r2_pic_lb35_10_ulb315_10_exp` | 126/1162 |
| 2 | r4 | `run_4_r4_pic_lb35_20_ulb315_20_exp` | 104/969 |
| 3 | r5 | `run_5_r5_pic_lb35_30_ulb315_30_exp` | 94/885 |
| 4 | r7 | `run_7_r7_pic_lb35_50_ulb315_50_exp` | 87/801 |
| 5 | r9 | `run_9_r9_pic_lb35_70_ulb315_70_exp` | 83/755 |
| 6 | cd2 | `run_14_cd2_pic_lb35_6_ulb315_6_step` | 147/1356 |
| 7 | cd3 | `run_15_cd3_pic_lb35_6_ulb315_6_partialrev` | 147/1356 |
| 8 | lu1 | `run_1_pic_lb35_5_ulb315_5_exp` | 157/1443 |
| 9 | lu2 | `run_2_pic_lb35_10_ulb315_10_exp` | 126/1162 |
| 10 | lu3 | `run_3_pic_lb35_5_ulb1170_1.0_exp` | 157/1170 |
| 11 | lu4 | `run_4_pic_lb35_10_ulb1170_1.0_exp` | 126/1170 |
| 12 | lu5 | `run_5_pic_lb35_5_ulb315_-5_pxe` | 157/1443 |
| 13 | lu6 | `run_6_pic_lb35_10_ulb315_-10_pxe` | 126/1162 |

六任务3/4无标签总量保持1170=9×130，名称为ulb1170。固定350训练候选/50验证/100测试的基础划分，L/U精确数量及原始样本身份均逐项匹配各自上一批正式实验；相同场景三版本完全共享L/U/验证/测试。

正式任务均从头训练，保留原seed=2、100 epochs、20,000步、batch=64、uratio=2、AdamW lr=0.0005、weight_decay=0.05、est_epoch=18。验证/测试各450/900样本，每项训练后重载最佳模型并独立测试900样本，最后核对模块状态、checkpoint和预测有限性及指标重算。

本次沿用seed=2，是同seed新一轮训练，不作为新增独立随机种子。

批次：`/media/docs_1/mete-expert/logs/selected13_39_20261107`。冻结代码在code，配置在configs，逐类样本在sample_counts.csv，索引在split_indices，验证记录在verification.json；正式配置另导出至config/selected13/full,no_ddde_kl,no_dea。

运行日志在gpu0/gpu1，训练问题写入issues_gpu*.jsonl，失败输出归档到failed_attempts后最多自动重试一次。队列有互斥锁阻止重复运行。准备过程中若遇问题会记录原因与修复；当前尚未发现模型或采样错误。

短训练批次logs/selected13_smoke_20261107覆盖39项真实短训练、最佳模型重载、900样本评估和产物审计；通过后才启动正式队列，短训练指标不用于论文结果。

查看进度：

```bash
/home/kv/.conda/envs/cw/bin/python status_signal_baselines.py logs/selected13_39_20261107
```

继续同一队列：bash run_selected13.sh。已完成项跳过，旧实验及模型均完整保留。

实时汇总位于summary/comparison.md，case编号对应上表。当前数据仍是用户确认的记录内随机划分。

39项短训练、重载、900样本测试和产物审计全部通过，无失败或重试。正式后台队列于2026-11-07T18:36:00.434508+08:00启动，PID=611880，独立会话运行，关闭聊天不会中断训练。
