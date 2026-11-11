# 24场景V1–V5消融与静态τ敏感性实验

按用户最终确认定义：r1–r12、CD1–CD6及六个L/U任务，共24个场景。每场景5个主版本，并补充静态τ=0、1、3；静态τ=2复用本轮新训练V4，静态τ=4复用本轮新训练V1。共120项主实验＋72项补充＝192项实际训练，两张GPU各96项。

| 版本 | 分布估计 | 分布EMA | 距离 | τ |
|---|---|---|---|---|
| V1 | 关闭 | 关闭 | 关闭 | 全程固定4 |
| V2 | DDDE有效秩 | 0.9 | 双参考L1 | 动态[0,4] |
| V3 | 高置信伪标签Histogram | 0.9 | 双参考对称KL | 动态[0,4] |
| V4 | 关闭 | 关闭 | 关闭 | 全程固定2 |
| V5 | DDDE有效秩 | 0.9 | 双参考对称KL | 动态[0,4] |
| S0 | 关闭 | 关闭 | 关闭 | 全程固定0 |
| S1 | 关闭 | 关闭 | 关闭 | 全程固定1 |
| S3 | 关闭 | 关闭 | 关闭 | 全程固定3 |

这里的EMA栏指分布EMA；模型权重EMA=0.999在全部版本保留。V4按用户补充说明修订为真实静态τ控制，不再将无效参与训练的DDDE/KL标为有效。所有版本保留DEA、lin1–lin8融合、ViT双专家、原专家及融合损失。

## 动态映射的精确定义

参考分布统一为9类均匀分布π_uni及反向标注先验π_rev=flip(π_l)。

V2：d_uni=Σ_c|π_u(c)−π_uni(c)|，d_rev=Σ_c|π_u(c)−π_rev(c)|，使用完整L1范数，不除以2。τ=4×softmax(−[mean(d_uni),mean(d_rev)])[1]。V2不调用对称KL；L1与KL的量纲/范围不同，这一版本比较的是两种明确的距离映射。

V3/V5：d为对称KL，D_sym(p,q)=0.5×(KL(p||q)+KL(q||p))，τ采用相同双参考softmax映射。V3在弱增强Tail伪标签置信度≥0.95的样本上计数并归一化为Histogram，V2/V5在相同筛选条件下使用类别有效秩。各动态版本均以EMA=0.9更新π_u，初始化均匀。

动态版本的前18个epoch预热τ=4，epoch18开始收集统计，19起使用上一epoch有效批次距离均值更新τ，epoch内固定。无有效统计时保留上一τ。静态版本从第一步起全程使用各自固定值，不进行估计/距离计算。τ只作为统计映射系数，无额外KL损失或可学习τ参数。

## 数据与训练

所有场景沿用此前已确认逐类N/L/U及原始样本索引：每类350训练候选、50验证、100测试，L/U不重叠，五主版本及敏感性版本共享同一场景所有样本身份。六任务3/4均U=1170=9×130。

seed=2，100 epochs，20,000步，batch=64，uratio=2，AdamW lr=0.0005、weight_decay=0.05，伪标签阈值0.95、温度T=0.5、模型EMA=0.999、无标签噪声，骨干和增强保持原设置。均从头训练，保留所有旧模型及结果。

| 场景 | 实际主任务名 | L/U |
|---|---|---:|
| r1 | `run_1_r1_pic_lb35_5_ulb315_5_exp` | 157/1443 |
| r2 | `run_2_r2_pic_lb35_10_ulb315_10_exp` | 126/1162 |
| r3 | `run_3_r3_pic_lb35_12_ulb315_12_exp` | 118/1103 |
| r4 | `run_4_r4_pic_lb35_20_ulb315_20_exp` | 104/969 |
| r5 | `run_5_r5_pic_lb35_30_ulb315_30_exp` | 94/885 |
| r6 | `run_6_r6_pic_lb35_40_ulb315_40_exp` | 90/834 |
| r7 | `run_7_r7_pic_lb35_50_ulb315_50_exp` | 87/801 |
| r8 | `run_8_r8_pic_lb35_60_ulb315_60_exp` | 83/774 |
| r9 | `run_9_r9_pic_lb35_70_ulb315_70_exp` | 83/755 |
| r10 | `run_10_r10_pic_lb35_80_ulb315_80_exp` | 80/737 |
| r11 | `run_11_r11_pic_lb35_90_ulb315_90_exp` | 79/724 |
| r12 | `run_12_r12_pic_lb35_100_ulb315_100_exp` | 78/712 |
| cd1 | `run_13_cd1_pic_lb35_6_ulb315_6_match` | 147/1356 |
| cd2 | `run_14_cd2_pic_lb35_6_ulb315_6_step` | 147/1356 |
| cd3 | `run_15_cd3_pic_lb35_6_ulb315_6_partialrev` | 147/1356 |
| cd4 | `run_16_cd4_pic_lb35_30_ulb315_30_fixed` | 131/848 |
| cd5 | `run_17_cd5_pic_lb30_30_ulb320_30_fixed` | 95/884 |
| cd6 | `run_18_cd6_pic_lb20_30_ulb330_30_fixed` | 60/919 |
| lu1 | `run_1_pic_lb35_5_ulb315_5_exp` | 157/1443 |
| lu2 | `run_2_pic_lb35_10_ulb315_10_exp` | 126/1162 |
| lu3 | `run_3_pic_lb35_5_ulb1170_1.0_exp` | 157/1170 |
| lu4 | `run_4_pic_lb35_10_ulb1170_1.0_exp` | 126/1170 |
| lu5 | `run_5_pic_lb35_5_ulb315_-5_pxe` | 157/1443 |
| lu6 | `run_6_pic_lb35_10_ulb315_-10_pxe` | 126/1162 |

按场景顺序r1–r12→CD1–CD6→LU1–LU6，每个场景排V1→V2→V3→V4→V5→S0→S1→S3。两卡按场景交替起始卡，各卡内部依次运行，每个版本均有12个场景在GPU0、12个在GPU1，合计各96项。

## 日志和结果

- variant_history.jsonl记录τ实际使用值、当前有效批次的KL或L1、上一epoch用于τ映射的距离均值（*_window）、当前EMA分布pi_estimated和原始估计pi_raw，以及版本/估计器/距离/固定值。
- 未启用的机制用null记录，不计算或伪造KL/L1/估计分布；动态版本预热或无有效批次时当前距离也为null。
- log.txt及TensorBoard保留适用的train/tau_T、train/KL_*或train/L1_*及训练/验证诊断。
- 每项训练后独立重载最佳模型并测试900样本，导出Accuracy、Macro-F1、逐类召回、混淆矩阵、逐样本预测和概率，最后核对checkpoint版本与数值有限性。

正式批次：`/media/docs_1/mete-expert/logs/v1v5_192_20261108`。配置在configs及config/v1v5_24，逐类数量在sample_counts.csv，索引在split_indices。主实验表comparison/main_results.md，敏感性表comparison/sensitivity.md，全部原始结果comparison/all_runs.csv。

## 验证和问题记录

8项数值/接口测试通过，包括V2不调用KL、V3不调用有效秩DDDE、静态版本关闭机制、Histogram/EMA、预热和待更新窗口恢复。真实GPU审计通过：V5在r1、r12、LU3、LU5上与原完整模型的初始化/损失/梯度/优化器更新/DDDE状态比特一致；八个模式均验证DEA梯度、固定/动态τ、checkpoint和独立评估恢复及版本错配拒绝。证据在gpu_audit.json和validation。

准备阶段GPU检查发现train_step通配签名与框架按形参名组装批次的机制不兼容，已改为原六个显式参数，并补回归测试；重新GPU检查通过。问题记录在setup_issues.jsonl。未改动原骨干、融合、专家损失或旧DDDE实现，仅增加独立算法入口metaexpert_v1v5及相应评估器。

正式训练前执行192项真实短训练、最佳模型重载、900样本独立测试及版本审计，全部通过后才启动正式队列。短训练批次logs/v1v5_smoke_20261108仅验证流程，不作为实验结论。

运行错误写入issues_gpu*.jsonl，失败输出归档到failed_attempts，最多自动重试一次。已修复问题保留原因及处理记录；同批次互斥锁避免重复启动。

查看进度：

```bash
/home/kv/.conda/envs/cw/bin/python status_signal_baselines.py logs/v1v5_192_20261108
```

继续同一队列使用bash run_v1v5.sh。当前仍为seed=2记录内划分，重复场景及同seed新训练不作为独立随机种子证据。

全部192项真实短训练/重载/独立测试/版本审计通过，无运行失败或重试。正式后台队列于2026-11-08T01:18:59.351747+08:00启动，PID=802623，两卡各96项，独立会话运行。
