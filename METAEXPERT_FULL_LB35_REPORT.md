# 当前完整模型：六种L/U分布任务

六任务均使用metaexpert_bacn，DDDE、双KL自适应τ及DEA全部启用，两个消融开关均为false。模型与此前已验证的完整模型逐文件一致。

按用户最新要求，任务3/4的无标签总量为1170=9×130，名称同步改为ulb1170；不再使用1162。

| 任务 | 名称 | 无标签分布 | L/U总量 | GPU |
|---|---|---|---:|---:|
| 1 | `run_1_pic_lb35_5_ulb315_5_exp` | 同向长尾 | 157/1443 | 0 |
| 2 | `run_2_pic_lb35_10_ulb315_10_exp` | 同向长尾 | 126/1162 | 1 |
| 3 | `run_3_pic_lb35_5_ulb1170_1.0_exp` | 均匀，每类130 | 157/1170 | 0 |
| 4 | `run_4_pic_lb35_10_ulb1170_1.0_exp` | 均匀，每类130 | 126/1170 | 1 |
| 5 | `run_5_pic_lb35_5_ulb315_-5_pxe` | 反向长尾 | 157/1443 | 0 |
| 6 | `run_6_pic_lb35_10_ulb315_-10_pxe` | 反向长尾 | 126/1162 | 1 |

GPU0依次运行1→3→5，GPU1依次运行2→4→6，各3项。每项100 epochs、20,000步，seed=2、batch=64、uratio=2、AdamW lr=0.0005、weight_decay=0.05、est_epoch=18；每类原始划分350训练、50验证、100测试。

L/U采用已验证的固定逐类计数分支。任务1/3/5标注样本身份相同，任务2/4/6标注样本身份相同；验证与测试索引和旧实验完全一致；L/U/验证/测试四集合互不重叠。

DDDE以Tail弱增强原始logits和共享特征估计分布，双KL依据上一epoch统计更新τ；前18个epoch预热，18开始收集，19起可更新。τ会与最佳checkpoint一起保存并在独立评估恢复，避免测试时回退固定4。DEA与专用融合损失保留，原融合预热时间保持。

每项依次执行训练、最佳模型独立900样本测试、产物审计。第三阶段核对DDDE状态、DEA设置、checkpoint有限性、预测概率、重算Accuracy与Macro-F1和逐类测试支持度。

正式批次：`/media/docs_1/mete-expert/logs/metaexpert_full_lb35_six_20261107`。配置在configs，逐类计数在sample_counts.csv，日志在gpu0/gpu1，模型及测试在models，实时结果在summary/comparison.md。可单独运行的正式YAML已导出至config/custom_sweep_full_lb35。

准备期间发现并修复：缓存目录匹配混入NPZ、摘要函数的字符串路径兼容、旧采样入口忽略缓存而重复采样。全部失败准备产物及异常保存在logs/metaexpert_full_lb35_setup_issues_20261107；当前批次setup_issues.md记录处理。旧缓存已按备份摘要恢复，当前批次不再使用或重写旧缓存。

实际运行故障会记录issues_gpu0.jsonl/issues_gpu1.jsonl，失败输出先归档后最多自动重试一次，不覆盖旧结果。

检查命令：

```bash
/home/kv/.conda/envs/cw/bin/python status_signal_baselines.py logs/metaexpert_full_lb35_six_20261107
```

需要继续同一批次时使用bash run_metaexpert_full_lb35.sh；已完成任务跳过，互斥锁防止同批次重复启动。

本数据每类只有一条采集记录，沿用用户确认的记录内随机划分。

全部六项短训练、重载、独立测试和产物审计通过；三种U分布的真实GPU动态τ/DEA梯度及状态恢复审计通过。正式队列于2026-11-07T02:17:26.926509+08:00启动，PID=194562，独立后台会话运行。
