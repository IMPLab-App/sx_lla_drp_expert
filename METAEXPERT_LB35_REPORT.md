# 当前 MetaExpert：lb35 六任务双 GPU 实验

正式批次：`logs/metaexpert_lb35_six_20261103`。只运行当前工作区的自有 MetaExpert 方法，共6项，不运行任何对比方法，也不使用历史基线或消融模型的训练实现。

GPU0：run_1→run_3→run_5；GPU1：run_2→run_4→run_6，各3项，双卡同时执行、每卡内部顺序执行。保留机器上现有其他计算进程。

| 任务 | 有标签头类数量 | 有标签比例 | 无标签数量参数 | 无标签分布 |
|---|---:|---:|---:|---|
| run_1_pic_lb35_5_ulb315_5_exp | 35 | 5 | 315 | exp / 5 |
| run_2_pic_lb35_10_ulb315_10_exp | 35 | 10 | 315 | exp / 10 |
| run_3_pic_lb35_5_ulb1162_1.0_exp | 35 | 5 | 1162 | 均匀 |
| run_4_pic_lb35_10_ulb1162_1.0_exp | 35 | 10 | 1162 | 均匀 |
| run_5_pic_lb35_5_ulb315_-5_pxe | 35 | 5 | 315 | pxe / -5 |
| run_6_pic_lb35_10_ulb315_-10_pxe | 35 | 10 | 315 | pxe / -10 |

均匀无标签配置1162为总数量，按现有采样器分配为130+8×129，完整保留1162个样本。长尾配置315是无标签头类数量。所有配置逐类核对样本需求不超过每类350个训练样本，有标签与无标签索引无重叠，train/eval/test也互不重叠。精确数量见批次`verification.json`。

模型沿用当前`vit_tiny_metaexpert`及当前`metaexpert.py`中的两专家、融合网络和损失。保持原融合预热规则（epoch > est_epoch+1时启用融合目标），正式配置est_epoch=18。正式训练为100 epochs、20,000步、batch=64、uratio=2、seed=2、AdamW lr=0.0005、weight_decay=0.05。执行环境使用逻辑GPU0，通过CUDA_VISIBLE_DEVICES分别映射物理GPU0/1。

当前原始信号数据：`raw_signal_dataset_G_1024.npz`，输入[1,1024]，9类各500个样本。70/10/20分层划分使用seed2，归一化只由训练集计算。数据SHA256：`6841e6bdee0c85603a215b9e9f011eac28eac83d6246a248be86e25b10c2eab8`。代码和数据均从当前工作区固定复制到本批次，不受后续工作区修改影响。

已确认并修复的问题：当前MetaExpert在加入路由/融合包装层之前创建优化器，导致后加入的参数未被优化。新增`finalize_model_setup`，在完整模型迁移CUDA之后重新构建优化器，并检查所有当前参数均被覆盖，无失效或重复参数引用。GPU实测路由层、融合层均有非零有限梯度且优化器一步更新改变了参数。检查尊重当前融合预热条件，没有提前启用正式训练的融合损失。

另外，校验工具已适配当前数据加载器基于类别名称指纹生成的索引缓存目录。独立评估保留当前`eval_confusion_matrix_new.py`流程，额外导出原始预测与概率以便重算Accuracy及Macro-F1。源文件修改前备份在`backups/metaexpert_lb35_20261103_212133`。

验证：六组数据数量/划分/一维形状通过；实际GPU优化器覆盖182个可训练参数张量，路由/融合梯度及参数更新通过；六任务短训练、最佳模型重载、平衡/不平衡/完整留出测试评估全部通过。短训练目录`logs/metaexpert_lb35_smoke_20261103`，只是验证流程，不作为正式实验结果。

正式六任务已于北京时间2026-11-03 22:51:25全部训练和评估完成，两张GPU队列正常退出，无失败记录。已复核6组最佳/最终模型文件、900样本的原始预测与评估完成状态，重算Accuracy和Macro-F1均一致。实时状态见`state/*.json`及`status.txt`，日志见`gpu0/`、`gpu1/`，运行异常写入`issues_gpu*.jsonl`。失败产物保留并最多从头重试一次；当前队列不会自动改写未知代码。

最佳模型和全部评估产物在`models/<任务名>/`，其中`evaluation/evaluation.json`包含heldout/balanced/imbalanced指标，`test_metrics.json`为统一留出集指标，`test_predictions.npz`和`test_probabilities.npz`可用于复核或后续绘图。每15秒更新`summary/comparison.md`和`summary/runs.csv`。

查看进度：

```bash
/home/kv/.conda/envs/cw/bin/python status_signal_baselines.py logs/metaexpert_lb35_six_20261103
```
