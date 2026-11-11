# 当前 MetaExpert 的11组实验

模块复核更正（2026-11-05）：本批次运行快照中的`metaexpert.py`没有导入或调用DDDE，Tail监督损失直接使用配置`la_tau_lb3=4.0`，不存在自适应τ更新。已核对快照与启动时SHA256指纹一致；已完成r1最佳模型也不含`cola_two_kl`状态。因此本批次是当前固定τ版本，不能标注为含DDDE/自适应τ的完整模型。用户已确认复用为无DDDE/自适应τ消融；11组与新完整模型关闭该模块时的初始化、前向、损失、梯度、优化器更新及数据索引一致性已实测通过。完整模型11组及无DEA11组已配置并通过22项短训练验证，后台监督器等待本批次全部完成后自动接续。详见`BACN_ABLATION_REPORT.md`和批次`MODULE_AUDIT.md`。

按用户给定的 C0–C8 逐类数量配置 r1–r5、CD1–CD6。使用当前工作区 MetaExpert / vit_tiny_metaexpert，真实一维输入 [1,1024]。正式训练沿用100 epochs、20,000步、batch=64、uratio=2、seed=2、AdamW lr=0.0005、weight_decay=0.05及est_epoch=18。

数据实际为每类500个样本，按现有seed=2随机划分为350训练、50验证、100测试。用户已确认沿用该划分。元数据每类只有一条来源记录，因此这些实验属于记录内测试，不能声明记录间独立泛化。

| 任务 | L | U | 完整训练集 |
|---|---:|---:|---:|
| r1 |157|1443|1600|
| r2 |126|1162|1288|
| r3 |118|1103|1221|
| r4 |104|969|1073|
| r5 |94|885|979|
| CD1 |147|1356|1503|
| CD2 |147|1356|1503|
| CD3 |147|1356|1503|
| CD4 |131|848|979|
| CD5 |95|884|979|
| CD6 |60|919|979|

所有任务使用显式full_train_counts和labeled_counts；U严格为完整训练集合减L。任务名中的比值保留用户命名，仅用于说明目标比例，实际采样以逐类数组为准。CD1–CD3完整训练样本身份相同；r5和CD4–CD6完整训练样本身份相同；CD4–CD6标注样本逐级嵌套。11组使用完全相同的验证/测试样本。

每批次固定复制代码、数据与配置，模型结果不会覆盖历史实验。GPU0按1/3/5/7/9/11运行，GPU1按2/4/6/8/10运行。保留已有GPU进程，每卡内部顺序执行，训练后自动重载最佳模型并在900个测试样本上评估。

2026-11-05 00:17:30（北京时间）已启动正式后台队列，主进程PID为3347884，GPU0/1工作进程PID为3347887/3347888。启动记录在批次`launch.json`，总日志在`launcher.log`。关闭当前会话不影响训练。

验证已完成：3项采样单元测试；11组真实数据数量、索引不重叠、全局随机状态独立性、固定验证/测试集合、CD完整集合身份一致及少标签集合嵌套检查；11组GPU短训练、最佳模型重载和900样本测试全部通过，无失败重试。短训练批次为`logs/bacn_11_smoke_20261105`，正式批次`verification.json`记录验证结果。短训练指标不作为正式实验结果。

正式批次为`logs/bacn_11_20261105`，绝对路径记录在`logs/bacn_11.latest`。批次内`configs/`是正式配置，另复制至`config/custom_sweep/`便于单独调用。`sample_counts.csv`是逐类样本表，`verification.json`及`split_indices/`记录数据验证与原始样本索引，`gpu0/`、`gpu1/`保存日志，`models/<任务名>/`保存模型、指标和预测，`summary/comparison.md`汇总Accuracy和Macro-F1。`status.txt`记录整批状态，`state/`记录每项任务状态，`issues_gpu*.jsonl`记录失败；最多自动重试一次并保留失败输出。

查看进度：

```bash
/home/kv/.conda/envs/cw/bin/python status_signal_baselines.py "$(<logs/bacn_11.latest)"
```

正式批次准备并验证后，`bash run_bacn_11.sh`可运行或继续队列；同批次使用文件锁阻止重复启动，已完成任务跳过。后台启动应将输出重定向至批次日志并通过nohup或独立会话执行。

准备新批次及验证：

```bash
/home/kv/.conda/envs/cw/bin/python prepare_bacn_11.py logs/bacn_11_NEW
/home/kv/.conda/envs/cw/bin/python logs/bacn_11_NEW/code/verify_bacn_11.py logs/bacn_11_NEW
```

短训练验证使用`prepare_bacn_11.py BATCH --smoke`，不作为正式实验结果。
