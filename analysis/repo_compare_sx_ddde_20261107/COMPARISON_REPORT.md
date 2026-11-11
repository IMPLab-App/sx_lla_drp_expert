# 本地完整模型与sx_ddde_meta_expert-比较

对比GitHub main当前提交：`324417233b7cfc735b8270ad27df8fbcd4cf7cf3`，提交日期2026-10-26。本地为当前工作区实际文件，并逐项核对其与2026-11-07完成的54项实验模型代码快照一致。

**结论：存在实质差异。DDDE/双KL数学实现和ViT骨干相同；多尺度融合结构不同。本地两次BACN实验不是远端SegFormerVectorMFF完整模型的同实现复现。**

## 文件核对范围

远端固定提交共有333个文件：47个字节完全一致，269个仅换行格式不同，17个存在实质文本差异。远端所有文件在本地都有对应路径；此外本地有远端不存在的新算法扩展和实验工具。

文件统计排除了只因CRLF/LF变化造成的伪差异。逐文件结果见file_inventory.csv；关键远端源码下载到remote并按Git blob SHA校验，差异文件保存在diffs。

## 算法及实验差异

| 内容 | 本地最近正式实验 | GitHub固定提交 | 影响 |
|---|---|---|---|
| 完整模型入口 | metaexpert_bacn，父类metaexpert保留固定τ基线 | metaexpert中直接集成DDDE及消融开关 | 同名配置不能默认视为同模型 |
| ViT骨干 | vit_tiny_metaexpert，Conv1d、一维信号、双专家 | 代码一致 | 骨干结构相同 |
| 多尺度融合 | lin1–lin8逐级加法与线性/SiLU变换 | SegFormerVectorMFF：各尺度Linear+LayerNorm+SiLU，拼接后压缩 | 主要结构差异，参数布局与初始化不同 |
| 路由及DEA | predict后softmax，再用专家置信度p1/p3重加权 | 相同基本公式 | 融合特征输入不同，学习出的权重可不同 |
| DDDE及τ | cola_ddde.py中的TwoKLDDDE | 同一数学实现，语法树一致 | 都是有效秩估计＋EMA＋均匀/反向先验双对称KL |
| 默认专家/融合目标 | 原专家监督和伪标签损失，原融合目标 | 默认分支相同；增加可选fix_fuse_mask等开关 | 未开额外开关时不应把差异解释成KL公式或默认损失重写 |
| 无DEA实现 | StaticExpertAverage移除聚合模块，原始logits等权平均，融合损失关闭 | 保留并冻结mff/predict，前向绕过并等权平均，融合损失关闭 | 消融目标相同，参数注册/保存与初始化流程不同；未证明逐步数值等价 |
| 其他消融 | 本次为Full/No-DDDE/No-DEA三版本 | 还提供no_mff、no_fuse_loss、fix_fuse_mask | 实验定义需按具体开关核对 |
| 逐类数据设置 | 支持full_train_counts/labeled_counts，局部seed固定抽样与原始sample_indices审计 | 原指数/均匀/反向采样，不读取显式逐类数组 | 直接复制CD场景YAML不能保证得到相同样本数或样本身份 |
| 索引缓存 | 类别名称指纹隔离；本次固定计数分支不依赖缓存 | raw_signal固定缓存目录 | 本地增强了类别排序和实验划分审计 |
| 模型状态保存 | DDDE状态＋bacn_module_settings，独立评估显式restore_ddde_state | DDDE状态＋cola_settings；hat/tau为register_buffer，随模型state_dict保存 | 完整checkpoint不能直接互换 |
| 续训处理 | 沿用父类基本保存/加载加DDDE扩展 | 另修正epoch/it保存并携带历史最佳模型 | 对续训路径也有差异，本次54项均从头训练 |
| 实验配置 | r1–r12及CD1–CD6，共54项；另6项L/U对照 | cola_two_kl_24为6场景×4版本，样例lb100/ulb250，γ50/100 | 场景与样本预算不同，已有结果不能直接当作同设置性能对照 |

## 多尺度融合的具体差别

本地融合：

```python
h1 = lin1(feat1)
h2 = lin2(feat2 + h1)
h3 = lin3(feat3 + h2)
h4 = lin4(feat4 + h3)
u1 = lin5(h1)
u2 = lin6(h2 + u1)
u3 = lin7(h3 + u2)
fused = lin8(h4 + u3)
```

GitHub融合：每个尺度独立Linear→LayerNorm→SiLU，再拼接四个投影向量，经过Linear/LayerNorm/SiLU融合。其shortcut没有加入forward输出，源码将该未使用分支冻结。

[远端融合及DEA代码](https://github.com/IMPLab-App/sx_ddde_meta_expert-/blob/324417233b7cfc735b8270ad27df8fbcd4cf7cf3/semilearn/imb_algorithms/metaexpert/metaexpert.py#L21)；本地对应semilearn/imb_algorithms/metaexpert/metaexpert.py第32行起。

## 已确认相同的内容

cola_ddde.py、vit_metaexpert.py、algorithmbase.py、datasets/utils.py、train.py经语法树比较一致。默认Head专家监督、两专家无监督目标以及原融合交叉熵表达式均在远端默认分支中匹配；Tail监督的动态τ由本地包装类同步，远端直接取cola.tau_dyn。两边融合损失均在epoch>est_epoch+1后启用。

双KL映射均为τ=τ_max×softmax(−[D_uni,D_rev])[1]，D为对称KL，对上一epoch有效批次取均值。不存在一边用KL、一边用另一种散度的差别。网络或样本选择不同仍可能产生不同的DDDE分布及τ轨迹。

[远端DDDE代码](https://github.com/IMPLab-App/sx_ddde_meta_expert-/blob/324417233b7cfc735b8270ad27df8fbcd4cf7cf3/semilearn/imb_algorithms/metaexpert/cola_ddde.py#L49)。

## 数据与评估的边界

两边真实数据入口均为raw_signal_dataset_G_1024.npz，基础70/10/20划分、训练池归一化及增强逻辑一致。远端README仍描述2048长度和另一个文件名，与实际loader常量不一致，应以代码为准。远端提交没有NPZ数据文件，无法确认两边实际信号数据字节相同。

本地评估增加逐样本概率/标签导出，并针对metaexpert_bacn显式恢复DDDE/τ。远端hat/tau是模型buffer，会随EMA模型state_dict加载；不能仅因远端没有本地restore_ddde_state方法就断言其独立评估一定固定τ=4。两边评估主预测规则仍使用路由权重融合专家原始logits。

双方共同存在的历史YAML没有实质内容差异；本地实际54项正式配置属于新增实验目录，不能与远端历史YAML混为一谈。

## 其他文件差异

- .gitignore
- COLA_TWO_KL_24_REPORT.md
- README.md
- cola_two_kl_queue.py
- eval_confusion_matrix.py
- eval_confusion_matrix_new.py
- run_cola_two_kl_24.sh
- semilearn/datasets/cv_datasets/raw_signal.py
- semilearn/imb_algorithms/bacon/bacon.py
- semilearn/imb_algorithms/cossl/cossl.py
- semilearn/imb_algorithms/cossl/utils.py
- semilearn/imb_algorithms/cpe/cpe.py
- semilearn/imb_algorithms/metaexpert/__init__.py
- semilearn/imb_algorithms/metaexpert/metaexpert.py
- semilearn/nets/__init__.py
- summarize_cola_two_kl_24.py
- tests/test_cola_queue.py

Bacon/CoSSL/CPE等对比方法和网络注册也有局部变化，这些不在本次MetaExpert主模型的执行路径内。

## 对目前结果的解释

当前两次r1–r12/CD1–CD6结果对应“线性多尺度融合＋DDDE＋双KL自适应τ＋DEA”的本地完整模型。若论文方法图或描述指定SegFormerVectorMFF，应先统一实现再验证性能，不能直接把本地结果标为远端SegFormer融合版本。

本次仅进行源码/配置/状态格式对比，未执行远端训练代码，也未改动本地训练实现。差异本身不足以证明远端模型更好或更差。
