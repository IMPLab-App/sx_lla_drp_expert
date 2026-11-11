# 官方Meta-Expert、本地与sx仓库的多尺度融合比较

官方固定提交：`b728ddb505b41c2546b0033fbed4e6af6a4a3c1b`；sx固定提交：`324417233b7cfc735b8270ad27df8fbcd4cf7cf3`。本地为已完成BACN实验使用的当前模型。

**如果判断标准是忠实于官方Meta-Expert多尺度融合结构，本地更符合：八个线性层的定义表达式和八步前向计算均与官方一致。sx仓库替换成了独立投影、LayerNorm、拼接压缩的融合头。**

上述一致性通过AST逐项比较验证，忽略空白和注释；不代表具体权重尺寸、骨干或整个模型一致。

| 项目 | 官方Meta-Expert | 当前本地 | sx_ddde_meta_expert- |
|---|---|---|---|
| 融合模块 | lin1–lin8 | lin1–lin8，同官方算子模板 | SegFormerVectorMFF |
| 合并方式 | 两轮逐级相加及Linear/SiLU变换 | 同官方八步公式 | 四个尺度独立Linear/LayerNorm/SiLU后拼接，再融合 |
| 融合内部LayerNorm | 无 | 无 | 有 |
| 输出维度 | 2*C4 | 2*C4；当前384维 | 2*C4；当前384维 |
| 特征来源 | 官方WRN的conv及不同block特征，经池化为向量 | ViT第3/6/9/12层的CLS向量 | 相同ViT中间CLS向量 |
| 通道配置 | WRN为[16,16*k,32*k,64*k] | 当前为[192,192,192,192] | 当前为[192,192,192,192] |
| 门控输入 | 融合特征+三个专家类别概率 | 融合特征+两个专家类别概率 | 融合特征+两个专家类别概率 |
| 门控输出 | 三个权重 | 两个权重 | 两个权重 |
| 与官方融合关系 | 原始实现 | 保留原结构并适配骨干 | 替换原融合头的改进实现 |

## 官方与本地相同的融合计算

```python
feat11 = self.lin1(feat1)
feat22 = self.lin2(feat2 + feat11)
feat33 = self.lin3(feat3 + feat22)
feat44 = self.lin4(feat4 + feat33)

feat111 = self.lin5(feat11)
feat222 = self.lin6(feat22 + feat111)
feat333 = self.lin7(feat33 + feat222)
feat444 = self.lin8(feat44 + feat333)
```

每个lin为Linear→SiLU。具体矩阵尺寸取决于backbone.channels；表达式相同不能直接推断官方WRN与本地ViT的参数形状相同。

[官方层定义](https://github.com/Yaxin-ML/Meta-Expert/blob/b728ddb505b41c2546b0033fbed4e6af6a4a3c1b/semilearn/imb_algorithms/metaexpert/metaexpert.py#L29)；[官方前向计算](https://github.com/Yaxin-ML/Meta-Expert/blob/b728ddb505b41c2546b0033fbed4e6af6a4a3c1b/semilearn/imb_algorithms/metaexpert/metaexpert.py#L109)。本地对应semilearn/imb_algorithms/metaexpert/metaexpert.py第32、111行。

## sx替换融合头

每个尺度独立Linear(C_i,D)→LayerNorm(D)→SiLU；四个投影结果拼接成4D，再通过Linear(4D,2C4)→LayerNorm→SiLU→Dropout→Linear(2C4,2C4)→LayerNorm→SiLU，最后额外SiLU输出。当前D=C4=192，因此拼接768维，输出384维。

拼接前各尺度分别保留，后续通过学习到的融合矩阵联合组合。官方/本地则在各阶段通过元素相加逐步混合跨层信息，计算路径不同。sx的shortcut定义后未参与forward输出，源码将其冻结。

[sx融合实现](https://github.com/IMPLab-App/sx_ddde_meta_expert-/blob/324417233b7cfc735b8270ad27df8fbcd4cf7cf3/semilearn/imb_algorithms/metaexpert/metaexpert.py#L21)。

## 什么意义下“最符合”

- 若目标是保留官方Meta-Expert的融合结构：保留本地lin1–lin8最符合。
- 若目标是实现SegFormer式独立投影、拼接压缩的向量融合头：sx实现符合这个改动目标，应写成对原融合头的替换。
- 若目标是航空柱塞泵诊断性能最好：仅看源码无法判断，需要固定数据划分、专家数、DDDE/τ、损失及训练预算后专门比较两种融合头。

## 整体复现的边界

官方WRN的特征来自不同CNN层级，具有不同空间分辨率/通道数，再经池化进入融合头。本地ViT从第3/6/9/12层抽取归一化CLS向量，维度相同且token序列没有逐阶段下采样，因此更精确的表述是多深度/多层向量特征融合。

官方有Head/Middle/Tail三个专家，门控输入为2*C4+3*K、输出3；本地与sx为两个专家，门控输入2*C4+2*K、输出2。当前9类ViT下门控输入402维。

因此本地保留了官方融合计算结构，但改了骨干、特征来源、专家数量，并增加DDDE/双KL自适应τ，不能声称整个模型完全复现官方。当前结果可描述为基于原Meta-Expert融合结构的ViT双专家扩展。

本次未改动训练代码，未执行外部训练；逐项AST验证见three_way_fusion_verification.json。
