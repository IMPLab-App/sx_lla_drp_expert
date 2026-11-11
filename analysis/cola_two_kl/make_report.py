"""Export scientific figures and a Chinese analysis from verified artifacts."""
import csv
import json
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score,average_precision_score
from analyze_results import OUT,VARIANTS,GROUPS,write_csv


def read(name):
    return list(csv.DictReader((OUT/name).open(encoding='utf-8-sig')))


def value(row,key): return float(row[key])
def pct(x): return f'{float(x)*100:.2f}'
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=220,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def main():
    metrics=read('metrics_by_run.csv');means=read('metrics_means.csv');classes=read('per_class_metrics.csv')
    taus=read('tau_diagnostics.csv');distributions=read('distribution_diagnostics.csv');epochs=read('epoch_history.csv')
    estimates=read('estimator_posthoc.csv');confidence=read('confidence_diagnostics.csv');paired=read('paired_test_diagnostics.csv')
    complement=[]
    for variant in ('full','no_ddde_kl'):
        for run in range(1,7):
            z=np.load(OUT/f'predictions_{variant}_{run}.npz');y=z['labels']
            h=z['head'].argmax(1);t=z['tail'].argmax(1);f=z['fused'].argmax(1)
            hc=h==y;tc=t==y;fc=f==y
            complement.append(dict(variant=variant,run=run,head_tail_agreement=float(np.mean(h==t)),
                tail_rescues_head_errors=int(np.sum(~hc&tc)),head_rescues_tail_errors=int(np.sum(hc&~tc)),
                both_wrong=int(np.sum(~hc&~tc)),label_aware_expert_selector_accuracy=float(np.mean(hc|tc)),
                fusion_rescues_head_errors=int(np.sum(~hc&fc)),fusion_loses_head_correct=int(np.sum(hc&~fc)),
                mean_tail_weight=float(z['w3'].mean())))
    write_csv('expert_complementarity.csv',complement)
    branch=read('branch_diagnostics.csv');geometry=read('feature_geometry.csv')
    metadata=json.loads((OUT/'data_metadata.json').read_text());names=metadata['class_names']
    held=[r for r in metrics if r['split']=='heldout']
    def row(variant,run):return next(r for r in held if r['variant']==variant and int(r['run'])==run)
    a=np.array([value(row('full',i),'accuracy') for i in range(1,7)])
    b=np.array([value(row('no_ddde_kl',i),'accuracy') for i in range(1,7)])
    fa=np.array([value(row('full',i),'macro_f1') for i in range(1,7)])
    fb=np.array([value(row('no_ddde_kl',i),'macro_f1') for i in range(1,7)])
    delta=(a-b)*100;df=(fa-fb)*100
    # Probability quality uses float64 and stable logsumexp, avoiding clipping
    # float32 probabilities when reporting cross-entropy.
    quality=[]
    for variant in ('full','no_ddde_kl'):
        for run in range(1,7):
            z=np.load(OUT/f'predictions_{variant}_{run}.npz');y=z['labels'];logits=z['fused'].astype(np.float64)
            logp=logits-logsumexp(logits,axis=1,keepdims=True);p=np.exp(logp);one=np.eye(9)[y]
            c=p.max(1);correct=p.argmax(1)==y;ece=0.
            for i in range(15):
                mask=(c>=i/15)&((c<(i+1)/15) if i<14 else (c<=1))
                if mask.any():ece+=mask.mean()*abs(correct[mask].mean()-c[mask].mean())
            quality.append(dict(variant=variant,run=run,macro_roc_auc=float(roc_auc_score(y,p,average='macro',multi_class='ovr')),
                macro_average_precision=float(average_precision_score(one,p,average='macro')),
                nll=float(-logp[np.arange(len(y)),y].mean()),brier=float(np.square(p-one).sum(1).mean()),ece15=float(ece)))
    write_csv('probability_metrics.csv',quality)
    p=np.array([float(r['exact_mcnemar_p']) for r in paired]);order=np.argsort(p);adjusted=np.empty(6);running=0
    for rank,index in enumerate(order):
        running=max(running,min(1.,p[index]*(6-rank)));adjusted[index]=running
    for index,r in enumerate(paired):r['holm_adjusted_p']=float(adjusted[index])
    write_csv('paired_test_diagnostics.csv',paired)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(12,8));x=np.arange(6)
    axes[0,0].bar(x-.18,delta,.36,label='Accuracy');axes[0,0].bar(x+.18,df,.36,label='Macro-F1')
    axes[0,0].axhline(0,color='k',lw=.8);axes[0,0].set(xticks=x,xticklabels=range(1,7),xlabel='Run',ylabel='Full - fixed tau (percentage points)',title='A. Gains depend on the scenario');axes[0,0].legend()
    for offset,variant,label,color in [(-.18,'full','Full','#31688e'),(.18,'no_ddde_kl','Fixed tau=4','#d38939')]:
        rec=[np.mean([float(r['recall']) for r in classes if r['variant']==variant and int(r['class_id'])==k])*100 for k in range(9)]
        axes[0,1].bar(np.arange(9)+offset,rec,.36,label=label,color=color)
    axes[0,1].set(ylim=(0,103),xticks=range(9),xticklabels=[f'C{k}' for k in range(9)],ylabel='Recall (%)',title='B. Recall changes concentrate in C3/C4/C6');axes[0,1].legend()
    for offset,key,label in [(-.24,'tv_prediction_hist','Prediction counts'),(0,'tv_selected_hist','Confident counts'),(.24,'tv_rank64_batch_mean','DDDE effective rank')]:
        axes[1,0].bar(x+offset,[value(r,key) for r in estimates],.24,label=label)
    axes[1,0].set(xticks=x,xticklabels=range(1,7),xlabel='Run',ylabel='Total variation from true prior',title='C. Frozen best model: unaugmented train U');axes[1,0].legend(fontsize=8)
    for offset,variant,label in [(-.18,'full','Full'),(.18,'no_ddde_kl','Fixed tau=4')]:
        group=[r for r in complement if r['variant']==variant]
        axes[1,1].bar(x+offset,[value(r,'head_tail_agreement')*100 for r in group],.36,label=label)
    axes[1,1].set(xticks=x,xticklabels=range(1,7),xlabel='Run',ylabel='Head/Tail prediction agreement (%)',ylim=(0,103),title='D. Head/Tail agreement differs by policy');axes[1,1].legend()
    fig.tight_layout();save(fig,'diagnostic_overview')
    fig,axes=plt.subplots(3,2,figsize=(11,10),sharex=True,sharey=True)
    for run,ax in enumerate(axes.flat,1):
        history=[r for r in epochs if int(r['run'])==run];info=taus[run-1]
        ax.plot([int(r['epoch']) for r in history],[value(r,'tau') for r in history],label='Actual dynamic tau')
        ax.axhline(4,color='#d38939',ls='--',label='Fixed tau=4');ax.axhline(value(info,'oracle_tau'),color='grey',ls=':',label='True prior through SAME map')
        ax.axvspan(0,18,color='grey',alpha=.12);ax.scatter(int(info['best_epoch'])-1,value(info,'best_tau'),s=35,color='crimson',zorder=4,label='Selected best checkpoint')
        ax.set(title=f'Run {run}: {info["scenario"]}',ylim=(0,4.15),xlabel='Epoch (zero-based)',ylabel='tau')
    axes[0,0].legend(fontsize=7,loc='lower left');fig.tight_layout();save(fig,'tau_trajectories')
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    for run,ax in zip((5,6),axes):
        records=[r for r in confidence if int(r['run'])==run];hist=[r for r in distributions if int(r['run'])==run]
        for offset,key,label,data in [(-.3,'true_prior','True frequency',records),(-.1,'predicted_prior','Prediction frequency',records),(.1,'simulated_rank_prior','Mean rank estimate (64 batches)',records),(.3,'best_pi_u','Historical EMA at best checkpoint',hist)]:
            ax.bar(np.arange(9)+offset,[value(r,key)*100 for r in data],.2,label=label)
        ax.set(xticks=range(9),xticklabels=[f'C{k}' for k in range(9)],ylabel='Class proportion (%)',title=f'Run {run}: reverse unlabeled distribution')
    axes[0].legend(fontsize=7);fig.tight_layout();save(fig,'distribution_estimation')
    fig,axes=plt.subplots(2,1,figsize=(11,7))
    for variant,ax in zip(('full','no_ddde_kl'),axes):
        z=np.array([[float(next(r for r in classes if r['variant']==variant and int(r['run'])==run and int(r['class_id'])==k)['recall'])*100 for k in range(9)] for run in range(1,7)])
        im=ax.imshow(z,vmin=50,vmax=100,cmap='RdYlGn',aspect='auto')
        ax.set(xticks=range(9),xticklabels=[f'C{k}' for k in range(9)],yticks=range(6),yticklabels=[f'Run {i}' for i in range(1,7)],title=variant)
        for i in range(6):
            for j in range(9):ax.text(j,i,f'{z[i,j]:.0f}',ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=axes,shrink=.7,label='Recall (%)');save(fig,'per_class_recall')
    # Main Chinese report; percentages are means across the six distinct settings.
    lines=['本分析结论：Full 的六组均值略优，但当前 DDDE/KL 并非稳定的逐场景、逐尾类增益。已复现 Full 和 No-DDDE/KL 共 12 个最佳 EMA 模型的原始 heldout 混淆矩阵；以下把实测事实、机制推断和待验证假设分开。没有修改训练代码、模型文件或官方评估结果。','',
    '当前源码与实验时版本存在差异，因此使用 `runtime_snapshot_verified/` 中由原始备份和会话保存差异重建的源码，只提取模型类进行冻结推理。12/12 混淆矩阵严格重现，为本次事后分析提供了版本、数据切分和预处理一致性核验。推理使用 CPU、四线程。','',
    '**1. 指标口径与头尾定义**','',
    'heldout 为 900 个样本，九类各 100。因此 Accuracy、Balanced Accuracy、Macro Recall、Micro-F1 数值相同；Weighted-F1 与 Macro-F1 也相同，不应把这些相等指标重复解释成独立证据。类别准确率按该真类被正确识别的比例计算，即该类 Recall；不使用包含大量真负例的 one-vs-rest Accuracy。组内 F1 是完整九分类各类 F1 的平均值。','',
    '配置 `cut1=1` 的路由分组为 C0（正常）与 C1–C8（故障），并不等价于将类别按数量分成三档，也不意味着 E1/E3 分类器只输出其中部分类别。本报告同时给出：①配置分组，②按有标签训练频数排序分成头 C0–C2、中 C3–C5、尾 C6–C8 三组。这是本分析自定的三等分口径，不声称是两篇参考论文的原始阈值。','',
    '| 类别 | 名称 | 有标签数量：50倍 / 100倍 | 分析频数组 |','|---|---|---:|---|']
    counts1=metadata['runs']['1']['labeled_counts'];counts2=metadata['runs']['2']['labeled_counts']
    for k,name in enumerate(names):lines.append(f'| C{k} | {name} | {counts1[k]} / {counts2[k]} | '+('头' if k<3 else '中' if k<6 else '尾')+' |')
    lines+=['','**2. 六组平均的完整比较（%）**','','| 模式 | Accuracy / BA / Macro-R | Macro-P | Macro-F1 | G-mean Recall | 头类 Recall | 中类 Recall | 尾类 Recall |','|---|---:|---:|---:|---:|---:|---:|---:|']
    mh={r['variant']:r for r in means if r['split']=='heldout'}
    for variant in VARIANTS:
        r=mh[variant];keys=['accuracy','macro_precision','macro_f1','gmean_recall','head_3_accuracy','middle_3_accuracy','tail_3_accuracy']
        lines.append('| '+variant+' | '+' | '.join(pct(r[k]) for k in keys)+' |')
    lines+=['','在与有标签先验一致的长尾测试子集上（每组 253 或 224 样本），均值如下。它与完整 heldout 是不同样本集，不能把两者差值全部归因于先验变化；尾类只有 1–5 个样本，类别指标波动很大。','', '| 模式 | Accuracy | Balanced Accuracy / Macro-R | Macro-P | Macro-F1 |','|---|---:|---:|---:|---:|']
    for r in means:
        if r['split']=='imbalanced':lines.append('| '+r['variant']+' | '+' | '.join(pct(r[k]) for k in ['accuracy','balanced_accuracy','macro_precision','macro_f1'])+' |')
    lines+=['','Full 的 Accuracy 不是唯一占优：其平均 AUROC/PR-AUC 略优，但 NLL、ECE 并不优于固定 τ。概率指标是本次冻结模型补算，均值如下：','', '| 模式 | Macro AUROC % | Macro AP % | NLL（低优） | Brier（低优） | ECE-15 %（低优） |','|---|---:|---:|---:|---:|---:|']
    for variant in ('full','no_ddde_kl'):
        group=[r for r in quality if r['variant']==variant];v=lambda k:np.mean([r[k] for r in group])
        lines.append(f'| {variant} | {pct(v("macro_roc_auc"))} | {pct(v("macro_average_precision"))} | {v("nll"):.4f} | {v("brier"):.4f} | {pct(v("ece15"))} |')
    lines+=['','**3. 为什么会有四组 No-DDDE/KL 更高：逐场景看，不能只看平均数**','', '| 组 | 情境 | Full Acc | 固定τ Acc | Full F1 | 固定τ F1 | ΔAcc（百分点） | ΔF1（百分点） | 净多识别样本 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for i in range(6):
        lines.append(f'| {i+1} | {metadata["runs"][str(i+1)]["scenario"]} | {a[i]*100:.2f} | {b[i]*100:.2f} | {fa[i]*100:.2f} | {fb[i]*100:.2f} | {delta[i]:+.2f} | {df[i]:+.2f} | {int(paired[i]["net_correct_difference"]):+d} |')
    lines+=[f'','Full 比固定 τ 平均高 {delta.mean():.2f} 个百分点，主要来自第 5 组的 +{delta[4]:.2f}。去掉第 5 组后，其余五组平均是 {np.delete(delta,4).mean():+.2f}；六组差值中位数为 {np.median(delta):+.2f}。所以“平均略好”不能写成“普遍有效”。',
    '按三类情境配对平均：matched 为 −0.22 个百分点，uniform 为 −0.56，reverse 为 +1.67；reverse 的两组本身也不一致，50 倍时显著受益，100 倍时近乎持平。','',
    '基于同一测试样本的精确 McNemar 检验：第 5 组 p=0.000117，六项 Holm 校正后约 0.000702；其他组 p≥0.296。它只反映这两个固定模型在当前样本上的配对差异，不能代替多训练种子重复；原始采样片段的独立性也不能仅凭这次随机切分保证。所有实验仅 seed=2。','',
    '**4. 头类与尾类：Full 的收益主要来自中频难类**','', '| 频数组 | Full Recall / 组准确率 % | 固定τ Recall % | Full 组 Macro-F1 % | 固定τ 组 Macro-F1 % |','|---|---:|---:|---:|---:|']
    for name,label in [('head_3','头 C0–C2'),('middle_3','中 C3–C5'),('tail_3','尾 C6–C8')]:
        lines.append('| '+label+' | '+' | '.join(pct(mh[v][name+suffix]) for suffix in ['_accuracy','_macro_f1'] for v in ['full','no_ddde_kl'])+' |')
    lines+=['','结论：头类接近饱和；Full 的中频组 Recall 高约 1.39 个百分点，最少样本三类 Recall 反而低约 0.56 个百分点。尾组 Macro-F1 都约 98.35%，几乎相同，这是 Precision/Recall 的权衡，不能只凭某一个指标认定全面提升或全面退步。',
    '若严格按配置 cut1=1，则 C0 正常识别率两者均 100%；C1–C8 故障组识别率 Full 94.40%、固定 τ 94.06%。这与“三等分尾类”结论不同，因此必须明确头尾定义。','',
    '| 类别 | Full P % | 固定τ P % | Full R % | 固定τ R % | Full F1 % | 固定τ F1 % |','|---|---:|---:|---:|---:|---:|---:|']
    for k in range(9):
        group={v:[r for r in classes if r['variant']==v and int(r['class_id'])==k] for v in ['full','no_ddde_kl']}
        values=[np.mean([value(r,key) for r in group[v]]) for key in ['precision','recall','f1'] for v in ['full','no_ddde_kl']]
        lines.append(f'| C{k} {names[k]} | '+' | '.join(pct(v) for v in values)+' |')
    lines+=['','C3 的 Recall 从 82.00% 提高到 87.00%，C4 的 Recall 从 78.50% 降到 77.00%，但其 Precision 从 80.89% 升到 84.05%，所以 C4 的 F1 反而升高。C6 的 Recall 从 92.83% 降到 91.17%，F1 也从 95.38% 降到 95.04%。C7/C8 虽然更少样本，Recall 却几乎都为 100%，说明样本稀少与类别难度不能等同。','',
    '第 4 组的 7 个净错误：Full 在 C3 多识别 7 个，C4 少识别 4 个，C6 少识别 10 个，总计 −7。第 5 组的 31 个净收益：C3 +21、C4 +2、C6 +8，收益来自几个具体的故障边界，而非所有尾类整体抬升。','',
    '**5. 机制证据一：DDDE 的有效秩不是类别频数，逆向情境存在系统偏差**','',
    '16400/16400 个估计阶段 batch 均有有效全局统计，未发现由“整批无高置信样本、未更新 τ、数值溢出”导致本轮差异的证据。但这不等于每一类都有充分统计。',
    '训练历史的 pi_u 与真实无标签先验相比，最佳模型处的 TV 距离（0 为完全一致）见下表。oracle τ 仅指把真实先验代入当前公式的结果，不是最优 τ，也没有用它训练模型。','',
    '| 组 | 最佳模型τ | 训练末τ | 真实先验经当前映射τ | 最佳pi_u的TV误差 |','|---|---:|---:|---:|---:|']
    for r in taus:lines.append(f'| {r["run"]} | {value(r,"best_tau"):.3f} | {value(r,"final_tau"):.3f} | {value(r,"oracle_tau"):.3f} | {value(r,"tv_best"):.3f} |')
    lines+=['','第 5/6 组 C8“弹簧失效”的真实先验分别为 39.31%/44.25%，历史最佳 pi_u 却仅 9.95%/9.79%。在冻结最佳模型上，对实际无标签训练子集做未增强的事后检查：该类的预测召回、0.95 置信覆盖和选中样本精确率均为 100%；按预测数量计数可恢复该类真实占比。64 个大小 128、带放回采样的模拟 batch 以原有效秩公式计算，C8 估计均值又降到 8.68%/11.26%，与历史偏差方向一致。',
    '因此，至少在这些冻结模型上，问题不只是低置信或伪标签错误。中心化 SVD 的有效秩反映类内特征多样性和几何结构，不与样本数量成正比；对样本多但表示集中、简单可分的类别，直接将有效秩归一化为类别比例会严重低估。小类的单样本 rank=1、批采样稀疏和重复样本也会影响估计。','',
    '| 组 | 原始预测计数 TV | 高置信预测计数 TV | 有效秩估计 TV（64批均值） |','|---|---:|---:|---:|']
    for r in estimates:lines.append(f'| {r["run"]} | {value(r,"tv_prediction_hist"):.4f} | {value(r,"tv_selected_hist"):.4f} | {value(r,"tv_rank64_batch_mean"):.4f} |')
    lines+=['','这些是最佳 EMA 模型在未增强训练 U 上的事后诊断，不是训练期间弱增强批次的逐步重放，更不是独立泛化成绩。完整训练历史 pi_u 的实际偏差与模拟结果同时列出，以免混淆。','',
    '**6. 机制证据二：分布估计正确，不等于当前两路 τ 策略最优**','',
    '当前 τ=4·softmax(−D_uni,−D_rev)[1] 只衡量分布相对距离，不直接优化平衡准确率、尾类 Recall 或 Macro-F1，也不感知 C3/C4/C6 的分类边界。去掉 KL_con 后也没有与“有标签同向先验”对应的第三参照。这个定制映射不能套用原始三路 CoLA 或 EnergyMatch 的结论。',
    '第 3/4 组真实无标签分布均匀，当前映射即使输入真实分布也给约 1.39/1.20，而非 4；实际最佳 τ 为 1.36/1.12，接近这个输出，但固定 4 的 Accuracy 仍更高。这直接说明“估计更准”并不自动推出“由该映射给出的 τ 更好”。反过来，oracle τ 本身也不能保证更高准确率。',
    'Tail 的监督目标使用 z_raw+τlog(pi_l)。例如 100 倍有标签不平衡时，C0/C8 之间的先验相对偏移为 τlog(100)：τ=4 时约 18.42，第 4 组最佳 τ≈1.123 时约 5.17。动态策略显著改变了训练时的边界要求；较小 τ 是否合适取决于实际类间可分性与专家互补，而非分布形状单独决定。','',
    '冻结 Full 模型权重，仅在推理前把 τ 临时改为 4，六组共 5400 个预测标签均未改变。因此本轮差异不能靠评估时“改一个 τ”直接修复，更符合训练过程中学到的表示、分类边界和专家分工变化。这个检查没有重新训练，不能被当作“固定 τ=4 训练”的替代实验。','',
    '**7. 机制证据三：专家互补、边界分离与各类难度**','',
    'Full 在第 1/3 组 Head/Tail 预测一致率为 99.78%/99.56%；固定 τ=4 为 90.56%/82.44%。动态 τ 较小时，两专家更趋同，可能减少聚合可利用的互补；不过这种关系不能仅凭六组观察证明单一因果。第 3 组固定 τ 的 Head 单独 Accuracy=97.89%，Tail 单独只有 82.22%，融合却达到 98.11%，说明不能用 Tail 专家整体准确率单独判断其价值。',
    '第 5 组 Full 的 Tail 单独 Accuracy=95.89%，固定 τ 的 Tail=92.22%，最终融合分别为 95.78%/92.33%；该组收益主要已出现在训练出的 Tail 分支，而不是推理时更改 τ 或门控产生。第 6 组固定 τ 的 Tail 对 C6 Recall=94%，融合后仅 86%，也说明现有聚合路径会限制专家优势的兑现。',
    '受 BACN 的类中心/边界视角启发，本报告补算 L2 归一化共享特征的“类中心距离平方 / 两类类内散度之和”（Fisher 分离比，越大越分离；这是本报告诊断量，不声称是论文原表指标）。C3/C4 的分离比第 3 组为 Full 2.34、固定 τ 3.48；第 4 组为 1.04/1.26；第 5 组为 0.95/0.71。六组中该比值变化方向与 Accuracy 差值方向一致，支持以难类边界质量解释表现，仍属于事后相关证据。','',
    '**8. 与两篇论文的关系及可借鉴的实验分析**','',
    '公开来源可访问的内容为摘要/引言和方法概述，没有取得完整实验表或作者代码。下面只引用已经核实的机制和问题设定，不虚构论文具体的消融数值、F1 表、头尾阈值或统计检验。','',
    '[Boundary-aware contrastive network（BACN）](https://www.sciencedirect.com/science/article/pii/S095741742503132X)关注未知长尾及有/无标签先验失配，通过类内聚合、类间分离、基于有效样本数的边界加权及距离向量对比学习，将有限标签约束保留在特征学习中。其公开摘要报告 13 种长尾情形。对本实验的启发是：必须看困难类间的混淆、类中心分离及不同先验情境，不能只看平均 Accuracy；其中“有效样本数”与 DDDE 的 SVD“有效秩”是不同概念。',
    '[EnergyMatch](https://www.sciencedirect.com/science/article/abs/pii/S0951832025012128)面向跨域、未知目标类分布，以双分类器、自由能对齐、高不确定样本移除和动态分布估计改善伪标签。我们没有实现这整条链：弱增强伪标签仍由原始 logits 产生，动态 τ 直接作用于 Tail 监督 LA 和融合输入；数据也未按源/目标工况独立划分。因此这两种设置不能直接比较论文 Accuracy，也不能把其动态分布模块的有效性直接移植到当前映射。',
    '本轮从同一 NPZ 做按类随机 70/10/20 切分，再构造有/无标签先验；这可以分析 label shift，但没有显式的跨工况 feature shift。未记录原始采集批次/机器/工况分组，不能据此断言存在数据泄漏，也不能声称已经排除了片段相关性或验证了跨域泛化。','',
    '**9. 建议的下一轮验证，按优先级排列（本次未启动新训练）**','',
    '1. 先只比较 Full 与固定 τ=4，至少 3–5 个相同种子，报告每组 Macro-F1、Balanced Accuracy、最少样本组 Recall 的均值与标准差，明确训练频数的头尾定义。当前 0.30 个百分点均值收益和四组小幅反向差异不足以证明稳定增益。',
    '2. 拆开“分布估计器”与“τ 决策”：相同训练流程分别使用固定 τ、置信预测计数 EMA、有效秩估计、已知训练 U 真先验的 oracle 诊断。oracle 只用于研究诊断，不能作为未知分布下可部署的方法。所有 τ/混合系数均仅用验证集选择，不能按测试结果调参。',
    '3. 暂不直接废除 DDDE；先检验“有效秩是置信/多样性质量因子”与“类别频率估计”的职责分离。可比较计数主导、有效秩辅助的估计，并记录每类选中数、覆盖率、伪标签精度、rank 和真实先验 TV。EnergyMatch 式不确定性筛选可作为独立对照，但它本身不能修复这里已高置信且预测正确的 C8 被有效秩低估的问题。',
    '4. 参照 BACN 的边界分析，对 C3/C4/C6 做类中心分离、类内散度、对称/方向性混淆、原始波形与频谱分析。若增加类中心约束或边界对比损失，必须作为新模块单独消融，不能把新增收益归到 DDDE/KL。',
    '5. 对真正跨工况任务采用工况/采集记录/机器互斥的数据划分，再借鉴 EnergyMatch 的 feature shift 与 label shift 联合评估。现有均匀测试集与长尾测试子集都应保留，并明确少数类样本数。','',
    '**10. 产物与复现**','',
    '- [指标总表](metrics_by_run.csv)、[均值](metrics_means.csv)、[逐类 P/R/F1](per_class_metrics.csv)。',
    '- [τ/估计分布诊断](tau_diagnostics.csv)、[各类真实/估计分布](distribution_diagnostics.csv)、[逐 epoch 轨迹](epoch_history.csv)。',
    '- [冻结专家分支](branch_diagnostics.csv)、[专家互补](expert_complementarity.csv)、[置信筛选](confidence_diagnostics.csv)、[估计器事后对比](estimator_posthoc.csv)、[特征分离](feature_geometry.csv)。',
    '- [冻结 τ 检查](frozen_tau_counterfactual.csv)、[配对检验](paired_test_diagnostics.csv)、[概率指标](probability_metrics.csv)、[复核来源](inference_provenance.json)。',
    '- 数值 CSV 使用比例 0–1，报告/图使用 %；差异使用百分点。训练种子仍只有 2。','',
    '![综合诊断](diagnostic_overview.png)','', '![逐类召回率](per_class_recall.png)','', '![tau轨迹](tau_trajectories.png)','', '![分布估计偏差](distribution_estimation.png)','',
    '各图同时提供同名 PDF。复现命令：','',
    '```bash\npython analysis/cola_two_kl/analyze_results.py\npython analysis/cola_two_kl/frozen_diagnostics.py --device cpu\npython analysis/cola_two_kl/make_report.py\n```']
    (OUT/'DDDE_KL_analysis_zh.md').write_text('\n'.join(lines)+'\n')
    print('Saved report and 4 figures in PNG/PDF')
    print('GROUP METRICS:')
    for group in ['head_3','middle_3','tail_3']:
        print(group,*[mh[v][group+k] for k in ['_accuracy','_macro_f1'] for v in ['full','no_ddde_kl']])
    print('QUALITY MEANS:')
    for v in ('full','no_ddde_kl'):
        r=[x for x in quality if x['variant']==v];print(v,{k:float(np.mean([x[k] for x in r])) for k in ['nll','ece15','macro_roc_auc','macro_average_precision']})


if __name__=='__main__':main()
