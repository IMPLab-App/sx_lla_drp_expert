"""Finish r12 attribution after validation-only temperature selection."""
import argparse
import csv
import json
from pathlib import Path
import time
import zipfile

import numpy as np


def summarize(batch, output):
    from r12_temperature_experiment import report
    report(batch)
    m=json.loads((batch/'manifest.json').read_text())
    selection=json.loads((batch/'selection.json').read_text())
    comparisons=[]
    reference=Path(m['reference_batch'])
    for version in ('V2','V5'):
        old=reference/'models'/f'run_12_r12_pic_lb35_100_ulb315_100_exp__{version}'/'variant_history.jsonl'
        task=next(t for t in m['tasks'] if t['version']==version and t['temperature']==1)
        new=Path(task['output'])/'variant_history.jsonl'
        a=[json.loads(s) for s in old.open()]
        b=[json.loads(s) for s in new.open()]
        equal=len(a)==len(b)
        max_difference=0.
        for r,s in zip(a,b):
            equal &= all(r[k]==s[k] for k in ('event','epoch','iteration','tau_used','pi_estimated','pi_raw'))
            max_difference=max(max_difference,abs(r['tau_used']-s['tau_used']),
                float(np.max(np.abs(np.asarray(r['pi_estimated'])-np.asarray(s['pi_estimated'])))))
        comparisons.append(dict(version=version,history_bitwise_equal=bool(equal),
                                max_tau_or_pi_difference=max_difference,original_records=len(a),replay_records=len(b)))
    (batch/'replay_history_comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    baseline=next(t for t in m['tasks'] if t['version']=='V5' and t['temperature']==1)
    val1=json.loads((Path(baseline['output'])/'validation_metrics.json').read_text())
    selected=selection['validation']
    test=json.loads((Path(selection['output'])/'test_metrics.json').read_text())
    gain=(selected['macro_f1']-val1['macro_f1'])*100
    text=(batch/'RESULTS.md').read_text()
    text+='\n## 诊断判断\n\n'
    text+=f'验证集选择T={selection["temperature"]}；相对T=1，验证Macro-F1变化{gain:+.2f}个百分点。\n\n'
    if selection['temperature']!=1 and gain>0:
        text+='同一DDDE和对称KL下，仅更改映射温度即可改善验证表现，说明映射尺度是一个影响因素；单个场景、单个种子不足以证明它是唯一或主要原因。\n\n'
    else:
        text+='当前温度网格未提供温度缩放改善验证表现的证据，不能把V5问题主要归因于映射尺度。\n\n'
    text+='复现日志与历史日志的比较：\n\n'
    for row in comparisons:
        text+=f'- {row["version"]}：历史状态完全一致={row["history_bitwise_equal"]}；tau或pi最大差={row["max_tau_or_pi_difference"]:.6g}。\n'
    text+='\n历史DDDE分析与带计数复现日志分别保留；不将重新采集的有效秩或伪标签数量标为原始历史记录。\n'
    text+='\n温度在测试前按验证集冻结；没有评估其余候选的测试集。当前测试集此前已用于既有实验，结果不是全新外部测试证据。\n'
    (batch/'RESULTS.md').write_text(text)
    output.mkdir(parents=True,exist_ok=True)
    (output/'COMPLETE_ANALYSIS.md').write_text(text)
    (output/'replay_history_comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in output.rglob('*'):
            if p.is_file(): z.write(p,'historical_and_analysis/'+str(p.relative_to(output)))
        for p in (batch/'analysis').rglob('*'):
            if p.is_file(): z.write(p,'instrumented_replays/'+str(p.relative_to(batch/'analysis')))
        for name in ('selection.json','RESULTS.md','replay_history_comparison.json','manifest.json'):
            z.write(batch/name,'experiment/'+name)
    print('Completed analysis:',output/'COMPLETE_ANALYSIS.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    parser.add_argument('--output',type=Path,default=Path('analysis/r12_ddde_temperature_20261109'))
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    while True:
        status=json.loads((args.batch/'status.json').read_text())['state']
        if status=='completed':break
        if status=='failed':raise RuntimeError('Experiment failed; inspect status.json and issue logs')
        if not args.watch:raise RuntimeError('Experiment not yet complete')
        time.sleep(10)
    summarize(args.batch.resolve(),args.output.resolve())
