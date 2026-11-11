"""Prepare the six original L/U scenarios with the current complete model."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

from prepare_metaexpert_lb35 import CASES as LEGACY_CASES, digest as base_digest, prepare as prepare_base

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT / 'logs/metaexpert_lb35_six_20261103'
FULL_REFERENCE = ROOT / 'logs/bacn_ablation_22_20261105'
CASES = [name.replace('ulb1162', 'ulb1170') for name in LEGACY_CASES]


def digest(path):
    return base_digest(Path(path))


def expected_counts(case):
    ratio = 5 if case % 2 else 10
    labeled = [int(35 * np.power(ratio, -c/8)) for c in range(9)]
    unlabeled = ([130]*9 if case in (3,4)
                 else [int(315 * np.power(ratio, -c/8)) for c in range(9)])
    if case >= 5:
        unlabeled = unlabeled[::-1]
    return labeled, unlabeled


def prepare(batch, smoke=False):
    batch = prepare_base(batch, smoke)
    code = batch/'code'
    reference_hashes = json.loads((FULL_REFERENCE/'source_sha256.json').read_text())
    # Check the actual current implementation, not merely its YAML name.
    model_hashes = {k:v for k,v in reference_hashes.items()
                    if k.startswith('semilearn/') or k in (
                        'train.py','eval_metaexpert_lb35.py','eval_confusion_matrix_new.py')}
    for name, expected in model_hashes.items():
        assert digest(code/name) == expected, f'Current model differs from verified Full: {name}'
    extras = ('prepare_metaexpert_full_lb35.py','verify_metaexpert_full_lb35.py',
              'run_metaexpert_full_lb35.py','verify_bacn_ablation.py','prepare_bacn_ablation.py')
    for name in extras:
        shutil.copy2(ROOT/name,code/name)
    assert digest(PREVIOUS/'data/raw_signal_dataset_G_1024.npz') == digest(batch/'data/raw_signal_dataset_G_1024.npz')
    m=json.loads((batch/'manifest.json').read_text())
    sample_rows=[]
    for task in m['tasks']:
        config=Path(task['config']);cfg=yaml.safe_load(config.read_text())
        previous_config=PREVIOUS/'configs'/config.name
        name=CASES[task['case']-1]
        labeled,unlabeled=expected_counts(task['case'])
        cfg.update(imb_algorithm='metaexpert_bacn',ablation_no_ddde_kl=False,
                   ablation_no_dea_aggregator=False,save_name=name,
                   full_train_counts=[a+b for a,b in zip(labeled,unlabeled)],labeled_counts=labeled)
        if task['case'] in (3,4):cfg['ulb_num_labels']=1170
        new_config=config.with_name(name+'.yaml')
        if new_config!=config:config.unlink()
        config=new_config
        config.write_text(yaml.safe_dump(cfg,sort_keys=False))
        task.update(method='full',variant='full',name=name,case_name=name,
                    config=str(config),output=str(batch/'models'/name),
                    labeled_counts=labeled,unlabeled_counts=unlabeled,
                    previous_config=str(previous_config),previous_config_sha256=digest(previous_config),
                    commands=[[str(code/'train.py'),'--c',str(config)],
                              [str(code/'eval_metaexpert_lb35.py'),'--c',str(config)],
                              [str(code/'verify_metaexpert_full_lb35.py'),str(batch),'--check-task',name]])
        for c in range(9):
            sample_rows.append(dict(case=task['case'],name=task['name'],class_id=c,
                                    class_name=m['class_names'][c],labeled=labeled[c],
                                    unlabeled=unlabeled[c],full=labeled[c]+unlabeled[c],
                                    validation=50,test=100))
    m.update(title='Current Full MetaExpert: six lb35 Consistent/Uniform/Reverse tasks',
             methods=['full'],previous_batch=str(PREVIOUS),full_reference_batch=str(FULL_REFERENCE),
             reference_model_sha256=model_hashes,
             previous_manifest_sha256=digest(PREVIOUS/'manifest.json'))
    m['protocol'].update(
        implementation='Current metaexpert_bacn Full: DDDE + two-KL adaptive tau + DEA',
        ddde_enabled=True,adaptive_tau_enabled=True,dea_enabled=True,
        deterministic_exact_count_sampling=True,ulb_balanced_total=1170,
        split='seed=2, 350/50/100 per class; deterministic fixed-count L/U',
        split_limitation='One source record per class; within-record evaluation only',
        fixed_test_samples=900,labeled_unlabeled_disjoint=True)
    (batch/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
    with (batch/'sample_counts.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(sample_rows[0]));writer.writeheader();writer.writerows(sample_rows)
    (batch/'source_sha256.json').write_text(json.dumps(
        {str(p.relative_to(code)):digest(p) for p in code.rglob('*') if p.is_file()},indent=2)+'\n')
    (batch/'config_sha256.json').write_text(json.dumps(
        {t['name']:digest(t['config']) for t in m['tasks']},indent=2)+'\n')
    (batch/'cache_sha256.json').write_text(json.dumps(
        {str(p.relative_to(batch/'data')):digest(p) for p in (batch/'data').rglob('*.npy')},indent=2)+'\n')
    (batch/'setup_issues.md').write_text(
        '# 配置核对与处理记录\n\n'
        '- 旧六任务模板默认imb_algorithm=metaexpert，不包含DDDE/自适应τ；新批次显式改为metaexpert_bacn，两个消融开关均false。\n'
        '- 当前模型、数据加载和训练代码已与先前完整模型逐文件比对一致，未替换为历史SegFormer结构。\n'
        '- 旧采样入口默认load_exist=False，会忽略缓存并重新采样；已改用现有固定逐类计数分支，未修改模型或数据加载源码。六任务的验证/测试仍与旧实验相同，L/U在当前批次固定。\n'
        '- 准备阶段发现缓存路径通配符同时命中NPZ数据文件，已增加is_dir过滤；失败的准备产物和异常记录归档到logs/metaexpert_full_lb35_setup_issues_20261107。\n'
        '- 准备阶段发现旧摘要函数只接受Path对象，已在新入口统一转换字符串路径；未改模型或采样逻辑。\n'
        '- 按用户最新要求，任务3/4无标签总量改为1170=9×130，任务名及配置同步更新；反向分布仅反转U，L保持正向长尾。\n'
        '- 训练后自动重载最佳模型并独立测试900样本，再检查checkpoint模块设置、有限性及预测指标。\n'
        '- 运行故障记录在issues_gpu0.jsonl/issues_gpu1.jsonl，完整日志在gpu0/gpu1；失败输出归档到failed_attempts后最多自动重试一次。\n'
        '- 六项真实短训练/重载/测试和独立GPU动态τ/融合梯度审计均通过后，才允许启动正式队列。\n')
    (batch/'README.md').write_text(
        '# 当前完整模型：六任务\n\n'
        'DDDE＋双KL自适应τ＋DEA，imb_algorithm=metaexpert_bacn，两个消融开关均false。\n'
        'GPU0依次run_1→run_3→run_5；GPU1依次run_2→run_4→run_6，各3项。\n'
        '沿用100 epochs、20,000步、batch=64、uratio=2、seed=2、est_epoch=18。L/U使用确定性逐类固定计数，验证/测试与旧六任务相同。\n'
        '任务3/4的U为1170=9×130，已按用户最新要求修改任务名；相同比例的L在正向/均匀/反向U三场景保持相同样本身份。\n'
        '样本数量见sample_counts.csv，设置与问题记录见setup_issues.md，验证证据见verification.json。\n'
        '正式训练日志在gpu0/gpu1，模型和独立900样本评估在models，实时结果在summary/comparison.md。\n'
        'smoke批次仅验证流程；正式批次采用冻结代码和配置，独立后台运行，互斥锁防止重复启动。\n')
    print(f'Enabled Full model for six tasks: {batch}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();prepare(args.batch,args.smoke)
