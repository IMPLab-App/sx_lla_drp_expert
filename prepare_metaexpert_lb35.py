"""Prepare only the current MetaExpert implementation for the six lb35 cases."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
CASES = ['run_1_pic_lb35_5_ulb315_5_exp', 'run_2_pic_lb35_10_ulb315_10_exp',
         'run_3_pic_lb35_5_ulb1162_1.0_exp', 'run_4_pic_lb35_10_ulb1162_1.0_exp',
         'run_5_pic_lb35_5_ulb315_-5_pxe', 'run_6_pic_lb35_10_ulb315_-10_pxe']
TEMPLATES = ['run_1_pic_lb100_5_ulb250_5_exp', 'run_2_pic_lb100_10_ulb250_10_exp',
             'run_3_pic_lb100_5_ulb315_1.0_exp', 'run_4_pic_lb100_10_ulb315_1.0_exp',
             'run_5_pic_lb100_5_ulb250_-5_pxe', 'run_6_pic_lb100_10_ulb250_-10_pxe']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(batch, smoke=False):
    batch = batch.resolve()
    batch.mkdir(parents=True, exist_ok=False)
    code = batch / 'code'
    for folder in ('semilearn', 'scripts'):
        shutil.copytree(ROOT / folder, code / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('train.py', 'eval_confusion_matrix_new.py', 'publication_plotting.py',
                 'eval_metaexpert_lb35.py', 'run_signal_baselines.py', 'status_signal_baselines.py',
                 'prepare_metaexpert_lb35.py', 'verify_metaexpert_lb35.py'):
        shutil.copy2(ROOT / name, code / name)
    data = batch / 'data'
    data.mkdir()
    source_data = ROOT / 'data/raw_signal_dataset_G_1024.npz'
    shutil.copy2(source_data, data)
    assert digest(data / source_data.name) == digest(source_data)
    with np.load(data / source_data.name, allow_pickle=False) as source:
        class_names = source['class_names'].tolist()
        assert source['data'].shape[1] == 1024 and len(class_names) == 9
        assert np.isfinite(source['data']).all()
    configs = batch / 'configs'
    configs.mkdir()
    tasks = []
    for case, (name, template) in enumerate(zip(CASES, TEMPLATES), 1):
        cfg = yaml.safe_load((ROOT / 'config/custom_sweep' / (template + '.yaml')).read_text())
        assert cfg['imb_algorithm'] == 'metaexpert' and cfg['net'] == 'vit_tiny_metaexpert'
        ratio = 5 if case % 2 else 10
        cfg.update(save_name=name, save_dir=str(batch / 'models'), data_dir=str(data),
                   num_labels=35, lb_imb_ratio=ratio, lb_imb_type='exp',
                   ulb_num_labels=1162 if case in (3, 4) else 315,
                   ulb_imb_ratio=1.0 if case in (3, 4) else (-ratio if case >= 5 else ratio),
                   ulb_imb_type='pxe' if case >= 5 else 'exp',
                   seed=2, gpu=0, num_workers=2, overwrite=False, resume=False, load_path=None,
                   save_eval_confusion=False)
        if smoke:
            cfg.update(epoch=3, num_train_iter=6, num_eval_iter=2, num_log_iter=1,
                       batch_size=8, eval_batch_size=128, num_workers=0,
                       est_epoch=0, use_tensorboard=False)
        config = configs / (name + '.yaml')
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        tasks.append(dict(method='metaexpert', case=case, case_name=name, repeat=1, seed=2,
                          gpu=(case - 1) % 2, name=name, config=str(config),
                          output=str(batch / 'models' / name),
                          commands=[[str(code / 'train.py'), '--c', str(config)],
                                    [str(code / 'eval_metaexpert_lb35.py'), '--c', str(config)]]))
    manifest = dict(title='当前 MetaExpert：lb35 六任务',
                    created=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                    code_snapshot=str(code), data_file=str(data / source_data.name),
                    data_sha256=digest(data / source_data.name), class_names=class_names,
                    methods=['metaexpert'], smoke=smoke, tasks=tasks,
                    protocol=dict(repeats=1, seeds=[2], epochs=3 if smoke else 100,
                                  train_steps=6 if smoke else 20000, num_labels=35,
                                  ulb_head=315, ulb_balanced_total=1162,
                                  gpu_cases={'0': [1, 3, 5], '1': [2, 4, 6]},
                                  implementation='Current workspace code; optimizer coverage repaired'))
    (batch / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    hashes = {str(p.relative_to(code)): digest(p) for p in code.rglob('*') if p.is_file()}
    (batch / 'source_sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
    (batch / 'setup_issues.md').write_text(
        '# 问题与处理\n\n'
        '- 仅运行当前工作区 MetaExpert，不启动八种对比方法，不使用先前基线或消融实验的模型快照。\n'
        '- 发现 MetaExpert 在包入路由/融合层之前构建优化器，导致新层未进入优化器；新增 finalize_model_setup，在 CUDA 迁移后重建优化器并执行完整参数覆盖校验。骨干、模型结构、损失和超参数保持当前代码设置。\n'
        '- 1162 不能整除9，按现有数据代码均匀分配为 [130,129,129,129,129,129,129,129,129]，不截断总数。\n'
        '- 当前数据加载器按类别名称指纹隔离索引缓存；校验脚本按实际缓存目录读取，保留现有防止类别重排混用索引的机制。\n'
        '- 评估增加逐样本预测概率导出，用于核对准确率和 Macro-F1；仍使用当前 eval_confusion_matrix_new.py 的真实融合与测试流程。\n'
        '- 代码、数据、配置均固定到独立批次；失败产物归档后最多重试一次，运行异常写入 issues_gpu*.jsonl。\n')
    (batch / 'README.md').write_text(
        '# 当前 MetaExpert：lb35 六任务\n\n'
        '仅当前自有方法，共6项；GPU0为run_1→run_3→run_5，GPU1为run_2→run_4→run_6，各3项。\n'
        '原配置中仅调整样本数量、比例和独立输出路径；正式训练100 epochs、20,000步、seed=2、batch=64、uratio=2，骨干vit_tiny_metaexpert，imb_algorithm=metaexpert。\n'
        '数据为同一份raw_signal_dataset_G_1024.npz，真实一维输入[1,1024]。lb35是头类数量；ulb315在长尾时是头类数量；均匀时1162为总量，分配130+8×129。\n'
        '当前其他进程已使用两张GPU，本批次保留这些进程并使用独立CUDA进程。\n\n'
        '修复说明见setup_issues.md，数据与配置校验见verification.json。\n'
        '训练和评估日志在gpu0/、gpu1/；模型、测试预测和混淆矩阵在models/；实时结果在summary/comparison.md。\n'
        '本队列运行当前模型；不复用先前对比方法训练代码快照。旧输出完整保留。\n')
    print(f'Prepared current-method {len(tasks)} tasks: {batch}', flush=True)
    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    prepare(args.batch, args.smoke)
