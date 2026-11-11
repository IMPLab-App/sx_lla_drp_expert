"""Freeze a fresh 18-scenario x 3-variant sweep after the six Full runs."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil

import yaml

ROOT=Path(__file__).resolve().parent
VARIANTS=('full','no_ddde_kl','no_dea')
SCENARIOS=[f'r{i}' for i in range(1,13)]+[f'cd{i}' for i in range(1,7)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(batch,dependency):
    batch,dependency=batch.resolve(),dependency.resolve()
    parent=json.loads((dependency/'manifest.json').read_text())
    assert len(parent['tasks'])==6 and parent['protocol']['ddde_enabled'] and not parent['smoke']
    old=ROOT/'logs/bacn_11_20261105';high=ROOT/'logs/bacn_r6_r12_21_20261106'
    om=json.loads((old/'manifest.json').read_text());hm=json.loads((high/'manifest.json').read_text())
    sources=[(old,t) for t in om['tasks'][:5]]
    sources += [(high,t) for t in hm['tasks'] if t['variant']=='full']
    sources += [(old,t) for t in om['tasks'][5:]]
    assert len(sources)==18
    batch.mkdir(parents=True,exist_ok=False)
    code=batch/'code';shutil.copytree(Path(parent['code_snapshot']),code,
        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    parent_hashes=json.loads((dependency/'source_sha256.json').read_text())
    for name,expected in parent_hashes.items():assert digest(code/name)==expected,name
    for name in ('prepare_bacn_18.py','verify_bacn_18.py','wait_bacn_18.py'):
        shutil.copy2(ROOT/name,code/name)
    data=batch/'data';data.mkdir();source=Path(parent['data_file']);shutil.copy2(source,data)
    assert digest(data/source.name)==parent['data_sha256']
    configs=batch/'configs';configs.mkdir();tasks=[];counts=[]
    for variant in VARIANTS:
        for case,((reference,source_task),scenario) in enumerate(zip(sources,SCENARIOS),1):
            cfg=yaml.safe_load(Path(source_task['config']).read_text())
            stem=re.sub(r'^run_\d+_',f'run_{case}_',source_task['case_name'])
            name=stem+'__'+variant
            cfg.update(save_name=name,save_dir=str(batch/'models'),data_dir=str(data),
                       imb_algorithm='metaexpert' if variant=='no_ddde_kl' else 'metaexpert_bacn',
                       overwrite=False,resume=False,load_path=None)
            cfg.pop('ablation_no_ddde_kl',None);cfg.pop('ablation_no_dea_aggregator',None)
            if variant!='no_ddde_kl':
                cfg.update(ablation_no_ddde_kl=False,ablation_no_dea_aggregator=variant=='no_dea')
            path=configs/(name+'.yaml');path.write_text(yaml.safe_dump(cfg,sort_keys=False))
            if reference==old:
                smoke=ROOT/('logs/bacn_11_smoke_20261105' if variant=='no_ddde_kl'
                            else 'logs/bacn_ablation_smoke_20261105')
                smoke_name=source_task['name']+('' if variant=='no_ddde_kl' else '__'+variant)
            else:
                smoke=ROOT/'logs/bacn_r6_r12_smoke_20261106'
                smoke_name=source_task['case_name']+'__'+variant
            task=dict(case=case,scenario=scenario,case_name=stem,name=name,
                      method=variant,variant=variant,repeat=1,seed=2,gpu=(case-1)%2,
                      config=str(path),output=str(batch/'models'/name),
                      full_counts=source_task['full_counts'],labeled_counts=source_task['labeled_counts'],
                      unlabeled_counts=source_task['unlabeled_counts'],
                      source_config=source_task['config'],source_config_sha256=digest(source_task['config']),
                      reference_indices=str(reference/'split_indices'/(source_task['name']+'.npz')),
                      historical_smoke_batch=str(smoke),historical_smoke_name=smoke_name,
                      commands=[[str(code/'train.py'),'--c',str(path)],
                                [str(code/'eval_metaexpert_lb35.py'),'--c',str(path)],
                                [str(code/'verify_bacn_18.py'),str(batch),'--check-task',name]])
            tasks.append(task)
            if variant=='full':
                for c in range(9):
                    counts.append(dict(scenario=scenario,case=case,class_id=c,
                        class_name=parent['class_names'][c],full=task['full_counts'][c],
                        labeled=task['labeled_counts'][c],unlabeled=task['unlabeled_counts'][c],
                        validation=50,test=100,target_ratio=cfg['lb_imb_ratio'],
                        actual_full_ratio=max(task['full_counts'])/min(task['full_counts'])))
    m=dict(title='BACN r1-r12 + CD1-CD6: fresh Full and two ablations (54 tasks)',
        created=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        code_snapshot=str(code),data_file=str(data/source.name),data_sha256=parent['data_sha256'],
        class_names=parent['class_names'],methods=list(VARIANTS),tasks=tasks,smoke=False,
        dependency_batch=str(dependency),dependency_manifest_sha256=digest(dependency/'manifest.json'),
        parent_source_sha256=parent_hashes,reference_model_sha256=parent['reference_model_sha256'],
        protocol=dict(repeats=1,seeds=[2],epochs=100,train_steps=20000,
            scenarios=SCENARIOS,gpu_tasks={'0':27,'1':27},fresh_training=True,
            fixed_test_samples=900,minimum_labeled_per_class=1,
            split='Existing seed=2, per-class 350 training / 50 validation / 100 test',
            split_limitation=parent['protocol']['split_limitation'],
            no_ddde_definition='Original MetaExpert; fixed tail tau=4; retain DEA and fusion losses',
            no_dea_definition='Remove entire dynamic aggregation and dedicated fusion losses; raw expert logits averaged 0.5/0.5; retain DDDE/adaptive tau',
            dependency='Start only after all six Full tasks finish training, held-out test and artifact audit',
            smoke_proof='Reuse the exact 54 matching completed historical short train/reload/test checks; no GPU used while predecessor trains'))
    (batch/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
    for file,value in [('source_sha256.json',{str(p.relative_to(code)):digest(p) for p in code.rglob('*') if p.is_file()}),
                       ('config_sha256.json',{t['name']:digest(t['config']) for t in tasks})]:
        (batch/file).write_text(json.dumps(value,indent=2)+'\n')
    with (batch/'sample_counts.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(counts[0]));writer.writeheader();writer.writerows(counts)
    (batch/'README.md').write_text(
        '# 18场景×3版本：54项全新正式训练\n\n'
        'r1–r12、CD1–CD6逐类数量与此前对应实验一致。完整模型、无DDDE/自适应τ、无DEA各18项。\n'
        '全部从头训练，seed=2、100 epochs、20,000步、batch=64、uratio=2，未复用任何正式模型结果。\n'
        'GPU0运行奇数场景，GPU1运行偶数场景；每卡先完整模型9项，再无DDDE9项，再无DEA9项，共27项。\n'
        '先等待当前6项完整模型训练、900样本独立测试及产物审计全部成功；准备和等待期间不启动额外GPU进程。\n'
        '沿用高长尾每类至少1标签；取整后的实际比值见sample_counts.csv，目标80/90/100实际头尾比均87.5。\n'
        '代码、数据、配置固定；逐任务日志在gpu0/gpu1，模型在models，状态在state，比较表在comparison。\n'
        '故障写入issues_gpu*.jsonl及supervisor_issues.jsonl，失败输出归档后最多自动重试一次。\n'
        '监督器使用互斥锁，依赖失败或调度器异常退出时记录错误并阻止新队列启动。\n')
    print(f'Prepared fresh {len(tasks)} tasks, 27 per GPU: {batch}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    parser.add_argument('--dependency',type=Path,default=ROOT/'logs/metaexpert_full_lb35_six_20261107')
    args=parser.parse_args();prepare(args.batch,args.dependency)
