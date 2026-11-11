"""Prepare the requested seven BACN and six L/U cases, three variants each."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil

import yaml

ROOT=Path(__file__).resolve().parent
VARIANTS=('full','no_ddde_kl','no_dea')
SCENARIOS=('r2','r4','r5','r7','r9','cd2','cd3','lu1','lu2','lu3','lu4','lu5','lu6')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(batch,smoke=False):
    batch=batch.resolve();reference=ROOT/'logs/bacn_18_54_20261107'
    six=ROOT/'logs/metaexpert_full_lb35_six_20261107'
    rm=json.loads((reference/'manifest.json').read_text());sm=json.loads((six/'manifest.json').read_text())
    assert (reference/'status.txt').read_text().strip()=='completed'
    assert (six/'status.txt').read_text().strip()=='completed'
    for name,expected in rm['reference_model_sha256'].items():assert digest(ROOT/name)==expected,name
    sources=[(reference,next(t for t in rm['tasks'] if t['scenario']==s and t['variant']=='full')) for s in SCENARIOS[:7]]
    sources += [(six,t) for t in sm['tasks']]
    assert len(sources)==13
    batch.mkdir(parents=True,exist_ok=False);code=batch/'code'
    shutil.copytree(Path(rm['code_snapshot']),code,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    source_hashes=json.loads((reference/'source_sha256.json').read_text())
    for name,expected in source_hashes.items():assert digest(code/name)==expected,name
    for name in ('prepare_selected13.py','verify_selected13.py','run_selected13.py'):
        shutil.copy2(ROOT/name,code/name)
    data=batch/'data';data.mkdir();source=Path(rm['data_file']);shutil.copy2(source,data)
    assert digest(data/source.name)==rm['data_sha256']
    configs=batch/'configs';configs.mkdir();tasks=[];count_rows=[]
    for case,((parent,old),scenario) in enumerate(zip(sources,SCENARIOS),1):
        original=yaml.safe_load(Path(old['config']).read_text())
        assert original['imb_algorithm']=='metaexpert_bacn'
        assert not original['ablation_no_ddde_kl'] and not original['ablation_no_dea_aggregator']
        full=original['full_train_counts'];labeled=original['labeled_counts']
        unlabeled=[n-l for n,l in zip(full,labeled)]
        stem=old['case_name']
        for variant in VARIANTS:
            cfg=dict(original);name=stem+'__'+variant
            cfg.update(save_name=name,save_dir=str(batch/'models'),data_dir=str(data),
                       imb_algorithm='metaexpert' if variant=='no_ddde_kl' else 'metaexpert_bacn',
                       overwrite=False,resume=False,load_path=None)
            cfg.pop('ablation_no_ddde_kl',None);cfg.pop('ablation_no_dea_aggregator',None)
            if variant!='no_ddde_kl':
                cfg.update(ablation_no_ddde_kl=False,ablation_no_dea_aggregator=variant=='no_dea')
            if smoke:
                cfg.update(epoch=3,num_train_iter=6,num_eval_iter=2,num_log_iter=1,
                    batch_size=8,eval_batch_size=128,num_workers=0,est_epoch=0,use_tensorboard=False)
            path=configs/(name+'.yaml');path.write_text(yaml.safe_dump(cfg,sort_keys=False))
            tasks.append(dict(case=case,scenario=scenario,case_name=stem,name=name,
                variant=variant,method=variant,repeat=1,seed=2,gpu=len(tasks)%2,
                config=str(path),output=str(batch/'models'/name),
                full_counts=full,labeled_counts=labeled,unlabeled_counts=unlabeled,
                source_config=old['config'],source_config_sha256=digest(old['config']),
                reference_indices=str(parent/'split_indices'/(old['name']+'.npz')),
                commands=[[str(code/'train.py'),'--c',str(path)],
                          [str(code/'eval_metaexpert_lb35.py'),'--c',str(path)],
                          [str(code/'verify_selected13.py'),str(batch),'--check-task',name]]))
        for c in range(9):
            count_rows.append(dict(case=case,scenario=scenario,task=stem,class_id=c,
                class_name=rm['class_names'][c],full=full[c],L=labeled[c],U=unlabeled[c],validation=50,test=100))
    m=dict(title='Selected 13 scenarios: current Full and two ablations (39 tasks)',
        created=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        code_snapshot=str(code),data_file=str(data/source.name),data_sha256=rm['data_sha256'],
        class_names=rm['class_names'],methods=list(VARIANTS),tasks=tasks,smoke=smoke,
        model_reference_batch=str(reference),reference_model_sha256=rm['reference_model_sha256'],
        parent_source_sha256=source_hashes,
        protocol=dict(repeats=1,seeds=[2],epochs=3 if smoke else 100,
            train_steps=6 if smoke else 20000,scenarios=list(SCENARIOS),fresh_training=True,
            task_order='Scenario-major, then Full / No-DDDE / No-DEA; tasks alternate physical GPU0/GPU1',
            gpu_tasks={'0':20,'1':19},fixed_test_samples=900,
            split='seed=2, 350 training pool / 50 validation / 100 test per class; exact fixed N/L',
            split_limitation=rm['protocol']['split_limitation'],
            full_model='Original Meta-Expert lin1-lin8 fusion + ViT dual experts + DDDE/two-KL adaptive tau + DEA',
            no_ddde_definition=rm['protocol']['no_ddde_definition'],
            no_dea_definition=rm['protocol']['no_dea_definition'],
            uniform_unlabeled_total=1170,uniform_unlabeled_per_class=130))
    (batch/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
    for filename,value in [('source_sha256.json',{str(p.relative_to(code)):digest(p) for p in code.rglob('*') if p.is_file()}),
                           ('config_sha256.json',{t['name']:digest(t['config']) for t in tasks})]:
        (batch/filename).write_text(json.dumps(value,indent=2)+'\n')
    with (batch/'sample_counts.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(count_rows[0]));w.writeheader();w.writerows(count_rows)
    (batch/'README.md').write_text(
        '# 指定13场景：39项全新训练\n\n'
        '顺序：r2、r4、r5、r7、r9、CD2、CD3、六个L/U分布任务。\n'
        '每场景依次排Full、No-DDDE/自适应τ、No-DEA；全局任务交替分到GPU0/1，各卡顺序执行，共20/19项。\n'
        '完整模型沿用当前lin1–lin8融合＋ViT双专家＋DDDE/双KL自适应τ＋DEA。\n'
        '无DDDE使用原metaexpert固定τ=4；无DEA移除整条动态聚合及专用融合损失，双专家原始logits固定平均。\n'
        '正式参数沿用seed=2、100 epochs、20,000步、batch=64、uratio=2。全部从头训练，不覆盖旧模型。\n'
        '六任务3/4的U=1170=9×130。所有任务固定L/U数量及原始索引，与各自上一次正式实验一致。\n'
        '启动前检查真实采样与39项短训练/最佳模型重载/900样本独立测试。smoke产物只验证流程。\n'
        '日志在gpu0/gpu1，异常在issues_gpu*.jsonl，失败输出归档到failed_attempts并最多自动重试一次。\n'
        '源代码、配置、数据冻结；每项训练后自动测试及checkpoint/预测有限性审计。\n'
        'case编号1–13按本次场景顺序，lu1–lu6对应六个完整任务名，见manifest与样本表。\n')
    print(f'Prepared {len(tasks)} tasks, GPU0=20/GPU1=19: {batch}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',type=Path)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args();prepare(args.batch,args.smoke)
