"""Freeze 120 main ablations plus 72 additional static-tau sensitivity runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import yaml

ROOT=Path(__file__).resolve().parent
VERSIONS=('V1','V2','V3','V4','V5','S0','S1','S3')
SCENARIOS=tuple([f'r{i}' for i in range(1,13)]+[f'cd{i}' for i in range(1,7)]+[f'lu{i}' for i in range(1,7)])


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(batch,smoke=False):
    from semilearn.imb_algorithms.metaexpert.five_variants import SPECS
    reference=ROOT/'logs/bacn_18_54_20261107';six=ROOT/'logs/metaexpert_full_lb35_six_20261107'
    rm=json.loads((reference/'manifest.json').read_text());sm=json.loads((six/'manifest.json').read_text())
    sources=[(reference,t) for t in rm['tasks'] if t['variant']=='full']+[(six,t) for t in sm['tasks']]
    assert len(sources)==24
    for name,h in rm['reference_model_sha256'].items():
        if name!='semilearn/imb_algorithms/metaexpert/__init__.py':assert digest(ROOT/name)==h,name
    batch=batch.resolve();batch.mkdir(parents=True,exist_ok=False);code=batch/'code'
    shutil.copytree(Path(rm['code_snapshot']),code,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    overlays=('semilearn/imb_algorithms/metaexpert/__init__.py',
              'semilearn/imb_algorithms/metaexpert/five_variants.py',
              'eval_metaexpert_v1v5.py','prepare_v1v5.py','verify_v1v5.py','verify_v1v5_gpu.py','run_v1v5.py')
    for name in overlays:shutil.copy2(ROOT/name,code/name)
    frozen_model={name:digest(code/name) for name in rm['reference_model_sha256']}
    frozen_model['semilearn/imb_algorithms/metaexpert/five_variants.py']=digest(code/'semilearn/imb_algorithms/metaexpert/five_variants.py')
    data=batch/'data';data.mkdir();source=Path(rm['data_file']);shutil.copy2(source,data)
    assert digest(data/source.name)==rm['data_sha256']
    configs=batch/'configs';configs.mkdir();tasks=[];sample_rows=[]
    for case,((parent,old),scenario) in enumerate(zip(sources,SCENARIOS),1):
        original=yaml.safe_load(Path(old['config']).read_text())
        for j,version in enumerate(VERSIONS):
            spec=SPECS[version];cfg=dict(original);name=old['case_name']+'__'+version
            cfg.update(save_name=name,save_dir=str(batch/'models'),data_dir=str(data),
                imb_algorithm='metaexpert_v1v5',five_variant=version,
                five_fixed_tau=spec['fixed_tau'] if spec['fixed_tau'] is not None else 4.,
                ablation_no_ddde_kl=not spec['dynamic'],ablation_no_dea_aggregator=False,
                overwrite=False,resume=False,load_path=None)
            if smoke:cfg.update(epoch=3,num_train_iter=6,num_eval_iter=2,num_log_iter=1,
                batch_size=8,eval_batch_size=128,num_workers=0,est_epoch=0,use_tensorboard=False)
            path=configs/(name+'.yaml');path.write_text(yaml.safe_dump(cfg,sort_keys=False))
            tasks.append(dict(case=case,scenario=scenario,name=name,case_name=old['case_name'],
                version=version,method=version,repeat=1,seed=2,gpu=(case-1+j)%2,
                config=str(path),output=str(batch/'models'/name),
                full_counts=original['full_train_counts'],labeled_counts=original['labeled_counts'],
                unlabeled_counts=[a-b for a,b in zip(original['full_train_counts'],original['labeled_counts'])],
                source_config=old['config'],source_config_sha256=digest(old['config']),
                reference_indices=str(parent/'split_indices'/(old['name']+'.npz')),
                commands=[[str(code/'train.py'),'--c',str(path)],
                          [str(code/'eval_metaexpert_v1v5.py'),'--c',str(path)],
                          [str(code/'verify_v1v5.py'),str(batch),'--check-task',name]]))
        for c in range(9):sample_rows.append(dict(case=case,scenario=scenario,class_id=c,
            class_name=rm['class_names'][c],full=original['full_train_counts'][c],L=original['labeled_counts'][c],
            U=original['full_train_counts'][c]-original['labeled_counts'][c],validation=50,test=100))
    m=dict(title='24 scenarios: V1-V5 and fixed-tau sensitivity (192 fresh runs)',
        code_snapshot=str(code),data_file=str(data/source.name),data_sha256=rm['data_sha256'],
        class_names=rm['class_names'],methods=list(VERSIONS),tasks=tasks,smoke=smoke,
        reference_model_sha256=frozen_model,base_model_sha256=rm['reference_model_sha256'],
        overlays=list(overlays),version_specs=SPECS,sensitivity_aliases={'0':'S0','1':'S1','2':'V4','3':'S3','4':'V1'},
        protocol=dict(repeats=1,seeds=[2],epochs=3 if smoke else 100,train_steps=6 if smoke else 20000,
            main_tasks=120,additional_sensitivity_tasks=72,total_tasks=192,gpu_tasks={'0':96,'1':96},
            scenarios=list(SCENARIOS),fresh_training=True,fixed_test_samples=900,
            split='Existing seed=2, 350/50/100 per class, exact-count L/U, same indices across all versions',
            split_limitation=rm['protocol']['split_limitation'],
            V2='DDDE effective rank + EMA; full L1 norm to uniform/reversed L prior; same softmax map, tau in [0,4]',
            V3='Confidence>=0.95 pseudo-label histogram + EMA; symmetric KL to same references; same dynamic tau',
            static='V1=4, V4=2, S0=0, S1=1, S3=3; no estimator or distance calculation',
            logging='variant_history.jsonl: tau, applicable KL/L1, estimated/raw distribution, estimator and module settings'))
    (batch/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
    for file,value in [('source_sha256.json',{str(p.relative_to(code)):digest(p) for p in code.rglob('*') if p.is_file()}),
                       ('config_sha256.json',{t['name']:digest(t['config']) for t in tasks})]:
        (batch/file).write_text(json.dumps(value,indent=2)+'\n')
    with (batch/'sample_counts.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(sample_rows[0]));w.writeheader();w.writerows(sample_rows)
    print(f'Prepared 120 main + 72 sensitivity runs, 96 per GPU: {batch}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('batch',type=Path);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();prepare(a.batch,a.smoke)
