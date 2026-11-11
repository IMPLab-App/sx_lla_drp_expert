"""Prepare independent matched LU Full configs; fixed-DEA configs are gated."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import yaml

ROOT=Path(__file__).resolve().parent
BATCH=ROOT/'logs/lla_drp_mff_dea_ablation_20261111'
REFERENCE=ROOT/'logs/lla_e3_replacement_20261111'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(batch):
    if any((batch/'lu_full').glob('*/training_history.jsonl')) or any((batch/'lu_wo_dea').glob('*/training_history.jsonl')):
        raise RuntimeError('Cannot update a snapshot after formal training has started')
    code=batch/'code';code.mkdir(exist_ok=True)
    shutil.copytree(ROOT/'semilearn',code/'semilearn',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    files=('train.py','train_lla_e3.py','lla_experiment_utils.py','verify_lla_e3.py',
           'verify_lla_e3_gpu.py','verify_lla_e3_ddp.py','prepare_lla_drp.py','train_lla_drp.py',
           'verify_lla_drp.py','verify_lla_drp_gpu.py','verify_lla_drp_ddp.py','run_lla_drp.py','summarize_lla_drp.py','chain_lla_drp.py','plot_lla_drp.py')
    for name in files:
        if (ROOT/name).exists():shutil.copy2(ROOT/name,code/name)
    (batch/'code_audit/source_sha256.json').write_text(json.dumps({str(p.relative_to(code)):sha(p) for p in code.rglob('*') if p.is_file()},indent=2)+'\n')


def prepare(batch):
    batch=batch.resolve();manifest=batch/'manifest.json'
    if manifest.exists():raise FileExistsError('Use a new experiment root; manifest already exists')
    old=json.loads((REFERENCE/'manifest.json').read_text())
    source=Path(old['data_file']);target=batch/'data'/source.name
    shutil.copy2(source,target);assert sha(target)==old['data_sha256']
    for directory in ('full','fixed_dea'): (batch/'configs'/directory).mkdir(exist_ok=True)
    tasks=[];reference=[]
    for i,case in enumerate(old['task_names'],1):
        source_task=next(t for t in old['tasks'] if t['case_name']==case and t['version']=='M0')
        cfg0=yaml.safe_load(Path(source_task['config']).read_text())
        split=Path(cfg0['fixed_split_file']);newsplit=batch/'split_indices'/split.name
        shutil.copy2(split,newsplit);assert sha(newsplit)==cfg0['fixed_split_sha256']
        for version,fixed in (('M0',False),('M2',True)):
            cfg=dict(cfg0);cfg.update(imb_algorithm='metaexpert_lla_drp',e3_adjust_mode='lla_hard',
                ablation_no_ddde_kl=True,ablation_no_dea_aggregator=False,ablation_no_mff=False,
                ablation_fixed_dea_weights=fixed,e3_drp_enabled=True,drp_momentum=.999,
                drp_n_sigma=2.,drp_per_class=False,drp_with_hard_mask=False,
                drp_epsilon=1e-8,drp_loss_epsilon=1e-12,mff_gradient_audit_window=100,
                data_dir=str(batch/'data'),fixed_split_file=str(newsplit),
                save_dir=str(batch/('lu_wo_dea' if fixed else 'lu_full')),save_name=case,
                study_variant=version,mff_implementation='original_lin1_lin8',
                resume=False,load_path=None,overwrite=False,
                stage='six_task_dynamic_weighting_ablation' if fixed else 'six_task_drp_increment',
                inference_protocol='fixed_half_E1raw_E3lla' if fixed else 'dynamic_DEA_E1raw_E3lla',
                e3_unlabeled_selection='continuous_DRP',fusion_training_loss_enabled=True,
                original_fusion_auxiliary_supervision_retained=True)
            path=batch/'configs'/('fixed_dea' if fixed else 'full')/(case+'.yaml')
            path.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
            tasks.append(dict(scenario=f'LU{i}',case_name=case,version=version,config=str(path),
                config_sha256=sha(path),output=str(Path(cfg['save_dir'])/case),gpu=(i-1)%2,
                fixed_split_sha256=sha(newsplit),seed=cfg['seed'],status='pending',
                enabled_after='drp_validation_gate' if fixed else 'old_18_runs_and_new_tests_passed'))
        for version,alias in (('M0','L0'),('V1','M1'),('V5','V5')):
            t=next(t for t in old['tasks'] if t['case_name']==case and t['version']==version)
            reference.append(dict(scenario=f'LU{i}',case_name=case,version=alias,source_version=version,
                config=t['config'],config_sha256=t['config_sha256'],output=t['output'],
                reuse='conditional_on_complete_checkpoints_and_matching_audit'))
    result=dict(title='E3-LLA + DRP, original lin1-lin8 MFF retained',tasks=tasks,references=reference,
        predecessor_batch=str(REFERENCE),predecessor_tasks=old['tasks'],
        data_file=str(target),data_sha256=sha(target),classes=old['classes'],task_names=old['task_names'],
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty_tree=True,
        code_snapshot=str(batch/'code'),active_stage='prepare_and_test',
        mff_resolution='User explicitly chose active lin1-lin8, not historical SegFormerVectorMFF',
        sage_commit='d3f3ca71abc9828daa8273cc2290187baeb9f4f7',
        protocol=dict(seed=2,split_seed=2,sampler_seed=0,epochs=100,iterations=20000,BN='independent_BNH_BNT',
            network_optimizer='unchanged AdamW .0005 / .05',lambda_optimizer='unchanged Adam .0005 / 0',
            selection='validation_macro_f1',drp_statistics='global_unbiased_variance_update_before_weight',
            drp_normalization='sum(weighted_H_and_T_CE)/(sum_H_weights+sum_T_weights+1e-12)',
            drp_warmup='none; active from step 0',fusion_warmup='unchanged epoch<=est_epoch+1 disabled',
            tail_class_ids=[6,7,8],test_for_selection=False),
        drp_gate=dict(mean_full_gt_lla_only=True,wins_over_lla_only_min=4,
            fixed_tau4_mean_max_drop=.005,
            fixed_tau4_mean_drop_description='No more than 0.5 percentage point mean F1 decline, preregistered engineering threshold',
            additional_reporting=['C5_recall','C8_recall','tail_macro_f1'],single_seed_exploratory=True),
        later_stages=dict(M2='only_after_drp_validation_gate',R_CD_54='never_automatically_start_requires_result_review'),
        test_metrics_status='not_evaluated')
    manifest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    shutil.copy2(REFERENCE/'code_audit/rcd_existing_definitions.json',batch/'code_audit/rcd_existing_definitions.json')
    snapshot(batch);print('Prepared 6 Full tasks; 6 fixed-DEA configs gated. No training started.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,default=BATCH)
    p.add_argument('--snapshot-only',action='store_true');a=p.parse_args()
    snapshot(a.batch.resolve()) if a.snapshot_only else prepare(a.batch)
