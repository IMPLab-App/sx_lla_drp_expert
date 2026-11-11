"""A single GPU's gated LU tasks. Stage transitions belong to chain_lla_drp."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
from lla_experiment_utils import require_idle_gpu,sha256


def wait_idle(gpu):
    deadline=time.monotonic()+90
    while True:
        try:return require_idle_gpu(gpu)
        except RuntimeError:
            if time.monotonic()>=deadline:raise
            time.sleep(2)


def verify_sources(batch):
    m=json.loads((batch/'manifest.json').read_text());code=Path(m['code_snapshot'])
    for name,expected in json.loads((batch/'code_audit/source_sha256.json').read_text()).items():
        if sha256(code/name)!=expected:raise RuntimeError('Frozen code mismatch: '+name)
    for marker in ('unit_tests/result.json','smoke_tests/result.json','unit_tests/ddp_result.json'):
        proof=json.loads((batch/marker).read_text())
        if proof['status']!='passed':raise RuntimeError('CPU verification incomplete: '+marker)
        for name,expected in proof['source_sha256'].items():
            if sha256(code/name)!=expected:raise RuntimeError('CPU proof differs from frozen code: '+name)
    for name,expected in json.loads((batch/'code_audit/protected_source_sha256.json').read_text()).items():
        source=Path(__file__).resolve().parent/name
        if sha256(source)!=expected:raise RuntimeError('Protected active LLA source changed: '+name)
    return m


def verify_predecessor(batch,verify_hashes=True):
    m=json.loads((batch/'manifest.json').read_text());old=json.loads((Path(m['predecessor_batch'])/'manifest.json').read_text())
    for t in old['tasks']:
        out=Path(t['output']);selection=out/'validation_selection.json';proof=out/'checkpoint_manifest.json'
        if not selection.exists() or not proof.exists():raise RuntimeError('Predecessor task not complete: '+t['case_name']+'/'+t['version'])
        record=json.loads(proof.read_text())
        if record['config_sha256']!=t['config_sha256']:raise RuntimeError('Predecessor config hash mismatch')
        if verify_hashes:
            for value in record['checkpoints'].values():
                if sha256(value['path'])!=value['sha256']:raise RuntimeError('Predecessor checkpoint hash mismatch')
    return old


def verify_reference_matching(batch):
    m=json.loads((batch/'manifest.json').read_text());rows=[]
    keys=('algorithm','net','num_classes','noise_ratio','hard_label','p_cutoff','T','ulb_loss_ratio','use_cat',
        'batch_size','uratio','eval_batch_size','ema_m','optim','lr','weight_decay','momentum','layer_decay',
        'amp','clip_grad','seed','train_seed','split_seed','num_workers','epoch','num_train_iter','num_warmup_iter',
        'num_eval_iter','est_epoch','cut1','cut2','beta1','beta2','la_tau_lb1','data_sha256','fixed_split_sha256',
        'full_train_counts','labeled_counts','include_lb_to_ulb','lla_init','lla_lr','lla_weight_decay','lla_optimizer',
        'lla_betas','lla_optimizer_epsilon','lla_epsilon','lla_q_stop_gradient','lla_scheduler','lla_clip_grad',
        'lla_tail_class_ids','gradient_accumulation_steps')
    for t in m['tasks']:
        if t['version']!='M0':continue
        cfg=yaml.safe_load(Path(t['config']).read_text())
        for r in (r for r in m['references'] if r['scenario']==t['scenario']):
            old=yaml.safe_load(Path(r['config']).read_text())
            differences={k:[cfg.get(k),old.get(k)] for k in keys if cfg.get(k)!=old.get(k)}
            if differences:raise RuntimeError('Reference not strictly matched: '+str(differences))
            if sha256(cfg['fixed_split_file'])!=sha256(old['fixed_split_file']):raise RuntimeError('Reference sample identity mismatch')
            rows.append(dict(scenario=t['scenario'],reference=r['version'],matched=True,checked_keys=list(keys),
                             reference_config=r['config'],config_sha256=r['config_sha256']))
    (batch/'code_audit/reference_matching.json').write_text(json.dumps(dict(status='passed',comparisons=rows),indent=2)+'\n')


def worker(batch,gpu,stage):
    lock=(batch/f'{stage}_gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=verify_sources(batch);verify_predecessor(batch,verify_hashes=False)
    if stage=='m2':
        gate=json.loads((batch/'comparison/drp_validation_gate.json').read_text())
        if not gate['eligible_for_m2']:raise RuntimeError('M2 is not enabled by the validation gate')
    code=Path(m['code_snapshot']);hardware=json.loads((batch/'smoke_tests'/f'gpu_result_{gpu}.json').read_text())
    if hardware['status']!='passed':raise RuntimeError('GPU preflight incomplete')
    for name,h in hardware['source_sha256'].items():
        if sha256(code/name)!=h:raise RuntimeError('GPU proof differs from frozen code')
    version='M0' if stage=='full' else 'M2'
    env=dict(os.environ,LLA_DRP_BATCH=str(batch),LLA_GPU_INDEX=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for t in m['tasks']:
        if t['version']!=version or t['gpu']!=gpu:continue
        out=Path(t['output']);marker=out/'checkpoint_manifest.json'
        if marker.exists() and (out/'validation_selection.json').exists():
            record=json.loads(marker.read_text());assert record['config_sha256']==t['config_sha256']
            for value in record['checkpoints'].values():assert sha256(value['path'])==value['sha256']
            continue
        if out.exists():raise RuntimeError('Partial output requires explicit recovery review: '+str(out))
        if sha256(t['config'])!=t['config_sha256']:raise RuntimeError('Configuration hash mismatch')
        wait_idle(gpu)
        log=out.parent/(out.name+'.console.log')
        with log.open('x') as stream:
            subprocess.run([sys.executable,str(code/'train_lla_drp.py'),'--c',t['config']],cwd=code,env=env,
                stdout=stream,stderr=subprocess.STDOUT,check=True)
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--gpu',type=int,choices=[0,1],required=True);p.add_argument('--stage',choices=['full','m2'],required=True)
    a=p.parse_args();sys.exit(worker(a.batch.resolve(),a.gpu,a.stage))
