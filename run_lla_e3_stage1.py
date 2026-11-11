"""Stage-1-only sequential queue. Busy GPUs cause a safe exit, never preemption."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from lla_experiment_utils import require_idle_gpu, sha256

DEFAULT=Path(__file__).resolve().parent/'logs/lla_e3_replacement_20261111'


def wait_for_idle(gpu):
    deadline=time.monotonic()+60
    while True:
        try:return require_idle_gpu(gpu)
        except RuntimeError:
            if time.monotonic()>=deadline:raise
            time.sleep(2)


def run(batch,gpu,shard_index=0,shards=1):
    m=json.loads((batch/'manifest.json').read_text())
    assert m['stage']==1 and len(m['tasks'])==18 and not m['stages2_and3_enqueued']
    if not 0<=shard_index<shards:raise ValueError('Invalid shard')
    lock=(batch/'stage1.lock').open('a')
    fcntl.flock(lock,(fcntl.LOCK_EX if shards==1 else fcntl.LOCK_SH)|fcntl.LOCK_NB)
    worker_lock=(batch/f'stage1_shard{shard_index}_of{shards}.lock').open('a')
    fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for file in ('unit_tests/result.json','smoke_tests/result.json'):
        result=json.loads((batch/file).read_text())
        if result['status']!='passed':raise RuntimeError('Verification not passed: '+file)
        if not result.get('source_sha256'):raise RuntimeError('Verification lacks source fingerprints: '+file)
        for name,h in result['source_sha256'].items():
            if sha256(Path(__file__).resolve().parent/name)!=h:
                raise RuntimeError('Source changed since verification: '+name)
    ddp=json.loads((batch/'unit_tests/ddp_result.json').read_text())
    if ddp['status']!='passed':raise RuntimeError('CPU DDP verification not passed')
    for file in ('code_audit/data_audit.json','code_audit/original_files_unchanged.json'):
        if json.loads((batch/file).read_text())['status']!='passed':raise RuntimeError('Audit failed: '+file)
    snapshot=Path(m['code_snapshot'])
    hashes=json.loads((batch/'code_audit/source_sha256.json').read_text())
    for name,h in hashes.items():
        if sha256(snapshot/name)!=h:raise RuntimeError('Snapshot checksum mismatch: '+name)
    try:state=wait_for_idle(gpu)
    except RuntimeError as e:
        (batch/'reports/TRAINING_STATUS.md').write_text('Stage 1 not started: '+str(e)+'\n\nNo GPU job was stopped. Later stages remain unstarted.\n')
        print(e);return 75
    # Hardware smoke runs first and uses independent short-run output. These
    # checks never evaluate the test set or count as scientific results.
    env=dict(os.environ,LLA_GPU_INDEX=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    if 'CUDA_VISIBLE_DEVICES' in env:
        raise RuntimeError('Run queue with CUDA_VISIBLE_DEVICES unset so physical GPU checks match torch indices')
    gpu_result=batch/'smoke_tests'/f'gpu_result_gpu{gpu}.json'
    if not gpu_result.exists():
        with (batch/'smoke_tests'/f'gpu_preflight_gpu{gpu}.log').open('a') as log:
            subprocess.run([sys.executable,str(snapshot/'verify_lla_e3_gpu.py'),'--batch',str(batch),'--gpu',str(gpu),'--result',str(gpu_result)],
                           cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    hardware=json.loads(gpu_result.read_text())
    if hardware['status']!='passed':raise RuntimeError('GPU preflight failed')
    for name,h in hardware['source_sha256'].items():
        if sha256(snapshot/name)!=h:raise RuntimeError('GPU verification source mismatch: '+name)
    if shards>1:
        # No formal worker starts until every GPU in this priority batch has
        # completed the full hardware preflight using this exact source.
        while True:
            ready=True
            for peer in range(shards):
                marker=batch/'smoke_tests'/f'gpu_result_gpu{peer}.json'
                if not marker.exists():
                    ready=False;break
                proof=json.loads(marker.read_text())
                if proof['status']!='passed':raise RuntimeError('Peer GPU preflight failed')
                for name,h in proof['source_sha256'].items():
                    if sha256(snapshot/name)!=h:raise RuntimeError('Peer GPU preflight source mismatch')
            if ready:break
            time.sleep(1)
    for t in m['tasks']:
        # All three models of a given LU use the same physical GPU.
        if (int(t['scenario'][2:])-1)%shards!=shard_index:continue
        out=Path(t['output'])
        if (out/'checkpoint_manifest.json').exists() and (out/'validation_selection.json').exists():
            saved=json.loads((out/'checkpoint_manifest.json').read_text())
            if saved['config_sha256']!=t['config_sha256']:raise RuntimeError('Completed configuration mismatch')
            for value in saved['checkpoints'].values():
                if sha256(value['path'])!=value['sha256']:raise RuntimeError('Checkpoint digest mismatch')
            continue
        if out.exists():raise RuntimeError('Partial output requires explicit review/resume: '+str(out))
        if sha256(t['config'])!=t['config_sha256']:raise RuntimeError('Config fingerprint mismatch')
        wait_for_idle(gpu)
        log_path=batch/'lu_replacement'/(out.name+'.console.log')
        with log_path.open('x') as log:
            subprocess.run([sys.executable,str(snapshot/'train_lla_e3.py'),'--c',t['config']],
                cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        (batch/'reports/TRAINING_STATUS.md').write_text('Last completed stage-1 task: '+out.name+'\n\nNo later stage launched.\n')
    if shards==1:
        subprocess.run([sys.executable,str(snapshot/'summarize_lla_e3.py'),'--batch',str(batch)],check=True,cwd=snapshot)
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,default=DEFAULT)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();sys.exit(run(a.batch.resolve(),a.gpu,a.shard_index,a.shards))
