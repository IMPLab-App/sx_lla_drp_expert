"""Read-only resource checks and experiment provenance helpers."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as s:
        for part in iter(lambda:s.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def gpu_status(index):
    rows=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True))))
    row=next(r for r in rows if int(r[0].strip())==index)
    uuid=row[1].strip()
    processes=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits'],text=True))))
    jobs=[dict(pid=int(r[1]),name=r[2].strip(),memory_mib=r[3].strip()) for r in processes
          if len(r)>=4 and r[0].strip()==uuid and 'gnome-remote-desktop-daemon' not in r[2]]
    paused_jobs=[]
    authorization=os.environ.get('LLA_PAUSED_PROCESS_MANIFEST')
    if authorization:
        import psutil
        pause=json.loads(Path(authorization).read_text())
        if pause.get('authorization')!='user_requested_priority_pause':
            raise ValueError('Invalid priority-pause authorization')
        allowed={p['pid']:p['created'] for p in pause['processes']}
        active=[]
        for job in jobs:
            if job['pid'] in allowed:
                try:
                    process=psutil.Process(job['pid'])
                    is_paused=(abs(process.create_time()-allowed[job['pid']])<.01
                               and process.status()==psutil.STATUS_STOPPED)
                except psutil.NoSuchProcess:
                    is_paused=False
                if is_paused:
                    paused_jobs.append(job);continue
            active.append(job)
        jobs=active
    return dict(index=index,uuid=uuid,utilization=int(row[2]),memory_mib=int(row[3]),
                total_memory_mib=int(row[4]),jobs=jobs,paused_jobs=paused_jobs)


def require_idle_gpu(index):
    state=gpu_status(index)
    if state['jobs'] or state['utilization']>10:
        raise RuntimeError('GPU busy; no job started or stopped: '+json.dumps(state,ensure_ascii=False))
    if state['total_memory_mib']-state['memory_mib']<8192:
        raise RuntimeError('Insufficient free GPU memory for priority training: '+json.dumps(state))
    return state


def verify_config_data(cfg):
    from semilearn.datasets.cv_datasets.raw_signal import DATASET_FILENAME
    if sha256(Path(cfg.data_dir)/DATASET_FILENAME)!=cfg.data_sha256:
        raise ValueError('Raw signal data fingerprint mismatch')
    if sha256(cfg.fixed_split_file)!=cfg.fixed_split_sha256:
        raise ValueError('Frozen split fingerprint mismatch')
