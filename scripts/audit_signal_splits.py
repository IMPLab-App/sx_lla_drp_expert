"""Record the common per-seed/per-case split fingerprints before training."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import yaml
from semilearn.datasets.cv_datasets.raw_signal import get_raw_signal, load_raw_signal_data, _stratified_split


def audit(batch):
    manifest=json.loads((batch/'manifest.json').read_text())
    data,targets,names=load_raw_signal_data(str(batch/'data'))
    rows=[]
    for task in manifest['tasks']:
        if task['method']!='acr':continue
        cfg=yaml.safe_load(Path(task['config']).read_text())
        args=SimpleNamespace(**cfg)
        np.random.seed(args.seed);random.seed(args.seed)
        with contextlib.redirect_stdout(io.StringIO()):
            lb,ulb,val,test,counts=get_raw_signal(args,data_dir=args.data_dir,return_test=True)
        splits=_stratified_split(targets,len(names),args.seed)
        for a,b in (('train','eval'),('train','test'),('eval','test')):
            assert not np.intersect1d(splits[a],splits[b]).size
        expected_lb=[int(100*args.lb_imb_ratio**(-c/8)) for c in range(9)]
        expected_ulb=[35]*9 if task['case'] in (3,4) else [int(250*abs(args.ulb_imb_ratio)**(-c/8)) for c in range(9)]
        if task['case']>=5:expected_ulb=expected_ulb[::-1]
        assert counts['lb_count']==expected_lb
        assert counts['ulb_count']==expected_ulb
        assert lb[0]['x_lb'].shape==(1,1024) and ulb[0]['x_ulb_s'].shape==(1,1024)
        row=dict(seed=args.seed,case=task['case'],counts=counts,normalization_mean=lb.weak_transform.mean,
                 normalization_std=lb.weak_transform.std)
        for key,ds in (('labeled',lb),('unlabeled',ulb),('eval',val),('test',test)):
            row[key+'_data_sha256']=hashlib.sha256(ds.data.tobytes()+np.asarray(ds.targets).tobytes()).hexdigest()
        rows.append(row)
    (batch/'split_audit.json').write_text(json.dumps(dict(shape=list(data.shape),class_names=names,splits=rows),indent=2,ensure_ascii=False)+'\n')
    print(f'PASS: audited {len(rows)} seed/case splits; [1,1024] inputs; expected long-tail counts; disjoint train/validation/test.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',type=Path)
    audit(parser.parse_args().batch.resolve())
