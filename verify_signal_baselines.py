"""Validate baseline protocol, 1-D interfaces, edge cases and completed smoke runs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from torch import nn
import yaml

from prepare_signal_baselines import METHODS, CASES
from semilearn.nets.signal_wrn import SignalWideResNet
from semilearn.imb_algorithms.bacon.bacon import BaCon
from semilearn.imb_algorithms.cossl.utils import WeightEMA
from scripts.signal_metrics import write_signal_metrics


def verify(batch, smoke=None):
    manifest=json.loads((batch/'manifest.json').read_text())
    tasks=manifest['tasks']
    assert len(tasks)==96
    assert len({t['name'] for t in tasks})==96
    assert hashlib.sha256(Path(manifest['data_file']).read_bytes()).hexdigest()==manifest['data_sha256']
    for gpu in (0,1):
        group=[t for t in tasks if t['gpu']==gpu]
        assert len(group)==48
        assert [t['case'] for t in group]==[c for c in range(1,7) for _ in METHODS]
        assert {t['seed'] for t in group}=={gpu+2}
    for task in tasks:
        cfg=yaml.safe_load(Path(task['config']).read_text())
        assert cfg['dataset']=='raw_signal' and cfg['num_train_iter']==20000 and cfg['epoch']==100
        assert cfg['seed']==task['seed'] and cfg['overwrite'] is False
        assert cfg['batch_size']==64 and cfg['uratio']==2
        assert cfg['data_dir']==str(batch/'data')
        assert task['case_name']==CASES[task['case']-1]
        assert cfg['lb_imb_ratio']==(50 if task['case']%2 else 100)
        assert cfg['ulb_num_labels']==(315 if task['case'] in (3,4) else 250)
        assert cfg['ulb_imb_ratio']==(1.0 if task['case'] in (3,4) else (-1 if task['case']>=5 else 1)*cfg['lb_imb_ratio'])
        assert cfg['ulb_imb_type']==('pxe' if task['case']>=5 else 'exp')
    # Gradients through every algorithm-specific head and all-native 1-D modules.
    torch.set_num_threads(2)
    for variant in ('plain','bacon','cpe'):
        model=SignalWideResNet(num_classes=9,variant=variant)
        assert not any(isinstance(m,(nn.Conv2d,nn.BatchNorm2d,nn.AdaptiveAvgPool2d)) for m in model.modules())
        out=model(torch.randn(2,1,1024))
        assert out['feat'].shape==(2,128)
        assert all(v.shape==(2,9) for k,v in out.items() if 'logits' in k)
        sum(v.square().mean() for v in out.values()).backward()
        assert model.conv1.weight.grad is not None
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert all(p.grad is not None for name, p in model.named_parameters() if 'classifier' in name or 'projection' in name)
    # Contrastive phase with selected samples and absent classes must remain finite.
    alg=BaCon.__new__(BaCon)
    alg.num_classes=9;alg.it=10000;alg.num_train_iter=20000;alg.cls_freq=torch.ones(9)
    feats=torch.randn(6,32,requires_grad=True)
    loss=alg.contrastive_loss(torch.randn(9,32),feats,torch.tensor([0,1,1]),
                              torch.tensor([[0,2,3],[1,2,3],[4,5,6]]),torch.ones(6))
    assert torch.isfinite(loss)
    loss.backward();assert torch.isfinite(feats.grad).all()
    # CoSSL EMA updates original floating tensors and copies integer BN counters.
    model=nn.Sequential(nn.Linear(2,2),nn.BatchNorm1d(2));ema=copy.deepcopy(model)
    updater=WeightEMA(model,ema,lr=0.01,alpha=0.5,wd=False)
    with torch.no_grad():model[0].weight.add_(2);model[1].num_batches_tracked.fill_(9)
    before=ema[0].weight.clone();updater.step()
    assert torch.allclose(ema[0].weight,before+1)
    assert ema[1].num_batches_tracked.item()==9
    with tempfile.TemporaryDirectory() as tmp:
        metrics=write_signal_metrics(tmp,[0,1,2],[0,-1,2],3)
        assert metrics['accuracy']==2/3 and metrics['rejected']==1
        assert metrics['macro_recall']==2/3
    if smoke:
        assert (smoke/'status.txt').read_text().strip()=='completed'
        smoke_tasks=json.loads((smoke/'manifest.json').read_text())['tasks']
        labels=None
        for task in smoke_tasks:
            state=json.loads((smoke/'state'/(task['name']+'.json')).read_text())
            assert state['state']=='completed'
            pred=np.load(Path(task['output'])/'test_predictions.npz')
            if labels is None:labels=pred['y_true']
            assert np.array_equal(labels,pred['y_true'])
            metrics=json.loads((Path(task['output'])/'test_metrics.json').read_text())
            assert metrics['accuracy']==float((pred['y_true']==pred['y_pred']).mean())
        assert len(smoke_tasks)==8
    print('PASS: 96 configs; two ordered 48-task GPU queues; native 1D head gradients; BaCon contrastive edge cases; CoSSL EMA; rejection metrics; 8-method identical held-out labels.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    parser.add_argument('--smoke',type=Path)
    args=parser.parse_args()
    verify(args.batch.resolve(),args.smoke.resolve() if args.smoke else None)
