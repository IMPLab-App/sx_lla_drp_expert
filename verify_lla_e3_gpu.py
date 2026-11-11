"""Deferred idle-GPU preflight; uses real data and never evaluates test labels."""
import argparse
import copy
from datetime import datetime
import gc
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import torch
from train import get_config
from train_lla_e3 import build_gpu_algorithm
from semilearn.imb_algorithms.metaexpert.lla_algorithm import LLAE3MetaExpert
from lla_experiment_utils import require_idle_gpu, verify_config_data
from verify_lla_e3 import equal, source_digests


def make(batch,output,gpu,mode='lla_hard',no_dea=False,case=1,amp=False):
    m=json.loads((batch/'manifest.json').read_text())
    t=next(t for t in m['tasks'] if t['scenario']==f'LU{case}' and t['version']=='M0')
    with patch.object(sys,'argv',['verify_gpu','--c',t['config']]):args=get_config()
    args.e3_adjust_mode=mode;args.ablation_no_ddde_kl=mode!='ddde_kl'
    args.ablation_no_dea_aggregator=no_dea
    args.epoch=3;args.num_train_iter=9;args.num_eval_iter=3;args.num_log_iter=1
    args.batch_size=4;args.uratio=2;args.eval_batch_size=128;args.num_workers=0
    args.est_epoch=0;args.amp=amp;args.gpu=gpu
    args.save_dir=str(output);args.save_name='audit'
    verify_config_data(args)
    a=build_gpu_algorithm(args,output/'audit')
    return a


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--result',type=Path)
    args=p.parse_args();batch=args.batch.resolve()
    require_idle_gpu(args.gpu)
    torch.set_num_threads(2)
    output=batch/'smoke_tests'/(datetime.now().strftime('gpu_attempt_%Y%m%d_%H%M%S')+f'_gpu{args.gpu}')
    output.mkdir(exist_ok=False)
    rows=[]
    scenarios=[('lla_hard',False,i,False) for i in range(1,7)]
    scenarios += [('fixed',False,1,False),('ddde_kl',False,1,False),('lla_hard',True,1,False),('lla_hard',False,1,True)]
    for mode,no_dea,case,amp in scenarios:
        name=f'{mode}_lu{case}_nodea{int(no_dea)}_amp{int(amp)}'
        a=make(batch,output/name,args.gpu,mode,no_dea,case,amp)
        initial=a.module.lla.lambda_logits.detach().clone() if mode=='lla_hard' else None
        a.train()
        if initial is not None:assert not torch.equal(initial,a.module.lla.lambda_logits)
        a.load_model(output/name/'audit/model_best.pth')
        before=a.evaluate(return_logits=True)['eval/logits']
        a.load_model(output/name/'audit/model_best.pth')
        after=a.evaluate(return_logits=True)['eval/logits'];np.testing.assert_array_equal(before,after)
        rows.append(dict(mode=mode,no_dea=no_dea,case=case,amp=amp,steps=9,passed=True))
        del a;gc.collect();torch.cuda.empty_cache()
    # Actual CUDA continuous vs interrupted/resumed run, including EMA/RNG.
    a=make(batch,output/'continuous',args.gpu);a.train();want=copy.deepcopy(a.get_save_dict())
    del a;gc.collect();torch.cuda.empty_cache()
    b=make(batch,output/'interrupted',args.gpu);b.train(stop_after=2)
    del b;gc.collect();torch.cuda.empty_cache()
    c=make(batch,output/'resumed',args.gpu);c.load_model(output/'interrupted/audit/interrupted_model.pth')
    c.train();got=c.get_save_dict()
    for key in ('model','ema_model','optimizer','lambda_optimizer','loss_scaler','rng',
                'sampler_state','it','epoch','best_it','best_eval_f1'):
        equal(want[key],got[key])
    result=dict(status='passed',source_sha256=source_digests(),device=f'cuda:{args.gpu}',runs=rows,attempt_output=str(output),
                resume_bitwise_equal=True,test_evaluated=False)
    (args.result or batch/'smoke_tests/gpu_result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
