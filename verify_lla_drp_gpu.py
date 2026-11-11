"""Deferred real-GPU preflight. Runs only after all predecessor LU tasks finish."""
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
from semilearn.imb_algorithms.metaexpert.lla_drp_algorithm import LLADRPMetaExpert
from semilearn.imb_algorithms.metaexpert.drp_e3 import DRPE3Controller
from lla_experiment_utils import require_idle_gpu,verify_config_data
from verify_lla_drp import source_digests
from verify_lla_e3 import equal


def make(batch,output,gpu,case=1,fixed=False,drp=True,mode='lla_hard',amp=False):
    m=json.loads((batch/'manifest.json').read_text());t=next(t for t in m['tasks'] if t['scenario']==f'LU{case}' and t['version']=='M0')
    with patch.object(sys,'argv',['gpu_audit','--c',t['config']]):args=get_config()
    args.e3_drp_enabled=drp;args.ablation_fixed_dea_weights=fixed
    args.e3_adjust_mode=mode;args.ablation_no_ddde_kl=mode!='ddde_kl'
    args.num_train_iter=9;args.epoch=3;args.num_eval_iter=3;args.num_log_iter=1
    args.batch_size=4;args.uratio=2;args.eval_batch_size=128;args.num_workers=0;args.est_epoch=0
    args.gpu=gpu;args.amp=amp;args.save_dir=str(output);args.save_name='audit'
    verify_config_data(args)
    return build_gpu_algorithm(args,output/'audit')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);args=p.parse_args();batch=args.batch.resolve()
    require_idle_gpu(args.gpu);torch.set_num_threads(2)
    output=batch/'smoke_tests'/(datetime.now().strftime('gpu_%Y%m%d_%H%M%S')+f'_{args.gpu}')
    output.mkdir(exist_ok=False)
    tiny=DRPE3Controller(9,momentum=0).cuda(args.gpu)
    p0=torch.ones(1,9,device=f'cuda:{args.gpu}')/9;tiny(p0)
    tiny.mu_max.fill_(.99);tiny.mu_gap.fill_(.99)
    weights,_,_=tiny(p0,update=False);assert ((weights>0)&(weights<=1)).all()
    rows=[]
    cases=[(f'Full_LU{i}',i,False,True,'lla_hard',False) for i in range(1,7)]
    cases += [('M2',1,True,True,'lla_hard',False),('L0',1,False,False,'lla_hard',False),
              ('M1',1,False,False,'fixed',False),('V5',1,False,False,'ddde_kl',False),
              ('Full_AMP',1,False,True,'lla_hard',True),('M2_AMP',1,True,True,'lla_hard',True)]
    for name,case,fixed,drp,mode,amp in cases:
        a=make(batch,output/name,args.gpu,case,fixed,drp,mode,amp);a.train()
        monitor=dict(active=a._mff_active_steps,nonzero=a._mff_nonzero_steps)
        if fixed:assert monitor['nonzero']>0
        a.load_model(output/name/'audit/model_best.pth');p0=a.evaluate(return_logits=True)['eval/logits']
        a.load_model(output/name/'audit/model_best.pth');p1=a.evaluate(return_logits=True)['eval/logits'];np.testing.assert_array_equal(p0,p1)
        rows.append(dict(mode=name,steps=9,passed=True,mff_gradient_monitor=monitor))
        del a;gc.collect();torch.cuda.empty_cache()
    resumes=[]
    for fixed in (False,True):
        name='M2' if fixed else 'Full'
        a=make(batch,output/(name+'_continuous'),args.gpu,fixed=fixed);a.train();want=copy.deepcopy(a.get_save_dict())
        del a;gc.collect();torch.cuda.empty_cache()
        a=make(batch,output/(name+'_interrupted'),args.gpu,fixed=fixed);a.train(stop_after=2)
        del a;gc.collect();torch.cuda.empty_cache()
        a=make(batch,output/(name+'_resumed'),args.gpu,fixed=fixed)
        a.load_model(output/(name+'_interrupted')/'audit/interrupted_model.pth');a.train();got=a.get_save_dict()
        for key in ('model','ema_model','optimizer','lambda_optimizer','loss_scaler','rng','sampler_state',
                    'drp_state','mff_gradient_monitor','it','epoch','best_it','best_eval_f1'):
            equal(want[key],got[key])
        resumes.append(dict(mode=name,bitwise_equal=True))
        del a,want,got;gc.collect();torch.cuda.empty_cache()
    result=dict(status='passed',gpu=args.gpu,source_sha256=source_digests(),attempt_output=str(output),runs=rows,
        resume=resumes,test_evaluated=False,zero_variance_positive_weight=True)
    (batch/'smoke_tests'/f'gpu_result_{args.gpu}.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
