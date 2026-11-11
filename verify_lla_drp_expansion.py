"""Real-signal CPU integration for all 54 configs and continuation recovery."""
import argparse
import contextlib
import gc
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
import numpy as np
import psutil
import torch
from train import get_config
from semilearn.imb_algorithms import get_imb_algorithm
from semilearn.imb_algorithms.metaexpert.lla_drp_algorithm import LLADRPMetaExpert
from semilearn.core.utils import get_net_builder
from lla_experiment_utils import sha256,verify_config_data
from run_lla_drp_expansion import execution_config,match,worker


def verify_failed_performance_gate_still_schedules(output):
    fixture=output/'queue';fixture.mkdir()
    for name in ('configs','tests','logs','failed_attempts','batch/smoke_tests','batch/comparison','code'):
        (fixture/name).mkdir(parents=True,exist_ok=True)
    canonical=fixture/'configs/task.yaml';canonical.write_text('resume: false\n')
    task=dict(scenario='LU1',version='M2',stage='lu_m2',gpu=0,
              config=str(canonical),config_sha256=sha256(canonical),output=str(fixture/'output'))
    manifest=dict(batch=str(fixture/'batch'),code_snapshot=str(fixture/'code'),tasks=[task])
    p=fixture/'manifest.json';p.write_text(json.dumps(manifest))
    (fixture/'tests/config_and_integration.json').write_text(json.dumps(
        dict(status='passed',manifest_sha256=sha256(p),source_sha256={})))
    (fixture/'batch/smoke_tests/gpu_result_0.json').write_text(json.dumps(dict(status='passed',source_sha256={})))
    (fixture/'batch/comparison/drp_validation_gate.json').write_text(json.dumps(dict(status='failed',eligible_for_m2=False)))
    calls=[]
    def execute(command,**kwargs):
        calls.append(command)
        out=Path(task['output']);out.mkdir();checkpoint=out/'model_best.pth';checkpoint.write_bytes(b'fixture')
        (out/'validation_selection.json').write_text('{}')
        (out/'checkpoint_manifest.json').write_text(json.dumps(dict(config_sha256=task['config_sha256'],
            selection='validation_macro_f1',test_evaluated=False,mff_gradient_monitor=dict(nonzero=1),
            checkpoints={'model_best.pth':dict(path=str(checkpoint),sha256=sha256(checkpoint))})))
    with patch('run_lla_drp_expansion.verify_sources'),patch('run_lla_drp_expansion.wait_idle'),patch('run_lla_drp_expansion.subprocess.run',side_effect=execute):
        assert worker(fixture,'lu_m2',0)==0
    assert len(calls)==1
    assert json.loads((fixture/'batch/comparison/drp_validation_gate.json').read_text())['eligible_for_m2'] is False


def verify(expansion):
    torch.set_num_threads(2)
    m=json.loads((expansion/'manifest.json').read_text());results=[]
    assert len(m['tasks'])==60 and len([t for t in m['tasks'] if t['stage']=='rcd'])==54
    assert m['performance_gate_controls_scheduling'] is False
    with tempfile.TemporaryDirectory(dir=expansion/'tests') as tmp:
        output=Path(tmp)
        for t in m['tasks']:
            assert sha256(t['config'])==t['config_sha256']
            if t['stage']!='rcd':continue
            with patch.object(sys,'argv',['verify','--c',t['config']]):args=get_config()
            verify_config_data(args)
            args.num_workers=0;args.distributed=False;args.gpu=0
            args.num_train_iter=9;args.epoch=3;args.num_eval_iter=3;args.num_log_iter=1
            args.est_epoch=0;args.batch_size=4;args.uratio=2;args.eval_batch_size=128
            args.save_dir=str(output/t['scenario']/t['version']);args.save_name='audit'
            random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
            with contextlib.redirect_stdout(io.StringIO()),patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self):
                a=get_imb_algorithm(args,get_net_builder(args.net,args.net_from_name),None,None)
            a.device=torch.device('cpu');a.finalize_model_setup();a.epoch=2
            with np.load(args.fixed_split_file,allow_pickle=False) as frozen:
                for name,key in [('train_lb','labeled'),('train_ulb','unlabeled'),('eval','validation'),('test','test')]:
                    assert np.array_equal(a.dataset_dict[name].sample_indices,frozen[key])
            inputs=a.process_batch(**next(iter(a.loader_dict['train_lb'])),**next(iter(a.loader_dict['train_ulb'])))
            a.out_dict,a.log_dict=a.train_step(**inputs)
            for value in a.out_dict.values():
                if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
            a.hooks_dict['ParamUpdateHook'].after_train_step(a)
            assert a.optimizer_steps==1
            assert a.log_dict['train/mff_gradient_norm']>0
            if t['version']=='M1':
                assert a.lambda_optimizer is None and a.drp is None
                assert not hasattr(a.module,'lla')
            else:
                assert a.log_dict['train/lambda_update_norm']>0
                assert a.drp.updates>0
            if t['version']=='M2':
                assert a.fixed_dea_weights and a._dea_last['w_H_mean']==.5 and a._dea_last['w_T_mean']==.5
            results.append(dict(scenario=t['scenario'],model=t['version'],status='passed',
                mff_gradient_norm=a.log_dict['train/mff_gradient_norm'],
                sample_indices_exact=True,finite_losses_and_optimizer_update=True))
            print('passed',t['scenario'],t['version'],flush=True)
            del a,inputs;gc.collect()
        # Recovery must preserve all original rows before trimming replay suffix.
        out=output/'partial';out.mkdir();(out/'metrics').mkdir()
        torch.save(dict(it=2,model={'weight':torch.ones(1)}),out/'latest_model.pth')
        history=''.join(json.dumps(dict(iteration=i))+'\n' for i in range(4))
        (out/'training_history.jsonl').write_text(history)
        import yaml
        canonical=output/'canonical.yaml';canonical.write_text(yaml.safe_dump(dict(resume=False,load_path=None)))
        task=dict(output=str(out),config=str(canonical),scenario='fixture',version='M2')
        cfg,recovery=execution_config(expansion,task,1)
        assert recovery['cursor']==2
        assert (Path(recovery['preserved_attempt'])/'training_history.jsonl').read_text()==history
        assert len((out/'training_history.jsonl').read_text().splitlines())==2
        assert yaml.safe_load(cfg.read_text())['resume'] is True
        # Supervisory hold must leave an independent worker making progress.
        counter=output/'counter';worker_code='import time; from pathlib import Path; p=Path('+repr(str(counter))+'); i=0\nwhile True:\n i+=1; p.write_text(str(i)); time.sleep(.02)'
        supervisor_code='import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",'+repr(worker_code)+']); time.sleep(30)'
        p=subprocess.Popen([sys.executable,'-c',supervisor_code],start_new_session=True)
        children=[]
        try:
            time.sleep(.3);root=psutil.Process(p.pid);children=root.children();assert len(children)==1
            root.suspend();before=int(counter.read_text());time.sleep(.2);after=int(counter.read_text());assert after>before
            root.resume()
        finally:
            try:psutil.Process(p.pid).resume()
            except psutil.NoSuchProcess:pass
            for c in children:
                try:c.terminate()
                except psutil.NoSuchProcess:pass
            p.terminate();p.wait(timeout=5);psutil.wait_procs(children,timeout=5)
        verify_failed_performance_gate_still_schedules(output)
    files=['prepare_lla_drp_expansion.py','run_lla_drp_expansion.py','summarize_lla_drp_expansion.py','verify_lla_drp_expansion.py']
    proof=dict(status='passed',manifest_sha256=sha256(expansion/'manifest.json'),
        source_sha256={f:sha256(f) for f in files},runs=results,
        parent_only_hold_preserves_worker_progress=True,partial_attempt_archived_before_resume=True,
        total_rcd_configurations=54,lu_m2_configs=6,performance_gate_controls_scheduling=False,
        failed_performance_gate_still_launches_m2=True,
        original_frozen_training_tests='CPU/Gloo/GPU full and M2 exact resume already passed',
        test_evaluated=False)
    (expansion/'tests/config_and_integration.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('expansion',type=Path)
    r=verify(p.parse_args().expansion.resolve());print('PASSED',len(r['runs']))
