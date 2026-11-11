"""CPU unit/real-signal smoke and deterministic checkpoint-resume verification.

Legacy MetaExpert constructs priors using Tensor.cuda. The CPU fixture redirects
ONLY those constructor calls to CPU; production training never patches CUDA.
"""
import argparse
import contextlib
import copy
import gc
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import random
import resource
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from semilearn.imb_algorithms.metaexpert.lla_e3 import LLAE3Controller, assert_parameter_partition
from semilearn.imb_algorithms.metaexpert.lla_algorithm import (LLAE3MetaExpert, LLAStaticExpertAverage,
                                                            LLAExpertNet, prediction_paths)
from semilearn.imb_algorithms.metaexpert.lla_runtime import capture_rng, restore_rng, IsolatedUpdateHook

ROOT=Path(__file__).resolve().parent
BATCH=ROOT/'logs/lla_e3_replacement_20261111'


def source_digests():
    names=['semilearn/imb_algorithms/metaexpert/'+n for n in
        ('lla_e3.py','lla_runtime.py','lla_algorithm.py','metaexpert.py','bacn_ablation.py','cola_ddde.py')]
    names += ['verify_lla_e3.py','semilearn/core/algorithmbase.py',
        'semilearn/datasets/cv_datasets/raw_signal.py','semilearn/nets/vit/vit_metaexpert.py',
        'train_lla_e3.py','lla_experiment_utils.py','verify_lla_e3_gpu.py','verify_lla_e3_ddp.py']
    return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}


def seed():
    random.seed(2);np.random.seed(2);torch.manual_seed(2)


def make_algorithm(output, mode='lla_hard', no_dea=False, legacy=False, workers=0, steps=6, epochs=2, case=1):
    from train import get_config
    from semilearn.imb_algorithms import get_imb_algorithm
    from semilearn.core.utils import get_net_builder
    m=json.loads((BATCH/'manifest.json').read_text())
    task=next(t for t in m['tasks'] if t['scenario']==f'LU{case}' and t['version']=='M0')
    with patch.object(sys,'argv',['verify','--c',task['config']]):args=get_config()
    args.e3_adjust_mode=mode
    args.ablation_no_ddde_kl=mode!='ddde_kl'
    args.ablation_no_dea_aggregator=no_dea
    if legacy:args.imb_algorithm='metaexpert_bacn'
    args.save_dir,args.save_name=str(output),'audit'
    args.num_workers=workers;args.distributed=False;args.gpu=0
    args.num_train_iter=steps;args.epoch=epochs;args.num_eval_iter=3;args.num_log_iter=1
    args.batch_size=4;args.uratio=2;args.eval_batch_size=128
    # Real backbone and all real split samples. Only short-run budget changes.
    seed()
    with contextlib.redirect_stdout(io.StringIO()),patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self):
        a=get_imb_algorithm(args,get_net_builder(args.net,args.net_from_name),None,None)
    a.device=torch.device('cpu')
    a.finalize_model_setup()
    a.epoch=0
    return a


def batch(a):
    lb=next(iter(a.loader_dict['train_lb']));u=next(iter(a.loader_dict['train_ulb']))
    return a.process_batch(**lb,**u)


def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b,(a,b)


class ControllerTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);seed()

    def test_zero_init_and_dimension(self):
        c=LLAE3Controller(9)
        torch.testing.assert_close(c.lambda_logits,torch.zeros(9))
        self.assertTrue(c.lambda_logits.requires_grad)
        for count in (0,1,-1,9.1):
            with self.assertRaises(ValueError):LLAE3Controller(count)
        with self.assertRaises(ValueError):LLAE3Controller(9,[0]*8)
        with self.assertRaises(ValueError):c.adjust(torch.zeros(2,8))

    def test_formula_no_inplace(self):
        c=LLAE3Controller(9,torch.arange(9).float())
        before=c.lambda_logits.clone();x=torch.randn(4,9)
        torch.testing.assert_close(c.adjust(x),x-torch.log_softmax(c.lambda_logits,0))
        torch.testing.assert_close(c.lambda_logits,before,rtol=0,atol=0)

    def test_class_average_exact_not_present_class_average(self):
        c=LLAE3Controller(9)
        z=torch.zeros(3,9);u=torch.empty(0,9)
        ll,lu,total=c.class_average_loss(z,torch.zeros(3,dtype=torch.long),u,u)
        torch.testing.assert_close(ll,torch.tensor(np.log(9)/9,dtype=torch.float))
        self.assertEqual(float(lu),0.)
        self.assertTrue(torch.isfinite(total))

    def test_extreme_missing_and_empty_are_finite(self):
        c=LLAE3Controller(9,[1000.]+[-1000.]*8)
        for count in (0,1,32):
            x=torch.randn(count,9)*1e4;u=torch.empty(0,9)
            ll,lu,loss=c.class_average_loss(x,torch.zeros(count,dtype=torch.long),u,u)
            loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(c.lambda_logits.grad).all())

    def test_nonfinite_rejected(self):
        c=LLAE3Controller(9)
        for value in (float('nan'),float('inf'),-float('inf')):
            x=torch.zeros(1,9);x[0,0]=value
            with self.assertRaises(FloatingPointError):c.adjust(x)

    def test_lambda_gradient_network_isolation(self):
        network=nn.Linear(2,9);c=LLAE3Controller(9)
        z=network(torch.randn(6,2));ll,lu,loss=c.class_average_loss(z,torch.tensor([0]*5+[8]),z,z)
        loss.backward()
        self.assertGreater(float(c.lambda_logits.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in network.parameters()))
        for p in c.parameters():p.grad=None
        c.adjust(network(torch.randn(4,2))).sum().backward()
        self.assertIsNone(c.lambda_logits.grad)
        self.assertTrue(all(p.grad is not None for p in network.parameters()))

    def test_q_gradient_strategy_explicit(self):
        c=LLAE3Controller(9);w=torch.randn(7,9);s=torch.randn(7,9)
        _,lu,_=c.class_average_loss(torch.randn(4,9),torch.tensor([0,1,3,8]),w,s)
        got=torch.autograd.grad(lu,c.lambda_logits)[0]
        q=c.adjust(w,False).softmax(-1).detach()
        expected=(-(q*c.adjust(s,False).log_softmax(-1)).sum(0)/(q.sum(0)+c.epsilon)).mean()
        want=torch.autograd.grad(expected,c.lambda_logits)[0]
        torch.testing.assert_close(got,want)
        c.q_stop_gradient=False
        _,full,_=c.class_average_loss(torch.randn(4,9),torch.tensor([0,1,3,8]),w,s)
        other=torch.autograd.grad(full,c.lambda_logits)[0]
        self.assertFalse(torch.allclose(got,other))

    def test_partition_duplicate_missing_and_overlap(self):
        m=nn.ModuleDict({'net':nn.Linear(2,9),'lla':LLAE3Controller(9)})
        n=torch.optim.Adam(m['net'].parameters());l=torch.optim.Adam(m['lla'].parameters())
        self.assertEqual(assert_parameter_partition(m,n,l)['lambda_tensors'],1)
        with self.assertRaises(RuntimeError):assert_parameter_partition(m,n)
        bad=torch.optim.Adam(m.parameters())
        with self.assertRaises(RuntimeError):assert_parameter_partition(m,bad,l)

    def test_amp_cpu_dual_step(self):
        from types import SimpleNamespace
        m=nn.ModuleDict({'net':nn.Linear(2,9),'lla':LLAE3Controller(9)})
        opt=torch.optim.Adam(m['net'].parameters(),lr=.001);lo=torch.optim.Adam(m['lla'].parameters(),lr=.001)
        with torch.autocast('cpu',dtype=torch.bfloat16):z=m['net'](torch.randn(5,2));base=z.float().square().mean()
        ll,lu,lc=m['lla'].class_average_loss(z,torch.tensor([0,0,0,7,8]),z,z)
        a=SimpleNamespace(out_dict={'loss':base,'lambda_loss':lc},optimizer=opt,lambda_optimizer=lo,
            use_amp=True,loss_scaler=torch.amp.GradScaler('cpu'),clip_grad=1.,args=SimpleNamespace(lla_clip_grad=1.),
            module=SimpleNamespace(lla=m['lla']),distributed=False,log_dict={},scheduler=None,lambda_scheduler=None,
            optimizer_steps=0,record_trajectory=lambda:None)
        initial=m['lla'].lambda_logits.clone();IsolatedUpdateHook().after_train_step(a)
        self.assertEqual(a.optimizer_steps,1);self.assertFalse(torch.equal(initial,m['lla'].lambda_logits))
        self.assertTrue(torch.isfinite(m['lla'].lambda_logits).all())


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup();gc.collect()

    def test_framework_signature(self):
        self.assertEqual(list(inspect.signature(LLAE3MetaExpert.train_step).parameters),
            ['self','x_lb','y_lb','y_lb_noised','x_ulb_w','x_ulb_s','y_ulb'])

    def test_real_model_gradient_isolation_and_no_ddde_or_old_tau(self):
        with patch('semilearn.imb_algorithms.metaexpert.bacn_ablation.TwoKLDDDE',side_effect=AssertionError('DDDE called')):
            a=make_algorithm(self.p)
        a.epoch=20
        data=batch(a)
        result,logs=a.train_step(**data)
        theta=[p for p in a.model.parameters() if p.requires_grad and p is not a.module.lla.lambda_logits]
        grads=torch.autograd.grad(result['lambda_loss'],theta,allow_unused=True,retain_graph=True)
        self.assertTrue(all(g is None for g in grads))
        self.assertIsNone(torch.autograd.grad(result['loss'],a.module.lla.lambda_logits,allow_unused=True,retain_graph=True)[0])
        result['loss'].backward()
        for p in (a.module.lin1[0].weight,a.module.predict[0].weight,a.module.backbone.aux_classifier2.weight):
            self.assertIsNotNone(p.grad);self.assertGreater(float(p.grad.abs().sum()),0)
        self.assertIsNone(a.module.lla.lambda_logits.grad)
        self.assertEqual(float(a.module.tau3),0.)
        self.assertIsNone(a.cola)
        a.assert_optimizer_coverage()
        # U labels must have no influence on any loss or pseudo-label statistic.
        a.model.zero_grad();rng=capture_rng()
        x,l=a.train_step(**data)
        restore_rng(rng);data['y_ulb']=torch.full_like(data['y_ulb'],999999)
        y,k=a.train_step(**data)
        torch.testing.assert_close(x['loss'],y['loss'],rtol=0,atol=0)
        torch.testing.assert_close(x['lambda_loss'],y['lambda_loss'],rtol=0,atol=0)

    def test_wodea_adjustment_and_losses(self):
        a=make_algorithm(self.p,no_dea=True);a.epoch=20
        with torch.no_grad():a.module.lla.lambda_logits.copy_(torch.arange(9).float())
        self.assertFalse(hasattr(a.module,'lin1'));self.assertFalse(hasattr(a.module,'predict'))
        data=batch(a);out=a.model(data['x_lb'])
        expected=.5*out['logits']+.5*a.module.lla.adjust(out['aux_logits2'])
        torch.testing.assert_close(out['fuse_logit_l'],expected)
        torch.testing.assert_close(prediction_paths(out,a.module.lla)['fusion_main'],expected)
        self.assertFalse(torch.allclose(expected,.5*(out['logits']+out['aux_logits2'])))
        loss,logs=a.train_step(**data)
        self.assertEqual(logs['train/total_fuse_loss'],0.)
        loss['lambda_loss'].backward()
        self.assertGreater(float(a.module.lla.lambda_logits.grad.abs().sum()),0)

    def test_eval_single_forward_and_ema_lambda(self):
        from semilearn.core.utils import EMA
        a=make_algorithm(self.p)
        a.ema=EMA(a.model,a.ema_m);a.ema.register()
        with torch.no_grad():a.module.lla.lambda_logits.copy_(torch.arange(9).float())
        online=a.module.lla.lambda_logits.clone()
        with patch.object(a.model,'forward',wraps=a.model.forward) as forward:
            result=a.evaluate()
        self.assertEqual(forward.call_count,len(a.loader_dict['eval']))
        torch.testing.assert_close(a.module.lla.lambda_logits,online,rtol=0,atol=0)
        record=json.loads(next((self.p/'audit/metrics').glob('*.json')).read_text())
        self.assertEqual(record['lambda_logits'],[0.]*9)
        self.assertTrue(0<=result['eval/F1']<=1)
        with self.assertRaises(RuntimeError):a.evaluate('test')

    def test_fixed_and_ddde_regression(self):
        for mode in ('fixed','ddde_kl'):
            a=make_algorithm(self.p/mode,mode=mode)
            b=make_algorithm(self.p/(mode+'old'),mode=mode,legacy=True)
            equal(a.model.state_dict(),b.model.state_dict())
            self.assertIsNone(a.lambda_optimizer)
            for ep in (0,20):
                a.epoch=b.epoch=ep;a.it=b.it=0
                if mode=='ddde_kl' and ep==20:
                    features=torch.arange(27*192).reshape(27,192).float()
                    logits=torch.eye(9).repeat_interleave(3,0)*20
                    for obj in (a,b):obj.cola.begin_epoch(18);obj.cola.observe(logits,features)
                data=batch(a);rng=capture_rng()
                x,_=a.train_step(**data);restore_rng(rng);y,_=b.train_step(**data)
                torch.testing.assert_close(x['loss'],y['loss'],rtol=0,atol=0)
                a.optimizer.zero_grad();b.optimizer.zero_grad()
                x['loss'].backward();y['loss'].backward()
                for (name,p),(name2,q) in zip(a.model.named_parameters(),b.model.named_parameters()):
                    self.assertEqual(name,name2)
                    if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
                a.optimizer.step();b.optimizer.step();equal(a.model.state_dict(),b.model.state_dict())
                if mode=='ddde_kl':equal(a.cola.state_dict(),b.cola.state_dict())
                a.model.eval()
                with torch.no_grad():
                    out=a.model(data['x_lb']);paths=prediction_paths(out)
                    torch.testing.assert_close(paths['fusion_main'],out['w1']*out['logits']+out['w3']*out['aux_logits2'])
                a.model.train()

    def test_empty_unlabeled_batch_and_corrected_hard_pseudo(self):
        a=make_algorithm(self.p);a.epoch=20
        data=batch(a)
        with torch.no_grad():
            a.module.lla.lambda_logits.fill_(10.)
            a.module.lla.lambda_logits[8]=-10.
        result,_=a.train_step(**data)
        counts,_=a._pending_pseudo
        self.assertEqual(int(counts[8]),len(data['x_ulb_w']))
        data['x_ulb_w']=data['x_ulb_w'][:0];data['x_ulb_s']=data['x_ulb_s'][:0];data['y_ulb']=data['y_ulb'][:0]
        result,_=a.train_step(**data)
        self.assertTrue(torch.isfinite(result['loss']))
        self.assertTrue(torch.isfinite(result['lambda_loss']))

    def test_selection_uses_macro_f1_and_not_accuracy(self):
        from types import SimpleNamespace
        from semilearn.imb_algorithms.metaexpert.lla_runtime import MacroF1SelectionHook
        values=iter([{'eval/F1':.2,'eval/top-1-acc':.9},{'eval/F1':.3,'eval/top-1-acc':.8},
                     {'eval/F1':.25,'eval/top-1-acc':.99}])
        a=SimpleNamespace(it=0,epoch=0,num_eval_iter=1,num_train_iter=3,log_dict={},
            best_eval_f1=-1.,best_eval_acc=0.,best_it=-1,best_epoch=-1,evaluate=lambda _:next(values))
        hook=MacroF1SelectionHook()
        for step in range(3):a.it=step;hook.after_train_step(a)
        self.assertEqual(a.best_it,1);self.assertEqual(a.best_eval_f1,.3);self.assertEqual(a.best_eval_acc,.8)

    def test_checkpoint_rejects_other_modes(self):
        a=make_algorithm(self.p/'lla');a._epoch_rng=capture_rng()
        a.save_model('initial.pth',self.p/'lla')
        b=make_algorithm(self.p/'fixed',mode='fixed')
        with self.assertRaisesRegex(ValueError,'protocol mismatch'):b.load_model(self.p/'lla/initial.pth')

    def test_cpu_data_parallel(self):
        a=make_algorithm(self.p)
        a.model=nn.DataParallel(a.model);a.ema_model=nn.DataParallel(a.ema_model)
        a.finalize_model_setup();loss,_=a.train_step(**batch(a))
        loss['loss'].backward();self.assertIsNone(a.module.lla.lambda_logits.grad)
        loss['lambda_loss'].backward();self.assertIsNotNone(a.module.lla.lambda_logits.grad)


def real_smoke(batch_root=BATCH):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    from datetime import datetime
    output=batch_root/'smoke_tests'/datetime.now().strftime('attempt_%Y%m%d_%H%M%S')
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for mode,no_dea in [('fixed',False),('ddde_kl',False),('lla_hard',False),('lla_hard',True)]:
        name=mode+('_no_dea' if no_dea else '')
        a=make_algorithm(output/name,mode=mode,no_dea=no_dea)
        initial=copy.deepcopy(a.model.state_dict());a.train()
        if mode=='lla_hard':
            assert not torch.equal(initial['lla.lambda_logits'],a.module.lla.lambda_logits)
        a.load_model(output/name/'audit/model_best.pth')
        before=a.evaluate(return_logits=True)['eval/logits']
        a.load_model(output/name/'audit/model_best.pth')
        after=a.evaluate(return_logits=True)['eval/logits'];np.testing.assert_array_equal(before,after)
        rows.append(dict(mode=name,real_signal_train_steps=6,validation_samples=450,
            lambda_changed=mode=='lla_hard',save_reload_predictions_equal=True,
            finite=True,test_evaluated=False))
        del a;gc.collect()
    resumes=[]
    for mode,no_dea in [('fixed',False),('ddde_kl',False),('lla_hard',False),('lla_hard',True)]:
        name=mode+('_no_dea' if no_dea else '')
        a=make_algorithm(output/('continuous_'+name),mode=mode,no_dea=no_dea,steps=8,epochs=2)
        a.train();want=copy.deepcopy(a.get_save_dict())
        b=make_algorithm(output/('interrupted_'+name),mode=mode,no_dea=no_dea,steps=8,epochs=2)
        b.train(stop_after=2)
        path=output/('interrupted_'+name)/'audit/interrupted_model.pth'
        c=make_algorithm(output/('resumed_'+name),mode=mode,no_dea=no_dea,steps=8,epochs=2)
        c.load_model(path);c.train();got=c.get_save_dict()
        for key in ('model','ema_model','optimizer','lambda_optimizer','scheduler','lambda_scheduler',
                    'loss_scaler','legacy_metaexpert','it','epoch','best_it','best_eval_f1','optimizer_steps','rng','sampler_state'):
            if key in want or key in got:
                try:equal(want[key],got[key])
                except AssertionError as error:raise AssertionError(f'Resume mismatch {name} / {key}: {error}') from error
        if mode=='ddde_kl':equal(want['cola_two_kl'],got['cola_two_kl'])
        resumes.append(dict(mode=name,interruption_step=2,total_steps=8,bitwise_equal=True,
            checked=['theta','lambda','theta_optimizer','lambda_optimizer','EMA','RNG','sampler','step',
                     'scheduler states (both None under legacy pic protocol)']))
        del a,b,c,want,got;gc.collect()
    # Original production worker counts (2 labeled, 4 unlabeled): replay must
    # also reconstruct prefetched augmentation streams.
    a=make_algorithm(output/'workers_continuous',workers=2,steps=8,epochs=2)
    a.train();want=copy.deepcopy(a.get_save_dict())
    b=make_algorithm(output/'workers_interrupted',workers=2,steps=8,epochs=2)
    b.train(stop_after=2)
    c=make_algorithm(output/'workers_resumed',workers=2,steps=8,epochs=2)
    c.load_model(output/'workers_interrupted/audit/interrupted_model.pth');c.train()
    got=c.get_save_dict()
    for key in ('model','ema_model','optimizer','lambda_optimizer','rng','sampler_state','it','best_eval_f1'):
        equal(want[key],got[key])
    resumes.append(dict(mode='lla_hard_workers2',labeled_workers=2,unlabeled_workers=4,
                        bitwise_equal=True,total_steps=8,interruption_step=2))
    del a,b,c,want,got;gc.collect()
    # Repeated train/eval calls must not retain graphs or grow without bound.
    a=make_algorithm(output/'memory');samples=[]
    import psutil
    process=psutil.Process()
    for i in range(16):
        a.it=i;a.epoch=20
        a.out_dict,a.log_dict=a.train_step(**batch(a))
        a.call_hook('after_train_step',hook_name='ParamUpdateHook')
        del a.out_dict;gc.collect()
        samples.append(process.memory_info().rss)
    growth=max(samples[8:])-min(samples[8:])
    assert growth < 128*1024**2, samples
    result=dict(status='passed',source_sha256=source_digests(),device='CPU',cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        real_data=True,full_original_backbone=True,short_runs=rows,resume=resumes,
        memory_rss_bytes=samples,memory_tail_growth_bytes=growth,
        memory_check='16 updates, latter 8 RSS range below 128 MiB; not a long-run guarantee',
        cuda_amp_and_multi_gpu_hardware_tested=False,test_set_evaluated=False)
    result['attempt_output']=str(output.resolve())
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (batch_root/'smoke_tests/result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('REAL SMOKE AND RESUME PASSED',flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true');p.add_argument('--units',action='store_true')
    args=p.parse_args()
    if torch.cuda.is_initialized():raise RuntimeError('CPU verification must not initialize CUDA')
    if args.smoke:real_smoke()
    else:
        suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        (BATCH/'unit_tests/result.json').write_text(json.dumps(dict(status='passed' if result.wasSuccessful() else 'failed',
            count=result.testsRun,failures=len(result.failures),errors=len(result.errors),device='CPU',source_sha256=source_digests()),indent=2)+'\n')
        sys.exit(not result.wasSuccessful())
