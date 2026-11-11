"""CPU numerical, integration and real-signal resume audits for LLA+DRP."""
import argparse
import ast
import contextlib
import copy
from datetime import datetime
import gc
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from semilearn.imb_algorithms.metaexpert.drp_e3 import DRPE3Controller
from semilearn.imb_algorithms.metaexpert.lla_drp_algorithm import LLADRPMetaExpert
from semilearn.imb_algorithms.metaexpert.lla_algorithm import prediction_paths
from semilearn.imb_algorithms.metaexpert.lla_runtime import capture_rng,restore_rng
from verify_lla_e3 import equal

ROOT=Path(__file__).resolve().parent
BATCH=Path(os.environ.get('LLA_DRP_BATCH',ROOT/'logs/lla_drp_mff_dea_ablation_20261111'))


def source_digests():
    files=['semilearn/imb_algorithms/metaexpert/'+name for name in
        ('drp_e3.py','lla_drp_algorithm.py','lla_e3.py','lla_algorithm.py','lla_runtime.py','metaexpert.py','bacn_ablation.py','cola_ddde.py')]
    files += ['verify_lla_drp.py','verify_lla_drp_gpu.py','verify_lla_drp_ddp.py','train_lla_drp.py','train_lla_e3.py','lla_experiment_utils.py',
              'semilearn/nets/vit/vit_metaexpert.py','semilearn/core/algorithmbase.py','semilearn/datasets/cv_datasets/raw_signal.py']
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}


def make(output,drp=True,fixed=False,mode='lla_hard',hard_mask=False,legacy=False,workers=0,steps=9,epochs=3,case=1):
    from train import get_config
    from semilearn.imb_algorithms import get_imb_algorithm
    from semilearn.core.utils import get_net_builder
    m=json.loads((BATCH/'manifest.json').read_text())
    task=next(t for t in m['tasks'] if t['scenario']==f'LU{case}' and t['version']=='M0')
    with patch.object(sys,'argv',['verify','--c',task['config']]):args=get_config()
    args.e3_drp_enabled=drp;args.ablation_fixed_dea_weights=fixed;args.e3_adjust_mode=mode
    args.ablation_no_ddde_kl=mode!='ddde_kl';args.drp_with_hard_mask=hard_mask
    if legacy:args.imb_algorithm='metaexpert_lla_e3'
    args.num_workers=workers;args.distributed=False;args.gpu=0
    args.num_train_iter=steps;args.epoch=epochs;args.num_eval_iter=3;args.num_log_iter=1
    args.est_epoch=0;args.batch_size=4;args.uratio=2;args.eval_batch_size=128
    args.save_dir=str(output);args.save_name='audit'
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    with contextlib.redirect_stdout(io.StringIO()),patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self):
        a=get_imb_algorithm(args,get_net_builder(args.net,args.net_from_name),None,None)
    a.device=torch.device('cpu');a.finalize_model_setup();a.epoch=0
    return a


def data(a):
    lb=next(iter(a.loader_dict['train_lb']));u=next(iter(a.loader_dict['train_ulb']))
    return a.process_batch(**lb,**u)


def official():
    tree=ast.parse((BATCH/'references/sage_utils.py').read_text())
    node=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='WeightingHook')
    namespace={'torch':torch,'MaskingHook':type('MaskingHook',(),{})}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned SAGE WeightingHook>','exec'),namespace)
    return namespace['WeightingHook']


class DRPNumericalTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(2)

    def test_initialization(self):
        c=DRPE3Controller(9)
        self.assertEqual(c.num_classes,9);self.assertAlmostEqual(float(c.mu_max),1/9,places=7)
        self.assertAlmostEqual(float(c.mu_gap),1/9,places=7)
        self.assertEqual(float(c.var_max),1);self.assertEqual(float(c.var_gap),1)
        self.assertEqual(len(list(c.parameters())),0)
        with self.assertRaises(ValueError):DRPE3Controller(9,per_class=True)

    def test_top_two_unbiased_moments_and_update_order(self):
        c=DRPE3Controller(3,momentum=0)
        p=torch.tensor([[.6,.3,.1],[.4,.4,.2]])
        w,confidence,gap=c(p)
        torch.testing.assert_close(confidence,torch.tensor([.6,.4]))
        torch.testing.assert_close(gap,torch.tensor([.3,0.]))
        torch.testing.assert_close(c.mu_max,confidence.mean())
        torch.testing.assert_close(c.var_max,confidence.var(unbiased=True))
        expected=torch.exp(-(confidence-c.mu_max).clamp_max(0).square()/(2*(c.var_max/4+1e-8))
                           -(gap-c.mu_gap).clamp_max(0).square()/(2*(c.var_gap/4+1e-8)))
        torch.testing.assert_close(w,expected)

    def test_official_reference_equivalence(self):
        from types import SimpleNamespace
        oracle=official()(9);c=DRPE3Controller(9)
        algorithm=SimpleNamespace(distributed=False,world_size=1)
        for _ in range(8):
            p=torch.randn(32,9).softmax(-1)
            expected=oracle.masking(algorithm,p,softmax_x_ulb=False)
            actual,_,_=c(p)
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
            for name,other in (('mu_max','prob_max_mu_t'),('mu_gap','prob_gap_mu_t'),('var_max','prob_max_var_t'),('var_gap','prob_gap_var_t')):
                torch.testing.assert_close(getattr(c,name),getattr(oracle,other),rtol=1e-6,atol=1e-7)

    def test_singleton_empty_and_zero_variance(self):
        c=DRPE3Controller(9,momentum=0)
        p=torch.ones(1,9)/9
        w,_,_=c(p);self.assertTrue(torch.isfinite(w).all());self.assertEqual(float(c.var_max),0)
        updates=int(c.updates);w,_,_=c(torch.empty(0,9));self.assertEqual(w.numel(),0);self.assertEqual(int(c.updates),updates)
        c.mu_max.fill_(.99);c.mu_gap.fill_(.99)
        w,_,_=c(p,update=False);self.assertTrue(((w>0)&(w<=1)).all())
        self.assertTrue(torch.isfinite(w).all())

    def test_detach_amp_and_range(self):
        c=DRPE3Controller(9)
        logits=torch.randn(11,9,requires_grad=True)
        with torch.autocast('cpu',dtype=torch.bfloat16):w,q,g=c(logits.softmax(-1))
        self.assertEqual(w.dtype,torch.float32);self.assertFalse(w.requires_grad)
        self.assertFalse(q.requires_grad);self.assertFalse(g.requires_grad)
        self.assertTrue(((w>0)&(w<=1)).all())

    def test_restore_and_next_update_exact(self):
        c=DRPE3Controller(9);c(torch.randn(5,9).softmax(-1))
        d=DRPE3Controller(9);d.load_state_dict(copy.deepcopy(c.state_dict()))
        p=torch.randn(7,9).softmax(-1)
        torch.testing.assert_close(c(p)[0],d(p)[0],rtol=0,atol=0);equal(c.state_dict(),d.state_dict())

    def test_invalid_probabilities_fail(self):
        c=DRPE3Controller(9)
        for p in (torch.zeros(2,9),torch.full((2,9),float('nan')),torch.ones(2,8)/8):
            with self.assertRaises(ValueError):c(p)

    def test_ema_formula(self):
        c=DRPE3Controller(3,momentum=.9);p=torch.tensor([[.6,.3,.1],[.4,.4,.2]])
        c(p);q=p.max(-1).values
        torch.testing.assert_close(c.mu_max,.9*torch.tensor(1/3)+.1*q.mean())
        torch.testing.assert_close(c.var_max,.9*torch.tensor(1.)+.1*q.var(unbiased=True))


class ModelIntegrationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup();gc.collect()

    def test_framework_signature(self):
        self.assertEqual(list(inspect.signature(LLADRPMetaExpert.train_step).parameters),
            ['self','x_lb','y_lb','y_lb_noised','x_ulb_w','x_ulb_s','y_ulb'])

    def test_disabled_regression_lla_fixed_and_ddde(self):
        for mode in ('lla_hard','fixed','ddde_kl'):
            a=make(self.p/('new_'+mode),drp=False,mode=mode)
            b=make(self.p/('old_'+mode),drp=False,mode=mode,legacy=True)
            equal(a.model.state_dict(),b.model.state_dict())
            for epoch in (0,3):
                a.epoch=b.epoch=epoch;a.it=b.it=0
                inputs=data(a);rng=capture_rng()
                x,l=a.train_step(**inputs);restore_rng(rng);y,k=b.train_step(**inputs)
                torch.testing.assert_close(x['loss'],y['loss'],rtol=0,atol=0)
                if mode=='lla_hard':torch.testing.assert_close(x['lambda_loss'],y['lambda_loss'],rtol=0,atol=0)
                a.optimizer.zero_grad();b.optimizer.zero_grad();x['loss'].backward();y['loss'].backward()
                for (name,p),(other,q) in zip(a.model.named_parameters(),b.model.named_parameters()):
                    self.assertEqual(name,other)
                    if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
                a.optimizer.step();b.optimizer.step();equal(a.model.state_dict(),b.model.state_dict())
                if a.cola is not None:equal(a.cola.state_dict(),b.cola.state_dict())

    def test_drp_only_changes_e3_unlabeled_objective(self):
        a=make(self.p/'full');b=make(self.p/'lla',drp=False)
        a.epoch=b.epoch=3
        equal(a.model.state_dict(),b.model.state_dict())
        inputs=data(a);rng=capture_rng();x,l=a.train_step(**inputs)
        restore_rng(rng);y,k=b.train_step(**inputs)
        self.assertEqual(l['train/sup_loss'],k['train/sup_loss'])
        self.assertEqual(l['train/unsup_loss'],k['train/unsup_loss'])
        self.assertEqual(l['train/sup_loss_e3'],k['train/sup_loss_e3'])
        self.assertEqual(l['train/total_fuse_loss'],k['train/total_fuse_loss'])
        torch.testing.assert_close(x['lambda_loss'],y['lambda_loss'],rtol=0,atol=0)
        self.assertGreater(l['train/unsup_loss_e3'],0)
        self.assertEqual(a._drp_last['effective_nonzero_samples'],len(inputs['x_ulb_w']))
        lam=a.module.lla.lambda_logits
        self.assertIsNone(torch.autograd.grad(x['loss'],lam,allow_unused=True,retain_graph=True)[0])
        theta=[p for p in a.model.parameters() if p.requires_grad and p is not lam]
        self.assertTrue(all(g is None for g in torch.autograd.grad(x['lambda_loss'],theta,allow_unused=True)))

    def test_weighted_ht_normalization(self):
        a=make(self.p);inputs=data(a);n=len(inputs['x_lb'])
        out=a.model(torch.cat((inputs['x_lb'],inputs['x_ulb_w'],inputs['x_ulb_s'])))
        weights=torch.linspace(.1,.8,len(inputs['x_ulb_w']))
        def weighting(probs):
            top=probs.topk(2,-1).values
            return weights,top[:,0],top[:,0]-top[:,1]
        with patch.object(a.drp,'forward',side_effect=weighting):_,got,_=a._lla_expert_loss(out,n,inputs['y_lb'])
        weak,_=out['aux_logits2'][n:].chunk(2);pseudo=a.module.lla.adjust(weak.detach()).argmax(-1)
        _,hs=out['aux_logitsH2'][n:].chunk(2);_,ts=out['aux_logitsT2'][n:].chunk(2)
        tw=weights*(pseudo>=a.cut1)
        want=((a.ce_loss(hs,pseudo,reduction='none')*weights).sum()+(a.ce_loss(ts,pseudo,reduction='none')*tw).sum())/(weights.sum()+tw.sum()+1e-12)
        torch.testing.assert_close(got,want)

    def test_fixed_dea_retains_mff_forward_and_effective_gradients(self):
        a=make(self.p,fixed=True);a.epoch=3
        inputs=data(a);seen=[];handles=[]
        for i in range(1,9):handles.append(getattr(a.module,f'lin{i}').register_forward_hook(lambda *args:seen.append(True)))
        captured=[];handles.append(a.model.register_forward_hook(lambda m,x,o:captured.append(o)))
        losses,logs=a.train_step(**inputs)
        for h in handles:h.remove()
        self.assertEqual(len(seen),8)
        out=captured[0];paths=prediction_paths(out,a.module.lla)
        torch.testing.assert_close(paths['fusion_main'],.5*out['logits']+.5*a.module.lla.adjust(out['aux_logits2']))
        self.assertTrue(torch.all(out['w1']==.5));self.assertTrue(torch.all(out['w3']==.5))
        self.assertIn('fuse_w_logit',out);self.assertIn('dynamic_w1',out)
        losses['loss'].backward()
        for i in range(1,9):
            p=getattr(a.module,f'lin{i}')[0].weight
            self.assertIsNotNone(p.grad);self.assertGreater(float(p.grad.norm()),0.)
        self.assertGreater(float(a.module.predict[0].weight.grad.norm()),0.)
        self.assertIsNone(a.module.lla.lambda_logits.grad)
        self.assertGreater(logs['train/total_fuse_loss'],0.)
        a.assert_optimizer_coverage()

    def test_mff_aux_loss_warmup_unchanged(self):
        a=make(self.p,fixed=True);inputs=data(a)
        for epoch in (0,1):
            a.epoch=epoch;_,log=a.train_step(**inputs);self.assertEqual(log['train/total_fuse_loss'],0.)
        a.epoch=2;_,log=a.train_step(**inputs);self.assertGreater(log['train/total_fuse_loss'],0.)

    def test_hard_mask_is_separate_diagnostic(self):
        a=make(self.p,hard_mask=True);_,logs=a.train_step(**data(a))
        self.assertEqual(a._drp_last['effective_nonzero_samples'],0)
        self.assertEqual(logs['train/unsup_loss_e3'],0.)
        self.assertTrue(a._drp_last['hard_mask_enabled'])

    def test_u_true_labels_never_read(self):
        a=make(self.p);inputs=data(a);rng=capture_rng();state=copy.deepcopy(a.drp.state_dict())
        x,_=a.train_step(**inputs);stats=copy.deepcopy(a.drp.state_dict())
        a.drp.load_state_dict(state);restore_rng(rng);inputs['y_ulb']=torch.full_like(inputs['y_ulb'],987654)
        y,_=a.train_step(**inputs)
        torch.testing.assert_close(x['loss'],y['loss'],rtol=0,atol=0)
        torch.testing.assert_close(x['lambda_loss'],y['lambda_loss'],rtol=0,atol=0)
        equal(stats,a.drp.state_dict())

    def test_eval_does_not_update_drp_and_uses_ema_lambda(self):
        from semilearn.core.utils import EMA
        a=make(self.p,fixed=True);a.train_step(**data(a));before=copy.deepcopy(a.drp.state_dict())
        a.ema=EMA(a.model,a.ema_m);a.ema.register()
        with torch.no_grad():a.module.lla.lambda_logits.copy_(torch.arange(9).float())
        online=a.module.lla.lambda_logits.detach().clone()
        with patch.object(a.model,'forward',wraps=a.model.forward) as forward:a.evaluate()
        self.assertEqual(forward.call_count,len(a.loader_dict['eval']))
        equal(before,a.drp.state_dict());torch.testing.assert_close(online,a.module.lla.lambda_logits,rtol=0,atol=0)
        with self.assertRaises(RuntimeError):a.evaluate('test')

    def test_amp_full_model(self):
        a=make(self.p,fixed=True);a.epoch=3
        a.use_amp=True;a.loss_scaler=torch.amp.GradScaler('cpu')
        a.amp_cm=lambda:torch.autocast('cpu',dtype=torch.bfloat16)
        a.out_dict,a.log_dict=a.train_step(**data(a))
        a.call_hook('after_train_step',hook_name='ParamUpdateHook')
        self.assertGreater(a.log_dict['train/mff_gradient_norm'],0.)
        self.assertTrue(torch.isfinite(a.module.lla.lambda_logits).all())


def smoke():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    output=BATCH/'smoke_tests'/datetime.now().strftime('cpu_%Y%m%d_%H%M%S');output.mkdir(exist_ok=False)
    modes=[('Full',True,False,'lla_hard',False),('M2',True,True,'lla_hard',False),
           ('L0',False,False,'lla_hard',False),('M1',False,False,'fixed',False),
           ('hard_mask_diagnostic',True,False,'lla_hard',True)]
    runs=[];resumes=[]
    for name,drp,fixed,mode,hard in modes:
        a=make(output/name,drp,fixed,mode,hard);initial=copy.deepcopy(a.model.state_dict());a.train()
        completed_mff_nonzero=a._mff_nonzero_steps
        a.load_model(output/name/'audit/model_best.pth');p=a.evaluate(return_logits=True)['eval/logits']
        a.load_model(output/name/'audit/model_best.pth');q=a.evaluate(return_logits=True)['eval/logits'];np.testing.assert_array_equal(p,q)
        if mode=='lla_hard':assert not torch.equal(initial['lla.lambda_logits'],a.module.lla.lambda_logits)
        if fixed:assert completed_mff_nonzero>0
        runs.append(dict(mode=name,steps=9,real_signal=True,original_backbone=True,save_reload_equal=True,
                         mff_nonzero_steps=completed_mff_nonzero,test_evaluated=False))
        del a;gc.collect()
    for name,fixed,workers in [('Full',False,0),('M2',True,0),('Full_workers2',False,2)]:
        a=make(output/('continuous_'+name),fixed=fixed,workers=workers);a.train();want=copy.deepcopy(a.get_save_dict())
        b=make(output/('interrupted_'+name),fixed=fixed,workers=workers);b.train(stop_after=2)
        c=make(output/('resumed_'+name),fixed=fixed,workers=workers)
        c.load_model(output/('interrupted_'+name)/'audit/interrupted_model.pth');c.train();got=c.get_save_dict()
        for key in ('model','ema_model','optimizer','lambda_optimizer','loss_scaler','rng','sampler_state',
                    'drp_state','mff_gradient_monitor','it','epoch','best_it','best_eval_f1'):
            try:equal(want[key],got[key])
            except AssertionError as error:raise AssertionError(f'{name} resume mismatch: {key}: {error}') from error
        resumes.append(dict(mode=name,interrupted_after=2,steps=9,workers=workers,bitwise_equal=True))
        del a,b,c,want,got;gc.collect()
    result=dict(status='passed',device='CPU',source_sha256=source_digests(),attempt_output=str(output),
                runs=runs,resume=resumes,test_evaluated=False,cuda_hardware_verified=False)
    (BATCH/'smoke_tests/result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DRP REAL SIGNAL SMOKE AND RESUME PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if torch.cuda.is_initialized():raise RuntimeError('CPU verification must not initialize CUDA')
    if args.smoke:smoke()
    else:
        tests=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
        result=unittest.TextTestRunner(verbosity=2).run(tests)
        (BATCH/'unit_tests/result.json').write_text(json.dumps(dict(status='passed' if result.wasSuccessful() else 'failed',
            count=result.testsRun,failures=len(result.failures),errors=len(result.errors),source_sha256=source_digests()),indent=2)+'\n')
        sys.exit(not result.wasSuccessful())
