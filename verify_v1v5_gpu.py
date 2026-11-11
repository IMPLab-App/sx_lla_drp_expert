"""Real CUDA audit: V5 old-model equivalence and all eight control modes."""
import argparse
import gc
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
from verify_bacn_ablation import make_algorithm,assert_state_equal,optimizer_signature


def audit(batch):
    m=json.loads((batch/'manifest.json').read_text());torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic=True;equivalence=[];modes=[]
    with tempfile.TemporaryDirectory(prefix='v1v5_gpu_') as folder:
        folder=Path(folder)
        for scenario in ('r1','r12','lu3','lu5'):
            t=next(t for t in m['tasks'] if t['scenario']==scenario and t['version']=='V5')
            old=make_algorithm(t['config'],'metaexpert_bacn',False,False,folder/'old')
            new=make_algorithm(t['config'],'metaexpert_v1v5',False,False,folder/'new')
            assert_state_equal(old.model.state_dict(),new.model.state_dict())
            assert optimizer_signature(old)==optimizer_signature(new)
            lb=next(iter(torch.utils.data.DataLoader(old.dataset_dict['train_lb'],batch_size=8)))
            ulb=next(iter(torch.utils.data.DataLoader(old.dataset_dict['train_ulb'],batch_size=8)))
            inputs=old.process_batch(**lb,**ulb)
            for epoch in (0,20):
                if epoch==20:
                    features=torch.randn(27,192,device='cuda');logits=torch.eye(9,device='cuda').repeat_interleave(3,0)*20
                    for a in (old,new):a.cola.begin_epoch(18);a.cola.observe(logits,features)
                old.epoch=new.epoch=epoch;old.it=new.it=0
                cpu,cuda=torch.get_rng_state(),torch.cuda.get_rng_state()
                old.optimizer.zero_grad(set_to_none=True);a,_=old.train_step(**inputs);a['loss'].backward()
                torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda)
                new.optimizer.zero_grad(set_to_none=True);b,_=new.train_step(**inputs);b['loss'].backward()
                torch.testing.assert_close(a['loss'],b['loss'],rtol=0,atol=0)
                for (name,p),(other,q) in zip(old.model.named_parameters(),new.model.named_parameters()):
                    assert name==other and (p.grad is None)==(q.grad is None)
                    if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0,msg=name)
                old.optimizer.step();new.optimizer.step();assert_state_equal(old.model.state_dict(),new.model.state_dict())
                assert_state_equal(old.cola.state_dict(),new.cola.state_dict())
            equivalence.append(dict(scenario=scenario,initialization_loss_gradients_updates_and_ddde_bitwise_equal=True))
            print('PASS V5 exact existing Full equivalence:',scenario,flush=True)
            del old,new,inputs,lb,ulb,a,b,p,q,features,logits;gc.collect();torch.cuda.empty_cache()
        for version in m['methods']:
            t=next(t for t in m['tasks'] if t['scenario']=='lu3' and t['version']==version)
            a=make_algorithm(t['config'],'metaexpert_v1v5',False,False,folder/version)
            lb=next(iter(torch.utils.data.DataLoader(a.dataset_dict['train_lb'],batch_size=8)))
            ulb=next(iter(torch.utils.data.DataLoader(a.dataset_dict['train_ulb'],batch_size=8)))
            inputs=a.process_batch(**lb,**ulb)
            if a.cola is not None:
                a.cola.begin_epoch(18)
                a.cola.observe(torch.eye(9,device='cuda').repeat_interleave(3,0)*20,torch.randn(27,192,device='cuda'))
            a.epoch,a.it=20,0;captured=[]
            hook=a.model.register_forward_hook(lambda mod,args,out:captured.append(out))
            if a.cola is None:
                with patch('semilearn.imb_algorithms.metaexpert.five_variants.symmetric_kl',side_effect=AssertionError('Static KL')),\
                     patch('semilearn.imb_algorithms.metaexpert.five_variants.effective_rank',side_effect=AssertionError('Static DDDE')):
                    result,logs=a.train_step(**inputs)
            else:result,logs=a.train_step(**inputs)
            hook.remove();assert torch.isfinite(result['loss'])
            a.optimizer.zero_grad(set_to_none=True);result['loss'].backward()
            for p in (a.model.predict[0].weight,a.model.lin1[0].weight,a.model.backbone.classifier.weight,a.model.backbone.aux_classifier2.weight):
                assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
            assert a.tau_lb3==float(a.model.tau3)
            if a.cola is None:assert a.tau_lb3==a.spec['fixed_tau']
            else:assert 0<=a.tau_lb3<4
            if version=='V2':assert 'train/KL_uni' not in logs
            a.optimizer.step();dest=folder/version/'checkpoint';a.save_model('audit.pth',str(dest))
            saved=torch.load(dest/'audit.pth',map_location='cpu',weights_only=False)
            if a.cola is not None:a.cola.pi_u.zero_()
            a.tau_lb3=-1;a.load_model(str(dest/'audit.pth'))
            a.ema_model.load_state_dict(saved['ema_model']);a.ema_model.tau3.fill_(-1);a.restore_ddde_state(saved)
            assert float(a.ema_model.tau3)==a.tau_lb3
            if a.cola is not None:assert_state_equal({k:v.cpu() for k,v in a.cola.state_dict().items()},saved['cola_two_kl'])
            else:assert 'cola_two_kl' not in saved
            wrong=dict(saved);wrong['five_variant_settings']=dict(saved['five_variant_settings'],version='wrong')
            try:a.restore_ddde_state(wrong)
            except ValueError:pass
            else:raise AssertionError('Mismatched version accepted')
            modes.append(dict(version=version,tau=a.tau_lb3,settings=a.variant_settings(),dea_gradients=True,checkpoint_and_eval_restore=True))
            print('PASS mode and restore:',version,flush=True)
            del a,inputs,lb,ulb,result,captured,p,saved,wrong;gc.collect();torch.cuda.empty_cache()
    result=dict(status='passed',V5_equivalence=equivalence,modes=modes,model_sha256=m['reference_model_sha256'])
    (batch/'gpu_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('batch',type=Path);a=p.parse_args();audit(a.batch.resolve())
