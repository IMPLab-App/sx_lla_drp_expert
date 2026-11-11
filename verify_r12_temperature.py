"""CUDA equivalence audit for non-invasive V2/V5 diagnostics at map T=1."""
import argparse
import gc
import json
from pathlib import Path
import tempfile

import torch

from verify_bacn_ablation import make_algorithm, assert_state_equal, optimizer_signature


def audit(batch):
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic=True
    manifest=json.loads((batch/'manifest.json').read_text())
    records=[]
    with tempfile.TemporaryDirectory(prefix='r12_temperature_cuda_') as temporary:
        for version in ('V2','V5'):
            task=next(t for t in manifest['tasks'] if t['version']==version and t['temperature']==1)
            old=make_algorithm(task['config'],'metaexpert_v1v5',False,False,Path(temporary)/'old')
            new=make_algorithm(task['config'],'metaexpert_temperature_audit',False,False,Path(temporary)/'new')
            assert_state_equal(old.model.state_dict(),new.model.state_dict())
            assert optimizer_signature(old)==optimizer_signature(new)
            assert 'test' not in new.loader_dict
            for key in ('train_lb','train_ulb','eval','test'):
                assert (old.dataset_dict[key].sample_indices==new.dataset_dict[key].sample_indices).all()
            lb=next(iter(torch.utils.data.DataLoader(old.dataset_dict['train_lb'],batch_size=8)))
            ulb=next(iter(torch.utils.data.DataLoader(old.dataset_dict['train_ulb'],batch_size=8)))
            inputs=old.process_batch(**lb,**ulb)
            for epoch in (0,20):
                if epoch==20:
                    logits=torch.eye(9,device='cuda').repeat_interleave(3,0)*20
                    features=torch.randn(27,192,device='cuda')
                    for algorithm in (old,new):
                        algorithm.cola.begin_epoch(18)
                        algorithm.cola.observe(logits,features)
                old.epoch=new.epoch=epoch
                old.it=new.it=0
                cpu,cuda=torch.get_rng_state(),torch.cuda.get_rng_state()
                old.optimizer.zero_grad(set_to_none=True)
                first,_=old.train_step(**inputs)
                first['loss'].backward()
                torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda)
                new.optimizer.zero_grad(set_to_none=True)
                second,_=new.train_step(**inputs)
                second['loss'].backward()
                torch.testing.assert_close(first['loss'],second['loss'],rtol=0,atol=0)
                for (name,p),(other,q) in zip(old.model.named_parameters(),new.model.named_parameters()):
                    assert name==other and (p.grad is None)==(q.grad is None)
                    if p.grad is not None:
                        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0,msg=name)
                old.optimizer.step();new.optimizer.step()
                assert_state_equal(old.model.state_dict(),new.model.state_dict())
                assert_state_equal(old.cola.state_dict(),new.cola.state_dict())
            records.append(dict(version=version,initialization_loss_gradients_update_and_controller_bitwise_equal=True,
                                split_identities_equal=True,automatic_test_loader_removed=True))
            del old,new,inputs,first,second,p,q,lb,ulb,features,logits
            gc.collect();torch.cuda.empty_cache()
    (batch/'gpu_equivalence_audit.json').write_text(json.dumps(dict(status='passed',records=records),indent=2)+'\n')
    print('PASS CUDA T=1 equivalence for V2 and V5',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    audit(parser.parse_args().batch.resolve())
