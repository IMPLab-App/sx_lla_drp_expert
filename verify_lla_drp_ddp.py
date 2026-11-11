"""CPU/Gloo two-rank DRP moments and Full/M2 resume equivalence audit."""
from datetime import datetime
import argparse
import copy
import json
from pathlib import Path
import tempfile
import torch
import torch.multiprocessing as mp
from verify_lla_drp import make,source_digests
from verify_lla_e3 import equal
from semilearn.imb_algorithms.metaexpert.drp_e3 import DRPE3Controller


def construct(output,rank,fixed):
    a=make(output,fixed=fixed)
    a.distributed=True;a.world_size=2;a.rank=rank;a.ngpus_per_node=2
    a.args.distributed=True;a.args.world_size=2;a.args.rank=rank
    a.loader_dict=a.set_data_loader()
    a.model=torch.nn.parallel.DistributedDataParallel(a.model,find_unused_parameters=True,broadcast_buffers=False)
    wrapper=torch.nn.Module();wrapper.add_module('module',a.ema_model);a.ema_model=wrapper
    a.finalize_model_setup()
    return a


def worker(rank,init_file,output):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.distributed.init_process_group('gloo',init_method='file://'+init_file,rank=rank,world_size=2)
    p=torch.tensor([[.6,.3,.1],[.4,.4,.2]]) if rank==0 else torch.tensor([[.8,.1,.1]])
    c=DRPE3Controller(3,momentum=0);c(p)
    global_p=torch.tensor([[.6,.3,.1],[.4,.4,.2],[.8,.1,.1]])
    torch.testing.assert_close(c.mu_max,global_p.max(-1).values.mean())
    torch.testing.assert_close(c.var_max,global_p.max(-1).values.var(unbiased=True))
    output=Path(output);rows=[]
    for fixed in (False,True):
        mode='M2' if fixed else 'Full'
        a=construct(output/(mode+'_continuous'),rank,fixed);a.train();want=copy.deepcopy(a.get_save_dict())
        b=construct(output/(mode+'_interrupted'),rank,fixed);b.train(stop_after=2)
        torch.distributed.barrier()
        c=construct(output/(mode+'_resumed'),rank,fixed)
        c.load_model(output/(mode+'_interrupted')/'audit/interrupted_model.pth');c.train();got=c.get_save_dict()
        for key in ('model','ema_model','optimizer','lambda_optimizer','drp_state','mff_gradient_monitor','rng','it','epoch','best_it','best_eval_f1'):
            try:equal(want[key],got[key])
            except AssertionError as e:raise AssertionError(f'{mode} rank={rank} key={key}: {e}') from e
        states=[None,None];torch.distributed.all_gather_object(states,c.drp.state_dict());equal(states[0],states[1])
        rows.append(dict(mode=mode,resume_bitwise_equal=True,global_drp_ema_identical=True,mff_nonzero_steps=c._mff_nonzero_steps))
    (output/f'rank{rank}.json').write_text(json.dumps(dict(rank=rank,passed=True,runs=rows),indent=2)+'\n')
    torch.distributed.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True);a=p.parse_args()
    output=a.batch.resolve()/'unit_tests'/datetime.now().strftime('gloo_%Y%m%d_%H%M%S');output.mkdir(exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='drp_gloo_') as temp:mp.spawn(worker,args=(str(Path(temp)/'init'),str(output)),nprocs=2,join=True)
    rows=[json.loads((output/f'rank{i}.json').read_text()) for i in (0,1)]
    (a.batch/'unit_tests/ddp_result.json').write_text(json.dumps(dict(status='passed',device='CPU',backend='gloo',
        world_size=2,ranks=rows,source_sha256=source_digests(),output=str(output)),indent=2)+'\n')
