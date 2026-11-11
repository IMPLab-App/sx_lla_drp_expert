"""Two-process CPU/Gloo full-model resume audit; no GPU contexts are created."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import tempfile
import torch
import torch.multiprocessing as mp
from verify_lla_e3 import make_algorithm, equal
from semilearn.imb_algorithms.metaexpert.lla_runtime import capture_rng


def construct(output, rank):
    a=make_algorithm(output,steps=6,epochs=2)
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
    output=Path(output)
    a=construct(output/'continuous',rank);a.train();want=copy.deepcopy(a.get_save_dict())
    b=construct(output/'interrupted',rank);b.train(stop_after=1)
    torch.distributed.barrier()
    c=construct(output/'resumed',rank);c.load_model(output/'interrupted/audit/interrupted_model.pth')
    c.train();got=c.get_save_dict()
    for key in ('model','ema_model','optimizer','lambda_optimizer','rng','it','epoch','best_it','best_eval_f1'):
        try:equal(want[key],got[key])
        except AssertionError as e:raise AssertionError(f'rank={rank}, key={key}: {e}') from e
    values=[torch.empty_like(c.module.lla.lambda_logits) for _ in range(2)]
    torch.distributed.all_gather(values,c.module.lla.lambda_logits.detach())
    equal(values[0],values[1])
    (output/f'rank{rank}.json').write_text(json.dumps(dict(rank=rank,passed=True,
        lambda_identical_across_ranks=True,continuous_resume_bitwise_equal=True),indent=2)+'\n')
    torch.distributed.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True);args=p.parse_args()
    output=args.batch.resolve()/'unit_tests'/datetime.now().strftime('ddp_%Y%m%d_%H%M%S')
    output.mkdir(exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='lla_gloo_') as temporary:
        mp.spawn(worker,args=(str(Path(temporary)/'init'),str(output)),nprocs=2,join=True)
    records=[json.loads((output/f'rank{i}.json').read_text()) for i in range(2)]
    (args.batch/'unit_tests/ddp_result.json').write_text(json.dumps(dict(status='passed',
        backend='gloo',device='CPU',world_size=2,ranks=records,output=str(output)),indent=2)+'\n')
