"""Independent GPU entry for the preregistered LLA-E3 comparison.

Does not invoke legacy train.main (which can delete existing output and convert
BN to SyncBN). Single-GPU stage-1 runs keep original independent BNH/BNT.
"""
import json
import os
from pathlib import Path
import random
import shutil
import numpy as np
import torch
from semilearn.imb_algorithms.metaexpert.lla_algorithm import LLAE3MetaExpert
from semilearn.imb_algorithms import get_imb_algorithm
from semilearn.core.utils import EMA, TBLog, get_net_builder, get_logger
from train import get_config
from lla_experiment_utils import require_idle_gpu, verify_config_data, sha256


def build_gpu_algorithm(args, output):
    """Construct original architecture; supports single GPU or torchrun DDP."""
    args.world_size=int(os.environ.get('WORLD_SIZE',getattr(args,'world_size',1)))
    args.rank=int(os.environ.get('RANK',0))
    local=int(os.environ.get('LOCAL_RANK',args.gpu or 0))
    args.gpu=local
    args.distributed=args.world_size>1
    if args.distributed and not torch.distributed.is_initialized():
        torch.distributed.init_process_group('nccl',init_method='env://')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.cuda.set_device(local)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    logger=get_logger(args.save_name,str(output),'INFO' if args.rank==0 else 'WARNING')
    a=get_imb_algorithm(args,get_net_builder(args.net,args.net_from_name),None,logger)
    a.model=a.model.cuda(local);a.ema_model=a.ema_model.cuda(local)
    # Original BNH/BNT retained; do not convert to shared/synchronized BN.
    if args.distributed:
        a.model=torch.nn.parallel.DistributedDataParallel(a.model,device_ids=[local],
            broadcast_buffers=False,find_unused_parameters=True)
        # Keep EMA names aligned with the DDP model without allocating a second
        # distributed reducer: EMAHook only needs a matching module prefix.
        wrapper=torch.nn.Module();wrapper.add_module('module',a.ema_model);a.ema_model=wrapper
    a.finalize_model_setup()
    return a


def main():
    args=get_config()
    args.gpu=int(os.environ.get('LLA_GPU_INDEX',args.gpu or 0))
    if args.imb_algorithm!='metaexpert_lla_e3':raise ValueError('Wrong independent algorithm entry')
    verify_config_data(args)
    output=Path(args.save_dir)/args.save_name
    # No directory removal. Existing output is allowed only for explicit resume
    # inside this new experiment root.
    if output.exists() and not args.resume:raise FileExistsError(output)
    rank=int(os.environ.get('RANK',0))
    require_idle_gpu(int(os.environ.get('LOCAL_RANK',args.gpu or 0)))
    if rank==0:
        output.mkdir(parents=True,exist_ok=args.resume)
        shutil.copy2(args.c,output/('resume_config.yaml' if args.resume else 'config.yaml'))
    a=build_gpu_algorithm(args,output)
    if args.resume:a.load_model(args.load_path)
    a.train()
    if not a.distributed or a.rank==0:
        checkpoints={p.name:dict(path=str(p.resolve()),sha256=sha256(p)) for p in output.glob('*.pth')}
        (output/'checkpoint_manifest.json').write_text(json.dumps(dict(
            config_sha256=sha256(args.c),checkpoints=checkpoints,selection='validation_macro_f1',
            test_evaluated=False,protocol=a.protocol_settings()),ensure_ascii=False,indent=2)+'\n')
    if a.distributed:torch.distributed.destroy_process_group()


if __name__=='__main__':main()
