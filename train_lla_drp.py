"""Independent entry for E3-LLA+DRP; does not edit active LLA-only source."""
import json
import os
from pathlib import Path
import shutil
from semilearn.imb_algorithms.metaexpert.lla_drp_algorithm import LLADRPMetaExpert
from train import get_config
from train_lla_e3 import build_gpu_algorithm
from lla_experiment_utils import require_idle_gpu, verify_config_data, sha256


def main():
    args=get_config()
    if args.imb_algorithm!='metaexpert_lla_drp':raise ValueError('Wrong study algorithm')
    args.gpu=int(os.environ.get('LLA_GPU_INDEX',args.gpu or 0))
    verify_config_data(args)
    out=Path(args.save_dir)/args.save_name
    if out.exists() and not args.resume:raise FileExistsError(out)
    require_idle_gpu(args.gpu)
    out.mkdir(parents=True,exist_ok=args.resume)
    shutil.copy2(args.c,out/('resume_config.yaml' if args.resume else 'config.yaml'))
    a=build_gpu_algorithm(args,out)
    if args.resume:a.load_model(args.load_path)
    a.train()
    source=Path(__file__).resolve().parent
    (out/'checkpoint_manifest.json').write_text(json.dumps(dict(config_sha256=sha256(args.c),
        checkpoints={p.name:dict(path=str(p.resolve()),sha256=sha256(p)) for p in out.glob('*.pth')},
        protocol=a.protocol_settings(),selection='validation_macro_f1',test_evaluated=False,
        source_sha256={str(p.relative_to(source)):sha256(p) for p in (source/'semilearn/imb_algorithms/metaexpert').glob('*.py')},
        mff_gradient_monitor=dict(active=a._mff_active_steps,nonzero=a._mff_nonzero_steps)),indent=2)+'\n')


if __name__=='__main__':main()
