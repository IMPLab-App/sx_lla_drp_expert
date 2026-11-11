"""Gate the detached Full-model six-case queue on actual verification."""
import argparse
import json
from pathlib import Path
import sys

from prepare_metaexpert_full_lb35 import digest


def main(batch):
    manifest=json.loads((batch/'manifest.json').read_text())
    proof=json.loads((batch/'verification.json').read_text())
    assert proof['status']=='passed' and proof['full_model_enabled'] and proof['tasks']==6
    if not manifest['smoke']:
        assert proof['all_six_smoke_train_reload_tests_passed']
        assert len(proof['actual_gpu_dynamic_execution'])==3
    for name,expected in json.loads((batch/'source_sha256.json').read_text()).items():
        assert digest(Path(manifest['code_snapshot'])/name)==expected,name
    configs=json.loads((batch/'config_sha256.json').read_text())
    for task in manifest['tasks']:assert digest(task['config'])==configs[task['name']]
    assert digest(manifest['data_file'])==manifest['data_sha256']
    for name,expected in json.loads((batch/'cache_sha256.json').read_text()).items():
        assert digest(batch/'data'/name)==expected,name
    from run_signal_baselines import main as run_queue
    sys.argv=[str(batch/'code/run_signal_baselines.py'),str(batch)]
    return run_queue()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    args=parser.parse_args();sys.exit(main(args.batch.resolve()))
