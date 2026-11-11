"""Verify frozen inputs, then enter the existing durable two-GPU queue."""
import argparse
import json
from pathlib import Path
import sys

from prepare_bacn_r6_r12 import digest


def main(batch):
    manifest=json.loads((batch/'manifest.json').read_text())
    audit=json.loads((batch/'verification.json').read_text())
    assert audit['status']=='passed' and audit['tasks']==21
    if not manifest['smoke']:
        assert audit['all_21_smoke_train_reload_tests_passed']
    for name,expected in json.loads((batch/'source_sha256.json').read_text()).items():
        assert digest(Path(manifest['code_snapshot'])/name)==expected,name
    hashes=json.loads((batch/'config_sha256.json').read_text())
    for task in manifest['tasks']:
        assert digest(task['config'])==hashes[task['name']],task['name']
    assert digest(manifest['data_file'])==manifest['data_sha256']
    sys.argv=[str(batch/'code/run_signal_baselines.py'),str(batch)]
    from run_signal_baselines import main as queue_main
    return queue_main()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    args=parser.parse_args()
    sys.exit(main(args.batch.resolve()))
