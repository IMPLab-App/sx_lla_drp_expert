"""Check frozen inputs and run the existing durable 20/19 two-GPU queue."""
import argparse
import sys

from pathlib import Path


def main(batch):
    import json
    from verify_selected13 import verify_inputs
    m=verify_inputs(batch);proof=json.loads((batch/'verification.json').read_text())
    assert proof['status']=='passed' and proof['tasks']==39
    if not m['smoke']:assert proof['all_39_smoke_train_reload_tests_passed']
    sys.argv=[str(batch/'code/run_signal_baselines.py'),str(batch)]
    from run_signal_baselines import main as queue_main
    return queue_main()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',type=Path)
    args=parser.parse_args();sys.exit(main(args.batch.resolve()))
