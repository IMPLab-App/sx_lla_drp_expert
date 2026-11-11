"""The successor must wait for all baseline tasks and stop on broken dependencies."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wait_bacn_ablation import dependency_ready


class DependencyTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='bacn_dependency_')
        self.root = Path(self.temporary.name)
        self.base, self.batch = self.root/'base', self.root/'next'
        self.base.mkdir(); self.batch.mkdir(); (self.base/'state').mkdir()
        tasks=[]
        for case in range(11):
            output=self.base/f'model{case}'
            output.mkdir()
            (output/'model_best.pth').touch()
            (output/'test_metrics.json').write_text(json.dumps(dict(samples=900)))
            name=f'case{case}'
            (self.base/'state'/(name+'.json')).write_text(json.dumps(dict(state='completed')))
            tasks.append(dict(name=name,output=str(output)))
        original=json.dumps(dict(code_snapshot='/expected/baseline/code',tasks=tasks)).encode()
        (self.base/'manifest.json').write_bytes(original)
        (self.batch/'manifest.json').write_text(json.dumps(dict(
            baseline_batch=str(self.base),baseline_manifest_sha256=hashlib.sha256(original).hexdigest())))
        (self.base/'status.txt').write_text('completed\n')

    def tearDown(self):
        self.temporary.cleanup()

    def test_all_eleven_completed_required(self):
        self.assertTrue(dependency_ready(self.batch))
        (self.base/'state/case10.json').write_text(json.dumps(dict(state='running')))
        with self.assertRaisesRegex(RuntimeError,'Incomplete'):
            dependency_ready(self.batch)

    def test_running_scheduler_waits_instead_of_starting(self):
        (self.base/'status.txt').write_text('running\n')
        (self.base/'queue.pid').write_text(str(os.getpid()))
        original_read = Path.read_bytes
        with patch.object(Path,'read_bytes',autospec=True) as read:
            read.side_effect=lambda p: (b'python\0/expected/baseline/code/run_signal_baselines.py\0'
                                       if str(p).startswith('/proc/') else original_read(p))
            self.assertFalse(dependency_ready(self.batch))

    def test_failure_or_manifest_change_blocks_successor(self):
        (self.base/'status.txt').write_text('completed_with_failures\n')
        with self.assertRaisesRegex(RuntimeError,'failed'):
            dependency_ready(self.batch)
        (self.base/'manifest.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'manifest changed'):
            dependency_ready(self.batch)

    def test_missing_checkpoint_blocks_successor(self):
        (self.base/'model10/model_best.pth').unlink()
        with self.assertRaisesRegex(RuntimeError,'Incomplete'):
            dependency_ready(self.batch)


if __name__=='__main__':
    unittest.main()
