"""A waiting sweep must not start on failed, incomplete or stale predecessors."""
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from prepare_bacn_18 import digest
from wait_bacn_18 import dependency_ready,main


class DependencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='bacn18_dependency_')
        self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);self.batch=root/'next';self.dep=root/'six'
        self.batch.mkdir();self.dep.mkdir();(self.dep/'state').mkdir()
        tasks=[]
        for i in range(6):
            name=f'task{i}';out=self.dep/name;out.mkdir()
            (out/'model_best.pth').write_text('checkpoint')
            (out/'test_metrics.json').write_text(json.dumps(dict(samples=900)))
            (out/'artifact_audit.json').write_text(json.dumps(dict(status='passed',samples=900)))
            (self.dep/'state'/(name+'.json')).write_text(json.dumps(dict(state='completed')))
            tasks.append(dict(name=name,output=str(out)))
        self.original=dict(tasks=tasks,code_snapshot='/frozen/six/code')
        (self.dep/'manifest.json').write_text(json.dumps(self.original))
        (self.dep/'status.txt').write_text('completed\n')
        (self.dep/'exit_codes.json').write_text(json.dumps(dict(codes=[0,0],completed=6)))
        (self.batch/'manifest.json').write_text(json.dumps(dict(dependency_batch=str(self.dep),
            dependency_manifest_sha256=digest(self.dep/'manifest.json'))))

    def test_all_six_train_test_artifact_checks_allow_start(self):
        self.assertTrue(dependency_ready(self.batch))

    def test_failed_predecessor_blocks_start(self):
        (self.dep/'status.txt').write_text('completed_with_failures\n')
        with self.assertRaisesRegex(RuntimeError,'failed'):dependency_ready(self.batch)

    def test_missing_artifact_audit_blocks_start(self):
        (self.dep/'task5/artifact_audit.json').unlink()
        with self.assertRaises(FileNotFoundError):dependency_ready(self.batch)

    def test_changed_manifest_blocks_start(self):
        (self.dep/'manifest.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'manifest changed'):dependency_ready(self.batch)

    def test_live_wrapper_queue_keeps_waiting(self):
        (self.dep/'status.txt').write_text('running\n');(self.dep/'queue.pid').write_text('987654')
        original_read=Path.read_bytes
        def read(path):
            if str(path)=='/proc/987654/cmdline':
                return b'python\0/frozen/six/code/run_metaexpert_full_lb35.py\0six'
            return original_read(path)
        with patch.object(Path,'read_bytes',read):self.assertFalse(dependency_ready(self.batch))

    def test_dead_predecessor_queue_blocks_start(self):
        (self.dep/'status.txt').write_text('running\n');(self.dep/'queue.pid').write_text('987654')
        original_read=Path.read_bytes
        def read(path):
            if str(path)=='/proc/987654/cmdline':raise FileNotFoundError(path)
            return original_read(path)
        with patch.object(Path,'read_bytes',read):
            with self.assertRaisesRegex(RuntimeError,'exited without completion'):dependency_ready(self.batch)

    def test_duplicate_supervisor_does_not_modify_existing_state(self):
        state=self.batch/'status.txt';state.write_text('waiting_for_dependency\n')
        with (self.batch/'successor.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):main(self.batch)
        self.assertEqual(state.read_text(),'waiting_for_dependency\n')


if __name__=='__main__':unittest.main()
