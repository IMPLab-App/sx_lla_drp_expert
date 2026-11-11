"""Operational tests use isolated files and tiny child processes, never training."""
import contextlib
import copy
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

import cola_two_kl_queue as queue


class TestQueue(unittest.TestCase):
    def test_second_queue_is_rejected(self):
        lock = queue.LOGS / 'queue.lock'
        with lock.open('a') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            result = subprocess.run(['bash', str(queue.ROOT / 'run_cola_two_kl_24.sh'), '--dry-run'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 75)
            self.assertIn('Another', result.stderr)

    def test_child_exit_codes_and_existing_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); log = root / 'child.log'
            task = dict(variant='test', run_name='isolated', seed=2)
            with patch.object(queue, 'LOGS', root), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'exit 7'):
                    queue.run_stage(task, 'train', [sys.executable, '-c', 'print("failure");raise SystemExit(7)'], log)
                self.assertIn('exit_code=7', log.read_text())
                self.assertEqual(json.loads((root/'current.json').read_text())['status'], 'failed')
                with self.assertRaises(FileExistsError):
                    queue.run_stage(task, 'train', [sys.executable, '-c', 'raise SystemExit(0)'], log)

    def test_failure_stops_before_second_task(self):
        tasks = [dict(variant='full', run_name=f'isolated{i}', seed=2, model_dir=f'model{i}', config='test.yaml') for i in range(24)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest = root/'manifest.json'; manifest.write_text(json.dumps(dict(tasks=tasks)))
            calls = []
            def child(task, stage, *args):
                calls.append((task['run_name'], stage))
                raise RuntimeError('simulated train failure')
            with patch.object(queue, 'ROOT', root), patch.object(queue, 'LOGS', root/'logs'), patch.object(queue, 'MANIFEST', manifest), \
                 patch.object(queue, 'validate_task'), patch.object(queue, 'completed', return_value=False), \
                 patch.object(queue, 'run_stage', side_effect=child), patch.object(queue.subprocess, 'run'), \
                 patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=1), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'simulated'):
                    queue.run_queue()
            self.assertEqual(calls, [('isolated0', 'train')])
            self.assertEqual(json.loads((root/'logs/queue.json').read_text())['status'], 'failed')

    def test_safe_skip_rejects_changed_artifacts_and_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); model=root/'models'; model.mkdir()
            state=dict(model={'x':torch.ones(1)},ema_model={'x':torch.ones(1)},optimizer={},best_it=2,epoch=1,it=4,
                       cola_two_kl={'tau_dyn':torch.tensor(2.)})
            torch.save(state,model/'latest_model.pth'); torch.save(state,model/'model_best.pth')
            evaluation=model/'evaluation'; evaluation.mkdir()
            split=dict(top1_accuracy=.5,balanced_accuracy=.5,macro_precision=.5,macro_recall=.5,macro_f1=.5)
            for key in ('confusion_matrix_csv','confusion_matrix_png'):
                p=evaluation/key;p.write_text('test artifact');split[key]=str(p)
            result=dict(complete=True,checkpoint=str(model/'model_best.pth'),heldout=split,balanced=split,imbalanced=split)
            (evaluation/'evaluation.json').write_text(json.dumps(result))
            task=dict(model_dir='models',variant='full',num_train_iter=4,config_sha256='isolated-test')
            with patch.object(queue, 'ROOT', root):
                self.assertFalse(queue.completed(task))
                marker=queue.validate_outputs(task); queue.atomic_json(model/'completed.json',marker)
                self.assertTrue(queue.completed(task))
                changed=copy.deepcopy(state);changed['model']['x'].add_(1);torch.save(changed,model/'latest_model.pth')
                with self.assertRaisesRegex(RuntimeError, 'fingerprints'):
                    queue.completed(task)
                changed['it']=3;torch.save(changed,model/'latest_model.pth')
                with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                    queue.completed(task)


    def test_dual_shards_are_disjoint_and_complete(self):
        tasks = json.loads(queue.MANIFEST.read_text())['tasks']
        first, second = queue.select_tasks(tasks, 0), queue.select_tasks(tasks, 1)
        self.assertEqual(len(first), 12)
        self.assertEqual(len(second), 12)
        key = lambda task: (task['variant'], task['run_name'])
        self.assertFalse(set(map(key, first)) & set(map(key, second)))
        self.assertEqual(first + second, tasks)
        self.assertEqual({t['variant'] for t in first}, {'full', 'no_mff'})
        self.assertEqual({t['variant'] for t in second}, {'no_dea_aggregator', 'no_ddde_kl'})

    def test_duplicate_task_ownership_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(queue, 'LOGS', Path(temp)):
            task = dict(variant='test', run_name='same_task')
            with queue.task_lock(task):
                with self.assertRaisesRegex(RuntimeError, 'already owned'):
                    with queue.task_lock(task):
                        self.fail('Duplicate ownership accepted')

    def test_worker_state_files_and_shared_status_are_separate(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(queue, 'LOGS', Path(temp)), contextlib.redirect_stdout(io.StringIO()):
            task = dict(variant='test', run_name='separate_states', seed=2)
            for worker in ('0','1'):
                with patch.dict(os.environ, COLA_QUEUE_WORKER=worker, GPU_ID=worker):
                    queue.status(task, 'train', 'running', pid=100 + int(worker))
            root = Path(temp)
            self.assertEqual(json.loads((root/'gpu_0/current.json').read_text())['gpu'], '0')
            self.assertEqual(json.loads((root/'gpu_1/current.json').read_text())['gpu'], '1')
            self.assertEqual(len((root/'status.tsv').read_text().splitlines()), 3)
            self.assertFalse((root/'current.json').exists())

    def test_legacy_handoff_only_at_committed_task_boundary(self):
        import cola_two_kl_handoff as handoff
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);request=root/'request.json';ready=root/'ready.json'
            request.write_text(json.dumps(dict(legacy_pid=123)))
            (root/'current.json').write_text(json.dumps(dict(stage='train',status='running')))
            with patch.object(handoff,'LOGS',root),patch.object(handoff,'REQUEST',request),patch.object(handoff,'READY',ready), \
                 patch.object(handoff.os,'getppid',return_value=123),patch.object(handoff,'is_legacy_scheduler',return_value=True), \
                 patch.object(handoff.os,'kill') as kill,contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'committed'):
                    handoff.after_summary()
                kill.assert_not_called()
                (root/'current.json').write_text(json.dumps(dict(stage='task',status='complete')))
                handoff.after_summary()
                self.assertTrue(ready.exists())
                self.assertEqual(kill.call_args.args[0],123)


    def test_real_safe_handoff_replaces_only_idle_scheduler(self):
        import signal
        import cola_two_kl_handoff as handoff
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'current.json').write_text(json.dumps(dict(stage='task',status='complete')))
            script=root/'cola_two_kl_queue.py'
            script.write_text("import os,sys,json,fcntl,subprocess\nfrom pathlib import Path\nroot=Path(sys.argv[1])\nlock=(root/'queue.lock').open('a')\nfcntl.flock(lock,fcntl.LOCK_EX)\n(root/'request.json').write_text(json.dumps(dict(legacy_pid=os.getpid())))\ncode=\"import sys;sys.path.insert(0,sys.argv[2]);from pathlib import Path;import cola_two_kl_handoff as h;h.LOGS=Path(sys.argv[1]);h.REQUEST=h.LOGS/'request.json';h.READY=h.LOGS/'ready.json';h.after_summary()\"\nsubprocess.run([sys.executable,'-c',code,str(root),sys.argv[2]],check=True)\nraise RuntimeError('Idle legacy scheduler should have been replaced')\n")
            old=subprocess.Popen([sys.executable,str(script),str(root),str(queue.ROOT)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            children=[]
            def start(worker):
                child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'])
                children.append(child)
                queue.atomic_json(root/'gpu_0/queue.json',dict(pid=child.pid,status='running'))
                return child.pid
            try:
                import time
                deadline=time.monotonic()+5
                while not (root/'request.json').exists():
                    if time.monotonic()>deadline:
                        self.fail('Test scheduler did not initialize')
                    time.sleep(.02)
                with patch.object(handoff,'LOGS',root),patch.object(handoff,'REQUEST',root/'request.json'), \
                     patch.object(handoff,'READY',root/'ready.json'),patch.object(handoff,'launch_worker',side_effect=start), \
                     contextlib.redirect_stdout(io.StringIO()):
                    handoff.supervise()
                self.assertEqual(old.wait(timeout=5),-signal.SIGKILL)
                self.assertEqual(json.loads((root/'handoff_supervisor.json').read_text())['status'],'complete')
                self.assertTrue((root/'ready.json').exists())
            finally:
                if old.poll() is None:
                    old.kill();old.wait()
                for child in children:
                    child.terminate();child.wait()

    def test_parallel_status_append_keeps_all_rows(self):
        import csv
        with tempfile.TemporaryDirectory() as temp:
            code="import sys,os;from pathlib import Path;import cola_two_kl_queue as q;q.LOGS=Path(sys.argv[1]);os.environ['COLA_QUEUE_WORKER']=sys.argv[2];os.environ['GPU_ID']=sys.argv[2];[q.status(dict(variant='test',run_name=sys.argv[2]+'_'+str(i),seed=2),'train','running',pid=i) for i in range(20)]"
            children=[subprocess.Popen([sys.executable,'-c',code,temp,str(worker)],cwd=queue.ROOT,stdout=subprocess.DEVNULL) for worker in (0,1)]
            for child in children:
                self.assertEqual(child.wait(timeout=10),0)
            with (Path(temp)/'status.tsv').open() as stream:
                rows=list(csv.DictReader(stream,delimiter='\t'))
            self.assertEqual(len(rows),40)
            self.assertEqual(len({r['run_name'] for r in rows}),40)


if __name__ == '__main__':
    unittest.main()
