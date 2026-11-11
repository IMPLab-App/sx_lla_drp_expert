"""Run a prepared raw-signal ablation manifest on two physical GPUs.

Usage: python run_signal_ablation_batch.py /absolute/path/to/batch
The batch contains a pinned code snapshot, configs, data and manifest.json.
"""

import argparse
import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def timestamp():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def write_metrics(batch):
    import numpy as np

    manifest = json.loads((batch / "manifest.json").read_text())
    rows, classes = [], []
    for task in manifest["tasks"]:
        for split in ("balanced", "imbalanced"):
            path = batch / "runs/models" / task["run"] / "confusion_matrix" / f"model_best_{split}.csv"
            with path.open(encoding="utf-8-sig", newline="") as handle:
                values = list(csv.reader(handle))
            names = values[0][1:]
            normalized = np.asarray([[float(x) for x in row[1:]] for row in values[1:]])
            ratio = 50 if task["case"] % 2 else 100
            if split == "balanced":
                support = np.full(9, 100, dtype=int)
            else:
                targets = np.array([int(100000 * np.power(ratio, -c / 8)) for c in range(9)])
                proportions = targets / targets[0]
                support = np.array([int(min(100 / proportions) * p) for p in proportions])
            raw = normalized * support[:, None]
            counts = np.rint(raw).astype(int)
            assert np.max(abs(raw - counts)) < 0.001
            assert np.array_equal(counts.sum(1), support)
            correct = counts.diagonal()
            predicted = counts.sum(0)
            precision = np.divide(correct, predicted, out=np.zeros(9), where=predicted > 0)
            recall = correct / support
            f1 = 2 * correct / (support + predicted)
            accuracy = correct.sum() / support.sum()
            log = (batch / f"runs/gpu{task['gpu']}" / f"{task['run']}.eval.log").read_text()
            reported = float(re.search(r"\[" + split + r"\] Test Accuracy: ([0-9.]+)", log).group(1))
            assert abs(accuracy - reported) < 0.000051
            rows.append({
                "case": task["case"], "variant": task["variant"], "test_set": split,
                "run": task["run"], "samples": int(support.sum()),
                "accuracy_pct": float(accuracy * 100),
                "macro_precision_pct": float(precision.mean() * 100),
                "macro_recall_pct": float(recall.mean() * 100),
                "macro_f1_pct": float(f1.mean() * 100),
                "weighted_f1_pct": float(np.average(f1, weights=support) * 100),
                "head_accuracy_pct": float(correct[0] / support[0] * 100),
                "tail_accuracy_pct": float(correct[1:].sum() / support[1:].sum() * 100),
            })
            for c, name in enumerate(names):
                classes.append({"case": task["case"], "variant": task["variant"],
                                "test_set": split, "class_id": c, "class_name": name,
                                "samples": int(support[c]), "correct": int(correct[c]),
                                "precision_pct": float(precision[c] * 100),
                                "recall_pct": float(recall[c] * 100), "f1_pct": float(f1[c] * 100)})
    output = batch / "metrics_summary"
    output.mkdir(exist_ok=True)
    for name, data in (("summary.csv", rows), ("per_class_metrics.csv", classes)):
        with (output / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    lines = ["# No-residual ablation results", "", "All values are percentages. Best checkpoints selected by validation accuracy.",
             "Head is class 0 (正常); tail is classes 1–8, following cut1=1.", "",
             "| Case | Variant | Balanced accuracy | Macro-F1 | Head accuracy | Tail accuracy |", "|---|---|---:|---:|---:|---:|"]
    for row in sorted((r for r in rows if r["test_set"] == "balanced"), key=lambda r: (r["case"], manifest["variants"].index(r["variant"]))):
        lines.append(f"| {row['case']} | {row['variant']} | {row['accuracy_pct']:.2f} | {row['macro_f1_pct']:.2f} | {row['head_accuracy_pct']:.2f} | {row['tail_accuracy_pct']:.2f} |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def worker(batch, gpu):
    manifest = json.loads((batch / "manifest.json").read_text())
    folder = batch / f"runs/gpu{gpu}"
    folder.mkdir(parents=True, exist_ok=True)
    status = folder / "status.tsv"
    (folder / "worker.pid").write_text(str(os.getpid()) + "\n")
    status.write_text("time\trun\tstage\tstatus\tcode\n")

    def record(run, stage, state, code=0):
        with status.open("a") as handle:
            handle.write(f"{timestamp()}\t{run}\t{stage}\t{state}\t{code}\n")
        print(f"[{timestamp()}] {run} {stage} {state} {code}", flush=True)

    tasks = [task for task in manifest["tasks"] if task["gpu"] == gpu]
    for task in tasks:
        record(task["run"], "prepare", "pending")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               MPLBACKEND="Agg", PYTHONUNBUFFERED="1", TZ="Asia/Shanghai")
    for task in tasks:
        run = task["run"]
        checkpoint = batch / "runs/models" / run / "model_best.pth"
        commands = [
            ("train", [sys.executable, "train.py", "--c", task["config"]]),
            ("eval", [sys.executable, "eval_confusion_matrix_new.py", "--c", task["config"],
                      "--load_path", str(checkpoint), "--gpu", "0"]),
        ]
        for stage, command in commands:
            record(run, stage, "running")
            with (folder / f"{run}.{stage}.log").open("wb") as handle:
                result = subprocess.run(command, cwd=manifest["code_snapshot"], env=env,
                                        stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            record(run, stage, "completed" if result.returncode == 0 else "failed", result.returncode)
            if result.returncode:
                return result.returncode
    record("queue", "all", "completed")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--worker", type=int, choices=(0, 1))
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    batch = args.batch.resolve()
    if args.report_only:
        write_metrics(batch)
        return 0
    if args.worker is not None:
        return worker(batch, args.worker)
    with (batch / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runs = batch / "runs"
        runs.mkdir(exist_ok=True)
        state = runs / "status.txt"
        if state.exists():
            raise RuntimeError("This batch has already been launched; use a fresh batch directory.")
        (batch / "queue.pid").write_text(str(os.getpid()) + "\n")
        state.write_text("running\n")
        processes = []
        manifest = json.loads((batch / "manifest.json").read_text())
        with (runs / "workers.tsv").open("w") as table:
            table.write("gpu\tpid\tcases\tvariants\n")
            for gpu in (0, 1):
                folder = runs / f"gpu{gpu}"
                folder.mkdir(exist_ok=True)
                with (folder / "queue.log").open("wb") as handle:
                    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(batch), "--worker", str(gpu)],
                                               stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
                processes.append(process)
                cases = "1,3,5" if gpu == 0 else "2,4,6"
                table.write(f"{gpu}\t{process.pid}\t{cases}\t{','.join(manifest['variants'])}\n")
                print(f"[{timestamp()}] GPU {gpu}: worker {process.pid}, cases {cases}", flush=True)
        # Poll both workers so a failure is visible even if the other queue continues.
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                state.write_text("worker_failed_other_worker_may_be_running\n")
            time.sleep(5)
        codes = [p.returncode for p in processes]
        (runs / "exit_codes.tsv").write_text(f"gpu\texit_code\n0\t{codes[0]}\n1\t{codes[1]}\n")
        if any(codes):
            state.write_text("failed\n")
            return 1
        state.write_text("summarizing\n")
        try:
            write_metrics(batch)
        except Exception:
            state.write_text("training_completed_report_failed\n")
            raise
        state.write_text("completed\n")
        print(f"[{timestamp()}] All 18 training/evaluation tasks completed; comparison metrics saved.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
