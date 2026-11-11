"""Back up and migrate raw-signal archives to the requested 0..8 fault labels."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil

import numpy as np


CLASS_NAMES = (
    "健康", "主轴疲劳", "缸体轴外圈疲劳", "缸体轴内圈疲劳",
    "柱塞磨损", "斜盘磨损", "配油盘磨损", "弹簧失效", "滑靴塌边",
)


def relabel(path, backup_dir):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    original_names = arrays["class_names"].tolist()
    names = ["健康" if name == "正常" else name for name in original_names]
    if len(names) != len(CLASS_NAMES) or set(names) != set(CLASS_NAMES):
        raise ValueError(f"Unexpected classes in {path}: {original_names}")
    targets = arrays["labels"]
    if not np.array_equal(np.unique(targets), np.arange(len(CLASS_NAMES))):
        raise ValueError(f"Invalid labels in {path}")
    if original_names == list(CLASS_NAMES):
        print(f"Already migrated: {path}")
        return

    old_to_new = np.array([CLASS_NAMES.index(name) for name in names])
    new_to_old = np.argsort(old_to_new)
    arrays["labels"] = old_to_new[targets].astype(targets.dtype)
    arrays["class_names"] = np.asarray(CLASS_NAMES)
    # Source filenames are indexed by class; signal rows and sample metadata stay put.
    if "source_files" in arrays:
        if arrays["source_files"].shape != (len(CLASS_NAMES),):
            raise ValueError(f"Unexpected source_files shape in {path}")
        arrays["source_files"] = arrays["source_files"][new_to_old]

    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / path.name
    if backup_path.exists():
        raise FileExistsError(backup_path)
    shutil.copy2(path, backup_path)
    temporary_path = path.with_suffix(".relabel.tmp.npz")
    try:
        with temporary_path.open("xb") as temporary_file:
            np.savez_compressed(temporary_file, **arrays)
        with np.load(temporary_path, allow_pickle=False) as check:
            for key, value in arrays.items():
                np.testing.assert_array_equal(check[key], value)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Migrated {path}: old -> new {old_to_new.tolist()}; backup: {backup_path}")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--backup-dir", type=Path, default=root / "backups" /
                        ("raw_signal_labels_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    args = parser.parse_args()
    paths = args.paths or sorted((root / "data").glob("raw_signal_dataset*.npz"))
    for path in paths:
        relabel(path, args.backup_dir)


if __name__ == "__main__":
    main()
