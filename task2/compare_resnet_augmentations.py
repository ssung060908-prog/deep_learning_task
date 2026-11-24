"""Run the two single-augmentation experiments on Task 2 fold 0, seed 42."""

import argparse
import csv
import json
import os
import sys
from importlib.util import find_spec
from pathlib import Path

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK / "code"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=TASK.parent / "data/T2_data")
    parser.add_argument("--output-root", type=Path, default=TASK.parent / "runs/resnet18_ablation")
    args = parser.parse_args()

    if not os.environ.get("TASK2_VISION_WHEEL") and (
        find_spec("torchvision") is None or find_spec("PIL") is None
    ):
        home = Path.home()
        wheel = home / "Downloads/torchvision-0.25.0-cp313-cp313-win_amd64.whl"
        pillow_python = home / ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
        if wheel.is_file() and pillow_python.is_file():
            os.environ["TASK2_VISION_WHEEL"] = str(wheel)
            os.environ["TASK2_PIL_PYTHON"] = str(pillow_python)

    from common import load_config, train
    from training_state import STATE_NAME

    root = args.output_root.resolve()
    config = load_config(TASK / "configs/geometry.json")
    config.update(data_root=str(args.data_root.resolve()), output_root=str(root))
    runs = {}
    for mode in ("flip_only", "rotate_only"):
        name = f"{mode}_f0_s42"
        dest = root / name
        settings = dict(config, geometry_mode=mode)
        if (dest / "training_summary.json").is_file():
            saved = read_json(dest / "config.json")
            if saved.get("smoke_test") or any(saved.get(k) != v for k, v in settings.items()):
                raise ValueError(f"Existing run has different settings: {dest}")
        elif (dest / STATE_NAME).is_file():
            train("resnet", settings, resume=dest)
        else:
            train("resnet", settings, run_name=name)
        runs[mode] = dest

    if len({(p / "split_manifest.csv").read_bytes() for p in runs.values()}) != 1:
        raise ValueError("The two training runs have different splits")
    archive = TASK.parent / "experiments/task2-resnet18-20261111/run_records"
    rows = []
    for policy, directory, source in [
        ("none", archive / "control_f0_s42", "archive"),
        *((mode, path, "current") for mode, path in runs.items()),
        ("flip_rotate", archive / "geometry_f0_s42", "archive"),
    ]:
        metrics = read_json(directory / "best_metrics.json")
        keys = ("epoch", "samples", "balanced_accuracy", "sensitivity", "specificity", "auroc")
        rows.append(dict(policy=policy, source=source, **{k: metrics[k] for k in keys}))
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['policy']:12s}  BA={row['balanced_accuracy']:.4f}  AUROC={row['auroc']:.4f}")


if __name__ == "__main__":
    main()
