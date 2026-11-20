"""Stage 1: freeze the existing split, then run CV-only comparisons."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
if __name__ == "__main__":
    local = ROOT / ".venv" / "Scripts" / "python.exe"
    if local.is_file() and Path(sys.executable).resolve() != local.resolve():
        raise SystemExit(subprocess.call([str(local), str(Path(__file__).resolve()), *sys.argv[1:]]))

import numpy as np
import pandas as pd
import torch
import first_version as core


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                          encoding="utf-8", newline="\n")


def empty_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_config(config):
    expected = set(core.configs(1, 0)[0])
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError(f"Config must have exactly these keys: {sorted(expected)}")
    hidden = config["hidden"]
    if not isinstance(hidden, list) or not hidden or any(type(n) is not int or n < 1 for n in hidden):
        raise ValueError("hidden must be a nonempty list of positive integer widths")
    for key, allowed in {"norm": {"none", "layer"}, "activation": {"relu", "gelu"},
                         "scaling": {"standard", "robust", "quantile"}}.items():
        if config[key] not in allowed:
            raise ValueError(f"Invalid {key}")
    for key in ("class_weight", "interactions", "domain_features"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    for key in ("lr", "weight_decay", "dropout", "smoothing"):
        if type(config[key]) not in (float, int) or not math.isfinite(config[key]):
            raise ValueError(f"{key} must be finite")
    if config["lr"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("lr must be positive and weight_decay nonnegative")
    if not 0 <= config["dropout"] < 1 or not 0 <= config["smoothing"] < 1:
        raise ValueError("dropout and smoothing must be in [0,1)")
    if type(config["batch_size"]) is not int or config["batch_size"] < 1:
        raise ValueError("batch_size must be a positive integer")
    return copy.deepcopy(config)


def validate_partition(protocol, frame, columns, y):
    n = len(y)
    def indices(values):
        if not isinstance(values, list) or any(type(v) is not int or not 0 <= v < n for v in values):
            raise ValueError("Invalid row index in protocol")
        if len(values) != len(set(values)):
            raise ValueError("Duplicate row index in protocol")
        return np.asarray(values, dtype=np.int64)
    dev = indices(protocol["development_rows"])
    holdout = indices(protocol["holdout_rows"])
    if set(dev) & set(holdout) or set(dev) | set(holdout) != set(range(n)):
        raise ValueError("Development/holdout partitions must be disjoint and cover the data")
    if len(np.unique(y[dev])) != 2 or len(np.unique(y[holdout])) != 2:
        raise ValueError("Both labels must occur in each partition")
    groups = pd.util.hash_pandas_object(frame[columns].apply(pd.to_numeric), index=False).to_numpy()
    if set(groups[dev]) & set(groups[holdout]):
        raise ValueError("Identical input records cross the holdout boundary")
    folds, seen = [], []
    if len(protocol["cv_folds"]) < 2:
        raise ValueError("At least two CV folds required")
    for fold in protocol["cv_folds"]:
        train, valid = indices(fold["train_rows"]), indices(fold["valid_rows"])
        if set(train) & set(valid) or set(train) | set(valid) != set(dev):
            raise ValueError("CV folds must partition development rows only")
        if len(np.unique(y[train])) != 2 or len(np.unique(y[valid])) != 2:
            raise ValueError("Both labels must occur in each CV train/validation partition")
        if set(groups[train]) & set(groups[valid]):
            raise ValueError("Identical input records cross a CV boundary")
        seen.extend(valid.tolist())
        folds.append((train, valid))
    if sorted(seen) != sorted(dev.tolist()):
        raise ValueError("Each development row must be validated exactly once")
    return dev, holdout, folds


def prepare(data, baseline_run, output):
    data, baseline_run = Path(data), Path(baseline_run).resolve()
    metrics = json.loads((baseline_run / "metrics.json").read_text(encoding="utf-8"))
    if sha(data) != metrics["data_sha256"]:
        raise ValueError("Dataset differs from baseline: cannot preserve its split")
    frame, columns, y = core.load_training(data)
    settings = metrics["arguments"]
    split_seed = int(settings["seed"])
    dev, holdout, folds, grouped = core.make_splits(
        frame, columns, y, settings["valid_size"], settings["folds"], split_seed)
    ids = np.full(len(y), -1, dtype=int)
    for number, (_, va) in enumerate(folds):
        ids[dev[va]] = number
    expected = pd.DataFrame({"row": np.arange(len(y)),
                            core.ID: frame[core.ID] if core.ID in frame else np.arange(len(y)),
                            "partition": np.where(ids == -1, "holdout", "development"), "cv_fold": ids})
    saved = pd.read_csv(baseline_run / "split.csv")
    # Preserve the exact baseline assignment; do not silently regenerate a different split.
    pd.testing.assert_frame_equal(saved, expected, check_dtype=False)
    protected = {p.relative_to(baseline_run).as_posix(): sha(p)
                 for p in sorted(baseline_run.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    protocol = dict(format_version=1, data_sha256=sha(data), columns=columns, rows=len(y),
                    split_seed=split_seed, valid_size=settings["valid_size"],
                    baseline_config=validate_config(metrics["config"]),
                    baseline_run=str(baseline_run), baseline_files_sha256=protected,
                    holdout_already_observed=True, identical_inputs_grouped=grouped,
                    development_rows=dev.tolist(), holdout_rows=holdout.tolist(),
                    cv_folds=[dict(train_rows=dev[tr].tolist(), valid_rows=dev[va].tolist()) for tr, va in folds])
    validate_partition(protocol, frame, columns, y)
    protocol["protocol_id"] = identity(protocol)
    out = empty_output(output)
    save(out / "protocol.json", protocol)
    shutil.copy2(baseline_run / "split.csv", out / "split.csv")
    save(out / "baseline_config.json", protocol["baseline_config"])
    print(f"Frozen baseline split: development={len(dev)}, holdout={len(holdout)}, folds={len(folds)}")
    print(f"Protocol: {out / 'protocol.json'}; no model training or holdout evaluation performed.")
    return protocol


def load_protocol(path, data):
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = {k: v for k, v in protocol.items() if k != "protocol_id"}
    if protocol.get("format_version") != 1 or identity(payload) != protocol.get("protocol_id"):
        raise ValueError("Protocol was modified or has unsupported format")
    if sha(data) != protocol["data_sha256"]:
        raise ValueError("Dataset content/order changed; frozen split cannot be reused")
    for name, digest in protocol["baseline_files_sha256"].items():
        path = Path(protocol["baseline_run"]) / name
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"Protected baseline file changed or missing: {name}")
    frame, columns, y = core.load_training(data)
    if columns != protocol["columns"] or len(frame) != protocol["rows"]:
        raise ValueError("Data schema does not match protocol")
    dev, holdout, folds = validate_partition(protocol, frame, columns, y)
    return protocol, frame, columns, y, dev, holdout, folds


def search(args):
    protocol, frame, columns, y, dev, _, folds = load_protocol(args.protocol, args.data)
    raw = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else protocol["baseline_config"]
    candidates = [raw] if isinstance(raw, dict) else raw
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Config JSON must be an object or a nonempty list of objects")
    candidates = [validate_config(c) for c in candidates]
    out = empty_output(args.output)
    save(out / "configs.json", candidates)
    save(out / "run.json", dict(mode="cv_only", protocol_id=protocol["protocol_id"],
                                data_sha256=protocol["data_sha256"], split_seed=protocol["split_seed"],
                                model_seed=args.model_seed, epochs=args.epochs, patience=args.patience,
                                device=args.device, threads=args.threads, holdout_evaluated=False,
                                core_code_sha256=sha(core.__file__), experiment_code_sha256=sha(__file__),
                                versions={p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "torch")}))
    # Only the frozen development rows are supplied to fit/transform and model evaluation.
    trials = []
    for number, config in enumerate(candidates):
        scores, epochs, histories = [], [], []
        for fold, (train, valid) in enumerate(folds):
            prep = core.Preprocessor(columns, config["scaling"], config["interactions"], config["domain_features"])
            prep.fit(frame.iloc[train])
            _, score, epoch, history = core.fit_model(
                prep.transform(frame.iloc[train]), y[train], config, args.epochs, args.patience,
                args.model_seed + fold, args.device, (prep.transform(frame.iloc[valid]), y[valid]))
            scores.append(float(score))
            epochs.append(int(epoch))
            histories.append(history)
            print(f"trial {number+1}/{len(candidates)}, fold {fold+1}/{len(folds)}: CV AUROC={score:.5f}, epoch={epoch}", flush=True)
        trials.append(dict(trial=number, config=config, scores=scores, epochs=epochs,
                           mean_auroc=float(np.mean(scores)), std_auroc=float(np.std(scores))))
        save(out / f"learning_curves_trial_{number+1}.json", histories)
        save(out / "search.json", trials)
    selected = max(trials, key=lambda t: (t["mean_auroc"], -t["std_auroc"]))
    selection = dict(config=selected["config"], epochs=max(1, int(np.median(selected["epochs"]))),
                     selected_cv_auroc=selected["mean_auroc"], protocol_id=protocol["protocol_id"],
                     data_sha256=protocol["data_sha256"], model_seed=args.model_seed,
                     split_seed=protocol["split_seed"], holdout_evaluated=False,
                     note="CV selection result only. No holdout evaluation, all-data refit, or submission weights.")
    save(out / "selection.json", selection)
    # Exact runnable sources are saved, without including any customer-level data.
    (out / "code").mkdir()
    shutil.copy2(__file__, out / "code" / "experiment.py")
    shutil.copy2(core.__file__, out / "code" / "first_version.py")
    print("CV-only comparison finished. Holdout was not evaluated; baseline weights were not changed.")
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Freeze and verify the existing baseline split; no training")
    prep.add_argument("--data", default=str(ROOT / "data" / "heloc_release.csv"))
    prep.add_argument("--baseline-run", default=str(ROOT / "runs" / "task1"))
    prep.add_argument("--output", default=str(ROOT / "runs" / "protocol_v1"))
    trial = sub.add_parser("search", help="CV only: never evaluate holdout or refit on all data")
    trial.add_argument("--data", default=str(ROOT / "data" / "heloc_release.csv"))
    trial.add_argument("--protocol", default=str(ROOT / "runs" / "protocol_v1" / "protocol.json"))
    trial.add_argument("--output", required=True)
    trial.add_argument("--config", help="JSON object/list; omitted uses the frozen baseline configuration")
    trial.add_argument("--model-seed", type=int, default=42)
    trial.add_argument("--epochs", type=int, default=200)
    trial.add_argument("--patience", type=int, default=25)
    trial.add_argument("--threads", type=int, default=4)
    trial.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.data, args.baseline_run, args.output)
    else:
        if min(args.epochs, args.patience, args.threads) < 1 or not 0 <= args.model_seed < 2**32 - 1000:
            parser.error("Positive counts and a model seed in [0, 2**32-1000) are required")
        torch.set_num_threads(args.threads)
        search(args)


if __name__ == "__main__":
    main()
