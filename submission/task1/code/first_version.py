"""HELOC: train and deploy ONE fully connected DNN. See README.md."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import random
import shutil
import warnings
from pathlib import Path

import sys
import subprocess

# Use this project environment even when the editor launches global Python.
if __name__ == "__main__":
    local_python = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
    if local_python.is_file() and Path(sys.executable).resolve() != local_python.resolve():
        raise SystemExit(subprocess.call([str(local_python), str(Path(__file__).resolve()), *sys.argv[1:]]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from torch import nn

TARGET = "is_at_risk"
ID = "sample_id"
CODES = (-9, -8, -7)
RATIO_FEATURES = (
    ("number_of_satisfactory_trades", "nr_total_trades"),
    ("nr_trades_insolvent_for_over_60_days", "nr_total_trades"),
    ("nr_trades_insolvent_for_over_90_days", "nr_total_trades"),
    ("nr_trades_initiated_in_last_year", "nr_total_trades"),
    ("nr_revolving_trades_with_balance", "nr_total_trades"),
    ("nr_installment_trades_with_balance", "nr_total_trades"),
    ("nr_banks_with_high_ratio", "nr_total_trades"),
    ("average_duration_of_resolution", "months_since_first_trade"),
    ("months_since_last_trade", "months_since_first_trade"),
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def numeric_frame(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    return frame.loc[:, columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)


class Preprocessor:
    """Fit only on each training partition; serialize pure data in the checkpoint."""
    def __init__(self, columns, scaling="standard", interactions=False, domain_features=False):
        self.columns = list(columns)
        self.scaling = scaling
        self.interactions = interactions
        self.domain_features = domain_features

    def clean(self, frame):
        raw = numeric_frame(frame, self.columns)
        flags = np.concatenate([(raw == c).astype(float) for c in CODES] +
                               [(~np.isfinite(raw)).astype(float)], axis=1)
        missing = np.isin(raw, CODES) | ~np.isfinite(raw)
        values = np.where(missing, np.nan, raw)
        if self.domain_features:
            extra = []
            for numerator, denominator in RATIO_FEATURES:
                if numerator in self.columns and denominator in self.columns:
                    a = values[:, self.columns.index(numerator)]
                    b = values[:, self.columns.index(denominator)]
                    ratio = np.full(len(raw), np.nan)
                    np.divide(a, b, out=ratio, where=np.isfinite(a) & np.isfinite(b) & (b > 0))
                    extra.append(ratio)
            if extra:
                derived = np.column_stack(extra)
                derived_missing = ~np.isfinite(derived)
                flags = np.concatenate([flags, derived_missing.astype(float)], axis=1)
                missing = np.concatenate([missing, derived_missing], axis=1)
                values = np.concatenate([values, derived], axis=1)
        return values, flags, missing

    def fit(self, frame):
        values, _, _ = self.clean(frame)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.median = np.nanmedian(values, axis=0)
        self.median = np.nan_to_num(self.median, nan=0.0)
        values = np.where(np.isnan(values), self.median, values)
        if self.scaling == "quantile":
            transformer = QuantileTransformer(n_quantiles=min(256, len(values)),
                                              output_distribution="uniform", random_state=0)
            transformer.fit(values)
            self.quantiles = transformer.quantiles_
            self.references = transformer.references_
        else:
            transformer = StandardScaler() if self.scaling == "standard" else RobustScaler()
            transformer.fit(values)
            self.center = transformer.mean_ if self.scaling == "standard" else transformer.center_
            self.scale = transformer.scale_
        return self

    def transform(self, frame):
        values, flags, missing = self.clean(frame)
        values = np.where(np.isnan(values), self.median, values)
        if self.scaling == "quantile":
            # Same symmetric interpolation as sklearn, including repeated quantiles.
            result = np.empty_like(values)
            for j in range(values.shape[1]):
                q, v = self.quantiles[:, j], values[:, j]
                result[:, j] = 0.5 * (np.interp(v, q, self.references) -
                                     np.interp(-v, -q[::-1], -self.references[::-1]))
                result[v == q[-1], j] = 1
                result[v == q[0], j] = 0
            values = (result - 0.5) * np.sqrt(12.0)
        else:
            values = (values - self.center) / self.scale
        values = np.clip(values, -8, 8)
        parts = [values, flags, missing.mean(axis=1, keepdims=True)]
        if self.interactions:
            a, b = np.triu_indices(values.shape[1], k=0)
            parts.append(values[:, a] * values[:, b])
        output = np.concatenate(parts, axis=1).astype(np.float32)
        if not np.isfinite(output).all():
            raise ValueError("Non-finite preprocessed values")
        return output

    def state(self):
        return {k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in vars(self).items()}

    @classmethod
    def restore(cls, state):
        obj = cls(state["columns"], state["scaling"], state["interactions"], state.get("domain_features", False))
        for k, v in state.items():
            setattr(obj, k, np.asarray(v) if k in
                    {"median", "quantiles", "references", "center", "scale"} else v)
        return obj


class DNN(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        layers = []
        width = input_dim
        for hidden in config["hidden"]:
            layers.append(nn.Linear(width, hidden))
            if config["norm"] == "layer":
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU() if config["activation"] == "relu" else nn.GELU())
            layers.append(nn.Dropout(config["dropout"]))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.inference_mode()
def probabilities(model, values, device, batch_size=2048):
    model.eval()
    if len(values) == 0:
        return np.empty(0, dtype=np.float32)
    return np.concatenate([model(torch.as_tensor(values[i:i + batch_size], device=device))
                           .sigmoid().cpu().numpy() for i in range(0, len(values), batch_size)])


def fit_model(x, y, config, epochs, patience, seed, device, valid=None):
    seed_all(seed)
    model = DNN(x.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    # Epoch-independent schedule so refitting to a selected epoch follows the same schedule.
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    weight = float((y == 0).sum() / (y == 1).sum()) if config["class_weight"] else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    xt = torch.as_tensor(x, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    best_score, best_epoch, best_state, history = -1.0, 0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(yt), device=device)
        total = 0.0
        for indices in order.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            logits = model(xt[indices])
            targets = yt[indices] * (1 - config["smoothing"]) + 0.5 * config["smoothing"]
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss became non-finite")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(indices)
        scheduler.step()
        row = {"epoch": epoch, "loss": total / len(y)}
        if valid is not None:
            score = float(roc_auc_score(valid[1], probabilities(model, valid[0], device)))
            row["auroc"] = score
            if score > best_score + 1e-5:
                best_score, best_epoch = score, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)
        if valid is not None and epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_score, best_epoch if valid is not None else epochs, history


def configs(count, seed):
    base = dict(hidden=[128, 64], norm="none", activation="relu", dropout=0.15,
                lr=0.001, weight_decay=0.001, batch_size=128, smoothing=0.0,
                class_weight=False, scaling="standard", interactions=False, domain_features=True)
    rng = random.Random(seed)
    result = [base]
    while len(result) < count:
        c = copy.deepcopy(base)
        c.update(hidden=rng.choice([[32], [64, 32], [128, 64], [256, 128, 64], [128]*4]),
                 norm=rng.choice(["none", "layer"]), activation=rng.choice(["relu", "gelu"]),
                 dropout=rng.choice([0.0, 0.1, 0.2, 0.35]), lr=10 ** rng.uniform(-3.7, -2.5),
                 weight_decay=10 ** rng.uniform(-6, -1.5), batch_size=rng.choice([64, 128, 256]),
                 smoothing=rng.choice([0.0, 0.0, 0.02]), class_weight=rng.choice([False, True]),
                 scaling=rng.choice(["standard", "robust", "quantile"]),
                 interactions=rng.choice([False, True]), domain_features=rng.choice([False, True]))
        result.append(c)
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_training(path):
    frame = pd.read_csv(path)
    if TARGET not in frame:
        raise ValueError(f"CSV must contain {TARGET}")
    columns = [c for c in frame.columns if c not in (TARGET, ID)]
    if len(columns) != 23:
        raise ValueError(f"Expected exactly 23 inputs, found {len(columns)}: {columns}")
    numeric_frame(frame, columns)
    y = pd.to_numeric(frame[TARGET], errors="raise")
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("Labels must contain both 0 and 1, without missing values")
    if ID in frame and (frame[ID].isna().any() or frame[ID].duplicated().any()):
        raise ValueError("sample_id must be non-missing and unique")
    return frame, columns, frame[TARGET].to_numpy(dtype=np.int64)


def make_splits(frame, columns, y, valid_size, folds, seed):
    # Customers with identical observed inputs stay together, even with different labels.
    numeric = frame[columns].apply(pd.to_numeric, errors="raise")
    groups = pd.util.hash_pandas_object(numeric, index=False).to_numpy()
    grouped = len(np.unique(groups)) < len(groups)
    if grouped:
        candidates = GroupShuffleSplit(n_splits=64, test_size=valid_size, random_state=seed)
        feasible = [(tr, va) for tr, va in candidates.split(frame, y, groups)
                    if len(np.unique(y[tr])) == 2 and len(np.unique(y[va])) == 2]
        if not feasible:
            raise ValueError("Cannot form a group-disjoint holdout containing both labels")
        dev, audit = min(feasible, key=lambda pair:
                         abs(len(pair[1]) / len(y) - valid_size) +
                         abs(y[pair[1]].mean() - y.mean()))
        cv = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
        splits = list(cv.split(dev, y[dev], groups[dev]))
    else:
        dev, audit = train_test_split(np.arange(len(y)), test_size=valid_size,
                                     stratify=y, random_state=seed)
        splits = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(dev, y[dev]))
    if min(np.bincount(y[dev])) < folds or any(
            len(np.unique(y[dev[indices]])) != 2 for pair in splits for indices in pair):
        raise ValueError("Not enough independent examples of each class; reduce --folds")
    return dev, audit, splits, grouped


def train(args):
    frame, columns, y = load_training(args.data)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Choose an empty output folder to preserve earlier runs")
    dev, audit, splits, grouped = make_splits(frame, columns, y, args.valid_size, args.folds, args.seed)
    fold_ids = np.full(len(y), -1, dtype=int)
    for fold, (_, va) in enumerate(splits):
        fold_ids[dev[va]] = fold
    pd.DataFrame({"row": np.arange(len(y)), ID: frame[ID] if ID in frame else np.arange(len(y)),
                  "partition": np.where(fold_ids == -1, "holdout", "development"),
                  "cv_fold": fold_ids}).to_csv(out / "split.csv", index=False)
    trials = []
    for trial, config in enumerate(configs(args.trials, args.seed)):
        scores, best_epochs = [], []
        for fold, (tr, va) in enumerate(splits):
            ti, vi = dev[tr], dev[va]
            prep = Preprocessor(columns, config["scaling"], config["interactions"], config["domain_features"]).fit(frame.iloc[ti])
            _, score, epoch, _ = fit_model(
                prep.transform(frame.iloc[ti]), y[ti], config, args.epochs, args.patience,
                args.seed + fold, args.device, (prep.transform(frame.iloc[vi]), y[vi]))
            scores.append(score)
            best_epochs.append(epoch)
            print(f"trial {trial+1}/{args.trials}, fold {fold+1}/{args.folds}: "
                  f"AUROC={score:.5f}, epoch={epoch}", flush=True)
        trials.append(dict(trial=trial, config=config, scores=scores, epochs=best_epochs,
                           mean_auroc=float(np.mean(scores)), std_auroc=float(np.std(scores))))
        write_json(out / "search.json", trials)
    best = max(trials, key=lambda t: (t["mean_auroc"], -t["std_auroc"]))
    config = best["config"]
    epochs = max(1, int(np.median(best["epochs"])))
    # Holdout is used once, only after all hyperparameters and epoch count are frozen.
    prep = Preprocessor(columns, config["scaling"], config["interactions"], config["domain_features"]).fit(frame.iloc[dev])
    model, _, _, history = fit_model(prep.transform(frame.iloc[dev]), y[dev], config,
                                     epochs, args.patience, args.seed, args.device)
    p = probabilities(model, prep.transform(frame.iloc[audit]), args.device)
    holdout_auc = float(roc_auc_score(y[audit], p))
    pd.DataFrame({ID: frame.iloc[audit][ID].to_numpy() if ID in frame else audit,
                  TARGET: y[audit], "probability": p}).to_csv(out / "holdout_predictions.csv", index=False)
    print(f"Untouched holdout AUROC: {holdout_auc:.5f}. Refitting one DNN on all supplied rows.", flush=True)
    prep = Preprocessor(columns, config["scaling"], config["interactions"], config["domain_features"]).fit(frame)
    x = prep.transform(frame)
    model, _, _, final_history = fit_model(x, y, config, epochs, args.patience, args.seed, args.device)
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "torch")}
    metadata = dict(config=config, epochs=epochs, seed=args.seed, rows=len(y),
                    selected_cv_auroc=best["mean_auroc"], holdout_auroc=holdout_auc,
                    holdout_rows=len(audit), identical_inputs_grouped=grouped, versions=versions,
                    data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                    arguments=vars(args),
                    note="CV is used for selection; holdout measures the development-only model. "
                         "Final model is refit on all supplied data; its private Test AUROC is unknown.")
    checkpoint = dict(format_version=1, input_dim=x.shape[1], config=config,
                      preprocessor=prep.state(), metadata=metadata,
                      state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()})
    torch.save(checkpoint, out / "task1.pt")
    # Verify the exact delivery path, including saved preprocessing.
    expected = probabilities(model, x[:32], args.device)
    actual = predict_proba(frame.iloc[:32], out / "task1.pt")
    np.testing.assert_allclose(expected, actual, rtol=1e-4, atol=1e-5)
    write_json(out / "metrics.json", metadata)
    write_json(out / "learning_curves.json", {"development": history, "final": final_history})
    (out / "requirements.txt").write_text("".join(f"{k}=={v}\n" for k, v in versions.items()), encoding="utf-8")
    (out / "code").mkdir()
    shutil.copy2(__file__, out / "code" / "task1.py")
    print(f"Saved and reload-verified: {out / 'task1.pt'}", flush=True)


def predict_proba(frame, weights, device="cpu"):
    """TA interface: raw DataFrame -> P(is_at_risk=1), original row order."""
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    prep = Preprocessor.restore(checkpoint["preprocessor"])
    model = DNN(checkpoint["input_dim"], checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return probabilities(model, prep.transform(frame), device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    t = commands.add_parser("train")
    t.add_argument("--data", default=str(Path(__file__).resolve().parent / "data" / "heloc_release.csv"), help="Provided labeled CSV (23 inputs + is_at_risk + optional sample_id)")
    t.add_argument("--output", default="runs/task1")
    t.add_argument("--trials", type=int, default=24)
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--patience", type=int, default=25)
    t.add_argument("--valid-size", type=float, default=0.2)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    p = commands.add_parser("predict")
    p.add_argument("--data", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--output", default="predictions.csv")
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nStart training: python first_version.py train")
        print("Quick check: python first_version.py train --output runs/quick --trials 2 --folds 2 --epochs 5 --patience 2")
        return
    args = parser.parse_args()
    if args.command == "train":
        if min(args.trials, args.epochs, args.patience, args.threads) < 1 or args.folds < 2:
            parser.error("Positive counts and at least two folds are required")
        if not 0.1 <= args.valid_size <= 0.4:
            parser.error("--valid-size must be between 0.1 and 0.4")
        torch.set_num_threads(args.threads)
        train(args)
    else:
        frame = pd.read_csv(args.data)
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        result = pd.DataFrame({ID: frame[ID] if ID in frame else np.arange(len(frame)),
                               "probability": predict_proba(frame, args.weights)})
        result.to_csv(args.output, index=False)
        print(f"Saved {len(result)} probabilities: {args.output}")


if __name__ == "__main__":
    main()
