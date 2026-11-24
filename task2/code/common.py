"""Task 2 training: independent, randomly initialized VGG/ResNet/GoogLeNet."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import GEOMETRY_POLICIES, ImageDataset, load_records, make_splits, split_summary
from training_state import STATE_NAME, atomic_save, capture_rng, check_metadata, check_progress, read_state, restore_rng

TASK_DIR = Path(__file__).resolve().parents[1]
ARCHITECTURES = {"vgg": "vgg11", "resnet": "resnet18", "googlenet": "googlenet"}
PREPROCESSING = {
    "version": 1, "color_mode": "RGB", "resize": "aspect_preserving_letterbox",
    "interpolation": "bilinear", "padding_color": [0, 0, 0],
    "pixel_scale": 255.0, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_config(path: str | Path | None = None) -> dict:
    """Relative data/output paths are always relative to the task2 folder."""
    path = Path(path) if path else TASK_DIR / "configs/baseline.json"
    return json.loads(path.read_text(encoding="utf-8"))


def task_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (TASK_DIR / path).resolve()


def setup_runtime(seed: int, cpu_threads: int = 4) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # GoogLeNet auxiliary adaptive pooling has no deterministic CUDA backward.
    # Warn for that operation while retaining deterministic kernels where available.
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu or auto.")
    return device


def build_model(model_name: str) -> nn.Module:
    """Use the original model families; never load external pretrained weights."""
    if model_name == "resnet" and os.environ.get("TASK2_VISION_WHEEL"):
        from local_cpu import resnet18
        return resnet18()
    import torchvision
    if model_name == "vgg":
        return torchvision.models.vgg11(weights=None, num_classes=2)
    if model_name == "resnet":
        return torchvision.models.resnet18(weights=None, num_classes=2)
    if model_name == "googlenet":
        return torchvision.models.googlenet(
            weights=None, num_classes=2, aux_logits=True,
            transform_input=False, init_weights=True,
        )
    raise ValueError(f"Unknown model {model_name!r}; expected {list(ARCHITECTURES)}")


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "samples": int(len(labels)),
        "threshold": float(threshold),
    }


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(records: pd.DataFrame, config: dict, training: bool) -> DataLoader:
    dataset = ImageDataset(records, task_path(config["data_root"]), image_size=config["image_size"],
                           augment=training and config.get("geometric_augmentation", False),
                           geometry_mode=config.get("geometry_mode", "flip_rotate"))
    return DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=training,
        num_workers=config["num_workers"], pin_memory=config["resolved_device"].startswith("cuda"),
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(config["seed"]),
    )


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
              class_weights: torch.Tensor, threshold: float,
              optimizer: torch.optim.Optimizer | None = None, aux_weight: float = 0.3) -> tuple[dict, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    loss_sum, weight_sum = 0.0, 0.0
    all_labels, all_probabilities = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(images)
            logits = output.logits if hasattr(output, "logits") else output
            numerator = F.cross_entropy(logits, labels, weight=class_weights, reduction="sum")
            # GoogLeNet's auxiliary classifiers are part of the same network.
            if training and hasattr(output, "aux_logits1"):
                for auxiliary in (output.aux_logits1, output.aux_logits2):
                    if auxiliary is not None:
                        numerator = numerator + aux_weight * F.cross_entropy(
                            auxiliary, labels, weight=class_weights, reduction="sum")
            denominator = class_weights[labels].sum()
            loss = numerator / denominator
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; inspect the inputs and learning rate.")
            if training:
                loss.backward()
                optimizer.step()
        loss_sum += numerator.detach().item()
        weight_sum += denominator.detach().item()
        all_labels.append(labels.detach().cpu().numpy())
        all_probabilities.append(logits.detach().softmax(dim=1)[:, 1].cpu().numpy())
    probabilities = np.concatenate(all_probabilities)
    metrics = classification_metrics(np.concatenate(all_labels), probabilities, threshold)
    metrics["loss"] = float(loss_sum / weight_sum)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metrics["elapsed_seconds"] = time.perf_counter() - started
    metrics["images_per_second"] = metrics["samples"] / metrics["elapsed_seconds"]
    return metrics, probabilities


def validate_config(config: dict) -> None:
    for key in ("epochs", "batch_size", "patience", "cpu_threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["batch_size"] < 2:
        raise ValueError("batch_size must be at least 2 for GoogLeNet BatchNorm")
    if type(config["image_size"]) is not int or config["image_size"] < 64:
        raise ValueError("image_size must be an integer >= 64")
    if type(config["num_workers"]) is not int or config["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    for key in ("learning_rate", "weight_decay", "aux_loss_weight"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if config["learning_rate"] == 0 or not 0 < config["threshold"] < 1:
        raise ValueError("learning_rate must be positive and threshold must be between 0 and 1")


def train(model_name: str, config: dict, smoke_test: bool = False, run_name: str | None = None,
          resume: str | Path | None = None) -> Path:
    """Train one model and save its best validation-BA checkpoint and predictions."""
    if model_name not in ARCHITECTURES:
        raise ValueError(f"Unknown model: {model_name}")
    config = dict(config)
    if smoke_test:
        config.update(epochs=1, batch_size=2, image_size=96, num_workers=0)
    config.setdefault("geometric_augmentation", False)
    if type(config["geometric_augmentation"]) is not bool:
        raise ValueError("geometric_augmentation must be a boolean")
    geometry_mode = config.get("geometry_mode", "flip_rotate")
    if geometry_mode not in GEOMETRY_POLICIES:
        raise ValueError(f"Unknown geometry_mode: {geometry_mode!r}")
    validate_config(config)
    setup_runtime(config["seed"], config["cpu_threads"])
    device = resolve_device(config["device"])
    config["resolved_device"] = str(device)
    records = make_splits(load_records(task_path(config["data_root"])), config["n_splits"], config["val_fold"])
    parts = {split: records.loc[records["split"] == split].reset_index(drop=True) for split in ("train", "valid")}
    # Keep the real split manifest; the smoke subset is recorded separately.
    if smoke_test:
        parts = {split: frame.groupby("label", group_keys=False).head(2).reset_index(drop=True)
                 for split, frame in parts.items()}
    counts = parts["train"]["label"].value_counts().reindex([0, 1]).to_numpy()
    weights = len(parts["train"]) / (2 * counts) if config["class_weighted_loss"] else np.ones(2)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    source_hashes = {
        str(source.relative_to(TASK_DIR)): hashlib.sha256(source.read_bytes()).hexdigest()
        for source in sorted((TASK_DIR / "code").glob("*.py"))
    }
    source_hashes["labels.csv"] = hashlib.sha256((task_path(config["data_root"]) / "labels.csv").read_bytes()).hexdigest()
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "numpy", "pandas", "scikit-learn")}
    if os.environ.get("TASK2_VISION_WHEEL"):
        from local_cpu import versions as local_versions
        versions.update(local_versions())
    else:
        versions.update({name: importlib.metadata.version(name) for name in ("torchvision", "Pillow")})
    versions["runtime"] = {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "cuda_build": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_total_memory_mib": torch.cuda.get_device_properties(device).total_memory / 2**20
        if device.type == "cuda" else None,
    }
    manifest_bytes = records.to_csv(index=False).encode("utf-8")
    metadata = {
        "model_name": model_name, "config": config, "smoke_test": smoke_test,
        "split_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_manifest": source_hashes, "environment": versions,
    }
    state = None
    if resume is not None:
        if run_name is not None:
            raise ValueError("Use --resume without --run-name; the existing run is continued")
        run_dir, state = read_state(resume)
        check_metadata(state["metadata"], metadata)
        check_progress(state)
        recorded_config = {**config, "model_name": model_name,
                           "architecture": ARCHITECTURES[model_name], "smoke_test": smoke_test}
        for filename, expected in (("source_manifest.json", source_hashes), ("environment.json", versions),
                                   ("config.json", recorded_config)):
            if json.loads((run_dir / filename).read_text()) != expected:
                raise ValueError(f"Refusing resume: saved {filename} differs")
        if (run_dir / "split_manifest.csv").read_bytes() != manifest_bytes:
            raise ValueError("Refusing resume: saved split_manifest.csv differs")
        for name, digest in source_hashes.items():
            if name.startswith("code/"):
                snapshot = run_dir / "source" / Path(name).name
                if not snapshot.is_file() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Refusing resume: original source snapshot differs: {name}")
    else:
        stamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f")
        run_name = run_name or f"{model_name}_{stamp}"
        if Path(run_name).name != run_name or run_name in (".", ".."):
            raise ValueError("run_name must be a single directory name")
        if smoke_test and not run_name.startswith("smoke_"):
            run_name = "smoke_" + run_name
        run_dir = task_path(config["output_root"]) / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        source_dir = run_dir / "source"
        source_dir.mkdir()
        for source in sorted((TASK_DIR / "code").glob("*.py")):
            shutil.copy2(source, source_dir / source.name)
        write_json(run_dir / "source_manifest.json", source_hashes)
        write_json(run_dir / "augmentation.json", {
            "training_only": True,
            "enabled": config["geometric_augmentation"],
            "policy": GEOMETRY_POLICIES[geometry_mode]
            if config["geometric_augmentation"] else "none",
            "validation_and_inference": "unchanged preprocessing version 1",
        })
        (run_dir / "split_manifest.csv").write_bytes(manifest_bytes)
        write_json(run_dir / "split_summary.json", split_summary(records))
        if smoke_test:
            pd.concat([parts["train"], parts["valid"]]).to_csv(run_dir / "smoke_subset.csv", index=False)
        write_json(run_dir / "config.json", {**config, "model_name": model_name,
                                           "architecture": ARCHITECTURES[model_name], "smoke_test": smoke_test})
        write_json(run_dir / "environment.json", versions)
    summary = split_summary(records)

    def log(message: str) -> None:
        print(message, flush=True)
        with (run_dir / "training.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    log(json.dumps({"run_dir": str(run_dir), "device": str(device), "splits": summary,
                    "smoke_test": smoke_test}, ensure_ascii=False))

    loaders = {split: make_loader(frame, config, split == "train") for split, frame in parts.items()}
    model = build_model(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    history, best_score, best_loss, stale_epochs = [], -1.0, float("inf"), 0
    best_result = None
    weights_path = run_dir / f"{model_name}.pt"
    completed_epoch, previous_elapsed, save_seconds = 0, 0.0, 0.0

    def checkpoint_for_current(result: dict) -> dict:
        return {
            "format_version": 1, "model_name": model_name, "architecture": ARCHITECTURES[model_name],
            "state_dict": model.state_dict(), "config": config, "epoch": result["epoch"],
            "valid_metrics": result["metrics"], "threshold": config["threshold"],
            "class_weights": weights.tolist(), "smoke_test": smoke_test,
            "initialization": "random; weights=None", "label_mapping": {0: "absent", 1: "present"},
            "preprocessing": {**PREPROCESSING, "image_size": config["image_size"]},
        }

    def write_best_records(result: dict, save_weights: bool) -> None:
        if save_weights:
            atomic_save(weights_path, checkpoint_for_current(result))
        write_json(run_dir / "best_metrics.json", {"epoch": result["epoch"], "smoke_test": smoke_test, **result["metrics"]})
        predictions = parts["valid"].copy()
        predictions["probability"] = result["probabilities"]
        predictions["prediction"] = (predictions["probability"].to_numpy() >= config["threshold"]).astype(int)
        predictions.to_csv(run_dir / "valid_predictions.csv", index=False)

    if state is not None:
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        completed_epoch = state["completed_epoch"]
        history, best_score, best_loss = state["history"], state["best_score"], state["best_loss"]
        stale_epochs, best_result = state["stale_epochs"], state["best_result"]
        previous_elapsed = state["elapsed_seconds"]
        save_seconds = state.get("state_save_seconds_before_latest", 0.0)
        # A committed state is authoritative. Recreate text files if interrupted
        # between the atomic state rename and the following report-file writes.
        recover_current_best = best_result["epoch"] == completed_epoch
        if not recover_current_best:
            if not weights_path.is_file():
                raise ValueError("Selected earlier best checkpoint is missing; refusing unsafe resume")
            selected = torch.load(weights_path, map_location="cpu", weights_only=True)
            if selected.get("epoch") != best_result["epoch"] or selected.get("model_name") != model_name:
                raise ValueError("Selected earlier best checkpoint disagrees with committed training state")
            del selected
        write_best_records(best_result, save_weights=recover_current_best)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        restore_rng(state["rng"], loaders)
        log(json.dumps({"resume_from_completed_epoch": completed_epoch,
                        "resume_policy": "atomic epoch boundary; reports reconciled from committed state"}))
        del state
    training_started = time.perf_counter()
    for epoch in range(completed_epoch + 1, config["epochs"] + 1):
        if stale_epochs >= config["patience"]:
            break
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_metrics, _ = run_epoch(model, loaders["train"], device, class_weights,
                                     config["threshold"], optimizer, config["aux_loss_weight"])
        valid_metrics, probabilities = run_epoch(model, loaders["valid"], device, class_weights, config["threshold"])
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row["peak_gpu_memory_mib"] = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        for prefix, metrics in (("train", train_metrics), ("valid", valid_metrics)):
            row.update({f"{prefix}_{key}": value for key, value in metrics.items()
                        if key not in ("confusion_matrix", "threshold", "samples")})
        history.append(row)
        score, valid_loss = valid_metrics["balanced_accuracy"], valid_metrics["loss"]
        improved = score > best_score or (score == best_score and valid_loss < best_loss)
        if improved:
            best_score, best_loss, stale_epochs = score, valid_loss, 0
            best_result = {"epoch": epoch, "metrics": valid_metrics, "probabilities": probabilities.tolist()}
        else:
            stale_epochs += 1
        scheduler.step()
        # Commit before report files: an interruption after this rename can be
        # repaired from the last state without repeating a committed epoch.
        save_seconds += atomic_save(run_dir / STATE_NAME, {
            "training_state_version": 1, "purpose": "epoch_boundary_resume", "metadata": metadata,
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "completed_epoch": epoch,
            "history": history, "best_score": best_score, "best_loss": best_loss,
            "stale_epochs": stale_epochs, "best_result": best_result,
            "elapsed_seconds": previous_elapsed + time.perf_counter() - training_started,
            "state_save_seconds_before_latest": save_seconds, "rng": capture_rng(loaders),
        })
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        log(json.dumps(row, allow_nan=False))
        if improved:
            write_best_records(best_result, save_weights=True)
        if stale_epochs >= config["patience"]:
            log(f"Early stopping after epoch {epoch}.")
            break
    write_json(run_dir / "training_summary.json", {
        "model_name": model_name, "smoke_test": smoke_test, "epochs_completed": len(history),
        "elapsed_seconds": previous_elapsed + time.perf_counter() - training_started,
        "best_valid_balanced_accuracy": best_score, "weights_path": str(weights_path),
        "resumable_state_path": str(run_dir / STATE_NAME), "state_save_seconds": save_seconds,
        "resumed_from_epoch": completed_epoch,
        "state_save_seconds_note": "cumulative; after resume excludes the last save before interruption",
    })
    log(f"Saved best validation BA={best_score:.4f}: {weights_path}")
    return run_dir


def load_trained_model(weights_path: str | Path, device: str = "auto") -> tuple[nn.Module, dict, torch.device]:
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    setup_runtime(checkpoint["config"]["seed"], checkpoint["config"]["cpu_threads"])
    resolved_device = resolve_device(device)
    model = build_model(checkpoint["model_name"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(resolved_device).eval(), checkpoint, resolved_device


def training_cli(model_name: str) -> None:
    parser = argparse.ArgumentParser(description=f"Task 2 {ARCHITECTURES[model_name]} from scratch")
    parser.add_argument("--config", type=Path, default=TASK_DIR / "configs/baseline.json")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--geometry-mode", choices=GEOMETRY_POLICIES,
                        help="Enable the selected training-only geometric augmentation")
    parser.add_argument("--device")
    parser.add_argument("--run-name")
    parser.add_argument("--resume", type=Path, help="existing run directory or last_training_state.pt; requires identical config/source/environment")
    parser.add_argument("--smoke-test", action="store_true", help="1 epoch, 4 images per split at 96px; not a performance result")
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ("epochs", "batch_size", "image_size", "device"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.geometry_mode is not None:
        config.update(geometric_augmentation=True, geometry_mode=args.geometry_mode)
    train(model_name, config, smoke_test=args.smoke_test, run_name=args.run_name, resume=args.resume)
