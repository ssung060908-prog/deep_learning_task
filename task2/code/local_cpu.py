"""Use torchvision's pure PyTorch ResNet definition and a separate Pillow interpreter."""
import ast
import atexit
import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch


def versions():
    wheel = Path(os.environ["TASK2_VISION_WHEEL"])
    with ZipFile(wheel) as archive:
        digest = hashlib.sha256(archive.read("torchvision/models/resnet.py")).hexdigest()
    pillow = subprocess.check_output([os.environ["TASK2_PIL_PYTHON"], "-c", "import PIL; print(PIL.__version__)"], text=True)
    return {"torchvision": wheel.name, "resnet_source_sha256": digest, "Pillow": pillow.strip()}


def resnet18():
    with ZipFile(os.environ["TASK2_VISION_WHEEL"]) as archive:
        tree = ast.parse(archive.read("torchvision/models/resnet.py").decode())
    names = {"conv3x3", "conv1x1", "BasicBlock", "Bottleneck", "ResNet"}
    tree.body = [node for node in tree.body if getattr(node, "name", None) in names]
    from typing import Callable, Optional, Union
    namespace = dict(torch=torch, nn=torch.nn, Tensor=torch.Tensor, Callable=Callable,
                     Optional=Optional, Union=Union, _log_api_usage_once=lambda model: None)
    exec(compile(tree, "torchvision/models/resnet.py", "exec"), namespace)
    return namespace["ResNet"](namespace["BasicBlock"], [2, 2, 2, 2], num_classes=2)


@lru_cache(maxsize=1)
def worker():
    proc = subprocess.Popen([os.environ["TASK2_PIL_PYTHON"], Path(__file__).with_name("local_pillow_worker.py")],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    atexit.register(proc.terminate)
    return proc


def pixels(path, size, augment, flip, angle):
    proc = worker()
    proc.stdin.write(json.dumps([str(path), size, augment, flip, angle]).encode() + b"\n")
    proc.stdin.flush()
    raw = proc.stdout.read(3 * size * size)
    if len(raw) != 3 * size * size:
        raise RuntimeError("Pillow worker stopped during preprocessing")
    return np.frombuffer(raw, dtype=np.uint8).reshape(size, size, 3).astype(np.float32) / 255.0
