"""Perform image operations in the interpreter that has Pillow installed."""
import json
import sys
from functools import lru_cache
from PIL import Image, ImageOps


@lru_cache(maxsize=32)
def load(path, size, augment):
    with Image.open(path) as source:
        image = source.convert("RGB")
    if augment:
        image.thumbnail((size, size), Image.Resampling.BILINEAR)
    return image


if __name__ == "__main__":
    for request in sys.stdin.buffer:
        path, size, augment, flip, angle = json.loads(request)
        image = load(path, size, augment)
        if flip:
            image = ImageOps.mirror(image)
        if angle:
            image = image.rotate(angle, Image.Resampling.BILINEAR, expand=True, fillcolor=0)
        image = ImageOps.pad(image, (size, size), method=Image.Resampling.BILINEAR, color=(0, 0, 0))
        sys.stdout.buffer.write(image.tobytes())
        sys.stdout.buffer.flush()
