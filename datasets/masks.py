"""Load hair masks in the pixel grid used by the reconstruction images."""

from pathlib import Path

import numpy as np
from PIL import Image


def load_hair_mask(path, image_size, *, required=False):
    """Return a float32 binary mask at (width, height), preserving label edges.

    The existing foreground threshold of 0.01 is applied after nearest-neighbor
    resizing. Missing optional masks cover the whole image; required masks fail.
    """
    path = Path(path)
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Hair mask requested but not found: {path}")
        width, height = image_size
        return np.ones((height, width), dtype=np.float32)

    with Image.open(path) as source:
        mask = source.convert("L")
        if mask.size != image_size:
            mask = mask.resize(image_size, resample=Image.Resampling.NEAREST)
        values = np.asarray(mask, dtype=np.float32) * (1.0 / 255.0)
    return (values > 0.01).astype(np.float32)
