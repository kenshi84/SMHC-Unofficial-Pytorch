import numpy as np
import pytest
from PIL import Image

from datasets.masks import load_hair_mask


def test_resized_mask_preserves_foreground_and_background_regions(tmp_path):
    path = tmp_path / "hair.png"
    Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8)).save(path)

    mask = load_hair_mask(path, (6, 4), required=True)

    assert mask.shape == (4, 6)
    assert mask.dtype == np.float32
    np.testing.assert_array_equal(mask[:2, :3], 0)
    np.testing.assert_array_equal(mask[:2, 3:], 1)
    np.testing.assert_array_equal(mask[2:, :3], 1)
    np.testing.assert_array_equal(mask[2:, 3:], 0)


def test_noninteger_resize_matches_image_grid_without_soft_labels(tmp_path):
    path = tmp_path / "hair.png"
    pixels = np.zeros((1024, 683), dtype=np.uint8)
    pixels[200:800, 100:500] = 254
    Image.fromarray(pixels).save(path)

    mask = load_hair_mask(path, (1600, 2400), required=True)

    assert mask.shape == (2400, 1600)
    assert set(np.unique(mask)) == {0.0, 1.0}
    assert mask[1200, 800] == 1
    assert mask[0, 0] == mask[-1, -1] == 0
    np.testing.assert_allclose(mask.mean(), (pixels > 0).mean(), atol=0.001)


def test_existing_foreground_threshold_is_preserved(tmp_path):
    path = tmp_path / "hair.png"
    Image.fromarray(np.array([[0, 1, 2, 3, 254, 255]], dtype=np.uint8)).save(path)

    mask = load_hair_mask(path, (6, 1), required=True)

    np.testing.assert_array_equal(mask, [[0, 0, 0, 1, 1, 1]])


def test_missing_required_mask_does_not_silently_disable_masking(tmp_path):
    with pytest.raises(FileNotFoundError, match="Hair mask requested"):
        load_hair_mask(tmp_path / "missing.png", (10, 12), required=True)
    np.testing.assert_array_equal(
        load_hair_mask(tmp_path / "missing.png", (10, 12)),
        np.ones((12, 10), dtype=np.float32),
    )
