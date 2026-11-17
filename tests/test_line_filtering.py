"""Check CUDA allocation bounds as well as outputs by running these tests with
compute-sanitizer --tool memcheck --error-exitcode 1 python -m pytest.
"""

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def smhc_module():
    if not torch.cuda.is_available():
        pytest.skip("Line filtering requires CUDA")

    import smhc

    try:
        yield smhc
    finally:
        smhc.destroy_dataframe()


@pytest.mark.parametrize("width", [1, 1024, 1025, 1600])
@pytest.mark.parametrize("use_mask", [False, True], ids=["unmasked", "masked"])
def test_filtering_preserves_valid_pixels_at_image_edges(smhc_module, width, use_mask):
    height = 3
    identity = torch.eye(3, dtype=torch.float32)
    translation = torch.zeros(3, dtype=torch.float32)
    projection = torch.cat((identity, translation[:, None]), dim=1)

    # Identical cameras project (x, y, 1) onto the corresponding image pixel.
    for camera_id in range(2):
        smhc_module.load_camera_params_to_dataframe(
            camera_id,
            height,
            width,
            projection,
            translation,
            identity,
            identity,
            identity,
            identity,
            translation,
            identity,
            translation,
            torch.zeros((3, 6), dtype=torch.float32),
            1.0,
            1.0,
        )

    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(x)
    positions = torch.stack((x, y, ones, ones), dim=-1).reshape(-1, 4)
    neighbor_positions = positions.clone()
    # A depth mismatch must still reject this pixel.
    neighbor_positions[0, 2] = 2.0
    directions = torch.zeros_like(positions)
    directions[:, 0] = 1.0
    orientation = torch.full_like(x, 0.25)
    confidence = torch.full_like(x, 4.0)
    mask = torch.ones_like(x) if use_mask else torch.empty(0)
    if use_mask:
        mask[1, -1] = 0.0

    for image_id, image_positions in enumerate((positions, neighbor_positions)):
        smhc_module.load_image_data_to_dataframe(
            image_id,
            height,
            width,
            ones,
            orientation,
            confidence,
            mask,
            torch.empty(0),
        )
        smhc_module.load_lpmvs_estimations_to_dataframe(
            image_id, image_positions, directions
        )

    params = smhc_module.LineFilteringHyperParameters(
        SimpleNamespace(
            line_filtering_position_consistency_thres=0.001,
            line_filtering_direction_diff_threshold_rad=math.radians(10.0),
            line_filtering_consistent_neighboring_views_threshold=1,
        )
    )
    int_map = torch.full((3, height, width), -1.0, device="cuda")
    smhc_module.run_multiview_filtering(
        params, 0, 1, torch.tensor([1], dtype=torch.int32), int_map
    )

    expected = torch.stack((ones, ones, torch.zeros_like(x)))
    expected[:, 0, 0] = torch.tensor([-1.0, 0.25, 4.0])
    if use_mask:
        expected[:, 1, -1] = torch.tensor([-1.0, 0.25, 4.0])

    # Includes every row boundary and the last valid pixel in the final block.
    torch.testing.assert_close(int_map.cpu(), expected, rtol=0, atol=0)
