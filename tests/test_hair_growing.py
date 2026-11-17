import math
from types import SimpleNamespace

import numpy as np
import pytest

from utils.graphics_utils import get_plucker_line_coordinates

torch = pytest.importorskip("torch")


@pytest.fixture
def smhc_module():
    if not torch.cuda.is_available():
        pytest.skip("The native camera dataframe requires CUDA")

    import smhc

    try:
        yield smhc
    finally:
        smhc.destroy_dataframe()


@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
@pytest.mark.parametrize(
    "independent_views", [True, False], ids=["consistent", "parallel"]
)
def test_growing_from_multiview_plane_intersections(
    smhc_module, scale, independent_views
):
    size = 64
    identity = torch.eye(3)
    intrinsics = torch.tensor([[64.0, 0.0, 32.0], [0.0, 64.0, 32.0], [0.0, 0.0, 1.0]])
    intrinsics_inverse = torch.linalg.inv(intrinsics)
    # These tensors must stay alive: the growing stage borrows their CPU storage.
    image = torch.ones((size, size))
    orientation = torch.zeros_like(image)
    confidence = torch.full_like(image, 10.0)
    camera_offsets = [-0.25, 0.0, 0.25] if independent_views else [0.0] * 3

    for camera_id, offset in enumerate(camera_offsets):
        center = torch.tensor([0.0, offset * scale, 0.0])
        translation = -center
        projection = intrinsics @ torch.cat((identity, translation[:, None]), dim=1)
        line_projection = torch.from_numpy(
            np.stack(
                [
                    np.flip(
                        get_plucker_line_coordinates(
                            projection[a].numpy(), projection[b].numpy()
                        )
                    )
                    for a, b in [(1, 2), (2, 0), (0, 1)]
                ]
            )
        )
        smhc_module.load_camera_params_to_dataframe(
            camera_id,
            size,
            size,
            projection,
            projection[:, 3].contiguous(),
            intrinsics_inverse,
            intrinsics,
            intrinsics_inverse,
            identity,
            translation,
            identity,
            center,
            line_projection,
            64.0,
            64.0,
        )
        smhc_module.load_image_data_to_dataframe(
            camera_id,
            size,
            size,
            image,
            orientation,
            confidence,
            torch.empty(0),
            torch.empty(0),
            load_to_cpu_containers=True,
        )

    params = smhc_module.MultiviewHairGrowingHyperParameters(
        SimpleNamespace(
            mvhg_reset_valid_views_period=1,
            mvhg_max_iter_num=0,  # One growing step at each tip.
            mvhg_background_ratio=0.9,
            mvhg_search_cone_opening_angle_rad=0.125,
            mvhg_search_cone_angular_resolution_rad=0.125,
            mvhg_search_cone_window_width=3.0,
            mvhg_search_cone_window_height=10.0,
            mvhg_orientation_2d_confidence_threshold=3.0,
            mvhg_orientation_2d_angle_difference_threshold_rad=math.radians(5.0),
            mvhg_valid_pixels_number_threshold=10,
            mvhg_minimum_views_with_valid_2d_direction=3,
            mvhg_growing_step_size=0.005 * scale,
            mvhg_direction_change_threshold_rad=math.radians(45.0),
            mvhg_background_color_threshold=0.01,
        )
    )
    positions = torch.tensor([[-0.01, 0.0, 1.0, 0.0], [0.01, 0.0, 1.0, 0.0]]) * scale
    directions = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)

    grown_positions, grown_directions, counts = smhc_module.run_multiview_hair_growing(
        params, 3, positions, directions, torch.tensor([2], dtype=torch.int32)
    )

    expected_x = [-0.015, -0.01, 0.01, 0.015] if independent_views else [-0.01, 0.01]
    assert counts.tolist() == [float(len(expected_x))]
    expected_positions = torch.tensor([[x, 0.0, 1.0] for x in expected_x])
    torch.testing.assert_close(
        grown_positions[:, :3] / scale, expected_positions, rtol=1e-5, atol=1e-7
    )
    expected_directions = torch.tensor([[1.0, 0.0, 0.0]] * len(expected_x))
    # Plane normals are constructed from float32 camera geometry, including
    # cancellation at the smallest scene scale.
    torch.testing.assert_close(
        grown_directions[:, :3].abs(), expected_directions, rtol=1e-5, atol=1e-5
    )
