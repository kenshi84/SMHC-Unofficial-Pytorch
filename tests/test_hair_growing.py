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


def grow_horizontal_strand(
    smhc_module,
    scale=1.0,
    independent_views=True,
    orientation_angle=0.0,
    confidence_value=10.0,
    image_values=None,
    hair_masks=None,
    minimum_views=3,
    growing_step=0.005,
    valid_pixels_threshold=10,
    only_background_removal=False,
    camera_centers=None,
    pipeline_path=None,
    use_hair_masks=True,
):
    size = 64
    identity = torch.eye(3)
    intrinsics = torch.tensor([[64.0, 0.0, 32.0], [0.0, 64.0, 32.0], [0.0, 0.0, 1.0]])
    intrinsics_inverse = torch.linalg.inv(intrinsics)
    camera_offsets = [-0.25, 0.0, 0.25] if independent_views else [0.0] * 3
    if camera_centers is None:
        camera_centers = [(0.0, offset, 0.0) for offset in camera_offsets]
    total_views = len(camera_centers)
    # These tensors must stay alive: the growing stage borrows their CPU storage.
    if image_values is None:
        image_values = [1.0] * total_views
    images = [torch.full((size, size), value) for value in image_values]
    orientation = torch.full((size, size), orientation_angle)
    confidence = (
        torch.as_tensor(confidence_value, dtype=torch.float32)
        .expand(size, size)
        .contiguous()
    )
    if hair_masks is None:
        hair_masks = [torch.empty(0) for _ in camera_centers]
    views = []

    for camera_id, coordinates in enumerate(camera_centers):
        center = torch.tensor(coordinates, dtype=torch.float32) * scale
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
            images[camera_id],
            orientation,
            confidence,
            hair_masks[camera_id],
            torch.empty(0),
            load_to_cpu_containers=True,
        )
        views.append(
            SimpleNamespace(
                u_id=camera_id,
                original_image=images[camera_id],
                gt_hair_mask=hair_masks[camera_id],
                ori2d_map=orientation,
                conf_map=confidence,
            )
        )

    hyper_args = SimpleNamespace(
        mvhg_reset_valid_views_period=1,
        mvhg_max_iter_num=0,  # One growing step at each tip.
        mvhg_background_ratio=0.9,
        mvhg_search_cone_opening_angle_rad=0.125,
        mvhg_search_cone_angular_resolution_rad=0.125,
        mvhg_search_cone_window_width=3.0,
        mvhg_search_cone_window_height=10.0,
        mvhg_orientation_2d_confidence_threshold=3.0,
        mvhg_orientation_2d_angle_difference_threshold_rad=math.radians(5.0),
        mvhg_valid_pixels_number_threshold=valid_pixels_threshold,
        mvhg_minimum_views_with_valid_2d_direction=minimum_views,
        mvhg_growing_step_size=growing_step * scale,
        mvhg_direction_change_threshold_rad=math.radians(45.0),
        mvhg_background_color_threshold=0.01,
    )
    params = smhc_module.MultiviewHairGrowingHyperParameters(hyper_args)
    positions = torch.tensor([[-0.01, 0.0, 1.0, 0.0], [0.01, 0.0, 1.0, 0.0]]) * scale
    directions = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)

    if pipeline_path is not None:
        from src.strand_accurate_reconstruction import SAR

        work = pipeline_path / "work"
        segments = work / "strand_segments"
        results = pipeline_path / "multiview_hairgrowing"
        segments.mkdir(parents=True)
        results.mkdir()
        np.save(segments / "segments_positions.npy", positions.numpy())
        np.save(segments / "segments_directions.npy", directions.numpy())
        np.save(segments / "points_num_per_segment.npy", np.array([2], dtype=np.int32))
        # Cameras are already loaded above; exercise the real mask-loading and
        # output-writing stage without running LPMVS on this synthetic scene.
        sar = SAR.__new__(SAR)
        sar.dataset = SimpleNamespace(views=views)
        sar.params = SimpleNamespace(
            **vars(hyper_args),
            work_path=str(work),
            output_path=str(pipeline_path),
            strand_segments_dir="strand_segments",
            multiview_growing_dir="multiview_hairgrowing",
            initial_depth_maps="",
            use_hair_masks=use_hair_masks,
            input_unit_scale=1.0,
            verbose=False,
        )
        sar.run_multiview_hair_growing()
        return tuple(
            torch.from_numpy(np.load(results / filename))
            for filename in (
                "final_positions.npy",
                "final_directions.npy",
                "final_points_num_per_strand.npy",
            )
        )

    return smhc_module.run_multiview_hair_growing(
        params,
        total_views,
        positions,
        directions,
        torch.tensor([2], dtype=torch.int32),
        only_apply_background_removal=only_background_removal,
    )


@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
@pytest.mark.parametrize(
    "independent_views", [True, False], ids=["consistent", "parallel"]
)
def test_growing_from_multiview_plane_intersections(
    smhc_module, scale, independent_views
):
    grown_positions, grown_directions, counts = grow_horizontal_strand(
        smhc_module, scale=scale, independent_views=independent_views
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


@pytest.mark.parametrize(
    "angle", [-0.125, 0.125], ids=["first-candidate", "last-candidate"]
)
def test_growing_accepts_candidates_at_both_cone_edges(smhc_module, angle):
    positions, directions, counts = grow_horizontal_strand(
        smhc_module, orientation_angle=angle
    )

    assert counts.tolist() == [4.0]
    growing_direction = torch.tensor([math.cos(angle), math.sin(angle), 0.0])
    tips = torch.tensor([[-0.01, 0.0, 1.0], [0.01, 0.0, 1.0]])
    expected_positions = torch.stack(
        [
            tips[0] - 0.005 * growing_direction,
            *tips,
            tips[1] + 0.005 * growing_direction,
        ]
    )
    torch.testing.assert_close(
        positions[:, :3], expected_positions, rtol=1e-5, atol=1e-7
    )
    torch.testing.assert_close(
        directions[[0, -1], :3].abs(),
        growing_direction.abs().repeat(2, 1),
        rtol=1e-5,
        atol=1e-5,
    )


def test_growing_stops_when_no_candidate_is_valid(smhc_module):
    positions, _, counts = grow_horizontal_strand(smhc_module, confidence_value=0.0)

    assert counts.tolist() == [2.0]
    torch.testing.assert_close(
        positions[:, :3], torch.tensor([[-0.01, 0.0, 1.0], [0.01, 0.0, 1.0]])
    )


@pytest.mark.parametrize("only_background_removal", [False, True])
def test_hair_masks_preserve_dark_foreground(smhc_module, only_background_removal):
    positions, _, counts = grow_horizontal_strand(
        smhc_module,
        image_values=[0.0] * 3,
        hair_masks=[torch.ones((64, 64))] * 3,
        only_background_removal=only_background_removal,
    )

    assert counts.tolist() == [2.0 if only_background_removal else 4.0]
    assert torch.isfinite(positions[:, :3]).all()


@pytest.mark.parametrize("only_background_removal", [False, True])
def test_hair_masks_reject_bright_background(smhc_module, only_background_removal):
    positions, _, counts = grow_horizontal_strand(
        smhc_module,
        hair_masks=[torch.zeros((64, 64))] * 3,
        only_background_removal=only_background_removal,
    )

    assert counts.numel() == positions.numel() == 0


def test_growing_stops_before_leaving_the_hair_mask(smhc_module):
    mask = torch.zeros((64, 64))
    mask[:, 31:33] = 1.0
    positions, _, counts = grow_horizontal_strand(
        smhc_module,
        hair_masks=[mask] * 3,
        growing_step=0.02,
        valid_pixels_threshold=1,
    )

    assert counts.tolist() == [2.0]
    torch.testing.assert_close(
        positions[:, :3], torch.tensor([[-0.01, 0.0, 1.0], [0.01, 0.0, 1.0]])
    )


def test_direction_windows_ignore_orientation_evidence_outside_hair(smhc_module):
    mask = torch.zeros((64, 64))
    mask[:, 31:33] = 1.0
    positions, _, counts = grow_horizontal_strand(
        smhc_module,
        hair_masks=[mask] * 3,
        confidence_value=(1.0 - mask) * 10.0,
    )

    assert counts.tolist() == [2.0]
    assert positions.shape[0] == 2


@pytest.mark.parametrize("minimum_views,expected_count", [(2, 4), (3, 0)])
def test_mask_support_uses_the_configured_view_count(
    smhc_module, minimum_views, expected_count
):
    positions, _, counts = grow_horizontal_strand(
        smhc_module,
        image_values=[0.0, 1.0, 1.0],
        hair_masks=[torch.zeros((64, 64)), torch.ones((64, 64)), torch.ones((64, 64))],
        minimum_views=minimum_views,
    )

    assert positions.shape[0] == expected_count
    assert counts.tolist() == ([float(expected_count)] if expected_count else [])


@pytest.mark.parametrize(
    "invalid_camera",
    [(0.0, 0.0, 2.0), (0.0, 5.0, 0.0), (0.0, 0.0, 1.0)],
    ids=["behind", "offscreen", "zero-depth"],
)
def test_unprojectable_views_do_not_supply_mask_support(smhc_module, invalid_camera):
    positions, _, counts = grow_horizontal_strand(
        smhc_module,
        hair_masks=[torch.ones((64, 64))] * 3,
        camera_centers=[(0.0, -0.25, 0.0), (0.0, 0.25, 0.0), invalid_camera],
        only_background_removal=True,
    )

    assert counts.numel() == positions.numel() == 0


@pytest.mark.parametrize("minimum_views,expected_count", [(2, 2), (3, 0)])
def test_views_without_masks_do_not_supply_mask_support(
    smhc_module, minimum_views, expected_count
):
    positions, _, counts = grow_horizontal_strand(
        smhc_module,
        hair_masks=[torch.empty(0), torch.ones((64, 64)), torch.ones((64, 64))],
        minimum_views=minimum_views,
        only_background_removal=True,
    )

    assert positions.shape[0] == expected_count
    assert counts.tolist() == ([float(expected_count)] if expected_count else [])


@pytest.mark.parametrize("only_background_removal", [False, True])
def test_unmasked_dark_background_still_gets_removed(
    smhc_module, only_background_removal
):
    positions, _, counts = grow_horizontal_strand(
        smhc_module,
        image_values=[0.0] * 3,
        only_background_removal=only_background_removal,
    )

    assert counts.numel() == positions.numel() == 0


@pytest.mark.parametrize(
    "use_masks,image_value,mask_value,expected_count",
    [(True, 0.0, 1.0, 4), (True, 1.0, 0.0, 0), (False, 1.0, 0.0, 4)],
    ids=["dark-hair", "bright-background", "masking-disabled"],
)
def test_sar_propagates_requested_hair_masks_to_growing(
    smhc_module, tmp_path, use_masks, image_value, mask_value, expected_count
):
    positions, _, counts = grow_horizontal_strand(
        smhc_module,
        image_values=[image_value] * 3,
        hair_masks=[torch.full((64, 64), mask_value)] * 3,
        pipeline_path=tmp_path,
        use_hair_masks=use_masks,
    )

    assert positions.shape[0] == expected_count
    assert counts.tolist() == ([expected_count] if expected_count else [])
