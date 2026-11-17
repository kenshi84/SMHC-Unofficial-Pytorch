"""Regression tests for the actual CUDA multi-view image cost."""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def lpmvs_options():
    pytest.importorskip("torch")
    from main import parse_args
    from smhc import LPMVSHyperParameters

    return LPMVSHyperParameters, parse_args


def test_default_and_legacy_configs_enable_zncc(lpmvs_options):
    parameters, parse_args = lpmvs_options
    args = parse_args([])
    assert parameters(args).photometric_method == "zncc"
    assert parameters(args).zncc_min_variance == pytest.approx(1e-6)
    del args.lpmvs_photometric_method
    del args.lpmvs_zncc_min_variance
    assert parameters(args).photometric_method == "zncc"
    assert parameters(args).zncc_min_variance == pytest.approx(1e-6)


def test_photometric_cli_options_reach_hyperparameters(lpmvs_options):
    parameters, parse_args = lpmvs_options
    args = parse_args(
        ["--lpmvs_photometric_method", "cosine", "--lpmvs_zncc_min_variance", "0.0001"]
    )

    assert parameters(args).photometric_method == "cosine"
    assert parameters(args).zncc_min_variance == pytest.approx(0.0001)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_photometric_variance_is_rejected(lpmvs_options, value):
    parameters, parse_args = lpmvs_options
    args = parse_args([])
    args.lpmvs_zncc_min_variance = value

    with pytest.raises(ValueError, match="lpmvs_zncc_min_variance"):
        parameters(args)


def test_unknown_photometric_method_is_rejected(lpmvs_options):
    parameters, parse_args = lpmvs_options
    args = parse_args([])
    args.lpmvs_photometric_method = "invalid"

    with pytest.raises(ValueError, match="lpmvs_photometric_method"):
        parameters(args)


@pytest.fixture(scope="module")
def multiview_image_cost(tmp_path_factory):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("LPMVS texture sampling requires CUDA")
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("The CUDA toolkit is required to compile the LPMVS cost test")

    root = Path(__file__).resolve().parents[1]
    csrc = root / "src/smhc/cuda/csrc"
    executable = tmp_path_factory.mktemp("lpmvs_cost") / "cost"
    build = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-O3",
            "--use_fast_math",
            "-I",
            str(csrc),
            str(root / "tests/native/lpmvs_photometric_cost.cu"),
            str(csrc / "device_dataframe.cu"),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr

    def evaluate(
        reference, neighbors, masks=None, depths=None, method="zncc", variance=1e-6
    ):
        images = np.asarray([reference, *neighbors], dtype=np.float32)
        assert images.shape[1:] == (5,)
        if masks is None:
            masks = np.ones_like(images)
        if depths is None:
            depths = np.zeros(len(images))
        lines = [str(len(neighbors))]
        for image, mask, depth in zip(images, masks, depths):
            lines.append(" ".join(map(str, [*image, *mask, depth])))
        result = subprocess.run(
            [str(executable), str(int(method == "zncc")), str(variance)],
            input="\n".join(lines),
            capture_output=True,
            text=True,
            check=True,
        )
        cost = float(result.stdout)
        assert np.isfinite(cost)
        assert -1e-6 <= cost <= 1 + 1e-6
        return cost

    return evaluate


def test_lpmvs_rejects_inverted_low_contrast_texture(multiview_image_cost):
    reference = np.array([0.49, 0.51, 0.49, 0.51, 0.49])

    assert multiview_image_cost(reference, [1 - reference]) == pytest.approx(
        1, abs=1e-5
    )


def test_lpmvs_matches_texture_despite_brightness_changes(multiview_image_cost):
    reference = np.array([0.1, 0.5, 0.3, 0.2, 0.6])

    assert multiview_image_cost(reference, [0.2 + 0.5 * reference]) == pytest.approx(
        0, abs=1e-5
    )


def test_lpmvs_does_not_reward_flat_patches(multiview_image_cost):
    assert multiview_image_cost([0.5] * 5, [[0.5] * 5]) == pytest.approx(1)


def test_lpmvs_counts_only_unmasked_sample_pairs(multiview_image_cost):
    reference = np.array([0.49, 0.51, 0.49, 0.51, 0.49])
    masks = [[1] * 5, [0, 0, 1, 1, 1]]

    assert multiview_image_cost(
        reference, [1 - reference], masks=masks
    ) == pytest.approx(1, abs=1e-5)


@pytest.mark.parametrize(
    "mask", [[0] * 5, [0, 0, 1, 0, 0]], ids=["no-pairs", "one-pair"]
)
def test_lpmvs_does_not_reward_insufficient_unmasked_samples(
    multiview_image_cost, mask
):
    reference = [0.1, 0.2, 0.3, 0.4, 0.5]

    assert multiview_image_cost(
        reference, [reference], masks=[[1] * 5, mask]
    ) == pytest.approx(1)


def test_flat_neighbors_do_not_dilute_photometric_cost(multiview_image_cost):
    reference = np.array([0.49, 0.51, 0.49, 0.51, 0.49])

    assert multiview_image_cost(reference, [1 - reference, [0.5] * 5]) == pytest.approx(
        1, abs=1e-5
    )


@pytest.mark.parametrize(
    "depths", [[1, 1, 2], [1, 2, 2]], ids=["one-visible", "none-visible"]
)
def test_occluded_neighbors_do_not_reduce_photometric_cost(
    multiview_image_cost, depths
):
    reference = np.array([0.49, 0.51, 0.49, 0.51, 0.49])

    assert multiview_image_cost(
        reference, [1 - reference, reference], depths=depths
    ) == pytest.approx(1, abs=1e-5)


def test_lpmvs_variance_threshold_controls_faint_texture(multiview_image_cost):
    reference = [0.4999, 0.5001, 0.4999, 0.5001, 0.4999]

    assert multiview_image_cost(reference, [reference]) == pytest.approx(1)
    assert multiview_image_cost(reference, [reference], variance=1e-9) == pytest.approx(
        0, abs=1e-5
    )


def test_lpmvs_can_use_the_previous_cosine_score(multiview_image_cost):
    reference = np.array([0.49, 0.51, 0.49, 0.51, 0.49])
    neighbor = 1 - reference
    expected = 0.5 * (
        1
        - reference
        @ neighbor
        / (np.linalg.norm(reference) * np.linalg.norm(neighbor) + 1e-8)
    )

    assert multiview_image_cost(
        reference, [neighbor], method="cosine"
    ) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("method", ["zncc", "cosine"])
def test_lpmvs_public_api_preserves_initial_hypotheses(method):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("LPMVS requires CUDA")
    import smhc
    from utils.graphics_utils import get_plucker_line_coordinates

    size = 16  # Complete CUDA tiles, including the random-state initializer.
    identity = torch.eye(3)
    zero = torch.zeros(3)
    projection = torch.cat((identity, zero[:, None]), dim=1)
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
    image = torch.linspace(0.1, 0.9, size).repeat(size, 1)
    try:
        for camera in range(2):
            smhc.load_camera_params_to_dataframe(
                camera,
                size,
                size,
                projection,
                zero,
                identity,
                identity,
                identity,
                identity,
                zero,
                identity,
                zero,
                line_projection,
                1.0,
                1.0,
            )
            smhc.load_image_data_to_dataframe(
                camera,
                size,
                size,
                image,
                torch.zeros_like(image),
                torch.ones_like(image),
                torch.empty(0),
                torch.empty(0),
            )
        params = smhc.LPMVSHyperParameters(
            SimpleNamespace(
                lpmvs_alpha=0.1,
                lpmvs_iterations=0,
                lpmvs_minimum_depth=0.5,
                lpmvs_maximum_depth=1.5,
                lpmvs_sampling_radius=2.0,
                lpmvs_samples_num=4,
                lpmvs_occlusion_depth_thresh=0.001,
                lpmvs_photometric_method=method,
                lpmvs_zncc_min_variance=1e-6,
            )
        )
        depths = torch.ones((size, size), device="cuda")
        directions = torch.zeros((3, size, size), device="cuda")
        directions[0] = 1
        positions, estimated_directions = smhc.run_lpmvs(
            params,
            size,
            size,
            -1.0,
            -1.0,
            0,
            1,
            torch.tensor([1], dtype=torch.int32),
            depths,
            directions,
        )
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        expected = torch.stack((x, y, torch.ones_like(x))).float()
        torch.testing.assert_close(positions[:3].cpu(), expected)
        torch.testing.assert_close(estimated_directions[:3], directions)
    finally:
        smhc.destroy_dataframe()
