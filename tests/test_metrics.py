import ctypes
import math
import re

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def calculate_metrics():
    if not torch.cuda.is_available():
        pytest.skip("Hair metrics build their search trees on CUDA")

    from smhc import calculate_hair_metrics

    return calculate_hair_metrics


def measure(capfd, calculate_metrics, directions, *, positions=None, distances=None):
    gt_positions = [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]
    angles = [5.0, 10.0, 20.0]
    if distances is None:
        distances = [0.0005] * len(angles)
    if positions is None:
        positions = gt_positions

    calculate_metrics(
        torch.tensor(distances, dtype=torch.float32),
        torch.tensor(angles, dtype=torch.float32),
        torch.tensor(gt_positions, dtype=torch.float32),
        torch.tensor([[1.0, 0.0, 0.0]] * 2, dtype=torch.float32),
        torch.tensor(positions, dtype=torch.float32),
        torch.tensor(directions, dtype=torch.float32),
    )
    # Native printf output is buffered when pytest redirects stdout to a file.
    ctypes.CDLL(None).fflush(None)
    output = capfd.readouterr().out
    rows = re.findall(
        r"for radius range:\s*(\S+),\s*angle range:\s*(\S+),\s*"
        r"precision:\s*(\S+),\s*recall:\s*(\S+),\s*f_score:\s*(\S+)",
        output,
    )
    assert len(rows) == len(angles), output
    results = [tuple(map(float, row)) for row in rows]
    # The public API and reported thresholds must both continue to use degrees.
    assert [row[1] for row in results] == angles
    assert [row[0] for row in results] == pytest.approx(distances)
    return [row[2:] for row in results]


def test_orthogonal_directions_fail_degree_thresholds_with_zero_f1(
    capfd, calculate_metrics
):
    results = measure(capfd, calculate_metrics, [[0.0, 1.0, 0.0]] * 2)

    assert results == [(0.0, 0.0, 0.0)] * 3


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_aligned_and_reversed_directions_match(capfd, calculate_metrics, sign):
    results = measure(capfd, calculate_metrics, [[sign, 0.0, 0.0]] * 2)

    assert results == [(100.0, 100.0, 100.0)] * 3


def test_each_degree_threshold_controls_direction_matches(capfd, calculate_metrics):
    directions = [
        [math.cos(math.radians(angle)), math.sin(math.radians(angle)), 0.0]
        for angle in [7.5, 15.0]
    ]

    results = measure(capfd, calculate_metrics, directions)

    assert results == [(0.0, 0.0, 0.0), (50.0, 50.0, 50.0), (100.0, 100.0, 100.0)]


def test_position_threshold_still_limits_matches(capfd, calculate_metrics):
    results = measure(
        capfd,
        calculate_metrics,
        [[1.0, 0.0, 0.0]] * 2,
        positions=[[0.0, 0.001, 0.0], [0.01, 0.001, 0.0]],
        distances=[0.0005, 0.002, 0.002],
    )

    assert results == [(0.0, 0.0, 0.0), (100.0, 100.0, 100.0), (100.0, 100.0, 100.0)]
