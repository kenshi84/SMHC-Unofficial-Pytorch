import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def forward_step(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("The CUDA toolkit is required to compile the native Euler step")

    csrc = Path(__file__).resolve().parents[1] / "src/smhc/cuda/csrc"
    build_dir = tmp_path_factory.mktemp("strand_generation")
    source = build_dir / "step.cu"
    executable = build_dir / "step"
    # Compile the actual implementation; this CPU step needs no CUDA device.
    source.write_text("""
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include "strand_segment_generation.cu"

int main(int argc, char** argv)
{
    const float distance = std::strtof(argv[1], nullptr);
    const float sign = std::strtof(argv[2], nullptr);
    const float3 origin = {0.01f, -0.005f, 0.003f};
    float3 direction = {sign, 0.f, 0.f};
    const float3 target = {origin.x + sign * distance, origin.y, origin.z};
    solution::pc_structs::detailed point(direction, target);
    float3 position = origin;
    float3 average_position = {0.f, 0.f, 0.f};
    float3 average_direction = {0.f, 0.f, 0.f};
    int neighbors = 0;
    solution::strand_segment_generation::forward_euler_step(
        &point, 1, origin, position, direction, average_position,
        average_direction, neighbors, 0.0002f, 0.002f, 0.00002f, 0.52f);
    std::cout << std::setprecision(9) << neighbors << ' '
              << (position.x - origin.x) * sign << ' '
              << (average_position.x - origin.x) * sign << '\\n';
}
""")
    build = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-O2",
            "--expt-relaxed-constexpr",
            "-I",
            str(csrc),
            "-I",
            str(csrc / "third_party/cudaKDTree"),
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr

    def step(distance, sign=1.0):
        result = subprocess.run(
            [str(executable), str(distance), str(sign)],
            capture_output=True,
            text=True,
            check=True,
        )
        return np.fromstring(result.stdout, sep=" ")

    return step


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["forward", "backward"])
@pytest.mark.parametrize("distance", [0.0002, 0.0004, 0.0018])
def test_euler_search_finds_points_at_uniform_step_intervals(
    forward_step, distance, sign
):
    neighbors, search_distance, advance = forward_step(distance, sign)

    assert neighbors == 1
    assert search_distance == pytest.approx(distance, abs=1e-8)
    # A gap search still advances the strand by only one Euler step.
    assert advance == pytest.approx(0.0002, abs=1e-8)


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["forward", "backward"])
def test_euler_search_does_not_reach_beyond_the_distance_limit(forward_step, sign):
    neighbors, search_distance, _ = forward_step(0.003, sign)

    assert neighbors == 0
    assert 0 < search_distance <= 0.002 + 1e-8


def test_separated_point_clouds_do_not_get_joined_across_a_long_gap():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Strand generation builds its search tree on CUDA")
    from smhc import StrandSegmentsGenerationHyperParametrs, generate_strand_segments

    params = StrandSegmentsGenerationHyperParametrs(
        SimpleNamespace(
            strand_generation_step_size=0.0002,
            strand_generation_length_threshold=0.002,
            strand_generation_neighboring_radius=0.00004,
            strand_generation_angle_difference_threshold_rad=0.52,
        )
    )
    x = torch.cat([torch.arange(21) * 0.0002, 0.007 + torch.arange(21) * 0.0002])
    positions = torch.zeros((len(x), 4))
    positions[:, 0] = x
    directions = torch.zeros_like(positions)
    directions[:, 0] = 1.0

    segments, _, counts = generate_strand_segments(params, positions, directions)

    assert len(counts) == 2
    ranges = []
    for strand in torch.split(segments, counts.int().tolist()):
        ranges.append((strand[:, 0].min().item(), strand[:, 0].max().item()))
    ranges.sort()
    assert ranges[0][1] <= 0.004 + 1e-7
    assert ranges[1][0] >= 0.007 - 1e-7
