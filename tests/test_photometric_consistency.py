"""Test the production correlation accumulator without requiring a CUDA device."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def correlation(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the photometric score tests")
    csrc = Path(__file__).resolve().parents[1] / "src/smhc/cuda/csrc"
    build_dir = tmp_path_factory.mktemp("photometric_consistency")
    source = build_dir / "correlation.cpp"
    executable = build_dir / "correlation"
    source.write_text("""
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include "photometric_consistency.h"

struct Color { float x, y, z, w; };

float read_float()
{
    std::string value;
    std::cin >> value;
    return std::strtof(value.c_str(), nullptr);
}

int main(int argc, char** argv)
{
    const bool use_zncc = std::atoi(argv[1]) != 0;
    const float variance = std::strtof(argv[2], nullptr);
    int count, channels;
    std::cin >> count >> channels;
    solution::lpmvs::PhotometricCorrelation<float> scalar;
    solution::lpmvs::PhotometricCorrelation<Color> color;
    for (int i = 0; i < count; ++i)
    {
        if (channels == 1)
        {
            const float reference = read_float();
            const float neighbor = read_float();
            scalar.add(reference, neighbor, use_zncc);
        }
        else
        {
            const Color reference = {read_float(), read_float(), read_float(), read_float()};
            const Color neighbor = {read_float(), read_float(), read_float(), read_float()};
            color.add(reference, neighbor, use_zncc);
        }
    }
    float result;
    const bool valid = channels == 1
        ? scalar.correlation(use_zncc, variance, result)
        : color.correlation(use_zncc, variance, result);
    std::cout << valid << ' ' << std::setprecision(9) << result << '\\n';
}
""")
    build = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(csrc),
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr

    def evaluate(reference, neighbor, variance=1e-6, method="zncc"):
        reference = np.asarray(reference, dtype=np.float32)
        neighbor = np.asarray(neighbor, dtype=np.float32)
        assert reference.shape == neighbor.shape
        channels = 1 if reference.ndim == 1 else reference.shape[1]
        assert channels in (1, 4)
        pairs = np.stack((reference, neighbor), axis=1)
        result = subprocess.run(
            [str(executable), str(int(method == "zncc")), str(variance)],
            input=f"{len(reference)} {channels}\n" + " ".join(map(str, pairs.ravel())),
            capture_output=True,
            text=True,
            check=True,
        )
        valid, value = map(float, result.stdout.split())
        assert np.isfinite(value)
        if valid:
            assert -1 <= value <= 1
        return bool(valid), value

    return evaluate


@pytest.mark.parametrize("amplitude", [0.01, 0.001, 0.0001])
def test_low_contrast_correlation_matches_centered_float64_reference(
    correlation, amplitude
):
    rng = np.random.default_rng(42)
    reference = (0.9 + amplitude * rng.normal(size=41)).astype(np.float32)
    neighbor = (0.8 + 0.6 * (reference - 0.9) + amplitude * rng.normal(size=41)).astype(
        np.float32
    )
    expected = np.corrcoef(reference.astype(np.float64), neighbor.astype(np.float64))[
        0, 1
    ]

    valid, value = correlation(reference, neighbor, variance=0)

    assert valid
    assert value == pytest.approx(expected, abs=2e-4)


@pytest.mark.parametrize(
    "reference,neighbor",
    [
        ([], []),
        ([0.5], [0.5]),
        ([0.5] * 4, [0.4, 0.6] * 2),
        ([0.4, 0.6] * 2, [0.5] * 4),
    ],
    ids=["empty", "one-pair", "constant-reference", "constant-neighbor"],
)
def test_insufficient_texture_is_rejected(correlation, reference, neighbor):
    valid, _ = correlation(reference, neighbor, variance=0)

    assert not valid


def test_variance_threshold_can_retain_faint_texture(correlation):
    reference = [0.4999, 0.5001] * 10

    assert not correlation(reference, reference)[0]
    valid, value = correlation(reference, reference, variance=1e-9)
    assert valid
    assert value == pytest.approx(1, abs=1e-6)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_samples_cannot_supply_a_match(correlation, invalid):
    assert not correlation([0.1, 0.2, 0.3], [0.1, invalid, 0.3])[0]


def test_rgb_uses_textured_channels_and_ignores_alpha(correlation):
    reference = [[x, 0.3, 0.4, float("nan")] for x in (0.49, 0.51, 0.49, 0.51)]
    neighbor = [[1 - x[0], 0.6, 0.7, 0.0] for x in reference]

    valid, value = correlation(reference, neighbor)

    assert valid
    assert value == pytest.approx(-1, abs=1e-6)


def test_rgb_without_textured_channels_is_rejected(correlation):
    assert not correlation([[0.3, 0.4, 0.5, 1.0]] * 4, [[0.6, 0.7, 0.8, 1.0]] * 4)[0]


def test_cosine_mode_preserves_the_previous_similarity(correlation):
    reference = np.array([0.49, 0.51, 0.49, 0.51])
    neighbor = 1 - reference
    expected = (
        reference
        @ neighbor
        / (np.linalg.norm(reference) * np.linalg.norm(neighbor) + 1e-8)
    )

    valid, value = correlation(reference, neighbor, method="cosine")

    assert valid
    assert value == pytest.approx(expected, abs=1e-6)
