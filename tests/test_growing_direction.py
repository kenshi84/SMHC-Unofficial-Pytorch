"""Exercise the production Eigen solver without requiring a CUDA device."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def solve_direction(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the Eigen solver tests")

    csrc = Path(__file__).resolve().parents[1] / "src/smhc/cuda/csrc"
    build_dir = tmp_path_factory.mktemp("growing_direction")
    source = build_dir / "solver.cpp"
    executable = build_dir / "solver"
    source.write_text("""
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include "growing_direction.h"

int main()
{
    int rows;
    std::cin >> rows;
    Eigen::MatrixX3d constraints(rows, 3);
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < 3; ++col)
        {
            std::string value;
            std::cin >> value;
            constraints(row, col) = std::strtod(value.c_str(), nullptr);
        }
    const auto direction = solution::multiview_hair_growing::solve_growing_direction(constraints);
    std::cout << std::setprecision(17) << direction.transpose() << '\\n';
}
""")
    build = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(csrc),
            "-I",
            str(csrc / "third_party/eigen"),
            str(source),
            "-o",
            str(executable),
        ],
        text=True,
        capture_output=True,
    )
    assert build.returncode == 0, build.stderr

    def solve(constraints):
        constraints = np.asarray(constraints, dtype=np.float64).reshape(-1, 3)
        result = subprocess.run(
            [str(executable)],
            input=f"{len(constraints)}\n"
            + " ".join(map(str, constraints.ravel().tolist())),
            text=True,
            capture_output=True,
            check=True,
        )
        direction = np.fromstring(result.stdout, sep=" ")
        assert direction.shape == (3,)
        assert np.isfinite(direction).all()
        return direction

    return solve


def assert_same_line(direction, expected, tolerance=1e-10):
    expected = np.asarray(expected, dtype=np.float64)
    expected /= np.linalg.norm(expected)
    assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-12)
    # Hair directions are unoriented: both signs represent the same line.
    assert abs(direction @ expected) > 1.0 - tolerance


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
def test_exact_plane_intersection_is_preserved_across_scales(solve_direction, scale):
    constraints = np.array([[2, -1, 0], [3, 0, -1], [5, -1, -1]]) * scale

    direction = solve_direction(constraints)

    assert_same_line(direction, [1, 2, 3])


def test_nearly_consistent_constraints_return_a_finite_unit_direction(solve_direction):
    constraints = [[1, 0, 1e-6], [0, 1, -1e-6], [1, 1, 0], [1, -1, 1e-6]]

    assert_same_line(solve_direction(constraints), [0, 0, 1])


def test_zero_residual_rows_do_not_break_reweighting(solve_direction):
    # Full rank initially, with two exact zero residuals along the best direction.
    constraints = np.diag([1.0, 2.0, 0.1])

    assert_same_line(solve_direction(constraints), [0, 0, 1])


def test_reweighting_reduces_an_outliers_effect(solve_direction):
    constraints = np.array(
        [[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, -1, 0], [1, 0, 1]],
        dtype=np.float64,
    )
    _, _, vectors = np.linalg.svd(constraints)
    initial_error = np.linalg.norm(vectors[-1, :2])

    direction = solve_direction(constraints)

    assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.norm(direction[:2]) < initial_error / 10.0


@pytest.mark.parametrize(
    "constraints",
    [
        [],
        [[1, 0, 0]],
        np.zeros((3, 3)),
        [[1, 0, 0], [2, 0, 0], [3, 0, 0]],
        np.eye(3),
    ],
    ids=["empty", "one-plane", "zero", "parallel-planes", "ambiguous-minimum"],
)
def test_underdetermined_direction_is_rejected(solve_direction, constraints):
    np.testing.assert_array_equal(solve_direction(constraints), np.zeros(3))


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_nonfinite_constraints_are_rejected(solve_direction, invalid_value):
    constraints = [[1, 0, 0], [0, 1, 0], [invalid_value, 0, 0]]

    np.testing.assert_array_equal(solve_direction(constraints), np.zeros(3))
