import numpy as np


def get_plucker_line_coordinates(
    in_vec1: np.ndarray, in_vec2: np.ndarray  # vector [4,]  # vector [4,]
) -> np.ndarray:
    assert in_vec1.shape[0] == 4
    assert in_vec2.shape[0] == 4

    out = np.zeros(6, dtype=np.float32)
    # first 3 dimensions of out will contain direction values
    out[0] = in_vec1[3] * in_vec2[0] - in_vec2[3] * in_vec1[0]
    out[1] = in_vec1[3] * in_vec2[1] - in_vec2[3] * in_vec1[1]
    out[2] = in_vec1[3] * in_vec2[2] - in_vec2[3] * in_vec1[2]
    # second 3 dimensions of out will contain momentum values
    out[3] = in_vec1[1] * in_vec2[2] - in_vec2[1] * in_vec1[2]
    out[4] = in_vec1[2] * in_vec2[0] - in_vec2[2] * in_vec1[0]
    out[5] = in_vec1[0] * in_vec2[1] - in_vec2[0] * in_vec1[1]

    return out
