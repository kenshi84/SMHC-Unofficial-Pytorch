"""Read the common millimetre-scale COLMAP input without converting input files.

Binary layout and camera conventions: https://colmap.github.io/format.html
"""

import os
import struct
from pathlib import Path

import numpy as np
import trimesh


def is_colmap_dataset(dataset_path):
    return (Path(dataset_path) / "sparse/cameras.bin").is_file()


def _read(stream, fmt):
    fmt = "<" + fmt
    size = struct.calcsize(fmt)
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"Truncated COLMAP file: {stream.name}")
    return struct.unpack(fmt, data)


def load_colmap_calibration(dataset_path):
    """Return legacy-compatible camera records and a head-based depth range in metres.

    World axes are preserved. COLMAP already stores world-to-camera poses in the
    OpenCV camera convention used by SMHC. Only translations need mm -> m scaling.
    """
    root = Path(dataset_path)
    intrinsics = {}
    with (root / "sparse/cameras.bin").open("rb") as stream:
        for _ in range(_read(stream, "Q")[0]):
            camera_id, model, width, height = _read(stream, "iiQQ")
            if model == 0:  # SIMPLE_PINHOLE
                fx, cx, cy = _read(stream, "3d")
                fy = fx
            elif model == 1:  # PINHOLE
                fx, fy, cx, cy = _read(stream, "4d")
            else:
                raise ValueError(
                    f"COLMAP camera {camera_id} uses unsupported model {model}. "
                    "Undistort images and cameras to PINHOLE or SIMPLE_PINHOLE first."
                )
            intrinsics[camera_id] = {
                "resolution": [height, width],
                "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            }

    cameras = []
    with (root / "sparse/images.bin").open("rb") as stream:
        file_size = os.fstat(stream.fileno()).st_size
        for _ in range(_read(stream, "Q")[0]):
            record = _read(stream, "i7di")
            image_id, camera_id = record[0], record[-1]
            name = bytearray()
            while (char := _read(stream, "c")[0]) != b"\0":
                name.extend(char)
            # Feature observations are not used by the reconstruction pipeline.
            num_points = _read(stream, "Q")[0]
            if stream.seek(24 * num_points, os.SEEK_CUR) > file_size:
                raise ValueError(f"Truncated COLMAP file: {stream.name}")
            extrinsics = trimesh.transformations.quaternion_matrix(record[1:5])
            extrinsics[:3, 3] = np.asarray(record[5:8]) * 0.001
            cameras.append(
                {
                    "camera_id": str(image_id),
                    "image_name": name.decode("utf-8"),
                    "intrinsics": intrinsics[camera_id],
                    "extrinsics": {"view_matrix": extrinsics},
                }
            )
    cameras.sort(key=lambda camera: int(camera["camera_id"]))
    if not cameras:
        raise ValueError(f"No registered images in {root / 'sparse/images.bin'}")

    head = trimesh.load(root / "head.obj", force="mesh", process=False)
    vertices = np.asarray(head.vertices) * 0.001
    if vertices.size == 0 or not np.isfinite(vertices).all():
        raise ValueError(f"No valid head vertices in {root / 'head.obj'}")
    # Include 150 mm of space around the head for hair. Explicit CLI depth limits
    # override this estimate for hairstyles that need a larger search volume.
    minimum_depth, maximum_depth = np.inf, -np.inf
    for camera in cameras:
        extrinsics = camera["extrinsics"]["view_matrix"]
        depths = vertices @ extrinsics[2, :3] + extrinsics[2, 3]
        minimum_depth = min(minimum_depth, float(depths.min()) - 0.15)
        maximum_depth = max(maximum_depth, float(depths.max()) + 0.15)
    minimum_depth = max(0.001, minimum_depth)
    if maximum_depth <= minimum_depth:
        raise ValueError("The head mesh is behind the COLMAP cameras.")
    return {"cameras": cameras}, (minimum_depth, maximum_depth)
