#!/usr/bin/env python3
"""Export SMHC's concatenated points/strand lengths to hairutil .data and .ply.

Usage: python scripts/export_hair.py output/real/20260415/iwata/multiview_hairgrowing
Coordinates and strand order are preserved; COLMAP reconstruction outputs use mm.
"""

import argparse
import fcntl
import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

INPUT_FILES = ("final_positions.npy", "final_points_num_per_strand.npy")


def timestamp():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds")


def fingerprint(directory):
    return {
        name: {
            "size": (directory / name).stat().st_size,
            "mtime_ns": (directory / name).stat().st_mtime_ns,
        }
        for name in INPUT_FILES
    }


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def export(directory, hairutil="hairutil"):
    directory = Path(directory).resolve()
    executable = shutil.which(str(hairutil))
    if executable is None:
        raise FileNotFoundError(f"hairutil executable not found: {hairutil}")
    with (directory / ".hairutil_export.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sources = fingerprint(directory)
        metadata_path = directory / "hairutil_export.json"
        if metadata_path.is_file():
            previous = json.loads(metadata_path.read_text())
            if (
                previous.get("status") == "SUCCEEDED"
                and previous.get("sources") == sources
            ):
                if all(
                    (directory / name).is_file()
                    and (directory / name).stat().st_size == details["bytes"]
                    for name, details in previous["outputs"].items()
                ):
                    return previous

        positions = np.load(
            directory / INPUT_FILES[0], mmap_mode="r", allow_pickle=False
        )
        counts = np.load(directory / INPUT_FILES[1], mmap_mode="r", allow_pickle=False)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("final_positions.npy must have shape (N, 3)")
        if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError(
                "final_points_num_per_strand.npy must be a 1D integer array"
            )
        if (
            len(counts) == 0
            or np.any(counts <= 0)
            or int(counts.sum(dtype=np.int64)) != len(positions)
        ):
            raise ValueError("Positive strand lengths must sum to the number of points")
        if len(counts) >= 2**32 or np.any(counts > 65536):
            raise ValueError("Strand counts exceed hairutil's .data/.ply format limits")
        points = np.asarray(positions, dtype="<f4")
        if not np.isfinite(points).all():
            raise ValueError("Positions contain nonfinite float32 coordinates")

        with tempfile.TemporaryDirectory(
            prefix=".hairutil-export-", dir=directory
        ) as temp:
            staging = Path(temp)
            data = staging / "final.data"
            with data.open("wb") as stream:
                stream.write(struct.pack("<I", len(counts)))
                offset = 0
                for count in counts:
                    count = int(count)
                    stream.write(struct.pack("<I", count))
                    stream.write(points[offset : offset + count].tobytes(order="C"))
                    offset += count

            commands = []
            with (directory / "hairutil_export.log").open("a", buffering=1) as log:
                log.write(
                    f"\n[{timestamp()}] Exporting {len(counts)} strands / {len(points)} points\n"
                )

                def run(*args, capture=False):
                    command = [executable, *map(str, args)]
                    commands.append(command)
                    result = subprocess.run(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    log.write(result.stdout)
                    if result.returncode:
                        raise RuntimeError(
                            f"hairutil failed ({result.returncode}): {result.stdout[-2000:]}"
                        )
                    return json.loads(result.stdout) if capture else None

                run("convert", "-i", data, "-o", "ply", "--no-autofix")
                ply = staging / "final.ply"
                info = run("info", "-i", ply, "--no-autofix", "-j", capture=True)
                if info["input"]["num_strands"] != len(counts) or info["input"][
                    "num_points"
                ] != len(points):
                    raise ValueError(
                        "hairutil's PLY strand/point counts differ from the NPY input"
                    )

                # Validate all coordinates and strand boundaries via hairutil's PLY reader.
                roundtrip = staging / "roundtrip"
                roundtrip.mkdir()
                run("convert", "-i", ply, "-o", "data", "-d", roundtrip, "--no-autofix")
                data_hash = digest(data)
                if digest(roundtrip / "final.data") != data_hash:
                    raise ValueError(
                        "PLY round trip changed coordinates or strand boundaries"
                    )
                if fingerprint(directory) != sources:
                    raise RuntimeError(
                        "NPY inputs changed during export; retry after reconstruction completes"
                    )

                outputs = {
                    "final.data": {"bytes": data.stat().st_size, "sha256": data_hash},
                    "final.ply": {"bytes": ply.stat().st_size, "sha256": digest(ply)},
                }
                metadata = {
                    "status": "SUCCEEDED",
                    "finished_at": timestamp(),
                    "sources": sources,
                    "source_sha256": {
                        name: digest(directory / name) for name in INPUT_FILES
                    },
                    "strands": len(counts),
                    "points": len(points),
                    "coordinate_transform": "none",
                    "strand_order": "preserved",
                    "hairutil_version": info["version"],
                    "hairutil_autofix": False,
                    "validation": "Exact .data -> hairutil .ply -> .data SHA-256 match",
                    "commands": commands,
                    "outputs": outputs,
                }
                for name in outputs:
                    (staging / name).replace(directory / name)
                temporary = staging / "metadata.json"
                temporary.write_text(json.dumps(metadata, indent=2) + "\n")
                temporary.replace(metadata_path)
                log.write(
                    f"[{timestamp()}] Saved final.data and final.ply; exact round trip verified\n"
                )
                return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--hairutil", default="hairutil")
    args = parser.parse_args()
    for directory in args.directories:
        result = export(directory, args.hairutil)
        print(
            f"{directory}: {result['strands']} strands / {result['points']} points -> final.data, final.ply",
            flush=True,
        )


if __name__ == "__main__":
    main()
