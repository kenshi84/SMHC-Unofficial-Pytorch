import os

import numpy as np
import torch
from rich.console import Console

from smhc import calculate_hair_metrics


def compute_quantitative_measures_pc_comparison(args):
    """
    Evaluating the reconstructions quantitatively by comparing gt and estimated point cloud,
    considering both estimated positions and their accompanied directions.
    """
    print("Calculating quantitative measures...")

    # loading the gt point cloud positions and directions
    gt_point_cloud_path = os.path.join(args.dataset_path, "surface_pcs.npy")
    gt_tangents_path = os.path.join(args.dataset_path, "surface_ts.npy")

    if not os.path.exists(gt_point_cloud_path):
        Console().print(
            "[yellow]info: was not able to find gt surface point cloud. Aborting quantitative measurment...[/yellow]"
        )
        return

    if not os.path.exists(gt_tangents_path):
        Console().print(
            "[yellow]info: was not able to find gt surface tangents. Aborting quantitative measurment...[/yellow]"
        )
        return

    with open(gt_point_cloud_path, "rb") as f:
        gt_positions_np = np.load(f)

    with open(gt_tangents_path, "rb") as f:
        gt_directions_np = np.load(f)

    # loading the estimated point cloud positions and directions
    final_estimated_positions_path = os.path.join(
        args.output_path, f"{args.multiview_growing_dir}/final_positions.npy"
    )
    final_estimated_directions_path = os.path.join(
        args.output_path, f"{args.multiview_growing_dir}/final_directions.npy"
    )

    if not os.path.exists(final_estimated_positions_path):
        Console().print(
            "[yellow]info: was not able to find the estimated point cloud. Aborting quantitative measurment...[/yellow]"
        )
        return

    if not os.path.exists(final_estimated_directions_path):
        Console().print(
            "[yellow]info: was not able to find the estimated directions. Aborting quantitative measurment...[/yellow]"
        )
        return

    with open(final_estimated_positions_path, "rb") as f:
        final_estimated_positions_np = np.load(f)

    with open(final_estimated_directions_path, "rb") as f:
        final_estimated_directions_np = np.load(f)

    # Common COLMAP inputs and final outputs are in mm; thresholds remain in m.
    gt_positions_np = gt_positions_np * args.input_unit_scale
    final_estimated_positions_np = final_estimated_positions_np * args.input_unit_scale

    gt_positions_torch = torch.from_numpy(gt_positions_np).to(torch.float32)
    gt_directions_torch = torch.from_numpy(gt_directions_np).to(torch.float32)

    final_estimated_positions_torch = torch.from_numpy(final_estimated_positions_np).to(
        torch.float32
    )
    final_estimated_directions_torch = torch.from_numpy(
        final_estimated_directions_np
    ).to(torch.float32)

    pos_thresholds = torch.tensor(
        args.distance_evaluation_thresholds, dtype=torch.float32
    )
    dir_thresholds = torch.tensor(
        args.direction_evaluation_thresholds, dtype=torch.float32
    )

    calculate_hair_metrics(
        pos_thresholds,
        dir_thresholds,
        gt_positions_torch,
        gt_directions_torch,
        final_estimated_positions_torch,
        final_estimated_directions_torch,
    )
