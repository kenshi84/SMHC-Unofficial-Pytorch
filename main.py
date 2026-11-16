import argparse
import os
import random
import sys
import uuid
from pathlib import Path

import numpy as np
import torch

from datasets.colmap import is_colmap_dataset
from datasets.dataset import Dataset
from src.strand_accurate_reconstruction import SAR
from utils.metrics_utils import compute_quantitative_measures_pc_comparison
from utils.misc_utils import load_cfg, save_cfg


def create_result_folders(args):

    def create_subfolder(root, folder, desc):
        fol = os.path.join(root, folder)
        if not os.path.exists(fol):
            print(f"{desc} will be saved at: {fol}")
        os.makedirs(fol, exist_ok=True)

    if not args.output_path:
        input_root = Path(__file__).resolve().parent / "input"
        try:
            case_path = (
                Path(args.dataset_path).resolve().relative_to(input_root.resolve())
            )
        except ValueError:
            case_path = (os.getenv("OAR_JOB_ID") or str(uuid.uuid4()))[:10]
        args.output_path = os.path.join("./output", str(case_path))

    # Set up output folder
    os.makedirs(args.output_path, exist_ok=True)
    args.work_path = os.path.join(args.output_path, "work")
    os.makedirs(args.work_path, exist_ok=True)

    # setup other necessary subfolders
    create_subfolder(args.work_path, args.ori2d_maps_dir, "2D orientation maps")
    create_subfolder(
        args.work_path, args.conf_maps_dir, "2D orientation confidence maps"
    )
    if not args.no_lpmvs:
        create_subfolder(
            args.work_path,
            args.estimates_dir,
            "positions and directions estimates from lpmvs",
        )
    if not args.no_lpmvs or not args.no_filtering:
        create_subfolder(args.work_path, args.int_maps_dir, "intermediate maps")
    if not args.no_filtering:
        create_subfolder(args.work_path, args.filtered_dir, "filtered point cloud")
    if not args.no_fusion:
        create_subfolder(args.work_path, args.fused_dir, "results of the line fusion")
    if not args.no_strand_segment_generation:
        create_subfolder(
            args.work_path, args.strand_segments_dir, "generated strand segments"
        )
    if not args.no_multiview_hair_growing:
        create_subfolder(
            args.output_path,
            args.multiview_growing_dir,
            "results of the multivew hair growing step",
        )

    if args.initial_depth_maps != "":
        create_subfolder(
            args.work_path,
            args.initial_depth_maps,
            "inferred depth and direction maps from previous runs of smhc",
        )


def main(args):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    create_result_folders(args)

    dataset = Dataset(args)
    # Include the resolved COLMAP depth range and unit conversion in the run config.
    save_cfg(path=f"{args.work_path}/cfg.yaml", args=args)
    sar = SAR(dataset, args)

    # running lpmvs
    if not args.no_lpmvs:
        sar.run_lpmvs()

    # running line filtering
    if not args.no_filtering:
        sar.run_line_filtering()

    # running line fusion
    if not args.no_fusion:
        sar.run_line_fusion()

    # running strand segment generation
    if not args.no_strand_segment_generation:
        sar.generate_strand_segments()

    # running multi-view hair growing
    if not args.no_multiview_hair_growing:
        sar.run_multiview_hair_growing()

    sar.cleanup()

    # computing quantitative metircs
    if args.calc_quantitative_measures:
        compute_quantitative_measures_pc_comparison(args)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser = load_cfg(cfg_path="cfgs/default.yaml", parser=parser)

    parser.add_argument("--dataset_path", default="", type=str)
    parser.add_argument("--output_path", default="", type=str)

    parser.add_argument("--hair_masks", default="hair", type=str)
    parser.add_argument("--initial_depth_maps", default="", type=str)
    parser.add_argument("--use_hair_masks", action="store_true", default=False)
    parser.add_argument("--no_lpmvs", action="store_true", default=False)
    parser.add_argument("--no_filtering", action="store_true", default=False)
    parser.add_argument("--no_fusion", action="store_true", default=False)
    parser.add_argument(
        "--no_strand_segment_generation", action="store_true", default=False
    )
    parser.add_argument(
        "--no_multiview_hair_growing", action="store_true", default=False
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--calc_quantitative_measures", action="store_true", default=False
    )
    parser.add_argument(
        "--distance_evaluation_thresholds",
        nargs="+",
        type=int,
        default=[0.0005, 0.001, 0.002],
    )
    parser.add_argument(
        "--direction_evaluation_thresholds",
        nargs="+",
        type=int,
        default=[5.0, 10.0, 20.0],
    )

    depth_defaults = {
        key: parser.get_default(key)
        for key in ("lpmvs_minimum_depth", "lpmvs_maximum_depth")
    }
    parser.set_defaults(**dict.fromkeys(depth_defaults))
    args = parser.parse_args(argv)
    if not is_colmap_dataset(args.dataset_path):
        for key, value in depth_defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
    return args


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
