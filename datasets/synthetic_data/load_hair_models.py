import json
import os
from argparse import ArgumentParser

import numpy as np
import torch
import trimesh
import trimesh.exchange
from PIL import Image
from tqdm import tqdm

from smhc import load_dot_bin_model, load_dot_hair_model


def extract_surface(
    positions: np.ndarray,
    directions: np.ndarray,
    cam_transforms_path: str,
    output_model_folder: str,
    surface_depth: float,
):

    with open(cam_transforms_path) as json_file:
        calib_data = json.load(json_file)

    # Project and save the ground truth in the same world frame used for rendering.
    if "strands_to_world_rotation" in calib_data:
        rotation = np.asarray(calib_data["strands_to_world_rotation"], dtype=np.float32)
        positions = positions @ rotation.T
        directions = directions @ rotation.T

    strands_hom = np.append(
        positions, np.ones((positions.shape[0], 1), dtype=np.float32), axis=1
    )
    strands_on_surface_ind = np.zeros(positions.shape[0], dtype=np.int8)

    print(f"Extracting points with maximum {surface_depth} depth from hair surface")
    progress_bar = tqdm(
        range(len(calib_data["cameras"])), desc="Camera Views Processed"
    )
    for i, c in enumerate(calib_data["cameras"]):

        # loading depth maps
        depth_img = np.load(
            os.path.join(output_model_folder, f"depth/depth_{i:06d}.npy")
        )
        depth_img = np.flip(depth_img, axis=0)
        depth_img_flattended = np.abs(np.reshape(depth_img, -1))

        height = c["intrinsics"]["resolution"][0]
        width = c["intrinsics"]["resolution"][1]

        extrinsics_cv = np.array(
            c["extrinsics"]["view_matrix"], dtype=np.float32
        ).reshape(4, 4)
        intrinsics_cv = np.array(
            c["intrinsics"]["camera_matrix"], dtype=np.float32
        ).reshape(3, 3)
        # intrinsics_cv[0, 0] /= resolution
        # intrinsics_cv[0, 2] /= resolution
        # intrinsics_cv[1, 1] /= resolution
        # intrinsics_cv[1, 2] /= resolution

        eye_mat_34 = np.zeros((3, 4), dtype=np.float32)
        eye_mat_34[0, 0] = 1.0
        eye_mat_34[1, 1] = 1.0
        eye_mat_34[2, 2] = 1.0

        cam_transform = intrinsics_cv @ eye_mat_34 @ extrinsics_cv  # 3x4 matrix

        strands_clip = (cam_transform @ strands_hom.T).T
        strands_depth = strands_clip[:, 2].copy()
        strands_ndc = strands_clip / strands_clip[:, 2:]
        strands_pixels = strands_ndc[:, :2].copy()
        strands_pixels = np.flip(strands_pixels.astype(np.int32), axis=1)

        strands_pixels_flattened = strands_pixels[:, 0] * width + strands_pixels[:, 1]
        strands_pixels_depth = depth_img_flattended[strands_pixels_flattened]
        surface_indication = (
            np.abs(strands_depth - strands_pixels_depth) < surface_depth
        )
        strands_on_surface_ind[surface_indication] = 1

        progress_bar.set_postfix(
            {"points#": f"{np.sum(strands_on_surface_ind, axis=0):07d}"}
        )
        progress_bar.update(1)
        if i == len(calib_data["cameras"]) - 1:
            progress_bar.close()

    surface_points = positions[strands_on_surface_ind == 1]
    ps_save_file = os.path.join(output_model_folder, "surface_pcs.npy")
    with open(ps_save_file, "wb") as f:
        np.save(f, surface_points)
    print(f"saved extracted surface positions at: {ps_save_file}")

    # for debug
    surface_strands_pc = trimesh.Trimesh(surface_points)
    write = surface_strands_pc.export(
        os.path.join(output_model_folder, "surface_pcs.ply")
    )

    surface_dirs = directions[strands_on_surface_ind == 1]
    ts_save_file = os.path.join(output_model_folder, "surface_ts.npy")
    with open(ts_save_file, "wb") as f:
        np.save(f, surface_dirs)
    print(f"saved extracted surface directions at: {ts_save_file}")


def interpolate_hair_strands(hair_pc, points_num_per_strand, finest_distance_measure):
    ps_acc = 0
    strands_interpolated = []
    points_num_per_strand_interpolated = []
    strands_dirs_interpolated = []

    print(
        f"Interpolating hair strand points to ensure distance between points is bellow {finest_distance_measure} ..."
    )
    progress_bar = tqdm(range(points_num_per_strand.shape[0]), desc="Progress")

    for idx, ps in enumerate(points_num_per_strand):
        if ps < 2:
            print(f"encountered a strand with less than 2 points at index: {idx}")
            ps_acc += ps
            continue
        strand = hair_pc[ps_acc : ps_acc + ps, :].clone()

        # find the maximum distance between consequitive positions in the strand
        diff = (
            strand[1:, :] - strand[:-1, :]
        )  # difference between consequitive points in one strand
        diff_norm = torch.linalg.norm(diff, axis=1)
        # max_dist = torch.max(diff_norm)
        diff_sum = torch.sum(diff_norm)

        # if max_dist < finest_distance_measure:
        #     vertices_target_num = ps
        # else:
        #     vertices_target_num = torch.ceil(max_dist / finest_distance_measure).int().item() * ps

        vertices_target_num = (
            torch.ceil(diff_sum / finest_distance_measure).int().item()
        )

        strand_torch = strand.unsqueeze(0)
        strand_interpolated = torch.nn.functional.interpolate(
            strand_torch.permute(0, 2, 1),
            vertices_target_num,
            align_corners=True,
            mode="linear",
        ).permute(0, 2, 1)
        strand_interpolated = strand_interpolated.squeeze(0)

        dirs_interpolated = torch.zeros(strand_interpolated.shape, dtype=torch.float32)
        dirs_interpolated[:-1, :] = (
            strand_interpolated[1:, :] - strand_interpolated[:-1, :]
        )
        dirs_interpolated[-1] = dirs_interpolated[-2]

        strands_interpolated.append(strand_interpolated)
        strands_dirs_interpolated.append(dirs_interpolated)
        points_num_per_strand_interpolated.append(vertices_target_num)

        ps_acc += ps
        progress_bar.update(1)
        if idx == points_num_per_strand.shape[0] - 1:
            progress_bar.close()

    return (
        torch.cat(strands_interpolated, dim=0),
        torch.cat(strands_dirs_interpolated, dim=0),
        torch.tensor(points_num_per_strand_interpolated),
    )


def load_hair(args, model_type: str):
    if model_type == ".hair":
        positions_scale = 0.003  # this number does look strange, it is emprically set with help of blender
        positions, points_per_strand = load_dot_hair_model(
            args.input_path, positions_scale
        )

        # .hair models are 90 degrees rotated on the z axis, consider that here
        rot_mat = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32)
        positions = (rot_mat @ positions.T).T

    elif model_type == ".bin":
        positions_scale = 0.001  # CT2Hair stores strand positions in millimeters.
        positions, points_per_strand = load_dot_bin_model(
            args.input_path, positions_scale
        )

    elif model_type == ".ply":
        with open(args.input_path, "rb") as f:
            # strands_mesh = trimesh.load_mesh(args.input_path, file_type='ply')
            strands_mesh = trimesh.exchange.ply.load_ply(f)

        positions = torch.from_numpy(strands_mesh["vertices"])

        pps_file = os.path.join(
            os.path.dirname(args.input_path), args.input_points_per_strand_list
        )
        if os.path.exists(pps_file):
            with open(pps_file, "rb") as f:
                points_per_strand = np.load(f)
            points_per_strand = torch.from_numpy(points_per_strand).int()

        elif args.fixed_points_num_per_strand > 0:
            if (positions.shape[0] % args.fixed_points_num_per_strand) != 0:
                raise ValueError(
                    "Hair point cloud size is not divisible by the provided fixed number of points per strand!"
                )

            strands_num = positions.shape[0] // args.fixed_points_num_per_strand
            points_per_strand = (
                torch.ones(strands_num).int() * args.fixed_points_num_per_strand
            )

        else:
            raise ValueError(
                "If the requested hair model is in .ply format, and there exists no additional per strand points number file, presumed fixed number of points per strand must be provided!"
            )

    else:
        raise ValueError("Unknown hair model!")

    (
        interpolated_positions,
        interpolated_directions,
        interpolated_points_num_per_strand,
    ) = interpolate_hair_strands(
        positions, points_per_strand.to(torch.int32), args.finest_distance_measure
    )

    if args.save_hair_model:
        # save the interpolated strands as ply file plus the accompanied points per strand number
        interpolated_strands_pc = trimesh.Trimesh(interpolated_positions.numpy())
        i_s_save_file = os.path.join(
            args.output_model_folder, "strands_interpolated.ply"
        )
        _ = interpolated_strands_pc.export(i_s_save_file)
        print(f"saved interpolated strands at: {i_s_save_file}")

        pps_save_file = os.path.join(
            args.output_model_folder, "strands_interpolated_points_per_strand.npy"
        )
        with open(pps_save_file, "wb") as f:
            np.save(f, interpolated_points_num_per_strand.numpy())
        print(f"saved interpolated strands' points per strand list at: {pps_save_file}")

    return (
        interpolated_positions.numpy(),
        interpolated_directions.numpy(),
    )


if __name__ == "__main__":
    print(os.getcwd())
    parser = ArgumentParser(conflict_handler="resolve")

    parser.add_argument("--input_path", default="", type=str)
    parser.add_argument("--input_points_per_strand_list", default="", type=str)
    parser.add_argument("--cameras_calibration_path", default="", type=str)
    parser.add_argument("--finest_distance_measure", default=0.00099, type=float)
    parser.add_argument("--hair_surface_depth", default=0.0004, type=float)
    parser.add_argument("--fixed_points_num_per_strand", default=100, type=int)
    parser.add_argument("--save_hair_model", default=False, action="store_true")
    parser.add_argument("--output_model_folder", default="", type=str)
    parser.add_argument("--extract_hair_surface", default=False, action="store_true")

    args, _ = parser.parse_known_args()

    if not os.path.exists(args.input_path):
        raise ValueError("Could not find the specified input hair model!")

    if not os.path.exists(args.output_model_folder):
        os.makedirs(args.output_model_folder)

    if args.input_path.endswith(".hair"):
        print("a dot hair model is being loaded ...")
        strand_points_positions, strand_points_directions = load_hair(
            args, model_type=".hair"
        )

    elif args.input_path.endswith(".bin"):
        print("a dot bin model is being loaded ...")
        strand_points_positions, strand_points_directions = load_hair(
            args, model_type=".bin"
        )

    elif args.input_path.endswith(".ply"):
        print("a dot ply model is being loaded ...")
        strand_points_positions, strand_points_directions = load_hair(
            args, model_type=".ply"
        )

    else:
        raise ValueError(
            "This type of hair model is currently not supported! Supported models are .hair, .bin and .ply"
        )

    if args.extract_hair_surface:
        test_depth_img_path = os.path.join(
            args.output_model_folder, f"depth/depth_000000.npy"
        )
        if not os.path.exists(test_depth_img_path):
            raise ValueError(
                "Could not find the rendererd depth maps which are necessary to extract hair surface. Make sure to first render the hair data using blender and then extract the surface point cloud."
            )

        extract_surface(
            strand_points_positions,
            strand_points_directions,
            args.cameras_calibration_path,
            args.output_model_folder,
            args.hair_surface_depth,
        )

    print("Done loading the model")
