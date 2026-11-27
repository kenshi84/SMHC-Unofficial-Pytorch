import json
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils.camera_utils import Camera
from utils.gabor import GaborFilterBank
from utils.graphics_utils import get_plucker_line_coordinates


class Dataset:
    """dataset parser."""

    def __init__(
        self,
        args,
        calibration_file="calibration.json",
        imgs_folder="image",
        imgs_name_trailing_zeros=6,
        imgs_name_prefix="img_",
        extension=".png",
        resolution=1,
        mask_images_using_alpha=False,
        pick_every_nth_camera=-1,
    ) -> None:

        if pick_every_nth_camera > 0:
            print(
                f"Only picking every {pick_every_nth_camera}-th of the cameras for further processing!"
            )

        with open(os.path.join(args.dataset_path, calibration_file)) as json_file:
            calib_data = json.load(json_file)

        self.views = []
        self.total_views_num = len(calib_data["cameras"])
        self.ori_calculator = GaborFilterBank(args)
        checked_views = 1
        progress_bar = tqdm(
            range(checked_views, self.total_views_num),
            desc="Loading multi-view dataset",
        )

        for i, c in enumerate(calib_data["cameras"]):

            if pick_every_nth_camera > 0:
                if i % pick_every_nth_camera != 0:
                    continue

            u_id = int(c["camera_id"].replace("C", ""))

            height = c["intrinsics"]["resolution"][0]
            width = c["intrinsics"]["resolution"][1]

            width = round(width / resolution)
            height = round(height / resolution)

            extrinsics_cv = np.array(
                c["extrinsics"]["view_matrix"], dtype=np.float32
            ).reshape(4, 4)
            intrinsics_cv = np.array(
                c["intrinsics"]["camera_matrix"], dtype=np.float32
            ).reshape(3, 3)
            intrinsics_cv[0, 0] /= resolution
            intrinsics_cv[0, 2] /= resolution
            intrinsics_cv[1, 1] /= resolution
            intrinsics_cv[1, 2] /= resolution

            extrinsics_cv_inv = np.linalg.inv(extrinsics_cv)

            eye_mat_34 = np.zeros((3, 4), dtype=np.float32)
            eye_mat_34[0, 0] = 1.0
            eye_mat_34[1, 1] = 1.0
            eye_mat_34[2, 2] = 1.0

            P = intrinsics_cv @ eye_mat_34 @ extrinsics_cv  # 3x4 matrix
            P_col3 = P[:, 3].copy()
            M = P[:, :3].copy()
            M_inv = np.linalg.inv(M)
            K = intrinsics_cv.copy()
            K_inv = np.linalg.inv(K)
            R = extrinsics_cv[:3, :3].copy()
            T = extrinsics_cv[:3, 3].copy()
            R_inv = extrinsics_cv_inv[:3, :3].copy()
            C = extrinsics_cv_inv[:3, 3].copy()
            fx = float(intrinsics_cv[0, 0])
            fy = float(intrinsics_cv[1, 1])

            line_projection_matrix = np.zeros((3, 6), dtype=np.float32)
            line_projection_matrix[0, :] = np.flip(
                get_plucker_line_coordinates(P[1, :], P[2, :])
            )
            line_projection_matrix[1, :] = np.flip(
                get_plucker_line_coordinates(P[2, :], P[0, :])
            )
            line_projection_matrix[2, :] = np.flip(
                get_plucker_line_coordinates(P[0, :], P[1, :])
            )

            if imgs_name_trailing_zeros == 5:
                image_path = os.path.join(
                    args.dataset_path,
                    f"{imgs_folder}/{imgs_name_prefix}{u_id:05d}{extension}",
                )
                depth_arr_path = os.path.join(
                    args.dataset_path, f"depth/depth_{u_id:05d}.npy"
                )
                hair_mask_path = os.path.join(
                    args.dataset_path, args.hair_masks, f'{imgs_name_prefix}{u_id:05d}.png'
                )
                body_mask_path = os.path.join(
                    args.dataset_path, f"body/{imgs_name_prefix}{u_id:05d}.png"
                )
                image_name = f"{imgs_name_prefix}{i:05d}"
            elif imgs_name_trailing_zeros == 6:
                image_path = os.path.join(
                    args.dataset_path,
                    f"{imgs_folder}/{imgs_name_prefix}{u_id:06d}{extension}",
                )
                depth_arr_path = os.path.join(
                    args.dataset_path, f"depth/depth_{u_id:06d}.npy"
                )
                hair_mask_path = os.path.join(
                    args.dataset_path, args.hair_masks, f'{imgs_name_prefix}{u_id:06d}.png'
                )
                body_mask_path = os.path.join(
                    args.dataset_path, f"body/{imgs_name_prefix}{u_id:06d}.png"
                )
                image_name = f"{imgs_name_prefix}{i:06d}"
            elif imgs_name_trailing_zeros == 4:
                image_path = os.path.join(
                    args.dataset_path,
                    f"{imgs_folder}/{imgs_name_prefix}{u_id:04d}{extension}",
                )
                depth_arr_path = os.path.join(
                    args.dataset_path, f"depth/depth_{u_id:04d}.npy"
                )
                hair_mask_path = os.path.join(
                    args.dataset_path, args.hair_masks, f'{imgs_name_prefix}{u_id:04d}.png'
                )
                body_mask_path = os.path.join(
                    args.dataset_path, f"body/{imgs_name_prefix}{u_id:04d}.png"
                )
                image_name = f"{imgs_name_prefix}{i:04d}"
            else:
                raise ValueError(
                    "Please provide the images with 4,5 or 6 trailing zeros in their names!"
                )

            if not os.path.exists(image_path):
                print(f"Could not find input image, {image_path}.")
                checked_views += 1
                continue

            gt_image_np = np.array(
                Image.open(image_path).convert("L"), dtype=np.float32
            ) * (1.0 / 255.0)

            if mask_images_using_alpha:
                if os.path.exists(body_mask_path):
                    alpha_mask_np = np.array(
                        Image.open(body_mask_path).convert("L"), dtype=np.float32
                    ) * (1.0 / 255.0)
                    alpha_mask_np[alpha_mask_np > 0.01] = 1.0
                    alpha_mask_np[alpha_mask_np < 0.01] = 0.0

                elif os.path.exists(depth_arr_path):
                    # we are using rendered depth values here to specify the alpha mask of the scene
                    with open(depth_arr_path, "rb") as f:
                        depth_arr = np.load(f)
                    depth_arr = np.flip(depth_arr, 0).copy()  # flip the vertical axis
                    alpha_mask_np = np.zeros_like(gt_image_np, dtype=np.float32)
                    alpha_mask_np[depth_arr < 10.0] = 1.0

                else:
                    print(
                        f"Could not find input alpha mask, {body_mask_path} or any depth data at {depth_arr_path} to derive alpha from."
                    )
                    alpha_mask_np = np.ones_like(gt_image_np, dtype=np.float32)

                gt_image_np = gt_image_np * alpha_mask_np

            if os.path.exists(hair_mask_path):
                hair_mask_np = np.array(
                    Image.open(hair_mask_path).convert("L"), dtype=np.float32
                ) * (1.0 / 255.0)
                hair_mask_np[hair_mask_np > 0.01] = 1.0
                hair_mask_np[hair_mask_np < 0.01] = 0.0
            else:
                print(f"Could not find input hair mask, {hair_mask_path}.")
                hair_mask_np = np.ones_like(gt_image_np, dtype=np.float32)

            int_map_file = os.path.join(args.output_path, f"int/int_map_{u_id:06d}.npy")
            if os.path.exists(int_map_file):
                with open(int_map_file, "rb") as f:
                    int_map_np = np.load(f)
            else:
                int_map_np = np.ones((height, width, 3), dtype=np.float32) * -1.0

            down_sampled_res = (width, height)
            if resolution == 1:
                gt_image = torch.from_numpy(gt_image_np).float()
                hair_mask = torch.from_numpy(hair_mask_np).float()
                int_map = torch.from_numpy(int_map_np).float().permute(2, 0, 1)
            else:
                orig_gt_image = Image.fromarray(gt_image_np)
                orig_hair_mask = Image.fromarray(hair_mask_np)

                gt_image = torch.from_numpy(
                    np.array(orig_gt_image.resize(down_sampled_res))
                )
                hair_mask = torch.from_numpy(
                    np.array(orig_hair_mask.resize(down_sampled_res))
                )

                int_map = []
                for j in range(3):
                    int_map.append(
                        torch.from_numpy(
                            np.array(
                                Image.fromarray(int_map_np[:, :, j]).resize(
                                    down_sampled_res
                                )
                            )
                        )
                    )
                int_map = torch.cat(int_map, dim=0)

            # calculate the 2d orientaion and confidence maps
            # check if 2d orienation map is already calculated, if this is the case load it
            # otherwise calculate the 2d orientation and confidence maps and save them in the specified directories
            ori2d_map_file = os.path.join(
                args.output_path, f"{args.ori2d_maps_dir}/img_{u_id:06d}.npy"
            )
            conf_map_file = os.path.join(
                args.output_path, f"{args.conf_maps_dir}/img_{u_id:06d}.npy"
            )

            if os.path.isfile(ori2d_map_file):
                with open(ori2d_map_file, "rb") as f:
                    ori2d_map_np = np.load(f)
                ori2d_map = torch.from_numpy(ori2d_map_np).float()

                with open(conf_map_file, "rb") as f:
                    conf_map_np = np.load(f)
                conf_map = torch.from_numpy(conf_map_np).float()

            else:
                # calculate the 2d orientation and confidence maps, also save them for future use
                ori2d, conf = self.ori_calculator.acquire_orientation_maps(gt_image)

                ori2d_map = ori2d.cpu()
                with open(ori2d_map_file, "wb") as f:
                    np.save(f, ori2d_map.clone().numpy())

                conf_map = conf.cpu()
                with open(conf_map_file, "wb") as f:
                    np.save(f, conf_map.clone().numpy())

            self.views.append(
                Camera(
                    u_id=u_id,
                    id=i,
                    P=P,
                    P_col3=P_col3,
                    M_inv=M_inv,
                    K=K,
                    K_inv=K_inv,
                    R=R,
                    T=T,
                    R_inv=R_inv,
                    C=C,
                    Line_proj_mat=line_projection_matrix,
                    Fx=fx,
                    Fy=fy,
                    image=gt_image,
                    gt_hair_mask=hair_mask,
                    int_map=int_map,
                    ori2d_map=ori2d_map,
                    conf_map=conf_map,
                    image_name=image_name,
                )
            )
            progress_bar.update(1)
            checked_views += 1
            if checked_views == self.total_views_num:
                progress_bar.close()

        # check if no views were loaded successfully
        if len(self.views) == 0:
            raise ValueError("No input data could be read from the dataset")

        print(f"Total number of loaded views: {len(self.views)} ...")
