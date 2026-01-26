import io
import json
import os
from argparse import ArgumentParser

import numpy as np
import pillow_avif
from PIL import Image
from tqdm import tqdm
from zipp import Path as ZipPath


def calc_view_vector(cam_extrinsics):
    cam_extrinsics_inv = np.linalg.inv(cam_extrinsics)
    camera_center_world_coord = cam_extrinsics_inv[:3, 3]
    center_pixel_depth_local_coord = np.array([0, 0, 1, 1], dtype=np.float32)
    vec = cam_extrinsics_inv @ center_pixel_depth_local_coord
    view_vector = vec[:3] - camera_center_world_coord
    return view_vector / np.linalg.norm(view_vector)


def isolate_views_around_central(views_calib_dict, central_view, acceptance_umbra):
    acceptance_umbra = (acceptance_umbra / 180.0) * np.pi  # ensure radian unit

    # get the central view vector
    central_view_id = -1
    for i, c in enumerate(views_calib_dict["cameras"]):
        u_id = int(c["camera_id"].replace("C", ""))
        if u_id == central_view:
            central_view_id = i
            cam_extrinsics = np.array(
                c["extrinsics"]["view_matrix"], dtype=np.float32
            ).reshape(4, 4)
            central_view_vec = calc_view_vector(cam_extrinsics)

    # ensure that the specified central view exists
    if central_view_id == -1:
        return views_calib_dict

    # loop through all provided views and select those in the acceptance umbra
    neighbors_list = {"cameras": []}
    for i, c in enumerate(views_calib_dict["cameras"]):
        u_id = int(c["camera_id"].replace("C", ""))
        cam_extrinsics = np.array(
            c["extrinsics"]["view_matrix"], dtype=np.float32
        ).reshape(4, 4)
        view_vec = calc_view_vector(cam_extrinsics)
        angle_diff = np.acos(np.dot(central_view_vec, view_vec))
        if angle_diff < acceptance_umbra:
            print(
                f"\t adding to the calibration file neighboring camera index {u_id} with angle difference {angle_diff} radians"
            )
            neighbors_list["cameras"].append(c)

    return neighbors_list


if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler="resolve")

    parser.add_argument("--input_path", default="", type=str)
    parser.add_argument("--output_path", default="", type=str)
    parser.add_argument("--frame_to_pick", default=34311, type=int)
    parser.add_argument("--central_view_to_select_around", default=401077, type=int)
    parser.add_argument(
        "--accepting_views_umbra_around_central_view", default=-1.0, type=float
    )

    args, _ = parser.parse_known_args()

    # making output directories
    os.makedirs(args.output_path, exist_ok=True)
    output_images_path = os.path.join(args.output_path, "image")
    os.makedirs(output_images_path, exist_ok=True)
    body_masks_path = os.path.join(args.output_path, "body")
    os.makedirs(body_masks_path, exist_ok=True)
    hair_masks_path = os.path.join(args.output_path, "hair")
    os.makedirs(hair_masks_path, exist_ok=True)

    calibration_path = os.path.join(args.input_path, "camera_calibration.json")

    with open(calibration_path) as json_file:
        calib_data = json.load(json_file)

    camera_calib_dict = {"cameras": []}
    checked_views = 1
    progress_bar = tqdm(
        range(checked_views, len(calib_data["KRT"])), desc="Loading multi-view dataset"
    )

    for i, c in enumerate(calib_data["KRT"]):
        cam_id = int(c["cameraId"])
        entry = {"camera_id": f"C{cam_id:06d}"}

        assert c["projectionModel"] == "Pinhole"

        extrinsics_cv = np.array(c["T"], dtype=np.float32).reshape(4, 4).T
        intrinsics_cv = np.array(c["K"], dtype=np.float32).reshape(3, 3).T
        # original data in comparison to the released ones have twice resolution
        # but the given intrinsics are according to the original data, so compensate
        # for the down sampling in the intrinsics here
        intrinsics_cv[0, 0] /= 2
        intrinsics_cv[0, 2] /= 2
        intrinsics_cv[1, 1] /= 2
        intrinsics_cv[1, 2] /= 2

        extrinsics_cv_inv = np.linalg.inv(extrinsics_cv)
        R_inv = extrinsics_cv_inv[:3, :3].copy()
        C = extrinsics_cv_inv[:3, 3].copy()

        # bring to metric space
        C_metric = C * 0.001
        new_ext_inv = np.eye(4, dtype=np.float32)
        new_ext_inv[:3, :3] = R_inv.copy()
        new_ext_inv[:3, 3] = C_metric.copy()

        new_ext = np.linalg.inv(new_ext_inv)

        entry["extrinsics"] = {
            "view_matrix": new_ext.reshape(-1).tolist(),
            "cam_position": C_metric.tolist(),
        }
        entry["intrinsics"] = {
            "camera_matrix": intrinsics_cv.reshape(-1).tolist(),
            "resolution": [2048, 1334],
        }
        camera_calib_dict["cameras"].append(entry)

        path = ZipPath(
            args.input_path + "/image/" + f"cam{cam_id}.zip",
            f"cam{cam_id}/{args.frame_to_pick:06d}.avif",
        )
        try:
            img_bytes = path.read_bytes()
        except:
            print(
                f"could not load image from camera {cam_id} at frame {args.frame_to_pick:06d}!"
            )
            checked_views += 1
            continue

        image = Image.open(io.BytesIO(img_bytes))

        image_save_path = os.path.join(output_images_path, f"img_{int(cam_id):06d}.png")
        image.save(image_save_path)

        masks_path = ZipPath(
            args.input_path + "/segmentation_parts/" + f"cam{cam_id}.zip",
            f"cam{cam_id}/{args.frame_to_pick:06d}.png",
        )
        masks_bytes = masks_path.read_bytes()
        masks = Image.open(io.BytesIO(masks_bytes))

        masks_np = np.array(masks, dtype=np.uint8)

        hair_mask_np = np.zeros_like(masks_np)
        body_mask_np = np.zeros_like(masks_np)

        hair_mask_np[masks_np == 1] = 255
        body_mask_np[masks_np > 0] = 255

        hair_mask = Image.fromarray(hair_mask_np)
        hair_mask.save(os.path.join(hair_masks_path, f"img_{int(cam_id):06d}.png"))

        body_mask = Image.fromarray(body_mask_np)
        body_mask.save(os.path.join(body_masks_path, f"img_{int(cam_id):06d}.png"))

        progress_bar.update(1)
        checked_views += 1
        if checked_views == len(calib_data["KRT"]):
            progress_bar.close()

    if (
        args.central_view_to_select_around > 0
        and args.accepting_views_umbra_around_central_view > 0
    ):
        camera_calib_dict = isolate_views_around_central(
            camera_calib_dict,
            args.central_view_to_select_around,
            args.accepting_views_umbra_around_central_view,
        )

    json_save_path = os.path.join(args.output_path, "calibration.json")
    with open(json_save_path, "w") as f:
        json.dump(camera_calib_dict, f, indent=4)
