import random

import numpy as np
import torch


class Camera:
    def __init__(
        self,
        u_id,
        id,
        P,
        P_col3,
        M_inv,
        K,
        K_inv,
        R,
        T,
        R_inv,
        C,
        Line_proj_mat,
        Fx,
        Fy,
        image,
        gt_hair_mask,
        int_map,
        ori2d_map,
        conf_map,
        image_name,
    ):
        super(Camera, self).__init__()

        self.id = id
        self.u_id = u_id

        """
        camera parameters, P, P_col34, M_inv, K, K_inv, R, t4, R_inv, c4 (Tensor) as following,

        projection formula x_31 = P_34 X_41 = K33 [ R_33 | t_31 ] X_41 = [ M_33 | P_col34_31] X_41
        extrinsic matrix : [ R | t4]
        extrinsic matrix inverse : [ R_inv | c4]
        """

        self.P = torch.tensor(P, dtype=torch.float32)
        self.P_col3 = torch.tensor(P_col3, dtype=torch.float32)
        self.M_inv = torch.tensor(M_inv, dtype=torch.float32)
        self.K = torch.tensor(K, dtype=torch.float32)
        self.K_inv = torch.tensor(K_inv, dtype=torch.float32)
        self.R_inv = torch.tensor(R_inv, dtype=torch.float32)
        self.C = torch.tensor(C, dtype=torch.float32)
        self.Fx = Fx
        self.Fy = Fy
        self.Px = self.K[0, 2]
        self.Py = self.K[1, 2]
        self.R = torch.tensor(R, dtype=torch.float32)
        self.T = torch.tensor(T, dtype=torch.float32)
        # self.znear = 0.01
        # self.zfar = 100.0
        self.Line_proj_mat = torch.tensor(Line_proj_mat, dtype=torch.float32)  # new

        self.image_name = image_name
        self.original_image = image.clamp(
            0.0, 1.0
        ).contiguous()  # move to device at dataloader to reduce VRAM requirement
        self.image_height = self.original_image.shape[0]
        self.image_width = self.original_image.shape[1]
        self.gt_hair_mask = gt_hair_mask.clamp(0.0, 1.0).contiguous()
        self.int_map = int_map.contiguous()
        self.ori2d_map = ori2d_map.contiguous()
        self.conf_map = conf_map.contiguous()

    def calc_point_from_depth_local_coord(
        self, pixel_location: torch.Tensor, depth: float
    ):
        return torch.tensor(
            [
                depth * (pixel_location[0] - self.Px) / self.Fx,
                depth * (pixel_location[1] - self.Py) / self.Fy,
                depth,
                1.0,
            ],
            dtype=torch.float32,
        )

    def calc_view_vector(self):

        cam_extrinsics_inv = torch.zeros((4, 4), dtype=torch.float32)
        cam_extrinsics_inv[:3, :3] = self.R_inv.clone()
        cam_extrinsics_inv[:3, 3] = self.C.clone()
        cam_extrinsics_inv[3, 3] = 1.0

        center_pixel = torch.tensor([self.Px, self.Py], dtype=torch.float32)
        vec = cam_extrinsics_inv @ self.calc_point_from_depth_local_coord(
            center_pixel, self.Fx
        )
        view_vector = vec[:3] - self.C
        return view_vector / torch.linalg.vector_norm(view_vector)

    def get_neighboring_views(
        self,
        all_views,
        min_angle_diff: float,
        max_angle_diff: float,
        max_neighbors_num: int,
        verbose: bool = False,
    ):
        neighbor_views = []

        ref_cam_view_vector = self.calc_view_vector()

        if verbose:
            print(
                f"Accepting neighboring cameras with viewing angle difference from {min_angle_diff} to {max_angle_diff} degrees!"
            )

        minimum_angle_radians = min_angle_diff * np.pi / 180.0
        maximum_angle_radians = max_angle_diff * np.pi / 180.0

        temp_neighbors_list = []
        for i, c in enumerate(all_views):
            view_vector = c.calc_view_vector()
            angle_diff = torch.acos(torch.dot(ref_cam_view_vector, view_vector))
            if (
                angle_diff > minimum_angle_radians
                and angle_diff < maximum_angle_radians
            ):
                temp_neighbors_list.append(i)
                if verbose:
                    print(
                        f"\t neighboring camera index {i} with angle difference {angle_diff} radians"
                    )

        if len(temp_neighbors_list) > max_neighbors_num:
            if verbose:
                print(
                    f"Too many detected neighboring views, selecting only {max_neighbors_num} of them on random."
                )
            random.shuffle(temp_neighbors_list)
            neighbor_views = temp_neighbors_list[:max_neighbors_num]
        else:
            neighbor_views = temp_neighbors_list

        return neighbor_views

    def project_3d_line_to_2d(self, estimated_direction, estimated_position):
        estimated_momentum = np.cross(estimated_position, estimated_direction)

        reverse_dir = np.zeros(
            (estimated_direction.shape[0], estimated_direction.shape[1], 6),
            dtype=np.float32,
        )
        reverse_dir[:, :, 0] = estimated_momentum[:, :, 2].copy()
        reverse_dir[:, :, 1] = estimated_momentum[:, :, 1].copy()
        reverse_dir[:, :, 2] = estimated_momentum[:, :, 0].copy()
        reverse_dir[:, :, 3] = estimated_direction[:, :, 2].copy()
        reverse_dir[:, :, 4] = estimated_direction[:, :, 1].copy()
        reverse_dir[:, :, 5] = estimated_direction[:, :, 0].copy()
        reverse_dir = reverse_dir.reshape(-1, 6)

        estimated_2d_line_params = (self.Line_proj_mat.numpy() @ reverse_dir.T).T
        ori_2d = np.arctan(
            -estimated_2d_line_params[:, 0] / (estimated_2d_line_params[:, 1] + 1.0e-8)
        )

        return ori_2d.reshape(
            estimated_direction.shape[0], estimated_direction.shape[1]
        )
