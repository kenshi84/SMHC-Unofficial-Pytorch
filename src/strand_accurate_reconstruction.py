import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from smhc import *


class SAR:
    """Container for smhc (strand-accurate multi-view hair reconstruction) logic and parameters."""

    def __init__(self, dataset, args) -> None:
        self.dataset = dataset
        self.params = args
        # loading the camera data, here all of the cameras are being sent to the dataframe
        self.load_all_camera_params()

    def acquire_image_maps(self, view_idx):
        """
        Acquire the maps given or computed for the input view index.
        These include, additional mask information, orientation maps, corresponding confidence values, initial depth or direction values
        """

        # load hair masks if using them was requested in the input arguments
        hair_mask = None
        if self.params.use_hair_masks:
            hair_mask = self.dataset.views[view_idx].gt_hair_mask

        idm_file = os.path.join(
            self.params.output_path,
            f"{self.params.initial_depth_maps}/img_{self.dataset.views[view_idx].u_id:06d}.npy",
        )
        if os.path.isfile(idm_file):
            with open(idm_file, "rb") as f:
                idm_np = np.load(f)
            idm = torch.from_numpy(idm_np).float()
        else:
            idm = None
            # if self.params.verbose:
            #     print(f"No initial depths data found for camera {self.dataset.views[view_idx].u_id:06d}, running lpmvs without any depth priors ...")

        img_maps = []

        img_maps.append(self.dataset.views[view_idx].original_image)
        img_maps.append(torch.empty(0) if hair_mask is None else hair_mask)
        img_maps.append(self.dataset.views[view_idx].ori2d_map)
        img_maps.append(self.dataset.views[view_idx].conf_map)
        img_maps.append(torch.empty(0) if idm is None else idm[3, :, :].clone())
        img_maps.append(torch.empty(0) if idm is None else idm[:3, :, :].clone())

        return img_maps

    def load_all_camera_params(self):
        """
        Provide camera information for all views in the devised data structure to be used during the run of algorithm
        """
        for idx, viewpoint_cam in enumerate(self.dataset.views):
            load_camera_params_to_dataframe(
                idx,
                viewpoint_cam.image_height,
                viewpoint_cam.image_width,
                viewpoint_cam.P,
                viewpoint_cam.P_col3,
                viewpoint_cam.M_inv,
                viewpoint_cam.K,
                viewpoint_cam.K_inv,
                viewpoint_cam.R,
                viewpoint_cam.T,
                viewpoint_cam.R_inv,
                viewpoint_cam.C,
                viewpoint_cam.Line_proj_mat,
                viewpoint_cam.Fx,
                viewpoint_cam.Fy,
            )

    def run_lpmvs(self):
        """
        GPU adaptation of Line-based patchmatch multi-view stereo (LPMVS)
        """
        if not self.params.verbose:
            progress_bar = tqdm(range(len(self.dataset.views)), desc="LPMVS Progress")

        # running lpmvs for each view independantly
        for idx, viewpoint_cam in enumerate(self.dataset.views):
            if self.params.verbose:
                print(f"Running patchmatch for view {viewpoint_cam.u_id} ...")

            neighboring_views = viewpoint_cam.get_neighboring_views(
                self.dataset.views,
                self.params.neighboring_views_mimimum_angle_diff,
                self.params.neighboring_views_maximum_angle_diff,
                self.params.maximum_neighboring_views_num,
                self.params.verbose,
            )

            # getting all of the provided image data for the view
            # loading an empty tensor as the mask images, causes the lpmvs algorithm to be computed for all pixels in the image
            # this can cause the runtime to become a bit longer!
            img, mask, ori2d, conf, ref_init_depth, ref_init_dirs = (
                self.acquire_image_maps(idx)
            )

            # regarding img_idx input parameter, index 0 is always representing the reference view,
            # the neighboring (source) views get index from 1 to number_of_neighbors
            # this is because the dataframe only contains information for computing results for a single view and after calculation
            # it vipes out all relevant containers (including all image maps).
            # In case you have considerably large VRAM you can change this behaviour and load all images for all views at once!)
            load_image_data_to_dataframe(
                0,
                viewpoint_cam.image_height,
                viewpoint_cam.image_width,
                img,
                ori2d,
                conf,
                mask,
                ref_init_depth,
                False,
            )

            # same process for loading the neighboring views also
            for nv_idx, nv in enumerate(neighboring_views):
                img, mask, ori2d, conf, init_depth, init_dirs = self.acquire_image_maps(
                    nv
                )
                neighbor_img_height = self.dataset.views[nv].image_height
                neighbor_img_width = self.dataset.views[nv].image_width
                load_image_data_to_dataframe(
                    nv_idx + 1,
                    neighbor_img_height,
                    neighbor_img_width,
                    img,
                    ori2d,
                    conf,
                    mask,
                    init_depth,
                    False,
                )

            neighboring_views_tensor = torch.tensor(
                neighboring_views, dtype=torch.int32
            )

            # since we are calculating the lpmvs for the first time here, we provided as the initial depth and direction empty tensors
            # also additional depth and direction perturbation parameters (delta_depth_map, delta_dir_map) are not provided here to
            # run the line refinement steps with the default parameters
            hyper_params = LPMVSHyperParameters(self.params)
            poses, dirs = run_lpmvs(
                hyper_params,
                viewpoint_cam.image_height,
                viewpoint_cam.image_width,
                -1.0,
                -1.0,
                idx,
                len(neighboring_views),
                neighboring_views_tensor,
                ref_init_depth.contiguous().cuda(),
                ref_init_dirs.contiguous().cuda(),
                self.params.verbose,
            )

            # saving the results
            pos_file = os.path.join(
                self.params.output_path,
                f"{self.params.estimates_dir}/pos_img_{viewpoint_cam.u_id:06d}.npy",
            )
            with open(pos_file, "wb") as f:
                np.save(f, poses.cpu().permute(1, 2, 0).numpy())
            dir_file = os.path.join(
                self.params.output_path,
                f"{self.params.estimates_dir}/dir_img_{viewpoint_cam.u_id:06d}.npy",
            )
            with open(dir_file, "wb") as f:
                np.save(f, dirs.cpu().permute(1, 2, 0).numpy())

            if not self.params.verbose:
                progress_bar.update(1)

    def extract_positions_and_directions_from_int_maps(self):
        """
        Convert the consistent (filtered) LMPVS estimtes into point cloud
        """
        # here there is an assumption that all images in the dataset have same size
        dataset_images_width = self.dataset.views[0].image_width
        dataset_images_height = self.dataset.views[0].image_height

        x_coord = (
            torch.arange(dataset_images_width)
            .view(1, -1)
            .repeat(dataset_images_height, 1)
        )  # +0.5 # height x width x 1
        y_coord = (
            torch.arange(dataset_images_height)
            .view(-1, 1)
            .repeat(1, dataset_images_width)
        )  # +0.5 # height x width x 1

        all_positions = []
        all_directions = []

        print(
            "Extracting estimated positions and directions from intermediate filtered maps ..."
        )
        progress_bar = tqdm(range(len(self.dataset.views)), desc="Viewpoints Processed")

        # getting reliable estimated positions and directions from int maps
        for idx, viewpoint_cam in enumerate(self.dataset.views):
            int_map_file = os.path.join(
                self.params.output_path,
                f"{self.params.int_maps_dir}/int_map_{viewpoint_cam.u_id:06d}.npy",
            )
            with open(int_map_file, "rb") as f:
                int_map = np.load(f)

            directions = torch.zeros(
                (viewpoint_cam.image_height, viewpoint_cam.image_width, 4),
                dtype=torch.float32,
            )
            directions[:, :, 0] = torch.from_numpy(int_map[:, :, 1].copy()).float()
            directions[:, :, 1] = torch.from_numpy(int_map[:, :, 2].copy()).float()
            directions[:, :, 2] = torch.nan_to_num(
                torch.sqrt(1 - directions[:, :, 0] ** 2 - directions[:, :, 1] ** 2),
                0.0,
                0.0,
            )

            # storing color values in the 4 element of directions here
            directions[:, :, 3] = viewpoint_cam.original_image.clone().squeeze(0)
            directions = directions.reshape(-1, 4)

            positions = torch.empty(
                (viewpoint_cam.image_height, viewpoint_cam.image_width, 4),
                dtype=torch.float32,
            )
            depth_map = torch.from_numpy(int_map[:, :, 0].copy()).float()
            pt_w = torch.empty(
                (viewpoint_cam.image_height, viewpoint_cam.image_width, 3),
                dtype=torch.float32,
            )
            pt_w[:, :, 0] = depth_map * x_coord - viewpoint_cam.P_col3[0]
            pt_w[:, :, 1] = depth_map * y_coord - viewpoint_cam.P_col3[1]
            pt_w[:, :, 2] = depth_map - viewpoint_cam.P_col3[2]
            pt_w = (viewpoint_cam.M_inv @ pt_w.reshape(-1, 3).T).T
            positions[:, :, :3] = pt_w.reshape(
                viewpoint_cam.image_height, viewpoint_cam.image_width, 3
            ).clone()
            positions[:, :, 3] = depth_map.clone()
            positions = positions.reshape(-1, 4)

            if self.params.use_hair_masks:
                # filter out non-hair parts and points with less than 0 depth values
                gt_hair_mask = viewpoint_cam.gt_hair_mask.reshape(-1)
                all_positions.append(
                    positions[(gt_hair_mask > 0.01) * (positions[:, 3] > 0)]
                )
                all_directions.append(
                    directions[(gt_hair_mask > 0.01) * (positions[:, 3] > 0)]
                )
            else:
                all_positions.append(positions[(positions[:, 3] > 0)])
                all_directions.append(directions[(positions[:, 3] > 0)])

            progress_bar.update(1)
            if idx == len(self.dataset.views) - 1:
                progress_bar.close()

        all_positions = torch.cat(all_positions, dim=0)
        all_directions = torch.cat(all_directions, dim=0)

        return (all_positions, all_directions)

    def run_line_filtering(self):
        """
        GPU adaptation of Line-filtering step on the estimations of LPMVS
        """

        if not self.params.verbose:
            progress_bar = tqdm(
                range(len(self.dataset.views)), desc="Line Filtering Progress"
            )

        for idx, viewpoint_cam in enumerate(self.dataset.views):
            if self.params.verbose:
                print(f"Running line filtering for view {idx} ...")

            neighboring_views = viewpoint_cam.get_neighboring_views(
                self.dataset.views,
                self.params.neighboring_views_mimimum_angle_diff,
                self.params.neighboring_views_maximum_angle_diff,
                self.params.maximum_neighboring_views_num,
                self.params.verbose,
            )

            # load the estimated positions and directions
            # in case of applying strand integration load the optimized results
            pos_file = os.path.join(
                self.params.output_path,
                f"{self.params.estimates_dir}/pos_img_{viewpoint_cam.u_id:06d}.npy",
            )
            dir_file = os.path.join(
                self.params.output_path,
                f"{self.params.estimates_dir}/dir_img_{viewpoint_cam.u_id:06d}.npy",
            )

            with open(pos_file, "rb") as f:
                poses = np.load(f)
            with open(dir_file, "rb") as f:
                dirs = np.load(f)
            poses = torch.from_numpy(poses.reshape(-1, 4)).float()
            dirs = torch.from_numpy(dirs.reshape(-1, 4)).float()

            # getting all of the provided image data for the view
            img, mask, ori2d, conf, ref_init_depth, ref_init_dirs = (
                self.acquire_image_maps(idx)
            )
            # same point regarding loading mask maps as empty tensors and image index parameter as explained in run_lpmvs function
            load_image_data_to_dataframe(
                0,
                viewpoint_cam.image_height,
                viewpoint_cam.image_width,
                img,
                ori2d,
                conf,
                mask,
                ref_init_depth,
                load_to_cpu_containers=False,
            )
            load_lpmvs_estimations_to_dataframe(0, poses, dirs)

            # same process for loading the neighboring views also
            for nv_idx, nv in enumerate(neighboring_views):

                # load the estimated positions and directions
                # in case of applying strand integration load the optimized results
                pos_file = os.path.join(
                    self.params.output_path,
                    f"{self.params.estimates_dir}/pos_img_{self.dataset.views[nv].u_id:06d}.npy",
                )
                dir_file = os.path.join(
                    self.params.output_path,
                    f"{self.params.estimates_dir}/dir_img_{self.dataset.views[nv].u_id:06d}.npy",
                )

                with open(pos_file, "rb") as f:
                    poses = np.load(f)
                with open(dir_file, "rb") as f:
                    dirs = np.load(f)
                poses = torch.from_numpy(poses.reshape(-1, 4)).float()
                dirs = torch.from_numpy(dirs.reshape(-1, 4)).float()

                img, mask, ori2d, conf, init_depth, init_dirs = self.acquire_image_maps(
                    nv
                )
                neighbor_img_height = self.dataset.views[nv].image_height
                neighbor_img_width = self.dataset.views[nv].image_width
                load_image_data_to_dataframe(
                    nv_idx + 1,
                    neighbor_img_height,
                    neighbor_img_width,
                    img,
                    ori2d,
                    conf,
                    mask,
                    init_depth,
                    load_to_cpu_containers=False,
                )
                load_lpmvs_estimations_to_dataframe(nv_idx + 1, poses, dirs)

            neighboring_views_tensor = torch.tensor(
                neighboring_views, dtype=torch.int32
            )

            # one point regarding calculation of intermediate maps is that, they are being computed incrementally
            # meaning if there are some results from previous runs of algorithm, they get loaded here and are only updated in the pixels
            # were there was previously no reliable estimation, other pixels are left unchanged and hence they will keep the reliable results
            # from previous runs.
            int_map = viewpoint_cam.int_map.cuda()
            hyper_params = LineFilteringHyperParameters(self.params)
            run_multiview_filtering(
                hyper_params,
                idx,
                len(neighboring_views),
                neighboring_views_tensor,
                int_map,
                verbose=self.params.verbose,
            )
            viewpoint_cam.int_map.copy_(int_map.cpu().contiguous())

            # saving the consistency maps for some debugging (this is not necessary, can be commented out!)
            consistency_map = int_map[0, :, :] > 0
            consistency_img = Image.fromarray(
                consistency_map.cpu().to(torch.uint8).numpy() * 255
            )
            consistency_img.save(
                os.path.join(
                    self.params.output_path,
                    f"{self.params.filtered_dir}/acc_img_{viewpoint_cam.u_id:06d}.png",
                )
            )

            # saving the int maps for later
            int_map_file = os.path.join(
                self.params.output_path,
                f"{self.params.int_maps_dir}/int_map_{viewpoint_cam.u_id:06d}.npy",
            )
            with open(int_map_file, "wb") as f:
                np.save(f, int_map.cpu().permute(1, 2, 0).numpy())

            if not self.params.verbose:
                progress_bar.update(1)

        all_positions, all_directions = (
            self.extract_positions_and_directions_from_int_maps()
        )
        filtered_results_dir = os.path.join(
            self.params.output_path, self.params.filtered_dir
        )
        with open(
            os.path.join(filtered_results_dir, "filtered_positions.npy"), "wb"
        ) as f:
            np.save(f, all_positions.clone().numpy())
        with open(
            os.path.join(filtered_results_dir, "filtered_directions.npy"), "wb"
        ) as f:
            np.save(f, all_directions.clone().numpy())

    def run_line_fusion(self):
        """
        GPU adaptation of Line-fusion step on the point cloud obtained from line filtering.
        CPU version is also provided but runs slower!
        """
        # load the filtered positions and directions
        estimates_dir = os.path.join(
            self.params.output_path, f"{self.params.filtered_dir}"
        )
        with open(os.path.join(estimates_dir, "filtered_positions.npy"), "rb") as f:
            filtered_positions_np = np.load(f)
        with open(os.path.join(estimates_dir, "filtered_directions.npy"), "rb") as f:
            filtered_directions_np = np.load(f)

        filtered_positions = torch.from_numpy(filtered_positions_np).float()
        filtered_directions = torch.from_numpy(filtered_directions_np).float()

        # fuse the input point cloud
        # you can do this step on CPU fy setting fuse_on_gpu parameter to False
        hyper_params = LineFusionHyperParameters(self.params)
        fused_positions, fused_directions = fuse_point_cloud(
            hyper_params,
            filtered_positions,
            filtered_directions,
            fuse_on_gpu=True,
            verbose=self.params.verbose,
        )

        # saving the fused output point cloud
        fused_results_dir = os.path.join(self.params.output_path, self.params.fused_dir)
        with open(os.path.join(fused_results_dir, "fused_positions.npy"), "wb") as f:
            np.save(f, fused_positions.clone().numpy())
        with open(os.path.join(fused_results_dir, "fused_directions.npy"), "wb") as f:
            np.save(f, fused_directions.clone().numpy())

    def generate_strand_segments(self):
        """
        Generate strand segments from the fused point cloud.
        """
        # load the fused positions and directions
        fused_results_dir = os.path.join(self.params.output_path, self.params.fused_dir)
        with open(os.path.join(fused_results_dir, "fused_positions.npy"), "rb") as f:
            fused_positions_np = np.load(f)
        with open(os.path.join(fused_results_dir, "fused_directions.npy"), "rb") as f:
            fused_directions_np = np.load(f)

        fused_positions = torch.from_numpy(fused_positions_np).float()
        fused_directions = torch.from_numpy(fused_directions_np).float()

        # generate the strand segments by applying forward Euler method on the given point cloud
        hyper_params = StrandSegmentsGenerationHyperParametrs(self.params)
        segments_positions, segments_directions, points_num_per_segment = (
            generate_strand_segments(
                hyper_params,
                fused_positions,
                fused_directions,
                verbose=self.params.verbose,
            )
        )

        # saving the generated strand segments
        results_dir = os.path.join(
            self.params.output_path, self.params.strand_segments_dir
        )
        with open(os.path.join(results_dir, "segments_positions.npy"), "wb") as f:
            np.save(f, segments_positions.clone().numpy())

        with open(os.path.join(results_dir, "segments_directions.npy"), "wb") as f:
            np.save(f, segments_directions.clone().numpy())

        with open(os.path.join(results_dir, "points_num_per_segment.npy"), "wb") as f:
            np.save(f, points_num_per_segment.clone().int().numpy())

    def run_multiview_hair_growing(self):
        """
        Elongate the generated strand segments consistent with the multiview input data.
        """
        # loading the initial strand segments
        strand_segments_dir = os.path.join(
            self.params.output_path, self.params.strand_segments_dir
        )
        with open(
            os.path.join(strand_segments_dir, "segments_positions.npy"), "rb"
        ) as f:
            segments_positions_np = np.load(f)
        with open(
            os.path.join(strand_segments_dir, "segments_directions.npy"), "rb"
        ) as f:
            segments_directions_np = np.load(f)
        with open(
            os.path.join(strand_segments_dir, "points_num_per_segment.npy"), "rb"
        ) as f:
            points_num_per_segment_np = np.load(f)

        segments_positions = torch.from_numpy(segments_positions_np).float()
        segments_directions = torch.from_numpy(segments_directions_np).float()
        points_num_per_segment = torch.from_numpy(points_num_per_segment_np).int()

        for idx in range(len(self.dataset.views)):
            # do this to first make sure that all the maps are loaded for the current view
            img, _, ori2d_map, conf_map, _, _ = self.acquire_image_maps(idx)
            load_image_data_to_dataframe(
                idx,
                img.size(0),
                img.size(1),
                img,
                ori2d_map,
                conf_map,
                torch.empty(0),
                torch.empty(0),
                load_to_cpu_containers=True,
            )

        hyper_params = MultiviewHairGrowingHyperParameters(self.params)
        final_positions, final_directions, points_num_per_strand = (
            run_multiview_hair_growing(
                hyper_params,
                len(self.dataset.views),
                segments_positions,
                segments_directions,
                points_num_per_segment,
                verbose=self.params.verbose,
            )
        )

        multiview_growing_results_dir = os.path.join(
            self.params.output_path, self.params.multiview_growing_dir
        )
        with open(
            os.path.join(multiview_growing_results_dir, "final_positions.npy"), "wb"
        ) as f:
            np.save(f, final_positions.clone().numpy()[:, :3])

        with open(
            os.path.join(multiview_growing_results_dir, "final_directions.npy"), "wb"
        ) as f:
            np.save(f, final_directions.clone().numpy()[:, :3])

        with open(
            os.path.join(
                multiview_growing_results_dir, "final_points_num_per_strand.npy"
            ),
            "wb",
        ) as f:
            np.save(f, points_num_per_strand.clone().int().numpy())

    def cleanup(self):
        destroy_dataframe()
