from jaxtyping import Float32, Int32
from torch import Tensor

import smhc.cuda as _C
from smhc.hyperparameters import (
    LineFilteringHyperParameters,
    LineFusionHyperParameters,
    LPMVSHyperParameters,
    MultiviewHairGrowingHyperParameters,
    StrandSegmentsGenerationHyperParametrs,
)
from smhc.type_checking import check_types


@check_types
def calculate_orientation_maps(
    image_data: Float32[Tensor, "img_height img_width"],
    gabor_kernels: Float32[Tensor, "K"],
    kernels_num: int,
    kernels_size: int = 17,
) -> tuple[
    Float32[Tensor, "img_height img_width"], Float32[Tensor, "img_height img_width"]
]:
    """Compute orientation maps and according confidence values from the DoG (difference of Gaussians) image.

    Args:
        image_data (Tensor): input DoG image.
        gabor_kernels (Tensor): all of the oriented gabor kernels appended to a single 1-dimensional tensor.
        kernels_num (int): number of all orented gabor kernels (this is usually 180 / kernel_angular_resolution).
        kernels_size (int): resolution of the kernels (set to 17 by default).

    Returns:
        A tuple of {Tensor, Tensor}:

        - **ori_map** (Tensor, with shape [img_height, img_width]): detected pixel-wise orientation data (in the range of [-pi/2, pi/2]).
        - **conf_map** (Tensor, with shape [img_height, img_width]): associated pixel-wise confidence values for the detected orientations (usally somewhere in the range of [0., 100.]).
    """

    ori_map, conf_map = _C.calculate_orientation_maps(
        image_data, gabor_kernels, kernels_num, kernels_size
    )
    return (ori_map, conf_map)


@check_types
def load_camera_params_to_dataframe(
    cam_idx: int,
    img_height: int,
    img_width: int,
    P: Float32[Tensor, "3 4"],
    P_col3: Float32[Tensor, "3"],
    M_inv: Float32[Tensor, "3 3"],
    K: Float32[Tensor, "3 3"],
    K_inv: Float32[Tensor, "3 3"],
    R: Float32[Tensor, "3 3"],
    t: Float32[Tensor, "3"],
    R_inv: Float32[Tensor, "3 3"],
    c: Float32[Tensor, "3"],
    line_proj_mat: Float32[Tensor, "3 6"],
    cams_fx: float,
    cams_fy: float,
) -> None:
    """Load the camera related info to the data structre that will reside in GPU or CPU and used through different steps of smhc.

    Args:
        cam_idx (int): index of the input camera.
        img_height (int): resolution of the image in y axis (vertical).
        img_width (int): resolution of the image in x axis (horizontal).
        camera parameters, P, P_col34, M_inv, K, K_inv, R, t4, R_inv, c4 (Tensor) as following,

        projection formula x_31 = P_34 X_41 = K33 [ R_33 | t_31 ] X_41 = [ M_33 | P_col3_31] X_41
        extrinsic matrix : [ R | t]
        extrinsic matrix inverse : [ R_inv | c]

        line_proj_mat (Tensor): projection of 6D Plücker line coordinates to lines in image space
                                (formulated by 3 parameters, a,b,c, ax+by+c=0) [(Hartley and Zisserman, Multiple view geometry in computer vision 2003, p.198)].
        cams_fx (float): focal length of the camera in x axis (specified in the intrisics matrix).
        cams_fy (float): focal length of the camera in y axis (specified in the intrisics matrix).

    Returns:
        None

    """

    _C.load_camera_params(
        cam_idx,
        img_height,
        img_width,
        P,
        P_col3,
        M_inv,
        K,
        K_inv,
        R,
        t,
        R_inv,
        c,
        line_proj_mat,
        cams_fx,
        cams_fy,
    )


@check_types
def load_image_data_to_dataframe(
    img_idx: int,
    img_height: int,
    img_width: int,
    img_gray: Float32[Tensor, "img_height img_width"],
    img_ori: Float32[Tensor, "img_height img_width"],
    img_conf: Float32[Tensor, "img_height img_width"],
    img_mask: Float32[Tensor, "img_height img_width"] | Float32[Tensor, "0"],
    initial_depth_map: Float32[Tensor, "img_height img_width"] | Float32[Tensor, "0"],
    load_to_cpu_containers: bool = False,
) -> None:
    """Load the image related info to the data structre that will reside in GPU or CPU and used through different steps of smhc.

    Args:
        img_idx (int): index of the input image, a bit caution here, this index does not correspond with the camera index.
                        In MVS setups there is a reference view and several neighboring (source views) which are used to find correspondenses.
                        In this implementation, index 0 always refers to the reference view and source views have indices from 1 up to the maximum allowed number of neighboring views.
        img_height (int): resolution of the image in y axis (vertical).
        img_width (int): resolution of the image in x axis (horizontal).

        img_gray (Tensor, with shape [img_height, img_width]): input intensity image.
        img_ori (Tensor, with shape [img_height, img_width]): computed orientation map of the input image.
        img_conf (Tensor, with shape [img_height, img_width]): corresponding confidence values of the computed orientations.

        img_mask (Tensor, with shape [img_height, img_width]): optional mask information, specifying the hair region in the input image.
                            this additional data can be used in the MVS step for better efficiency and accuracy!
                            but it is not without its drawbacks, since masking is potentionally not consistent in the multi-view setup,
                            hence resulting in reconstruction defficiencies specially in the silhouette regions.

        initial_depth_map (Tensor, with shape [img_height, img_width]): optional initial depth information for the input image.
                            this additional data can be used in the MVS step for better accuracy!
                            the depth map can come from any depth estimation method, e.g. it can be from priory fitted gaussians in the Gaussian Splatting [Kerbl et al, 2023] technique.

        load_to_cpu_containers (bool): specifying whether the provided input image info should reside in cpu, which in this case
                                        no copying from CPU to GPU will take place and only pointers to the torch tensors will be borrowed (!)
                                        for further uses in smhc.

    Returns:
        None

    """

    _C.load_image_data(
        img_idx,
        img_height,
        img_width,
        img_gray,
        img_ori,
        img_conf,
        img_mask,
        initial_depth_map,
        load_to_cpu_containers,
    )


@check_types
def load_lpmvs_estimations_to_dataframe(
    img_idx: int,
    estimated_positions: Float32[Tensor, "img_num_pixels 4"],
    estimated_direcitons: Float32[Tensor, "img_num_pixels 4"],
) -> None:
    """Load the MVS depth and direction estimations to the GPU to do line filtering step.

    Args:
        img_idx (int): index of the input image, a bit caution here, this index does not correspond with the camera index.
                        In MVS setups there is a reference view and several neighboring (source) views which are used to find correspondenses.
                        In this implementation, index 0 always refers to the reference view and source views have indices from 1 up to the maximum allowed number of neighboring views.


        estimated_positions (Tensor, with shape [img_height*img_width, 4]): estimated depth values from MVS will result in position estimations
                                for each pixel in the input image. Take caution here, the estimations are flattened.
        estimated_directions (Tensor, with shape [img_height*img_width, 4]): estimated direction values from MVS for each pixel in the input
                                image. Take caution here, the estimations are flattened.


    Returns:
        None

    """

    _C.load_patchmatch_estimations(img_idx, estimated_positions, estimated_direcitons)


def destroy_dataframe():
    _C.destroy_dataframe()


@check_types
def run_lpmvs(
    params: LPMVSHyperParameters,
    img_height: int,
    img_width: int,
    delta_depth_amp: float,
    delta_dir_amp: float,
    ref_view_idx: int,
    neighboring_views_number: int,
    neighboring_views_idx: Int32[Tensor, "S"],
    initial_depths: Float32[Tensor, "img_height img_width"] | Float32[Tensor, "0"],
    initial_directions: (
        Float32[Tensor, "3 img_height img_width"] | Float32[Tensor, "0"]
    ),
    verbose: bool = False,
) -> tuple[
    Float32[Tensor, "4 img_height img_width"], Float32[Tensor, "4 img_height img_width"]
]:
    """Line-based patchmatch multi-view stereo (LPMVS).

    Args:
        img_height (int): resolution of the image in y axis (vertical).
        img_width (int): resolution of the image in x axis (horizontal).
        params (): input data structure containing LPMVS hyper parameters (for more details refer to the original paper by Nam et al. 2019)
        delta_depth_amp (float): additional control over the amplitude of depth perturbations in the line hypothesis refinement steps
                            (default value of -1. will result in the amplitude to be as difference of maximum_depth and minimum_depth provided as hyper parameters).
        delta_dir_amp (float): additional control over the amplitude of direction perturbations in the line hypothesis refinement steps
                    (default value of -1. will result in the amplitude to be 1. for each component of the direction vector).
        ref_view_idx (int): index of the input image that correspondes to the camera indices.
        neighboring_views_number (int): number of neighboring views
        neighboring_views_idx (Tensor, with shape [neighboring_views_number]): list of neighboring views indices, these correspond to the camera indices.

        initial_depths (Tensor, with shape [img_height, img_width]): optional input, if any estimation of depth for input image is available
                            it can be used for better initialization for LPMVS.
        initial_directions (Tensor, with shape [3, img_height, img_width]): optional input, if any estiation of 3D directions for image pixels
                            is available it can be used for better initialization for LMPVS.

        verbose (bool): whether to have additional info printed into to the console regarding procession of the LMPVS algorithm (default: false).

    Returns:
        A tuple of {Tensor, Tensor}:

        - **estimated_positions** (Tensor, with shape [4, img_height, img_width]): detected pixel-wise 3D positions.
        - **estimated_directions** (Tensor, with shape [4, img_height, img_width]): detected pixel-wise 3D directions.
    """

    estimated_positions, estimated_directions = _C.run_line_based_patchmatch_mvs(
        img_height,
        img_width,
        params.alpha,
        params.iterations,
        params.minimum_depth,
        params.maximum_depth,
        params.sampling_radius,
        params.samples_num,
        params.occlusion_depth_thresh,
        delta_depth_amp,
        delta_dir_amp,
        ref_view_idx,
        neighboring_views_number,
        neighboring_views_idx,
        initial_depths,
        initial_directions,
        verbose,
    )

    return (estimated_positions, estimated_directions)


@check_types
def run_multiview_filtering(
    params: LineFilteringHyperParameters,
    ref_view_id: int,
    neighboring_views_number: int,
    neighboring_views_idx: Int32[Tensor, "S"],
    int_map: Float32[Tensor, "3 img_height img_width"],
    verbose: bool = False,
) -> None:
    """Line filtering step applied to estimations of LPMVS.

    Args:
        ref_view_idx (int): index of the input image that correspondes to the camera indices.
        neighboring_views_number (int): number of neighboring views
        neighboring_views_idx (Tensor, with shape [neighboring_views_number]): list of neighboring views indices, these correspond to the camera indices.

        int_map (Tensor, with shape [3, img_height, img_width]): intermediate maps that will contain filtered estimations of LPMVS for valid pixels
                    (valid here refers to pixels that passed filtering), and further 2D information (orientation and confidence) for the invalid
                    pixels (those that failed filtering).

                    A point regarding calculation of intermediate maps. They are being computed incrementally
                    meaning if there are some results from previous runs of algorithm, they get loaded here and are only updated in the pixels
                    were there was previously no valid estimation, other pixels are left unchanged and hence they will keep the reliable results
                    from previous runs.

        params (): input data structure containing line filtering hyper parameters (for more details refer to the original paper by Nam et al. 2019)
        verbose (bool): whether to have additional info printed into to the console regarding procession of the filtering algorithm (default: false).

    Returns:
        Explicitly none, however note that, int_map is both input and output of this function

    """

    _C.filter_pc(
        ref_view_id,
        neighboring_views_number,
        neighboring_views_idx,
        int_map,
        params.position_consistency_thres,
        params.direction_diff_threshold_rad,
        params.consistent_neighboring_views_threshold,
        verbose,
    )


@check_types
def fuse_point_cloud(
    params: LineFusionHyperParameters,
    positions: Float32[Tensor, "num_points 4"],
    directions: Float32[Tensor, "num_points 4"],
    fuse_on_gpu: bool = True,
    verbose: bool = False,
) -> tuple[Float32[Tensor, "num_points 4"], Float32[Tensor, "num_points 4"]]:
    """Point cloud fusion step applied to the filtered point cloud.

    Args:
        positions (Tensor, with shape [num_points, 4]): input positions of the point cloud.
        directions (Tensor, with shape [num_points, 4]): input directions of the point cloud.
        params (): input data structure containing line fusion hyper parameters (for more details refer to the original paper by Nam et al. 2019)

        fuse_on_gpu (bool): whether to fuse point cloud on GPU (provides better efficiency) or CPU. (default: True)
        verbose (bool): whether to have additional info printed into to the console regarding procession of the fusion algorithm (default: False).

    Returns:
        A tuple of {Tensor, Tensor}:

        - **fused_positions** (Tensor, with shape [num_points, 4]): output (fused) 3D positions.
        - **fused_directions** (Tensor, with shape [num_points, 4]): output (fused) 3D directions.

    """
    fused_positions, fused_directions = _C.fuse_pc(
        positions,
        directions,
        params.position_update_threshold,
        params.search_radius,
        params.sigma_pos,
        params.sigma_dir,
        fuse_on_gpu,
        verbose,
    )
    return (fused_positions, fused_directions)


@check_types
def generate_strand_segments(
    params: StrandSegmentsGenerationHyperParametrs,
    positions: Float32[Tensor, "num_points 4"],
    directions: Float32[Tensor, "num_points 4"],
    verbose: bool = False,
) -> tuple[Float32[Tensor, "X 4"], Float32[Tensor, "X 4"], Float32[Tensor, "Y"]]:
    """Genrating strand segments from the provided point cloud.

    Args:
        positions (Tensor, with shape [num_points, 4]): input positions of the point cloud.
        directions (Tensor, with shape [num_points, 4]): input directions of the point cloud.
        params (): input data structure containing strand segment generation hyper parameters (for more details refer to the original paper by Nam et al. 2019)
        verbose (bool): whether to have additional info printed into to the console regarding procession of the strand generation algorithm (default: False).

    Returns:
        A tuple of {Tensor, Tensor, Tensor}:

        - **strands_positions** (Tensor, with shape [__un_known__, 4]): output 3D positions of the strands vertices.
        - **strands_directions** (Tensor, with shape [__un_known__, 4]): output 3D directions of the strands vertices.
        - **points_per_strand** (Tensor, with shape [__un_known__]): output list of number of points per strand segments, used to specify individual segments.

    """

    strands_positions, strands_directions, points_per_strand = (
        _C.generate_strand_segments(
            positions,
            directions,
            params.step_size,
            params.length_threshold,
            params.neighboring_radius,
            params.angle_difference_threshold_rad,
            verbose,
        )
    )

    return (strands_positions, strands_directions, points_per_strand)


@check_types
def run_multiview_hair_growing(
    params: MultiviewHairGrowingHyperParameters,
    total_views_num: int,
    positions: Float32[Tensor, "num_points 4"],
    directions: Float32[Tensor, "num_points 4"],
    points_num_per_strand: Int32[Tensor, "num_segments"],
    only_apply_background_removal: bool = False,
    verbose: bool = False,
) -> tuple[Float32[Tensor, "X 4"], Float32[Tensor, "X 4"], Float32[Tensor, "Y"]]:
    """Apply multi-view hair growing on the generated strand segments.

    Args:
        total_views_num (int): total number of input camera views.
        positions (Tensor, with shape [num_points, 4]): input positions of the strand segments.
        directions (Tensor, with shape [num_points, 4]): input directions of the strand segments.
        points_num_per_strand (Tensor, with shape [num_segments]): input list of number of points per strand segments, used to specify individual segments.
        params (): input data structure containing multi-view hair growing hyper parameters (for more details refer to the original paper by Nam et al. 2019)

        only_apply_background_removal (bool): if only omission of back ground strands is intended (default: False).
        verbose (bool): whether to have additional info printed into to the console regarding procession of the multi-view hair growing algorithm (default: False).

    Returns:
        A tuple of {Tensor, Tensor, Tensor}:

        - **strands_positions** (Tensor, with shape [__un_known__, 4]): output 3D positions of the strands vertices.
        - **strands_directions** (Tensor, with shape [__un_known__, 4]): output 3D directions of the strands vertices.
        - **out_points_per_strand** (Tensor, with shape [__un_known__]): output list of number of points per strand segments, used to specify individual segments.

    """

    strands_positions, strands_directions, out_points_per_strand = (
        _C.run_multiview_hair_growing(
            total_views_num,
            positions,
            directions,
            points_num_per_strand,
            params.reset_valid_views_period,
            params.max_iter_num,
            params.background_ratio,
            params.search_cone_opening_angle_rad,
            params.search_cone_angular_resolution_rad,
            params.search_cone_window_width,
            params.search_cone_window_height,
            params.orientation_2d_confidence_threshold,
            params.orientation_2d_angle_difference_threshold_rad,
            params.valid_pixels_number_threshold,
            params.minimum_views_with_valid_2d_direction,
            params.growing_step_size,
            params.direction_change_threshold_rad,
            params.background_color_threshold,
            only_apply_background_removal,
            verbose,
        )
    )

    return (strands_positions, strands_directions, out_points_per_strand)


@check_types
def calculate_hair_metrics(
    position_thresholds: Float32[Tensor, "num_thresholds"],
    direction_thresholds: Float32[Tensor, "num_thresholds"],
    gt_positions: Float32[Tensor, "gt_num_points 3"],
    gt_directions: Float32[Tensor, "gt_num_points 3"],
    estimated_positions: Float32[Tensor, "num_points 3"],
    estimated_directions: Float32[Tensor, "num_points 3"],
    verbose: bool = False,
) -> None:
    """Compute quantitative metrics on the estimated strands.

        Metrics are calculated as the following. Assume we have two sets of points. For each point in set1 we check in set2 if there exists a
        point in the specified vicinity (positional threshold) with compatible direction (directional threshold). If so we mark the point in
        set1 as valid and otherwise as invalid.
        Precision of estimations is computed by fixing set1 as the estimated and set2 as the ground truth strands.
        Recall is computed the other way by choosing set1 as the groud truth and set2 as the estimated strands.

    Args:
        position_thresholds (Tensor, with shape [__num_thresholds__]): different thresholds (meter) on the tolerated positional error, e.g. [0.0005, 0.001, 0.002]
        direction_thresholds (Tensor, with shape [__num_thresholds__]): different thresholds (degrees) on the tolerated directional error, e.g. [5°, 10°, 20°]
        gt_positions (Tensor, with shape [gt_num_points, 3]): 3D postion of vertices in the ground truth strands.
        gt_directions (Tensor, with shape [gt_num_points, 3]): 3D direction of ground truth strands at the place of their vertices.
        estimated_positions (Tensor, with shape [num_points, 3]): 3D position of estimated strand segments.
        estimated_directions (Tensor, with shape [num_points, 3]): 3D direction of strand segments at the place of their vertices
        verbose (bool): whether to have additional info printed into to the console regarding procession of the quantitative measures calculation (default: False).

    Returns:
        None, calculated precision, recall and F1_score will be printed to the console for different thresholds.

    """
    _C.calculate_hair_metrics(
        position_thresholds,
        direction_thresholds,
        gt_positions,
        gt_directions,
        estimated_positions,
        estimated_directions,
        verbose,
    )


@check_types
def load_dot_hair_model(
    file_name: str, positions_scale: float = 1.0
) -> tuple[Float32[Tensor, "X 3"], Float32[Tensor, "Y"]]:
    """Load hair model with '.hair' file extension.

    Args:
        file_name (str): path to the hair model.
        positions_scale (float): optional input, if additional scaling of the strands vertices is required

    Returns:
        A tuple of {Tensor, Tensor}:

        - **strands_positions** (Tensor, with shape [__un_known__, 3]): output 3D positions of the strands vertices.
        - **out_points_per_strand** (Tensor, with shape [__un_known__]): output list of number of points per strand segments, used to specify individual segments.

    """
    strands_positions, out_points_per_strand = _C.load_dot_hair_model(
        file_name, positions_scale
    )

    return (strands_positions, out_points_per_strand)


@check_types
def load_dot_bin_model(
    file_name: str, positions_scale: float = 1.0
) -> tuple[Float32[Tensor, "X 3"], Float32[Tensor, "Y"]]:
    """Load hair model with '.bin' file extension.

    Args:
        file_name (str): path to the hair model.
        positions_scale (float): optional input, if additional scaling of the strands vertices is required

    Returns:
        A tuple of {Tensor, Tensor}:

        - **strands_positions** (Tensor, with shape [__un_known__, 3]): output 3D positions of the strands vertices.
        - **out_points_per_strand** (Tensor, with shape [__un_known__]): output list of number of points per strand segments, used to specify individual segments.

    """

    strands_positions, out_points_per_strand = _C.load_dot_bin_model(
        file_name, positions_scale
    )

    return (strands_positions, out_points_per_strand)


@check_types
def load_dot_data_model(
    file_name: str, positions_scale: float = 1.0
) -> tuple[Float32[Tensor, "num_points 3"], Float32[Tensor, "num_strands"]]:
    """Read XYZ positions and per-strand point counts from a hairutil .data file.

    The file starts with an int32 strand count. Each strand stores an int32 point
    count followed by that many float32 XYZ triples. Positions are returned in
    file order and multiplied by positions_scale.
    """
    return _C.load_dot_data_model(file_name, positions_scale)
