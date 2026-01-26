class LPMVSHyperParameters:
    def __init__(self, args) -> None:
        self.alpha = args.lpmvs_alpha
        self.iterations = args.lpmvs_iterations
        self.minimum_depth = args.lpmvs_minimum_depth
        self.maximum_depth = args.lpmvs_maximum_depth
        self.sampling_radius = args.lpmvs_sampling_radius
        self.samples_num = args.lpmvs_samples_num
        self.occlusion_depth_thresh = args.lpmvs_occlusion_depth_thresh


class LineFilteringHyperParameters:
    def __init__(self, args) -> None:
        self.position_consistency_thres = args.line_filtering_position_consistency_thres
        self.direction_diff_threshold_rad = (
            args.line_filtering_direction_diff_threshold_rad
        )
        self.consistent_neighboring_views_threshold = (
            args.line_filtering_consistent_neighboring_views_threshold
        )


class LineFusionHyperParameters:
    def __init__(self, args) -> None:
        self.position_update_threshold = args.line_fusion_position_update_threshold
        self.search_radius = args.line_fusion_search_radius
        self.sigma_pos = args.line_fusion_sigma_pos
        self.sigma_dir = args.line_fusion_sigma_dir


class StrandSegmentsGenerationHyperParametrs:
    def __init__(self, args) -> None:
        self.step_size = args.strand_generation_step_size
        self.length_threshold = args.strand_generation_length_threshold
        self.neighboring_radius = args.strand_generation_neighboring_radius
        self.angle_difference_threshold_rad = (
            args.strand_generation_angle_difference_threshold_rad
        )


class MultiviewHairGrowingHyperParameters:
    def __init__(self, args) -> None:
        self.reset_valid_views_period = args.mvhg_reset_valid_views_period
        self.max_iter_num = args.mvhg_max_iter_num
        self.background_ratio = args.mvhg_background_ratio
        self.search_cone_opening_angle_rad = args.mvhg_search_cone_opening_angle_rad
        self.search_cone_angular_resolution_rad = (
            args.mvhg_search_cone_angular_resolution_rad
        )
        self.search_cone_window_width = args.mvhg_search_cone_window_width
        self.search_cone_window_height = args.mvhg_search_cone_window_height
        self.orientation_2d_confidence_threshold = (
            args.mvhg_orientation_2d_confidence_threshold
        )
        self.orientation_2d_angle_difference_threshold_rad = (
            args.mvhg_orientation_2d_angle_difference_threshold_rad
        )
        self.valid_pixels_number_threshold = args.mvhg_valid_pixels_number_threshold
        self.minimum_views_with_valid_2d_direction = (
            args.mvhg_minimum_views_with_valid_2d_direction
        )
        self.growing_step_size = args.mvhg_growing_step_size
        self.direction_change_threshold_rad = args.mvhg_direction_change_threshold_rad
        self.background_color_threshold = args.mvhg_background_color_threshold
