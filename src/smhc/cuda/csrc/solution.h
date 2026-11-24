
#ifndef SOLUTION_H_INCLUDED
#define SOLUTION_H_INCLUDED

#include <torch/extension.h>
#include <tuple>


namespace solution
{

    std::tuple<torch::Tensor, torch::Tensor>
    calculate_orientation_maps(
        const torch::Tensor& img,
        const torch::Tensor& kernels,
        const int kernels_num,
        const int kernels_size
    );

    void destroy_dataframe();

    void load_camera_parameters_to_dataframe(
        const int cam_idx,
        const int img_height,
        const int img_width,
        const torch::Tensor& cams_P_cpu,
        const torch::Tensor& cams_P_col34_cpu,
        const torch::Tensor& cams_M_inv_cpu,
        const torch::Tensor& cams_K_cpu,
        const torch::Tensor& cams_K_inv_cpu,
        const torch::Tensor& cams_R_cpu,
        const torch::Tensor& cams_t4_cpu,
        const torch::Tensor& cams_R_inv_cpu,
        const torch::Tensor& cams_c4_cpu,
        const torch::Tensor& cams_line_proj_mat_cpu,
        const float cams_fx_cpu,
        const float cams_fy_cpu
    );

    void load_img_data_to_dataframe(
            const int img_idx,
            const int img_height,
            const int img_width,
            const torch::Tensor& img_gray,
            const torch::Tensor& img_ori,
            const torch::Tensor& img_conf,
            const torch::Tensor& img_mask,
            const torch::Tensor& initial_depth_map,
            const bool load_to_cpu_containers
    );

    void load_lpmvs_estimations_to_dataframe(
        const int img_idx,
        const torch::Tensor& estimated_positions,
        const torch::Tensor& estimated_direcitons
    );

    std::tuple<torch::Tensor, torch::Tensor>
    run_lpmvs(
        const int img_rows,
        const int img_cols,
        const float alpha,
        const int iterations,
        const float minimum_depth,
        const float maximum_depth,
        const float sampling_radius,
        const int samples_num,
        const float occulusion_depth_thresh,
        const float delta_depth_amp,
        const float deleta_dir_amp,
        const int ref_view_id,
        const int view_selection_subset_number,
        const torch::Tensor& view_selection_subset_cpu,
        const torch::Tensor& initial_depths,
        const torch::Tensor& initial_directions,
        const bool verbose=false
    );


    int run_line_filtering(
        const int ref_view_id,
        const int view_selection_subset_number,
        const torch::Tensor& view_selection_subset_cpu,
        torch::Tensor& int_map,
        const float position_consistency_threshold,
        const float direction_diff_threshold_rad,
        const int consistent_neighboring_views_threshold,
        const bool verbose=false
    );

    std::tuple<torch::Tensor, torch::Tensor>
    run_line_fusion(
        const torch::Tensor& positions,
        const torch::Tensor& directions,
        // algorithm parameters
        const float position_update_threshold,
        const float search_radius,
        const float sigma_pos,
        const float sigma_dir,
        //
        const bool& fuse_on_gpu=true,
        const bool verbose=false
    );

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    run_strand_segment_generation(
        const torch::Tensor& positions,
        const torch::Tensor& directions,
        // algorithm parameters
        const float strand_generation_step_size,
        const float strand_generation_length_threshold,
        const float strand_generation_neighboring_radius,
        const float strand_generation_angle_difference_threshold_rad,
        //
        const bool verbose=false
    );

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    run_multiview_hair_growing(
        const int total_views_num,
        const torch::Tensor& positions,
        const torch::Tensor& directions,
        const torch::Tensor& points_num_per_strand,
        // algorithm parameters
        const int reset_valid_views_period,
        const int max_iter_num,
        const float background_ratio,
        const float search_cone_opening_angle_rad,
        const float search_cone_angular_resolution_rad,
        const float search_cone_window_width,
        const float search_cone_window_height,
        const float orientation_2d_confidence_threshold,
        const float orientation_2d_angle_difference_threshold_rad,
        const int valid_pixels_number_threshold,
        const int minimum_views_with_valid_2d_direction,
        const float growing_step_size,
        const float direction_change_threshold_rad,
        const float background_color_threshold,
        //
        const bool only_remove_background,
        const bool verbose=false
    );

    void calculate_hair_metrics(
        const torch::Tensor& position_thresholds,
        const torch::Tensor& direction_thresholds,
        const torch::Tensor& gt_positions,
        const torch::Tensor& gt_directions,
        const torch::Tensor& estimated_positions,
        const torch::Tensor& estimated_directions,
        const bool verbose=false
    );

    std::tuple<torch::Tensor, torch::Tensor>
    load_dot_hair_model(
        const std::string& file_name,
        const float positions_scale
    );

    std::tuple<torch::Tensor, torch::Tensor>
    load_dot_bin_model(
        const std::string& file_name,
        const float positions_scale
    );

    std::tuple<torch::Tensor, torch::Tensor>
    load_dot_data_model(
        const std::string& file_name,
        const float positions_scale
    );

}

#endif