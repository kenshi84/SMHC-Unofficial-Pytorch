
#include <torch/extension.h>
#include <ATen/ATen.h>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <tuple>
#include <functional>
#include <cstdlib>
#include <cmath>

#include "device_dataframe.h"
#include "orientation.h"
#include "lpmvs.h"
#include "line_filtering.h"
#include "line_fusion.h"
#include "strand_segment_generation.h"
#include "multiview_hair_growing.h"
#include "metrics.h"
#include "hair_models.h"

namespace solution
{

    std::function<float*(size_t N)> resize_functional_float(torch::Tensor& t) 
    {
	    auto lambda = [&t](size_t N) 
        {
	    	t.resize_({(long long)N});
	    	return reinterpret_cast<float*>(t.contiguous().data_ptr());
	    };

	    return lambda;
    }


    std::tuple<torch::Tensor, torch::Tensor>
    calculate_orientation_maps(
        const torch::Tensor& img,
        const torch::Tensor& kernels,
        const int kernels_num,
        const int kernels_size
    )
    {
        if (img.ndimension() != 2)
        {
            AT_ERROR("For calculating orientation maps, make sure that the input images have dimensions [img_height, img_width]!");
        }

        if (kernels.ndimension() != 1 || kernels.size(0) != (kernels_num * kernels_size * kernels_size))
        {
            AT_ERROR("For calculating orientation maps, all gabor kernels must be flattened and appended to form a single tensor!");
        }

        const int img_rows = img.size(0);
        const int img_cols = img.size(1);

        torch::Tensor orientation_img = torch::zeros({img_rows, img_cols}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
        torch::Tensor confidence_img = torch::zeros({img_rows, img_cols}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

        gabor::calculate_orientation_maps(
            img.contiguous().data<float>(),
            img_rows,
            img_cols,
            kernels.contiguous().data<float>(),
            kernels_num,
            kernels_size,
            orientation_img.contiguous().data<float>(),
            confidence_img.contiguous().data<float>()
        );

        return std::make_tuple(orientation_img, confidence_img);
    }


    void destroy_dataframe()
    {
        destroy_ddf_instance();
    }

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
    )
    {
        load_camera_parameters(
            cam_idx,
            img_height,
            img_width,
            cams_P_cpu.contiguous().data<float>(),
            (float4*)cams_P_col34_cpu.contiguous().data<float>(),
            cams_M_inv_cpu.contiguous().data<float>(),
            cams_K_cpu.contiguous().data<float>(),
            cams_K_inv_cpu.contiguous().data<float>(),
            cams_R_cpu.contiguous().data<float>(),
            (float4*)cams_t4_cpu.contiguous().data<float>(),
            cams_R_inv_cpu.contiguous().data<float>(),
            (float4*)cams_c4_cpu.contiguous().data<float>(),
            cams_line_proj_mat_cpu.contiguous().data<float>(),
            cams_fx_cpu,
            cams_fy_cpu
        );
    }

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
    )
    {
        if (img_gray.ndimension() != 2 || img_ori.ndimension() != 2 || img_conf.ndimension() != 2)
        {
            AT_ERROR("Make sure that the input intensity image, orientation and confidence map have dimensions [img_height, img_width]!");
        }

        if (img_gray.size(0) != img_height || img_gray.size(1) != img_width)
        {
            AT_ERROR("Make sure that the input intensity image, orientation and confidence map have dimensions [img_height, img_width]!");
        }

        if (img_ori.size(0) != img_height || img_ori.size(1) != img_width)
        {
            AT_ERROR("Make sure that the input intensity image, orientation and confidence map have dimensions [img_height, img_width]!");
        }

        if (img_conf.size(0) != img_height || img_conf.size(1) != img_width)
        {
            AT_ERROR("Make sure that the input intensity image, orientation and confidence map have dimensions [img_height, img_width]!");
        }

        if (img_mask.ndimension() > 1 && img_mask.ndimension() != 2)
        {
            AT_ERROR("Make sure that the provided mask have dimensions [img_height, img_width]!");
        }

        if (initial_depth_map.ndimension() > 1 && initial_depth_map.ndimension() != 2)
        {
            AT_ERROR("Make sure that the provided initial depth values have dimensions [img_height, img_width]!");
        }

        const float* img_mask_ptr = (img_mask.size(0)==0) ? nullptr : img_mask.contiguous().data<float>();
        const float* img_initial_depth_ptr = (initial_depth_map.size(0)==0) ? nullptr : initial_depth_map.contiguous().data<float>();

        load_img_data(
            img_idx,
            img_height,
            img_width,
            img_gray.contiguous().data<float>(),
            img_ori.contiguous().data<float>(),
            img_conf.contiguous().data<float>(),
            img_mask_ptr,
            img_initial_depth_ptr,
            load_to_cpu_containers
        );
    }


    void load_lpmvs_estimations_to_dataframe(
        const int img_idx,
        const torch::Tensor& estimated_positions,
        const torch::Tensor& estimated_directions
    )
    {
        if (estimated_positions.ndimension() != 2 || estimated_positions.size(1) != 4) 
        {
	        AT_ERROR("estimated positions tensor must have dimensions (image_height*image_width, 4)!");
        }

        if (estimated_directions.ndimension() != 2 || estimated_directions.size(1) != 4) 
        {
	        AT_ERROR("estimated directions tensor must have dimensions (image_height*image_width, 4)!");
        }

        if (estimated_positions.size(0) != estimated_directions.size(0))
        {
            AT_ERROR("estimated directions and positions tensor must have a same first dimension that corresponds to image_height*image_width!");

        }

        load_lpmvs_estimations(
            img_idx,
            estimated_positions.contiguous().data<float>(),
            estimated_directions.contiguous().data<float>()
        );
    }

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
        const bool verbose
    )
    {
        if (view_selection_subset_cpu.size(0) != view_selection_subset_number)
        {
            AT_ERROR("How is it that the number of neighboring views does not match with the length of their index list!");
        }

        if (initial_depths.ndimension() != 2 && initial_depths.ndimension() > 1)
        {
            AT_ERROR("Make sure that the provided initial depth values have dimensions [img_height, img_width]!");
        }

        if ((initial_directions.ndimension() == 3 && initial_directions.size(0) != 3) || (initial_directions.ndimension() != 3 && initial_directions.ndimension() > 1))
        {
            AT_ERROR("Make sure that the provided initial directions have dimensions [3, img_height, img_width]!");
        }

        torch::Tensor final_positions = torch::zeros({4, img_rows, img_cols}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
        torch::Tensor final_directions = torch::zeros({4, img_rows, img_cols}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

        const float* initial_depths_ptr = (initial_depths.size(0)==0) ? nullptr : initial_depths.contiguous().data<float>();
        const float* initial_directions_ptr = (initial_directions.size(0)==0) ? nullptr : initial_directions.contiguous().data<float>();
        
        lpmvs::run(
            alpha,
            iterations,
            minimum_depth,
            maximum_depth,
            sampling_radius,
            samples_num,
            occulusion_depth_thresh,
            delta_depth_amp,
            deleta_dir_amp,
            ref_view_id,
            view_selection_subset_number,
            view_selection_subset_cpu.contiguous().data<int>(),
            final_positions.contiguous().data<float>(),
            final_directions.contiguous().data<float>(),
            initial_depths_ptr,
            initial_directions_ptr,
            verbose
        );

        return std::make_tuple(final_positions, final_directions);
    }

    int run_line_filtering(
        const int ref_view_id,
        const int view_selection_subset_number,
        const torch::Tensor& view_selection_subset_cpu,
        torch::Tensor& int_map,
        const float position_consistency_threshold,
        const float direction_diff_threshold_rad,
        const int consistent_neighboring_views_threshold,
        const bool verbose
    )
    {
        if (view_selection_subset_cpu.size(0) != view_selection_subset_number)
        {
            AT_ERROR("How is it that the number of neighboring views does not match with the length of their index list!");
        }

        if (int_map.ndimension() != 3 || int_map.size(0) != 3)
        {
            AT_ERROR("make sure that the input intermediate map has dimension (3, img_height, img_width)!");
        }

        // torch::Tensor int_map = torch::zeros({3, img_rows, img_cols}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
        // torch::Tensor consistency_results = torch::zeros({img_rows, img_cols}, torch::dtype(torch::kUInt8).device(torch::kCUDA, 0));

        line_filtering::run(
            ref_view_id,
            view_selection_subset_number,
            view_selection_subset_cpu.contiguous().data<int>(),
            int_map.contiguous().data<float>(),
            position_consistency_threshold,
            direction_diff_threshold_rad,
            consistent_neighboring_views_threshold,
            verbose
        );

        return 0;
    }

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
        const bool& fuse_on_gpu,
        const bool verbose
    )
    {
        if (positions.ndimension() != 2 || positions.size(1) != 4) 
        {
	        AT_ERROR("positions tensor must have dimensions (num_points, 4)!");
        }

        if (directions.ndimension() != 2 || directions.size(1) != 4) 
        {
	        AT_ERROR("directions tensor must have dimensions (num_points, 4)!");
        }

        if (positions.size(0) != directions.size(0))
        {
            AT_ERROR("directions and positions tensor must have a same first dimension that corresponds to num_points!");

        }

        torch::Tensor positions_has_nan = torch::any(torch::isnan(positions)).to(torch::kByte);
        torch::Tensor directions_has_nan = torch::any(torch::isnan(directions)).to(torch::kByte);

        if (positions_has_nan.item<bool>())
        {
            AT_ERROR("provided positions tensor has nan values, trust me, you do not want to give nan values to the cukd builders!");
        }

        if (directions_has_nan.item<bool>())
        {
            AT_ERROR("provided directions tensor has nan values, this can cause problems later!");
        }

        const int points_num = positions.size(0);

        torch::Tensor out_positions = torch::zeros({points_num, 4}, torch::dtype(torch::kFloat32).device(torch::kCPU, 0));
        torch::Tensor out_directions = torch::zeros({points_num, 4}, torch::dtype(torch::kFloat32).device(torch::kCPU, 0));

        line_fusion::run(
            positions.contiguous().data<float>(),
            directions.contiguous().data<float>(),
            points_num,
            position_update_threshold,
            search_radius,
            sigma_pos,
            sigma_dir,
            // 
            out_positions.contiguous().data<float>(),
            out_directions.contiguous().data<float>(),
            fuse_on_gpu,
            verbose
        );

        return std::make_tuple(out_positions, out_directions);

    }

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
        const bool verbose
    )
    {
        if (positions.ndimension() != 2 || positions.size(1) != 4) 
        {
	        AT_ERROR("positions tensor must have dimensions (num_points, 4)!");
        }

        if (directions.ndimension() != 2 || directions.size(1) != 4) 
        {
	        AT_ERROR("directions tensor must have dimensions (num_points, 4)!");
        }

        if (positions.size(0) != directions.size(0))
        {
            AT_ERROR("directions and positions tensor must have a same first dimension that corresponds to num_points!");

        }

        torch::Tensor positions_has_nan = torch::any(torch::isnan(positions)).to(torch::kByte);
        torch::Tensor directions_has_nan = torch::any(torch::isnan(directions)).to(torch::kByte);

        if (positions_has_nan.item<bool>())
        {
            AT_ERROR("provided positions tensor has nan values, trust me, you do not want to give nan values to the cukd builders!");
        }

        if (directions_has_nan.item<bool>())
        {
            AT_ERROR("provided directions tensor has nan values, this can cause problems later!");
        }

        const int points_num = positions.size(0);

        torch::Tensor out_positions = torch::empty({0}, torch::dtype(torch::kFloat32));
        torch::Tensor out_directions = torch::empty({0}, torch::dtype(torch::kFloat32));
        torch::Tensor out_points_per_strand = torch::empty({0}, torch::dtype(torch::kFloat32));

        std::function<float*(size_t)> out_positions_buffer = resize_functional_float(out_positions);
        std::function<float*(size_t)> out_directions_buffer = resize_functional_float(out_directions);
        std::function<float*(size_t)> out_points_per_strand_buffer = resize_functional_float(out_points_per_strand);


        strand_segment_generation::run(
            positions.contiguous().data<float>(),
            directions.contiguous().data<float>(),
            points_num,
            strand_generation_step_size,
            strand_generation_length_threshold,
            strand_generation_neighboring_radius,
            strand_generation_angle_difference_threshold_rad,
            //
            out_positions_buffer,
            out_directions_buffer,
            out_points_per_strand_buffer,
            verbose
        );

        return std::make_tuple(out_positions.reshape({-1, 4}), out_directions.reshape({-1, 4}), out_points_per_strand);

    }


    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    run_multiview_hair_growing(
        const int total_views_num,
        const torch::Tensor& positions,
        const torch::Tensor& directions,
        const torch::Tensor& points_num_per_strand,
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
        const bool verbose
    )
    {
        if (total_views_num < 1)
        {
            AT_ERROR("how is it that total number of views is less than 1? make sure that you are providing a reasonable (!) number of input views for the multi-view hair growing step!");
        }

        if (positions.ndimension() != 2 || positions.size(1) != 4) 
        {
	        AT_ERROR("positions tensor must have dimensions (num_points, 4)!");
        }

        if (directions.ndimension() != 2 || directions.size(1) != 4) 
        {
	        AT_ERROR("directions tensor must have dimensions (num_points, 4)!");
        }

        if (positions.size(0) != directions.size(0))
        {
            AT_ERROR("directions and positions tensor must have a same first dimension that corresponds to num_points!");
        }

        const int strand_segments_num = points_num_per_strand.size(0);

        torch::Tensor out_positions = torch::empty({0}, torch::dtype(torch::kFloat32));
        torch::Tensor out_directions = torch::empty({0}, torch::dtype(torch::kFloat32));
        torch::Tensor out_points_per_strand = torch::empty({0}, torch::dtype(torch::kFloat32));

        std::function<float*(size_t)> out_positions_buffer = resize_functional_float(out_positions);
        std::function<float*(size_t)> out_directions_buffer = resize_functional_float(out_directions);
        std::function<float*(size_t)> out_points_per_strand_buffer = resize_functional_float(out_points_per_strand);

        multiview_hair_growing::run(
            total_views_num,
            positions.contiguous().data<float>(),
            directions.contiguous().data<float>(),
            points_num_per_strand.contiguous().data<int>(),
            strand_segments_num,
            reset_valid_views_period,
            max_iter_num,
            background_ratio,
            search_cone_opening_angle_rad,
            search_cone_angular_resolution_rad,
            search_cone_window_width,
            search_cone_window_height,
            orientation_2d_confidence_threshold,
            orientation_2d_angle_difference_threshold_rad,
            valid_pixels_number_threshold,
            minimum_views_with_valid_2d_direction,
            growing_step_size,
            direction_change_threshold_rad,
            background_color_threshold,
            //
            out_positions_buffer,
            out_directions_buffer,
            out_points_per_strand_buffer,
            only_remove_background,
            verbose
        );

        return std::make_tuple(out_positions.reshape({-1, 4}), out_directions.reshape({-1, 4}), out_points_per_strand);
    }

   void calculate_hair_metrics(
        const torch::Tensor& position_thresholds,
        const torch::Tensor& direction_thresholds,
        const torch::Tensor& gt_positions,
        const torch::Tensor& gt_directions,
        const torch::Tensor& estimated_positions,
        const torch::Tensor& estimated_directions,
        const bool verbose
    )
    {
        if (position_thresholds.ndimension() != 1 || direction_thresholds.ndimension() != 1)
        {
            AT_ERROR("direction and position threshold tensor must have dimensions (thresholds_num,)!");
        }

        if (position_thresholds.size(0) != direction_thresholds.size(0))
        {
            AT_ERROR("you need to provide the same number of thresholds for position and direction!");
        }

        if (gt_positions.ndimension() != 2 || gt_positions.size(1) != 3) 
        {
	        AT_ERROR("gt positions tensor must have dimensions (num_points, 3)!");
        }

        if (gt_directions.ndimension() != 2 || gt_directions.size(1) != 3) 
        {
	        AT_ERROR("gt directions tensor must have dimensions (num_points, 3)!");
        }

        if (gt_positions.size(0) != gt_directions.size(0))
        {
            AT_ERROR("gt directions and positions tensor must have a same first dimension that corresponds to gt num_points!");

        }

        if (estimated_positions.ndimension() != 2 || estimated_positions.size(1) != 3) 
        {
	        AT_ERROR("estimated positions tensor must have dimensions (num_points, 3)!");
        }

        if (estimated_directions.ndimension() != 2 || estimated_directions.size(1) != 3) 
        {
	        AT_ERROR("estimated directions tensor must have dimensions (num_points, 3)!");
        }

        if (estimated_positions.size(0) != estimated_directions.size(0))
        {
            AT_ERROR("estimated directions and positions tensor must have a same first dimension that corresponds to estimated num_points!");

        }

        const int thresholds_num = position_thresholds.size(0);
        const int gt_points_num = gt_positions.size(0);
        const int estimated_points_num = estimated_positions.size(0);

        metrics::run(
            position_thresholds.contiguous().data<float>(),
            direction_thresholds.contiguous().data<float>(),
            thresholds_num,
            gt_positions.contiguous().data<float>(),
            gt_directions.contiguous().data<float>(),
            gt_points_num,
            estimated_positions.contiguous().data<float>(),
            estimated_directions.contiguous().data<float>(),
            estimated_points_num,
            verbose
        );

    }

    

    std::tuple<torch::Tensor, torch::Tensor>
    load_dot_hair_model(
        const std::string& file_name,
        const float positions_scale
    )
    {
        if (std::filesystem::is_directory(file_name) || !std::filesystem::exists(file_name))
        {
        	AT_ERROR("could not find the given .hair model file!");
        }
        
        torch::Tensor positions = torch::empty({0}, torch::dtype(torch::kFloat32));
        torch::Tensor points_per_strand = torch::empty({0}, torch::dtype(torch::kFloat32));

        std::function<float*(size_t)> positions_buffer = resize_functional_float(positions);
        std::function<float*(size_t)> points_per_strand_buffer = resize_functional_float(points_per_strand);

        hair_models::load_dot_hair_model_file(
            file_name,
            positions_scale,
            positions_buffer,
            points_per_strand_buffer
        );

        return std::make_tuple(positions.reshape({-1, 3}), points_per_strand);
    }

    std::tuple<torch::Tensor, torch::Tensor>
    load_dot_bin_model(
        const std::string& file_name,
        const float positions_scale
    )
    {
        if (std::filesystem::is_directory(file_name) || !std::filesystem::exists(file_name))
        {
        	AT_ERROR("could not find the given .bin model file!");
        }
        
        torch::Tensor positions = torch::empty({0}, torch::dtype(torch::kFloat32));
        torch::Tensor points_per_strand = torch::empty({0}, torch::dtype(torch::kFloat32));

        std::function<float*(size_t)> positions_buffer = resize_functional_float(positions);
        std::function<float*(size_t)> points_per_strand_buffer = resize_functional_float(points_per_strand);

        hair_models::load_dot_bin_model_file(
            file_name,
            positions_scale,
            positions_buffer,
            points_per_strand_buffer
        );

        return std::make_tuple(positions.reshape({-1, 3}), points_per_strand);
    }

}