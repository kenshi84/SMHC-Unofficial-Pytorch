#include <omp.h>
#include <sstream>
#include <vector>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <cmath>
#include <cfloat>

#include <Eigen/Core>
#include <Eigen/Eigen>

// added for timing
#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>
#include <iomanip>

#include "multiview_hair_growing.h"
#include "point_cloud_structs.h"
#include "device_dataframe.h"

namespace solution::multiview_hair_growing
{

inline float calc_strand_length(
    const std::vector<pc_structs::plain>& strand
)
{
    if ((int)strand.size() < 2)
        return 0.f;
    float length = 0.f;
    for (int i = 1; i < strand.size(); ++i)
    {
        float4 pos = strand[i].position;
        float4 pos_prev = strand[i - 1].position;
        sub4(pos, pos_prev);
        length += get_normf3(pos);
    }
    return length;
}

inline bool does_window_contain(
    const std::vector<int2> window_pixels,
    const int2 pixel_to_check
)
{
    for (auto& p : window_pixels)
    {
        if (p.x == pixel_to_check.x && p.y == pixel_to_check.y)
            return true;
    }
    return false;
}

void validate_directional_window(
    const float3& tip_pixel_coord,
    const float& search_angle,
    const bool& search_downwards,
    const float* orientation_image,
    const float* confidence_image,
    const int imgs_rows,
    const int imgs_cols,
    //
    const float search_cone_window_width,
    const float search_cone_window_height,
    const float orientation_2d_confidence_threshold,
    const float orientation_2d_angle_difference_threshold_rad,
    const int valid_pixels_number_threshold,
    // 
    float& window_score,
    bool& is_valid
)
{
    
    float2 sample_aux;
    int2 sample;
    float downwards = search_downwards ? 1.f : -1.f;
    float search_angle_perp = search_angle + M_PI_2;

    int valid_pixels_num = 0;

    // constructing the search window
    for (int i = 0; i < search_cone_window_height; ++i)
    {
        
        sample_aux.x = tip_pixel_coord.x + i * std::cos(search_angle) * downwards;
        sample_aux.y = tip_pixel_coord.y + i * std::sin(search_angle) * downwards;

        for (int j = -search_cone_window_width / 2; j < search_cone_window_width / 2; ++j)
        {
            sample.x = int(std::floor(sample_aux.x + j * std::cos(search_angle_perp)));
            sample.y = int(std::floor(sample_aux.y + j * std::sin(search_angle_perp)));

            if (sample.y < 0 || sample.y >= imgs_rows || sample.x < 0 || sample.x >= imgs_cols)
            {
                continue;
            }

            int p_index = sample.y * imgs_cols + sample.x;
            float image_2d_ori_angle = orientation_image[p_index];
            float image_2d_conf = confidence_image[p_index];

            if (image_2d_conf < orientation_2d_confidence_threshold)
                continue;

            float angle_difference = std::abs(M_PI_2 - std::abs(std::abs(search_angle - image_2d_ori_angle) - M_PI_2));

            if (angle_difference > orientation_2d_angle_difference_threshold_rad)
                continue;

            valid_pixels_num++;
            window_score += angle_difference;
        }
        
    }

    is_valid = valid_pixels_num > valid_pixels_number_threshold;
    window_score /= valid_pixels_num;

}

inline void calc_cam_center_2d_dir_plane_normal(
    const host_dataframe* hdf,
    const int cam_idx,
    const float4 tip_position,
    const float& angle_2d,
    float3& plane_normal
)
{
    float3 pixel_coord;
    matvecmul4_hom(hdf->P_cpu[cam_idx], tip_position, pixel_coord);
    pixel_coord.x /= pixel_coord.z;
    pixel_coord.y /= pixel_coord.z;

    float3 next_pixel_coord;
    next_pixel_coord.x = pixel_coord.x + std::cos(angle_2d);
    next_pixel_coord.y = pixel_coord.y + std::sin(angle_2d);

    float4 pt;
    float3 pt_w;
    pt.x = hdf->fx[cam_idx] * next_pixel_coord.x - hdf->P_col34[cam_idx].x;
    pt.y = hdf->fx[cam_idx] * next_pixel_coord.y - hdf->P_col34[cam_idx].y;
    pt.z = hdf->fx[cam_idx] - hdf->P_col34[cam_idx].z;
    matvecmul4(hdf->M_inv_cpu[cam_idx], pt, pt_w);

    sub4(pt_w, tip_position);
    float3 aux;
    aux.x = tip_position.x - hdf->c4[cam_idx].x;
    aux.y = tip_position.y - hdf->c4[cam_idx].y;
    aux.z = tip_position.z - hdf->c4[cam_idx].z;

    cross_product(aux, pt_w, plane_normal);
}



inline Eigen::Vector3d get_smallest_eigen_vector_inverse_iter(Eigen::Matrix3d A)
{
    const Eigen::Matrix3d A_inv = A.inverse();
    // handle the singular case
    if (!A_inv.allFinite())
    {
        return Eigen::Vector3d::Zero();
    }

    // normalized initial guess
    Eigen::Vector3d x(1., 1., 1.);
    x.normalize();

    const double eps = 1e-10; 
    const int iter_max = 100;

    for (int iter = 0; iter < iter_max; ++iter)
    {
        const Eigen::Vector3d x_prev = x;

        // inverse iteration
        x = A_inv * x;
        x.normalize();

        // check if vectors are aligned
        if (std::abs(x.dot(x_prev)) > (1.0 - eps)) 
        {
            break;
        }
    }
    
    return x;
}


void find_valid_3d_growing_direction(
    const host_dataframe* hdf,
    const int total_views_num,
    std::vector<int>& valid_views_cam_idx,
    const float4& tip_position,
    const float4& next_to_tip_position,
    const float3& reverse_dir,
    const float3& reverse_momentum,
    //
    const float search_cone_opening_angle_rad,
    const float search_cone_angular_resolution_rad,
    const float search_cone_window_width,
    const float search_cone_window_height,
    const float orientation_2d_confidence_threshold,
    const float orientation_2d_angle_difference_threshold_rad,
    const int valid_pixels_number_threshold,
    const int minimum_views_with_valid_2d_direction,
    //
    float4& growing_dir,
    bool& found_dir,
    // for debug
    const int strand_id
)
{
    const bool already_picked_valid_views = !valid_views_cam_idx.empty();
    std::vector<int>::iterator cam_idx_pos_in_already_picked_valid_views;
    std::vector<float> valid_views_2d_angle;

    // loop over the camera views
    for (int cam_idx = 0; cam_idx < total_views_num; ++cam_idx)
    {
        if (already_picked_valid_views)
        {
            // cam index is not in the valid views
            cam_idx_pos_in_already_picked_valid_views = std::find(valid_views_cam_idx.begin(), valid_views_cam_idx.end(), cam_idx);
            if (cam_idx_pos_in_already_picked_valid_views == valid_views_cam_idx.end())
            {
                continue;
            }
        }

        float3 pixel_coord;
        matvecmul4_hom(hdf->P_cpu[cam_idx], tip_position, pixel_coord);
        pixel_coord.x /= pixel_coord.z;
        pixel_coord.y /= pixel_coord.z;

        float3 next_pixel_coord;
        matvecmul4_hom(hdf->P_cpu[cam_idx], next_to_tip_position, next_pixel_coord);
        next_pixel_coord.x /= next_pixel_coord.z;
        next_pixel_coord.y /= next_pixel_coord.z;

        bool downward_dir = next_pixel_coord.x < pixel_coord.x;

        // project 3d line to 2d
        // 2D line parameters a,b of ax + by + c = 0
        float2 line_2d;
        // const float* line_proj_mat = camera.line_projection_matrix.val;
        plucker_line_project_to_2d(reverse_dir,
            reverse_momentum,
            hdf->line_proj_mat_cpu[cam_idx],
            line_2d);

        // perform the conic search around the strand orientation 
        float angle = atanf(-line_2d.x / line_2d.y);

        std::vector<float> scores;
        float score = 0.f;
        bool is_valid = false;
        
        // performing angular search in a 2d cone
        float min_search_angle = angle - search_cone_opening_angle_rad;
        float max_search_angle = angle + search_cone_opening_angle_rad;
        for (float search_angle = min_search_angle;
            search_angle < max_search_angle + search_cone_angular_resolution_rad;
            search_angle += search_cone_angular_resolution_rad)
        {

            // constructing the window for each searching angle and compute its score
            validate_directional_window(
                pixel_coord,
                search_angle,
                downward_dir,
                hdf->imgs_ori_cpu[cam_idx],
                hdf->imgs_conf_cpu[cam_idx],
                hdf->rows,
                hdf->cols,
                //
                search_cone_window_width,
                search_cone_window_height,
                orientation_2d_confidence_threshold,
                orientation_2d_angle_difference_threshold_rad,
                valid_pixels_number_threshold,
                // 
                score,
                is_valid
            );

            if (!is_valid)
                score = -1;
            scores.push_back(score);

            score = 0.f;
            is_valid = false;

        }

        // pick the search angle with lowest difference as the 2d direction for the view
        float min_score = FLT_MAX;
        int min_score_angle_idx = -1;
        for (int s = 0; s < scores.size(); ++s)
        {
            if (scores[s] < 0.f)
                continue;
            if (scores[s] < min_score)
            {
                min_score = scores[s];
                min_score_angle_idx = s;
            }
        }

        if (min_score_angle_idx > 0)
        {
            float min_score_angle = min_search_angle + min_score_angle_idx * search_cone_angular_resolution_rad;
            valid_views_2d_angle.push_back(min_score_angle);

            if (!already_picked_valid_views)
            {
                valid_views_cam_idx.push_back(cam_idx);
            }
        }
        else if (already_picked_valid_views)
        {
            valid_views_cam_idx.erase(cam_idx_pos_in_already_picked_valid_views);
        }

    } // loop over all views


    // if number of valid views is beneath a threshold return 
    if (valid_views_cam_idx.size() < minimum_views_with_valid_2d_direction)
    {
        found_dir = false; growing_dir.x = 0.f; growing_dir.y = 0.f; growing_dir.z = 0.f; return;
    }

    // perform the svd for the views with valid direction
    Eigen::MatrixX3d A(valid_views_cam_idx.size(), 3);
    int valid_views_num = 0;
    for (int v = 0; v < valid_views_cam_idx.size(); ++v)
    {
        // getting normal of the plane constructed from the camera center and the 2d angle
        int valid_view_cam_idx = valid_views_cam_idx[v];
        float3 plane_normal;
        calc_cam_center_2d_dir_plane_normal(
            hdf,
            valid_view_cam_idx,
            tip_position,
            valid_views_2d_angle[v],
            plane_normal
        );
        A(valid_views_num, 0) = plane_normal.x;
        A(valid_views_num, 1) = plane_normal.y;
        A(valid_views_num, 2) = plane_normal.z;
        valid_views_num++;
    }

    // do noise removal iterations
    // first iteration
    Eigen::Matrix3d AtA = A.transpose() * A;    // AtA: 3 x 3 matrix
    Eigen::Vector3d x = get_smallest_eigen_vector_inverse_iter(AtA);
    if (x == Eigen::Vector3d(0., 0., 0.))
    {
        found_dir = false; growing_dir.x = 0.f; growing_dir.y = 0.f; growing_dir.z = 0.f; return;
    }
    // watch out for nan values
    if (x(0) != x(0) || x(1) != x(1) || x(2) != x(2))
    {
        found_dir = false; growing_dir.x = 0.f; growing_dir.y = 0.f; growing_dir.z = 0.f; return;
    }
    // compute weights squared inverse
    Eigen::VectorXd weights_si = (A * x).array().square().inverse();
    // apply row-wise weighting
    A = A.array().colwise() * weights_si.array();
    // normalize by the absolute maximum for stability
    A /= A.array().abs().maxCoeff();

    // second iteration
    AtA = A.transpose() * A;
    x = get_smallest_eigen_vector_inverse_iter(AtA);
    if (x == Eigen::Vector3d(0., 0., 0.))
    {
        found_dir = false; growing_dir.x = 0.f; growing_dir.y = 0.f; growing_dir.z = 0.f; return;
    }
    // watch out for nan values
    if (x(0) != x(0) || x(1) != x(1) || x(2) != x(2))
    {
        found_dir = false; growing_dir.x = 0.f; growing_dir.y = 0.f; growing_dir.z = 0.f; return;
    }
    growing_dir.x = x(0);
    growing_dir.y = x(1);
    growing_dir.z = x(2);
    found_dir = true;
    return;

}


void check_strand_tip(
    const bool first_tip,
    std::vector<pc_structs::plain>& strand,
    host_dataframe* hdf,
    const int total_views_num,
    const int strand_idx,
    //
    const int reset_valid_views_period,
    const int max_iter_num,
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
    const float background_color_threshold
)
{

    float4 growing_dir;
    bool valid_growing_dir = true;
    std::vector<int> valid_views_cam_idx;
    int iters_num = 0;

    int tip_idx = 0;
    int one_before_tip_idx = 1;

    while (valid_growing_dir)
    {
        if (iters_num > max_iter_num)
            break;

        if (iters_num % reset_valid_views_period == 0)
            valid_views_cam_idx.clear();


        if (!first_tip)
        {
            tip_idx = strand.size() - 1;
            one_before_tip_idx = tip_idx - 1;
        }

        float3 line_direction;
        float3 aux;

        subout(strand[tip_idx].position, strand[one_before_tip_idx].position, aux);
        set_equal_vec3(line_direction, strand[tip_idx].direction);

        float auxdot = dot_product(aux, line_direction);
        if (auxdot < 0)
        {
            line_direction.x *= -1;
            line_direction.y *= -1;
            line_direction.z *= -1;
        }

        float3 line_momentum;
        cross_product(strand[tip_idx].position, line_direction, line_momentum);
        // plucker_line3(strand[end_idx].position, strand[end_idx-1].position, line_direction, line_momentum);

        float3 reverse_dir;
        float3 reverse_momentum;
        plucker_line_reverse(line_direction,
            line_momentum,
            reverse_dir,
            reverse_momentum);
        
        find_valid_3d_growing_direction(
            hdf,
            total_views_num,
            valid_views_cam_idx,
            strand[tip_idx].position,
            strand[one_before_tip_idx].position,
            reverse_dir,
            reverse_momentum,
            //
            search_cone_opening_angle_rad,
            search_cone_angular_resolution_rad,
            search_cone_window_width,
            search_cone_window_height,
            orientation_2d_confidence_threshold,
            orientation_2d_angle_difference_threshold_rad,
            valid_pixels_number_threshold,
            minimum_views_with_valid_2d_direction,
            //
            growing_dir,
            valid_growing_dir,
            strand_idx
        );

        if (valid_growing_dir)
        {
            // check whether the growing direction exhibits sudden changes
            auxdot = dot_product(growing_dir, line_direction);
            if (auxdot < 0)
            {
                growing_dir.x *= -1;
                growing_dir.y *= -1;
                growing_dir.z *= -1;
            }
            float dir_diff = std::acos(dot_product(growing_dir, line_direction) / (get_normf3(growing_dir) * get_normf3(line_direction)));
            // in case growing_dir and line_direction are completely aligned acos() can generate nan
            if (dir_diff != dir_diff)
                dir_diff = 0.f;
            if (dir_diff > direction_change_threshold_rad)
            {
                valid_growing_dir = false;
                break;
            }

            float4 new_tip = strand[tip_idx].position;
            new_tip.x += growing_dir.x * growing_step_size;
            new_tip.y += growing_dir.y * growing_step_size;
            new_tip.z += growing_dir.z * growing_step_size;

            // check if we ended up in background
            for (int cam_idx = 0; cam_idx < total_views_num; ++cam_idx)
            {

                float3 pixel_coord;
                matvecmul4_hom(hdf->P_cpu[cam_idx], new_tip, pixel_coord);

                pixel_coord.x /= pixel_coord.z;
                pixel_coord.y /= pixel_coord.z;

                if (pixel_coord.y < 0 || pixel_coord.y >= hdf->rows || pixel_coord.x < 0 || pixel_coord.x >= hdf->cols)
                {
                    // valid_growing_dir = false;
                    // break;
                    continue;
                }

                int pixel_index = (int)(std::floor(pixel_coord.y)) * hdf->cols + (int)(std::floor(pixel_coord.x));
                float pixel_gray_scale_value = hdf->imgs_gray_cpu[cam_idx][pixel_index];

                if (pixel_gray_scale_value < background_color_threshold)
                {
                    valid_growing_dir = false;
                    break;
                }
            }

            if (valid_growing_dir)
            {
                if (first_tip)
                {
                    strand.insert(strand.begin(), { growing_dir, new_tip });
                }
                else
                {
                    strand.push_back({ growing_dir, new_tip });
                }
            }


        }

        iters_num++;
        
    }

}


int run(
    const int total_views_num,
    const float* positions,
    const float* directions,
    const int* points_num_per_strand,
    const int strand_segments_num,
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
    std::function<float* (size_t)> out_positions_buffer,
    std::function<float* (size_t)> out_directions_buffer,
    std::function<float* (size_t)> out_points_per_strand_buffer,
    const bool only_remove_background,
    const bool verbose
)
{
    host_dataframe* hdf = get_hdf_instance();

    std::vector<std::vector<pc_structs::plain>> strands;
    strands.resize(strand_segments_num);

    int point_acc = 0;
    for (int s_l = 0; s_l < strand_segments_num; ++s_l)
    {
        int points_num_in_strand = points_num_per_strand[s_l];

        for (int s_p = 0; s_p < points_num_in_strand; ++s_p)
        {
            float4 position = {positions[4*point_acc], positions[4*point_acc+1], positions[4*point_acc+2], 0.f};
            float4 direction = {directions[4*point_acc], directions[4*point_acc+1], directions[4*point_acc+2], 0.f};
            strands[s_l].push_back({direction, position});
            point_acc++;
        }
    }

    int processed_strands_num = 0;

    // for test and debug
    std::vector<float> strands_length;
    std::vector<int> strands_points_num;

    // start timing
    auto start = std::chrono::high_resolution_clock::now();

    if (!only_remove_background)
    {

    printf("Multi-view hair growing ...\n");

#pragma omp parallel for num_threads(OMP_NUM_THREADS)
    for (int i = 0; i < strands.size(); ++i)
    {
        
        std::vector<pc_structs::plain>& strand = strands[i];
        
        if (strand.size() < 2) {
            printf("very small strand segment with less than 2 vertices encountered at index %d\n", i);
            continue;
        }
        
        // first tip of the strand
        check_strand_tip(
            true,
            strand,
            hdf,
            total_views_num,
            i,
            reset_valid_views_period,
            max_iter_num,
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
            background_color_threshold
        );

        // second tip of the strand
        check_strand_tip(
            false,
            strand,
            hdf,
            total_views_num,
            i,
            reset_valid_views_period,
            max_iter_num,
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
            background_color_threshold
        );

#pragma omp critical
        {
            processed_strands_num++;
            if (processed_strands_num % 10 == 0)
                printf("\rProcess... %d / %zd (%.2f%%)", processed_strands_num, strands.size(), 100.f * processed_strands_num / strands.size());
        }

    } // loop over strands

        printf("\n");
    }

    // final touch, check and delete the strands that lie on the background
    printf("Final background consistency check ...\n");
    processed_strands_num = 0;

    std::vector<std::vector<pc_structs::plain>> final_strands;

#pragma omp parallel for num_threads(OMP_NUM_THREADS)
    for (int s_idx = 0; s_idx < strands.size(); ++s_idx)
    {
        std::vector<pc_structs::plain>& strand = strands[s_idx];
        bool add_to_final_strands = true;
        int background_points_num = 0;

        for (int v_idx = 0; v_idx < strand.size(); ++v_idx)
        {
            for (int cam_idx = 0; cam_idx < total_views_num; ++cam_idx)
            {

                float3 pixel_coord;
                matvecmul4_hom(hdf->P_cpu[cam_idx], strand[v_idx].position, pixel_coord);

                pixel_coord.x /= pixel_coord.z;
                pixel_coord.y /= pixel_coord.z;

                if (pixel_coord.y < 0 || pixel_coord.y >= hdf->rows || pixel_coord.x < 0 || pixel_coord.x >= hdf->cols)
                {
                    // background_points_num++;
                    // break;
                    continue;
                }

                int pixel_index = (int)(std::floor(pixel_coord.y)) * hdf->cols + (int)(std::floor(pixel_coord.x));
                float pixel_gray_scale_value = hdf->imgs_gray_cpu[cam_idx][pixel_index];

                if (pixel_gray_scale_value < background_color_threshold)
                {
                    background_points_num++;
                    break;
                }
                
            } // loop over all camera views

        } // loop over all strand vertices

        add_to_final_strands = ((float)background_points_num / strand.size()) < background_ratio;
        if (add_to_final_strands)
        {
#pragma omp critical
            {
                final_strands.push_back(strand);
                float strand_length = calc_strand_length(strand);
                strands_length.push_back(strand_length);
                strands_points_num.push_back(strand.size());
            }
        }
                
#pragma omp critical
        {
            processed_strands_num++;
            if (processed_strands_num % 10 == 0)
                printf("\rProcess... %d / %zd (%.2f%%)", processed_strands_num, strands.size(), 100.f * processed_strands_num / strands.size());
        }
    } // loop over strands

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (verbose)
    {
        printf("Ellapsed time: %16.10f ms\n", static_cast<float>(duration.count()));
    }

    printf("\n");

    float max_length = -1.f;
    float length_sum = 0.f;
    int total_points = 0;
    for (auto& l : strands_length)
    {
        length_sum += l;
        if (l > max_length)
            max_length = l;

    }
    for (auto& pn : strands_points_num)
    {
        total_points += pn;
    }

    //
    // sending the data to the specified output tensors
    float* output_positions = out_positions_buffer(4 * total_points);
    float* output_directions = out_directions_buffer(4 * total_points);
    float* output_points_per_strand = out_points_per_strand_buffer(final_strands.size());

    point_acc = 0;
    int strand_acc = 0;
    // loop over strands
    for (auto& s : final_strands)
    {
        // loop over vertices in each strand
        for (auto& p : s)
        {
            output_positions[4*point_acc] =     p.position.x;
            output_positions[4*point_acc+1] =   p.position.y;
            output_positions[4*point_acc+2] =   p.position.z;

            output_directions[4*point_acc] =     p.direction.x;
            output_directions[4*point_acc+1] =   p.direction.y;
            output_directions[4*point_acc+2] =   p.direction.z;

            point_acc++;

        }
        output_points_per_strand[strand_acc] = s.size();
        strand_acc++;
    }

    if (verbose)
    {
        printf("total number of segments is: %zd\n", strands.size());
        printf("total number of strands in the forground: %zd\n", final_strands.size());
        printf("total number of segments lengths is: %zd\n", strands_length.size());
        printf("maximum of the segment lengths is : %f\n", max_length);
        printf("average of the segment lengths is : %f\n", length_sum / (float)strands.size());
        printf("total number of vertices is: %d\n", total_points);
    }
    

    return 0;

} // run_multiview_hair_growing funciton



} // namespace solution::multiview_hair_growing
