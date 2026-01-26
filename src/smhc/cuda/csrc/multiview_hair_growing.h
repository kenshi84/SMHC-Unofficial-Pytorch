#ifndef MULTIVIEW_HAIR_GROWING_H_INCLUDED
#define MULTIVIEW_HAIR_GROWING_H_INCLUDED

#include <string>
#include <functional>

#include "auxiliary.h"

namespace solution
{
    namespace multiview_hair_growing
    {

        // namespace params
        // {
        //     // this variable specifies the priod which the neighboring views with valid growing 
        //     // direction are selected, you can set it to 1, but it will result in a longer
        //     // runtime for the algorithm.
        //     constexpr int reset_valid_views_period = 20;  // for straight 100, for curly 10
        //     constexpr int max_iter_num = 1000;
        //     constexpr float background_ratio = 0.9f;
        //     constexpr float search_cone_opening_angle{ 5.f };
        //     constexpr float search_cone_opening_angle_rad{ search_cone_opening_angle * M_PI / 180.0f };
        //     constexpr float search_cone_angular_resolution{ 1.f };
        //     constexpr float search_cone_angular_resolution_rad{ search_cone_angular_resolution * M_PI / 180.f };
        //     constexpr float search_cone_window_width{ 3.f };
        //     constexpr float search_cone_window_height{ 10.f };
        //     constexpr float orientation_2d_confidence_threshold{ 3.f };
        //     constexpr float orientation_2d_angle_difference_threshold{ 5.f };
        //     constexpr float orientation_2d_angle_difference_threshold_rad{ orientation_2d_angle_difference_threshold * M_PI / 180.f };
        //     constexpr int valid_pixels_number_threshold{ 10 };
        //     constexpr int minimum_views_with_valid_2d_direction{ 6 };
        //     constexpr float growing_step_size{ 0.0003f }; // a bit more than length of a pixel which is 0.2mm
        //     constexpr float direction_change_threshold{ 45.f };
        //     constexpr float direction_change_threshold_rad{ direction_change_threshold * M_PI / 180.f };
        //     constexpr float background_color_threshold{ 3.f / 255.f };
        // }

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
            const bool only_remove_background=false,
            const bool verbose=false
        );
        
    }
}


#endif