#ifndef STRAND_SEGMENT_GENERATION_H_INCLUDED
#define STRAND_SEGMENT_GENERATION_H_INCLUDED

#include <string>
#include <functional>

#include "auxiliary.h"
namespace solution
{

    namespace strand_segment_generation
    {
        // alternatively if you are sure of certain parameters and you don't want to change them later
        // you can define them here as constexpr variables which gives you some additional possibilities
        // like doing some compile time optimizations such as unrolling for loops in cuda kernels
        // namespace params
        // {
        //     constexpr float strand_generation_step_size{ 0.0002f }; // 0.0001f
        //     constexpr float strand_generation_length_threshold{ 0.002f }; // 2mm
        //     constexpr float strand_generation_neighboring_radius{ 0.0002f };
        //     constexpr float strand_generation_angle_difference_threshold{ 30.f };
        //     constexpr float strand_generation_angle_difference_threshold_rad{ strand_generation_angle_difference_threshold * M_PI / 180.0f };
        // }

        int run(
            const float* positions,
            const float* directions,
            const int points_num,
            //  algorithm parameters
            const float strand_generation_step_size,
            const float strand_generation_length_threshold,
            const float strand_generation_neighboring_radius,
            const float strand_generation_angle_difference_threshold_rad,
            //
            std::function<float* (size_t)> out_positions_buffer,
            std::function<float* (size_t)> out_directions_buffer,
            std::function<float* (size_t)> out_points_per_strand_buffer,
            const bool verbose=false
        );
        


    }

}

#endif