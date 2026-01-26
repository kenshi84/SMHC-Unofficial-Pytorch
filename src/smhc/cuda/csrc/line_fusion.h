#ifndef LINE_FUSION_H_INCLUDED
#define LINE_FUSION_H_INCLUDED

#include <string>
#include <functional>

#include "auxiliary.h"
namespace solution
{

    namespace line_fusion
    {
        // alternatively if you are sure of certain parameters and you don't want to change them later
        // you can define them here as constexpr variables which gives you some additional possibilities
        // like doing some compile time optimizations such as unrolling for loops in cuda kernels
        // namespace params
        // {
        //     constexpr float position_update_threshold{ 0.00002f };
        //     constexpr float neighbors_search_radius{ 0.002f };
        //     constexpr float sigma_pos{ 0.0002f }; // 0.0001mm relative to radius of each strand
        //     constexpr float sigma_dir{ M_PI / 6.0f };
        //     constexpr float sigma_dir_pow2{ sigma_dir * sigma_dir }; 
        // }

        int run(
            const float* positions,
            const float* directions,
            const int points_num,
            //  algorithm parameters
            const float position_update_threshold,
            const float search_radius,
            const float sigma_pos,
            const float sigma_dir,
            //
            float* out_positions,
            float* out_directions,
            const bool fuse_on_gpu=true,
            const bool verbose=false
        );
        


    }

}

#endif