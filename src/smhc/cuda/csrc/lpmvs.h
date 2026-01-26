#ifndef LPMVS_H_INCLUDED
#define LPMVS_H_INCLUDED


#include <string>

#include <cuda_runtime.h>

namespace solution
{

    namespace lpmvs
    {

        // alternatively if you are sure of certain parameters and you don't want to change them later
        // you can define them here as constexpr variables which gives you some additional possibilities
        // like doing some compile time optimizations such as unrolling for loops in cuda kernels
        // namespace params
        // {
        //     constexpr float alpha{ 0.1f }; // cost function color and orientation term weighting 
		// 						           // cost = (1 - alpha) * cost_orientation + alpha * cost_color
        //     constexpr int iterations{ 8 }; // total number of patchmatch iterations
        //     constexpr float minimum_depth{ 0.4f }; // minimum pixel depth when initalizing lines at random
        //     constexpr float maximum_depth{ 1.4f }; // maximum pixel depth when initalizing lines at random
        //    
        //     constexpr float sampling_radius{ 10.f };
        //     constexpr int half_samples_num{ 20 };
        // }

        int run(
            // lpmvs algorithm parameters
            const float alpha,
            const int iterations,
            const float minimum_depth,
            const float maximum_depth,
            const float sampling_radius,
            const int samples_num,
            const float occulusion_depth_thresh,
            // some additional control over the perturbations in the line refinement steps
            const float delta_depth_amp,
            const float deleta_dir_amp,
            // reference view index and other source neighboring views indices
            const int ref_view_id,
            const int view_selection_subset_number,
            const int* view_selection_subset_cpu,
            float* final_positions,
            float* final_directions,
            // also if you have a better guess (than random values) for initial position and direction values
            const float* initial_depths=nullptr,
            const float* initial_directions=nullptr,
            const bool verbose=false
        );


    }
}

#endif
