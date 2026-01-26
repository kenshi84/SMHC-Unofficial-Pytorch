#ifndef LINE_FILTERING_H_INCLUDED
#define LINE_FILTERING_H_INCLUDED

#include <cuda_runtime.h>

#include "auxiliary.h"

namespace solution
{
    namespace line_filtering
    {

        // namespace params
        // {
	    //     constexpr int maximum_neighboring_views{ 20 };
	    //     constexpr float position_consistency_threshold{ 0.001f };
	    //     constexpr float direction_diff_threshold{ 10.f };
	    //     constexpr float direction_diff_threshold_rad{ direction_diff_threshold * M_PI / 180.0f };
	    //     constexpr int consistent_neighboring_views_threshold{ 2 };
        // }

        int run(
            const int ref_view_id,
            const int view_selection_subset_number,
            const int* view_selection_subset_cpu,
            float* int_map,
            // algorithm parameters
            const float position_consistency_threshold,
            const float direction_diff_threshold_rad,
            const int consistent_neighboring_views_threshold,
            //
            const bool verbose=false
        );
    }
}

#endif