#ifndef METRICS_H_INCLUDED
#define METRICS_H_INCLUDED


namespace solution
{
    namespace metrics
    {

        // namespace params
        // {
        //     // constexpr int metric_thresholds_number = 4;
        //     // constexpr float metric_position_difference_thresholds[] = { 0.0002f, 0.0005f, 0.001f, 0.002f };
        //     // constexpr float metric_direction_difference_thresholds[] = { 2.f * M_PI / 180.0f, 
        //     //                                                             5.f * M_PI / 180.0f, 
        //     //                                                             10.f * M_PI / 180.0f, 
        //     //                                                             20.f * M_PI / 180.0f };
        // }

        int run(
            const float* position_thresholds,
            const float* direction_thresholds,
            const int thresholds_num,
            const float* gt_positions,
            const float* gt_directions,
            const int gt_particles_num,
            const float* estimated_positions,
            const float* estimated_directions,
            const int estimated_points_num,
            const bool verbose=false
        );

    }
}

#endif