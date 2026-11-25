#ifndef METRICS_H_INCLUDED
#define METRICS_H_INCLUDED


namespace solution
{
    namespace metrics
    {

        // Position thresholds are in metres; direction thresholds are in degrees.
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