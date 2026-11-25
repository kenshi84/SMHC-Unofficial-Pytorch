#include <omp.h>
#include <sstream>
#include <vector>
#include <fstream>
#include <filesystem>
#include <cmath>

#include "auxiliary.h"
#include "metrics.h"
#include "point_cloud_structs.h"

namespace solution::metrics
{

void process_neighbor(
    const float3& ref_position,
    const float3& ref_direction,
    const float3& neighbor_position,
    float3& neighbor_direction,
    const float search_radius,
    const float angle_diff_thresh,
    bool& verified

)
{
    // check the distance to fall within the search radius
    float3 aux;
    subout(neighbor_position, ref_position, aux);
    const float sqr_dist = norm2(aux);
    if (sqrtf(sqr_dist) > search_radius)
        return;

    float dir_diff = acosf(dot_product(ref_direction, neighbor_direction) / (get_normf3(ref_direction) * get_normf3(neighbor_direction)));
    // in case neighbor_dir and ref_dir are completely aligned acosf() can generate nan
    if (dir_diff != dir_diff)
        dir_diff = 0.f;
    dir_diff = M_PI_2 - fabsf(dir_diff - M_PI_2);
    if (dir_diff > angle_diff_thresh)
        return;

    verified = true;
    return;
}


void traverse_tree(
    const pc_structs::detailed* pc,
    const int points_num,
    const float3& ref_position,
    const float3& ref_direction,
    const float search_radius,
    const float angle_diff_thresh,
    bool& verified
)
{
    struct stack_entry {
        int   node_id;
        float dist;
    };
    stack_entry stack_base[30];
    stack_entry* stack_ptr = stack_base;

    int curr = 0;

    while (true)
    {
        while (curr < points_num)
        {
            const int  curr_dim = pc[curr].split_dim;

            pc_structs::detailed curr_node = pc[curr];
           
            process_neighbor(
                ref_position,
                ref_direction,
                curr_node.position,
                curr_node.direction,
                search_radius,
                angle_diff_thresh,
                verified
            );

            if (verified)
                return;

            const float node_coord = pc_structs::detailed_traits::get_coord(curr_node, curr_dim);
            const float query_coord = (curr_dim == 2) ? ref_position.z : (curr_dim ? ref_position.y : ref_position.x);
            const bool  left_is_close = query_coord < node_coord;
            const int   l_child = 2 * curr + 1;
            const int   r_child = l_child + 1;

            const int close_child = left_is_close ? l_child : r_child;
            const int far_child = left_is_close ? r_child : l_child;

            const float dist_to_plane = fabsf(query_coord - node_coord);
            if (dist_to_plane < search_radius && far_child < points_num)
            {
                stack_ptr->node_id = far_child;
                stack_ptr->dist = dist_to_plane;
                ++stack_ptr;
            }
            curr = close_child;
        }


        while (true)
        {
            if (stack_ptr == stack_base)
                return;
            --stack_ptr;
            if (stack_ptr->dist >= search_radius)
                continue;
            curr = stack_ptr->node_id;
            break;
        }
    }
}



void evaluate_for_particle_set(
    const pc_structs::detailed* src_pc,
    const pc_structs::detailed* dst_pc,
    const int src_points_num,
    const int dst_points_num,
    const float distance_threshold,
    const float angle_threshold,
    float& result,
    const bool verbose
)
{
    int verified_points_num = 0;
    int totoal_processed_points = 0;

#pragma omp parallel for num_threads(OMP_NUM_THREADS)
        for (int p_idx = 0; p_idx < src_points_num; ++p_idx)
        {
#pragma omp critical
        {
            totoal_processed_points++;
            if (totoal_processed_points % 10000 == 0 && verbose)
                printf("\rProcess... %d / %zd (%.2f%%)", totoal_processed_points, src_points_num, 100.f * totoal_processed_points / src_points_num);
        }

        bool verified = false;
        traverse_tree(
            dst_pc,
            dst_points_num,
            src_pc[p_idx].position,
            src_pc[p_idx].direction,
            distance_threshold,
            angle_threshold,
            verified
        );
            
        if (verified)
        {
#pragma omp critical
            {
                verified_points_num++;
            }
        }

    }

    printf("\n");

    result = 100.f * verified_points_num / src_points_num;
    return;
    
}


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
    const bool verbose
)
{

    // loading the gt particles
    const int gt_particles_size_byte = gt_particles_num * sizeof(pc_structs::detailed);
    pc_structs::detailed* gt_particles = (pc_structs::detailed*)malloc(gt_particles_size_byte);
    memset(gt_particles, 0, gt_particles_size_byte);
    
    // loading the gt point cloud data, into the position+direction structure array
    for(int i{0}; i < gt_particles_num; ++i)
    {
        gt_particles[i] = {{gt_directions[3*i + 0], gt_directions[3*i+1], gt_directions[3*i+2]},
                            {gt_positions[3*i + 0], gt_positions[3*i+1], gt_positions[3*i+2]}};
    }


    // container for the estimated particles
    int estimated_particles_size_byte = estimated_points_num * sizeof(pc_structs::detailed);
    pc_structs::detailed* estimated_particles = (pc_structs::detailed*)malloc(estimated_particles_size_byte);

    // loading the estimated strands into the position+direction structure array
    for (int p_i=0; p_i < estimated_points_num; ++p_i)
    {
        float3 pos = {estimated_positions[3*p_i], estimated_positions[3*p_i+1], estimated_positions[3*p_i+2]};
        float3 dir = {estimated_directions[3*p_i], estimated_directions[3*p_i+1], estimated_directions[3*p_i+2]};
        estimated_particles[p_i] = { dir, pos };
    }


    // building kd tree structures for faster neighborhood traversal
    // building kd tree for the gt particles
    pc_structs::detailed* d_gt_pc;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_gt_pc, gt_particles_size_byte));
    CUDA_SAFE_CALL(cudaMemcpy(d_gt_pc, gt_particles, gt_particles_size_byte, cudaMemcpyHostToDevice));

    cukd::box_t<float3>* d_world_bounds = 0;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_world_bounds, sizeof(*d_world_bounds)));

    printf("KD tree construction for gt particles... \n");
    cukd::buildTree<pc_structs::detailed, pc_structs::detailed_traits>(d_gt_pc, gt_particles_num, d_world_bounds);

    CUDA_CHECK_ERRORS("kd tree construction failure!");
    // constructing the kd-tree is essentially just reordering the input points
    // so here we upload the re-arranged points back to the initial point cloud
    CUDA_SAFE_CALL(cudaMemcpy(gt_particles, d_gt_pc, gt_particles_size_byte, cudaMemcpyDeviceToHost));

    // do the whole kd tree stuff for the estimated particles as well
    pc_structs::detailed* d_estimated_pc;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_estimated_pc, estimated_particles_size_byte));
    CUDA_SAFE_CALL(cudaMemcpy(d_estimated_pc, estimated_particles, estimated_particles_size_byte, cudaMemcpyHostToDevice));
    printf("KD tree construction for estimated particles... \n");
    cukd::buildTree<pc_structs::detailed, pc_structs::detailed_traits>(d_estimated_pc, estimated_points_num, d_world_bounds);

    CUDA_CHECK_ERRORS("kd tree construction failure!");
    CUDA_SAFE_CALL(cudaMemcpy(estimated_particles, d_estimated_pc, estimated_particles_size_byte, cudaMemcpyDeviceToHost));

    for (int i = 0; i < thresholds_num; ++i)
    {
        float precision;
        float recall;

        // Public thresholds are in degrees; angular comparisons use radians.
        const float angle_threshold_rad = direction_thresholds[i] * static_cast<float>(M_PI / 180.0);

        // evaluating for gt particles results in recall
        evaluate_for_particle_set(gt_particles, 
            estimated_particles, 
            gt_particles_num, 
            estimated_points_num, 
            position_thresholds[i], 
            angle_threshold_rad,
            recall,
            verbose
        );

        // evaluating for estimated particles results in precision
        evaluate_for_particle_set(estimated_particles, 
            gt_particles, 
            estimated_points_num, 
            gt_particles_num, 
            position_thresholds[i], 
            angle_threshold_rad,
            precision,
            verbose
        );

        const float precision_recall_sum = precision + recall;
        const float f_score = precision_recall_sum > 0.f
            ? 2.f * (precision * recall) / precision_recall_sum
            : 0.f;
        printf("for radius range: %.4f, \t angle range: %.4f, \t precision: %.4f, \t recall: %.4f, \t f_score:%.4f\n",
            position_thresholds[i], direction_thresholds[i], precision, recall, f_score);

    }


    CUDA_SAFE_CALL(cudaFree(d_gt_pc));
    CUDA_SAFE_CALL(cudaFree(d_estimated_pc));

    free(gt_particles);
    free(estimated_particles);

    return 0;
}

}