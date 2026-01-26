#include <vector>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <omp.h>

#include "cukd/fcp.h"
#include "cukd/knn.h"

#include "line_fusion.h"
#include "point_cloud_structs.h"

namespace solution::line_fusion
{

inline __host__ __device__ 
void project_point_to_plane(
    const float3& ref_position,
    const float3& ref_direction,
    const float3& neighbor_position,
    const float3& neighbor_direction,
    float3& projected_point
)
{
    // projected_position = neighbor_position +
    //                   [ref_direction.(ref_position - neighbor_position) / ref_direction.neighbor_direction] * neighbor_direction
    float aux_f;
    subout(ref_position, neighbor_position, projected_point);
    aux_f = dot_product(ref_direction, projected_point) / (dot_product(ref_direction, neighbor_direction) + 1.e-7f);
    addscaleout(neighbor_position, aux_f, neighbor_direction, projected_point);
}


inline __host__ __device__
float calc_bilateral_weight(
    const float3& ref_position,
    const float3& ref_direction,
    const float3& neighbor_position,
    float3& neighbor_direction,
    const float sigma_pos,
    const float sigma_dir
)
{
    float3 aux;
    subout(ref_position, neighbor_position, aux);

    float aux_f1 = get_normf3(aux) / (2 * sigma_pos);
    aux_f1 *= aux_f1;
    aux_f1 *= 2;

    float aux_f2 = acosf(dot_product(neighbor_direction, ref_direction) / (get_normf3(neighbor_direction) * get_normf3(ref_direction)));
    // in case neighbor_dir and ref_dir are completely aligned acosf() can generate nan
    if (aux_f2 != aux_f2)
        aux_f2 = 0.f;

    if (aux_f2 > M_PI_2)
    {
        neighbor_direction.x = -neighbor_direction.x;
        neighbor_direction.y = -neighbor_direction.y;
        neighbor_direction.z = -neighbor_direction.z;
    }

    aux_f2 = M_PI_2 - fabsf(aux_f2 - M_PI_2);
    aux_f2 = (aux_f2 * aux_f2) / (2 * sigma_dir * sigma_dir);

    return expf(-aux_f1 - aux_f2);
}


inline __host__ __device__
void process_neighbor(
    const float3& ref_position,
    const float3& ref_direction,
    const float3& neighbor_position,
    float3& neighbor_direction,
    const float& sqr_dist,
    float3& pos_acc,
    float3& dir_acc,
    float& weight_acc,
    const float search_radius,
    const float sigma_pos,
    const float sigma_dir
)
{
    if (sqrtf(sqr_dist) > search_radius)
        return;

    // project point to the ref point's plane
    float3 projected_position;
    project_point_to_plane(
        ref_position,
        ref_direction,
        neighbor_position,
        neighbor_direction,
        projected_position
    );

    // calculate the bilateral weight
    float bilateral_weight = calc_bilateral_weight(
        ref_position,
        ref_direction,
        projected_position,
        neighbor_direction,
        sigma_pos,
        sigma_dir
    );

    // accumulate the result for this neighboring point
    pos_acc.x += bilateral_weight * projected_position.x;
    pos_acc.y += bilateral_weight * projected_position.y;
    pos_acc.z += bilateral_weight * projected_position.z;

    dir_acc.x += bilateral_weight * (neighbor_direction.x / get_normf3(neighbor_direction));
    dir_acc.y += bilateral_weight * (neighbor_direction.y / get_normf3(neighbor_direction));
    dir_acc.z += bilateral_weight * (neighbor_direction.z / get_normf3(neighbor_direction));

    weight_acc += bilateral_weight;
}


inline __host__ __device__
void traverse_tree(
    const pc_structs::detailed* pc,
    const int points_num,
    const float3& ref_position,
    const float3& ref_direction,
    float3& pos_acc,
    float3& dir_acc,
    float& weight_acc,
    const float search_radius,
    const float sigma_pos,
    const float sigma_dir
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
            const int  curr_dim = pc_structs::detailed_traits::get_dim(pc[curr]);

            pc_structs::detailed curr_node = pc[curr];
            float3 aux;
            subout(curr_node.position, ref_position, aux);
            const float sqr_dist = norm2(aux);

            process_neighbor(
                ref_position, 
                ref_direction, 
                curr_node.position, 
                curr_node.direction, 
                sqr_dist, 
                pos_acc, 
                dir_acc, 
                weight_acc,
                search_radius,
                sigma_pos,
                sigma_dir
            );

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


__global__ 
void mean_shift_kernel(
    const pc_structs::detailed* __restrict__ pc,
    const int points_num,
    pc_structs::detailed* __restrict__ d_fused_pc,
    const float search_radius,
    const float sigma_pos,
    const float sigma_dir,
    const float position_update_threshold
)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= points_num) return;

    float3 ref_position = pc[tid].position;
    float3 ref_direction = pc[tid].direction;

    float3 pos_acc;
    pos_acc.x = 0.f;
    pos_acc.y = 0.f;
    pos_acc.z = 0.f;

    float3 dir_acc;
    dir_acc.x = 0.f;
    dir_acc.y = 0.f;
    dir_acc.z = 0.f;

    float weight_acc = 0.f;

    int loop_count = 0;

    // do while loop here until the position update gets less than line_fusion_params::position_update_threshold
    float update_distance = 1000.f;

    do {
        loop_count++;

        // in the first iteration the point itself also appears in its neighborhood and thus
        // it does not require to be added to the accumulators
        if (loop_count != 1)
        {
            add4(pos_acc, ref_position);
            add4(dir_acc, ref_direction);
            weight_acc += 1;
        }

        traverse_tree(
            pc, 
            points_num, 
            ref_position, 
            ref_direction, 
            pos_acc, 
            dir_acc, 
            weight_acc,
            search_radius,
            sigma_pos,
            sigma_dir
        );

        vecdiv4(pos_acc, (weight_acc + 1.e-7f));
        vecdiv4(dir_acc, (weight_acc + 1.e-7f));

        sub4(ref_position, pos_acc);
        update_distance = get_normf3(ref_position);

        ref_position.x = pos_acc.x;
        ref_position.y = pos_acc.y;
        ref_position.z = pos_acc.z;

        ref_direction.x = dir_acc.x;
        ref_direction.y = dir_acc.y;
        ref_direction.z = dir_acc.z;

        pos_acc.x = 0.f;
        pos_acc.y = 0.f;
        pos_acc.z = 0.f;

        dir_acc.x = 0.f;
        dir_acc.y = 0.f;
        dir_acc.z = 0.f;

        weight_acc = 0.f;

        if (loop_count > 100)
        {
            printf("overwhelming while loop in thread %d, with %d iterations!\n", tid, loop_count);
            break;
        }
    } while (update_distance > position_update_threshold);

    // writing to the global memory
    d_fused_pc[tid].direction = ref_direction;

    // setting the split dim of the fused point cloud as the filtered one would mean that for strand
    // generation step also calculation of neighboring points and the structure of kd-tree would remain
    // the same as the line fusion step which would not be correct;
    // d_fused_pc[tid].split_dim = pc[tid].split_dim;
    d_fused_pc[tid].split_dim = 0;
    d_fused_pc[tid].mark_to_delete = 0;

    d_fused_pc[tid].position = ref_position;
}



void apply_mean_shift(
    pc_structs::detailed* initial_pc,
    pc_structs::detailed* fused_pc,
    const int points_num,
    const float search_radius,
    const float sigma_pos,
    const float sigma_dir,
    const float position_update_threshold
)
{

    int processed_points_num = 0;

#pragma omp parallel for num_threads(OMP_NUM_THREADS)
    for (int point_idx = 0; point_idx < points_num; point_idx++)
    {

#pragma omp critical
        {
            processed_points_num++;
            if (processed_points_num % 10000 == 0)
                printf("\rProcess... %d / %d (%.2f%%)", processed_points_num, points_num, 100.f * processed_points_num / points_num);

        }

        float3 ref_position  = initial_pc[point_idx].position;
        float3 ref_direction = initial_pc[point_idx].direction;

        float3 pos_acc;
        pos_acc.x = 0.f;
        pos_acc.y = 0.f;
        pos_acc.z = 0.f;
        float3 dir_acc;
        dir_acc.x = 0.f;
        dir_acc.y = 0.f;
        dir_acc.z = 0.f;
        float weight_acc = 0.f;

        // to prevent extravagent loop iterations!
        int loop_count = 0;

        // do while loop here until the position update gets less than line_fusion_params::position_update_threshold
        float update_distance = 1000.f;

        do {
            loop_count++;

            // in the first iteration the point itself also appears in its neighborhood and thus
            // it does not require to be added to the accumulators
            if (loop_count != 1)
            {
                add4(pos_acc, ref_position);
                add4(dir_acc, ref_direction);
                weight_acc += 1;
            }

            traverse_tree(
                initial_pc, 
                points_num, 
                ref_position, 
                ref_direction, 
                pos_acc, 
                dir_acc, 
                weight_acc,
                search_radius,
                sigma_pos,
                sigma_dir
            );

            vecdiv4(pos_acc, (weight_acc + 0.0000001f));
            vecdiv4(dir_acc, (weight_acc + 0.0000001f));

            sub4(ref_position, pos_acc);
            update_distance = get_normf3(ref_position);

            ref_position.x = pos_acc.x;
            ref_position.y = pos_acc.y;
            ref_position.z = pos_acc.z;

            ref_direction.x = dir_acc.x;
            ref_direction.y = dir_acc.y;
            ref_direction.z = dir_acc.z;

            pos_acc.x = 0.f;
            pos_acc.y = 0.f;
            pos_acc.z = 0.f;

            dir_acc.x = 0.f;
            dir_acc.y = 0.f;
            dir_acc.z = 0.f;

            weight_acc = 0.f;

            if (loop_count > 100)
            {
                printf("overwhelming while loop in point %d, with %d iterations!\n", point_idx, loop_count);
                break;
            }

        } while (update_distance > position_update_threshold);


        // writing to the global memory
        fused_pc[point_idx].direction = ref_direction;
        
        // setting the split dim of the fused point cloud as the filtered one would mean that for strand
        // generation step also calculation of neighboring points and the structure of kd-tree would remain
        // the same as the line fusion step which would not be correct;
        // d_fused_pc[tid].split_dim = pc[tid].split_dim;
        fused_pc[point_idx].split_dim = 0;
        fused_pc[point_idx].mark_to_delete = 0;

        fused_pc[point_idx].position = ref_position;

    }


}




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
        const bool fuse_on_gpu,
        const bool verbose
)
{
    uint64_t total_points_size_byte = points_num * sizeof(pc_structs::detailed);
    pc_structs::detailed* inital_pc = (pc_structs::detailed*)malloc(total_points_size_byte);

    pc_structs::detailed* fused_pc = (pc_structs::detailed*)malloc(total_points_size_byte);
    memset(fused_pc, 0, total_points_size_byte);

    // loading the point cloud data
    for(int i{0}; i < points_num; ++i)
    {
        inital_pc[i] = {{directions[4*i + 0], directions[4*i+1], directions[4*i+2]},
                            {positions[4*i + 0], positions[4*i+1], positions[4*i+2]}};
    }

    pc_structs::detailed* d_initial_pc;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_initial_pc, total_points_size_byte));
    CUDA_SAFE_CALL(cudaMemcpy(d_initial_pc, inital_pc, total_points_size_byte, cudaMemcpyHostToDevice));

    // some timing variables 
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    float milliseconds = 0.f;

    // needed for cukd builder
    cukd::box_t<float3>* d_world_bounds = 0;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_world_bounds, sizeof(*d_world_bounds)));

    CUDA_SAFE_CALL(cudaEventRecord(start));

    printf("KD tree construction ... \n");
    cukd::buildTree<pc_structs::detailed, pc_structs::detailed_traits>(d_initial_pc, points_num, d_world_bounds);
    
    printf("Line fusion ... \n");
    if (fuse_on_gpu)
    {
        pc_structs::detailed* d_fused_pc;
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_fused_pc, total_points_size_byte));
        CUDA_SAFE_CALL(cudaMemset(d_fused_pc, 0, total_points_size_byte));

        dim3 block_size = dim3(128);
        dim3 grid_size = dim3(div_round_up(points_num, 128));

        mean_shift_kernel << <grid_size, block_size >> > (
            d_initial_pc, 
            points_num, 
            d_fused_pc,
            search_radius,
            sigma_pos,
            sigma_dir,
            position_update_threshold
        );
        cudaDeviceSynchronize();
        CUDA_CHECK_ERRORS("line fusion kernel launch failure");

        CUDA_SAFE_CALL(cudaMemcpy(fused_pc, d_fused_pc, total_points_size_byte, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaFree(d_fused_pc));

        for(int i{0}; i < points_num; ++i)
        {
            if (fused_pc[i].position.x != fused_pc[i].position.x)
                printf("found nan values at index: %d\n", i);
        }

    }
    else
    {
        // constructing the kd-tree is essentially just reordering the input points
        // so here we upload the re-arranged points back to the initial point cloud
        CUDA_SAFE_CALL(cudaMemcpy(inital_pc, d_initial_pc, total_points_size_byte, cudaMemcpyDeviceToHost));

        apply_mean_shift(
            inital_pc, 
            fused_pc, 
            points_num,
            search_radius,
            sigma_pos,
            sigma_dir,
            position_update_threshold
        );
    }

    CUDA_SAFE_CALL(cudaEventRecord(stop));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, start, stop));

    if (verbose)
    {
        printf("Line fusion is compelete!\n");
        printf("Ellapsed time: %16.10f ms\n", milliseconds);
    }

    // loading the fused point cloud data to the output tensors
    for(int i{0}; i < points_num; ++i)
    {

        out_positions[4*i + 0] = fused_pc[i].position.x;
        out_positions[4*i + 1] = fused_pc[i].position.y;
        out_positions[4*i + 2] = fused_pc[i].position.z;
        out_positions[4*i + 3] = 0.f;

        out_directions[4*i + 0] = fused_pc[i].direction.x;
        out_directions[4*i + 1] = fused_pc[i].direction.y;
        out_directions[4*i + 2] = fused_pc[i].direction.z;
        out_directions[4*i + 3] = 0.f;
    }

    CUDA_SAFE_CALL(cudaFree(d_world_bounds));
    CUDA_SAFE_CALL(cudaFree(d_initial_pc));
    free(inital_pc);
    free(fused_pc);

    return 0;

}
        

}