#include <vector>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <omp.h>

// for timing
#include <chrono>

#include "cukd/fcp.h"
#include "cukd/knn.h"

#include "strand_segment_generation.h"
#include "point_cloud_structs.h"


namespace solution::strand_segment_generation
{

inline float calc_dist_to_line(
    const pc_structs::detailed& p,
    const pc_structs::detailed& line
)
{
    // distance from position q to line with plucker coord (l_hat, m) (where l_hat is normalized direction) 
    // can be calculated as the norm || m - q x l_hat||
    float3 line_momentum;
    cross_product(line.position, line.direction, line_momentum);
    float3 aux;
    cross_product(p.position, line.direction, aux);
    sub4(aux, line_momentum);
    return get_normf3(aux);
}


inline float calc_strand_length(
    const std::vector<pc_structs::detailed>& strand
)
{
    if ((int)strand.size() < 2)
        return 0.f;
    float length = 0.f;
    for (int i = 1; i < strand.size(); ++i)
    {
        float3 pos = strand[i].position;
        float3 pos_prev = strand[i-1].position;
        sub4(pos, pos_prev);
        length += get_normf3(pos);
    }
    return length;
}


void find_neighboring_points(
    pc_structs::detailed* pc,
    const int& points_num,
    std::vector<int>& neighbors_indices,
    const float neighboring_radius,
    const float3& ref_position
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

            pc_structs::detailed& curr_node = pc[curr];
            float3 aux;
            subout(curr_node.position, ref_position, aux);
            const float sqr_dist = norm2(aux);

            if (sqrtf(sqr_dist) < neighboring_radius)
            {
                neighbors_indices.push_back(curr);
            }

            const float node_coord = pc_structs::detailed_traits::get_coord(curr_node, curr_dim);
            const float query_coord = (curr_dim == 2) ? ref_position.z : (curr_dim ? ref_position.y : ref_position.x);
            const bool  left_is_close = query_coord < node_coord;
            const int   l_child = 2 * curr + 1;
            const int   r_child = l_child + 1;

            const int close_child = left_is_close ? l_child : r_child;
            const int far_child = left_is_close ? r_child : l_child;

            const float dist_to_plane = fabsf(query_coord - node_coord);
            if (dist_to_plane < neighboring_radius && far_child < points_num)
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
            if (stack_ptr->dist >= neighboring_radius)
                continue;
            curr = stack_ptr->node_id;
            break;
        }
    }

}


void mark_points_close_to_strand(
    pc_structs::detailed* pc,
    const int& points_num,
    const std::vector<pc_structs::detailed>& strand_points,
    const float strand_generation_neighboring_radius
)
{
    for (auto& p : strand_points)
    {
        std::vector<int> neighbors_indices;
        find_neighboring_points(
            pc,
            points_num,
            neighbors_indices,
            2 * strand_generation_neighboring_radius,
            p.position
        );

        // mark point in the cylinder defined by strand segment
        for (auto& idx : neighbors_indices)
        {
            float dist_to_line = calc_dist_to_line(pc[idx], p);
            if (dist_to_line < strand_generation_neighboring_radius)
                pc[idx].mark_to_delete = 1;
        }
    }

}


void forward_euler_step(
    pc_structs::detailed* pc,
    const int& points_num,
    const float3& prev_move_position,
    float3& move_position,
    float3& move_direction,
    float3& avg_position,
    float3& avg_direction,
    int& neighbors_num,
    const float strand_generation_step_size,
    const float strand_generation_length_threshold,
    const float strand_generation_neighboring_radius,
    const float strand_generation_angle_difference_threshold_rad
)
{
    // Each candidate distance is measured from the same starting position.
    const float3 search_origin = move_position;
    for (float move_step = strand_generation_step_size;
        move_step < strand_generation_length_threshold;
        move_step += strand_generation_step_size)
    {
        addscaleout(search_origin, move_step, move_direction, move_position);
        std::vector<int> neighbors_indices;
        find_neighboring_points(
            pc,
            points_num,
            neighbors_indices,
            strand_generation_neighboring_radius,
            move_position
        );

        for (auto& n_idx : neighbors_indices)
        {
            pc_structs::detailed& curr_node = pc[n_idx];
            if (curr_node.mark_to_delete == 0)
            {
                float angle_diff = acosf(dot_product(curr_node.direction, move_direction) / (get_normf3(curr_node.direction) * get_normf3(move_direction)));

                // handling the possible nan values
                if (angle_diff != angle_diff)
                    angle_diff = 0.f;

                angle_diff = M_PI_2 - fabsf(angle_diff - M_PI_2);

                if (angle_diff < strand_generation_angle_difference_threshold_rad)
                {

                    avg_position.x += curr_node.position.x;
                    avg_position.y += curr_node.position.y;
                    avg_position.z += curr_node.position.z;

                    if (dot_product(curr_node.direction, move_direction) < 0.f)
                    {
                        avg_direction.x -= (curr_node.direction.x / get_normf3(curr_node.direction));
                        avg_direction.y -= (curr_node.direction.y / get_normf3(curr_node.direction));
                        avg_direction.z -= (curr_node.direction.z / get_normf3(curr_node.direction));
                    }
                    else
                    {
                        avg_direction.x += (curr_node.direction.x / get_normf3(curr_node.direction));
                        avg_direction.y += (curr_node.direction.y / get_normf3(curr_node.direction));
                        avg_direction.z += (curr_node.direction.z / get_normf3(curr_node.direction));
                    }

                    neighbors_num++;
                }
            }
        }

        if (neighbors_num > 0)
        {
            vecdiv4(avg_position, (neighbors_num + 0.0000001f));
            vecdiv4(avg_direction, (neighbors_num + 0.0000001f));
            float avg_direction_norm = get_normf3(avg_direction);
            vecdiv4(avg_direction, avg_direction_norm);

            if (move_step != strand_generation_step_size)
            {
                subout(avg_position, prev_move_position, avg_direction);
                float avg_direction_norm = get_normf3(avg_direction);
                vecdiv4(avg_direction, avg_direction_norm);
                addscaleout(prev_move_position, strand_generation_step_size, avg_direction, avg_position);
            }

            break;
        }
    }
    
}



void generate_strand_segments(
    pc_structs::detailed* pc,
    const int points_num,
    const float strand_generation_step_size,
    const float strand_generation_length_threshold,
    const float strand_generation_neighboring_radius,
    const float strand_generation_angle_difference_threshold_rad,
    std::function<float* (size_t)> out_positions_buffer,
    std::function<float* (size_t)> out_directions_buffer,
    std::function<float* (size_t)> out_points_per_strand_buffer,
    const bool verbose
)
{
    int max_segment_steps = 10000;

    std::vector<std::vector<pc_structs::detailed>> strands_points;
    std::vector<float> strands_length;
    size_t total_strand_points = 0;

    printf("Strand generation ...\n");
    int processed_points_num = 0;

    // parallelisation here is not straight forward, cause I have not yet find a good way (compliant with multi-threading) for
    // marking the neighborhood of each of the generated segments to not be considered for the rest of the segment generation process
    // side note: maybe the reordering that happens due to kd-tree reconstruction somehow arranges the points in a way that it might not
    // be even necessary to consider a special solution for thread safe marking of the neighborhood!!! 
// #pragma omp parallel for num_threads(NUM_THREADS)
    for (int point_idx = 0; point_idx < points_num; point_idx++)
    {

// #pragma omp critical
        {
            processed_points_num++;
            if (processed_points_num % 10000 == 0)
                printf("\rProcess... %d / %d (%.2f%%)", processed_points_num, points_num, 100.f * processed_points_num / points_num);

        }

        if (pc[point_idx].mark_to_delete == 1)
            continue;

        float3 origin_position = pc[point_idx].position;
        float3 origin_direction = pc[point_idx].direction;
        float direction_norm = get_normf3(origin_direction);
        origin_direction.x /= direction_norm;
        origin_direction.y /= direction_norm;
        origin_direction.z /= direction_norm;

        float3 move_position;
        float3 prev_move_position;
        float3 move_direction;

        float3 avg_position;
        float3 avg_direction;

        int neighbors_num;
        int loop_count;

        std::vector<pc_structs::detailed> forward_strand_points;
        std::vector<pc_structs::detailed> backward_strand_points;
        std::vector<pc_structs::detailed> strand_points;

        for (int i = 1; i > -2; i -= 2) // i=-1: backward direction, i=+1: forward direction
        {
            set_equal_vec3(move_position, origin_position);
            set_equal_vec3(prev_move_position, origin_position);

            scale_vec4(origin_direction, i);
            set_equal_vec3(move_direction, origin_direction);

            zero_init_vec3(avg_position);
            zero_init_vec3(avg_direction);

            neighbors_num = 0;
            loop_count = 0;

            while (true)
            {
                loop_count++;
                forward_euler_step(
                    pc,
                    points_num,
                    prev_move_position,
                    move_position,
                    move_direction,
                    avg_position,
                    avg_direction,
                    neighbors_num,
                    strand_generation_step_size,
                    strand_generation_length_threshold,
                    strand_generation_neighboring_radius,
                    strand_generation_angle_difference_threshold_rad
                );

                // preventing severe direction changes in the strand under production
                // float3 aux;
                // subout(avg_position, move_position, aux);
                // if (dot_product(aux, move_direction) < 0)
                //     break;

                if (neighbors_num < 1)
                    break;

                // update the origin point
                set_equal_vec3(move_position, avg_position);
                set_equal_vec3(move_direction, avg_direction);

                sub4(prev_move_position, move_position);
                if (get_normf3(prev_move_position) < 0.0000001)
                    break;

                set_equal_vec3(prev_move_position, move_position);

                // add new point to the strand vector 
                if (i == 1) // forward direction
                {
                    forward_strand_points.push_back({ move_direction, move_position });
                }
                else // backward direction
                {
                    backward_strand_points.push_back({ move_direction, move_position });
                }

                zero_init_vec3(avg_position);
                zero_init_vec3(avg_direction);
                neighbors_num = 0;
                if (loop_count >= max_segment_steps)
                {
                    break;
                }

            } // while loop to find the consequitive points on strand segment on both directions

            // if the loop count reaches the maximum segment steps it is an indication for erronous segment generation
            if (i == 1 && loop_count >= max_segment_steps) // forward direction
            {
                forward_strand_points.clear();
            }
            else if (i == -1 && loop_count >= max_segment_steps)// backward direction
            {
                backward_strand_points.clear();
            }

        } // for loop doing strand generation on both directions

        // merging forward and backward strand vectors
        for (int j = 0; j < backward_strand_points.size(); ++j)
            strand_points.push_back(backward_strand_points[backward_strand_points.size() - j - 1]);

        for (int j = 1; j < forward_strand_points.size(); ++j)
            strand_points.push_back(forward_strand_points[j]);

        mark_points_close_to_strand(pc, points_num, strand_points, strand_generation_neighboring_radius);
        float strand_length = calc_strand_length(strand_points);
        if (strand_length > strand_generation_length_threshold)
        {
// #pragma omp critical
            {
                strands_points.push_back(strand_points);
                strands_length.push_back(strand_length);
                total_strand_points += strand_points.size();
            }
        }

    } // for loop going through all points in the point cloud


    //
    // sending the data to the specified output tensors
    float* output_positions = out_positions_buffer(4 * total_strand_points);
    float* output_directions = out_directions_buffer(4 * total_strand_points);
    float* output_points_per_strand = out_points_per_strand_buffer(strands_points.size());

    int point_acc = 0;
    int strand_acc = 0;
    // loop over strands
    for (auto& s : strands_points)
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

    printf("\n");

    float max_length = -1.f;
    float length_sum = 0.f;
    for (auto& l : strands_length)
    {
        length_sum += l;
        if (l > max_length)
            max_length = l;

    }

    if (verbose)
    {
        printf("total number of segments is: %zd\n", strands_points.size());
        printf("total number of segments lengths is: %zd\n", strands_length.size());
        printf("maximum of the segment lengths is : %f\n", max_length);
        printf("average of the segment lengths is : %f\n", length_sum / (float)strands_points.size());
    }
    
}



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
        const bool verbose
)
{
    uint64_t total_points_size_byte = points_num * sizeof(pc_structs::detailed);
    pc_structs::detailed* initial_pc = (pc_structs::detailed*)malloc(total_points_size_byte);

    // loading the point cloud data
    for(int i{0}; i < points_num; ++i)
    {
        initial_pc[i] = {{directions[4*i + 0], directions[4*i+1], directions[4*i+2]},
                            {positions[4*i + 0], positions[4*i+1], positions[4*i+2]}};
    }


    // constructing the kd-tree here for tracking segments neighborhood
    pc_structs::detailed* d_initial_pc;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_initial_pc, total_points_size_byte));
    CUDA_SAFE_CALL(cudaMemcpy(d_initial_pc, initial_pc, total_points_size_byte, cudaMemcpyHostToDevice));

    // start timing
    auto start = std::chrono::high_resolution_clock::now();

    // needed for cukd builder
    cukd::box_t<float3>* d_world_bounds = 0;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_world_bounds, sizeof(*d_world_bounds)));

    printf("KD tree construction ... \n");
    cukd::buildTree<pc_structs::detailed, pc_structs::detailed_traits>(d_initial_pc, points_num, d_world_bounds);

    CUDA_CHECK_ERRORS("kd tree construction failiure!");

    // constructing the kd-tree is essentially just reordering the input points
    // so here we upload the re-arranged points back to the initial point cloud
    CUDA_SAFE_CALL(cudaMemcpy(initial_pc, d_initial_pc, total_points_size_byte, cudaMemcpyDeviceToHost));
    
    generate_strand_segments(
        initial_pc, 
        points_num,
        strand_generation_step_size,
        strand_generation_length_threshold,
        strand_generation_neighboring_radius,
        strand_generation_angle_difference_threshold_rad,
        out_positions_buffer,
        out_directions_buffer,
        out_points_per_strand_buffer,
        verbose
    );

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (verbose)
    {
        printf("Ellapsed time: %16.10f ms\n", static_cast<float>(duration.count()));
    }

    printf("Strand segments generation is compelete!\n");

    CUDA_SAFE_CALL(cudaFree(d_world_bounds));
    CUDA_SAFE_CALL(cudaFree(d_initial_pc));
    free(initial_pc);

    return 0;

}
        

}
