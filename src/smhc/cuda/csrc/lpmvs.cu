#include <iostream>
#include <stdexcept>
#include "auxiliary.h"

#include "device_dataframe.h"
#include "lpmvs.h"

#define EXTRA_POINTS_AXIS_ALIGNED_3_PIXEL_FAR 1
#define EXTRA_POINTS_AXIS_ALIGNED_5_PIXEL_FAR  1
#define EXTRA_POINTS_DIAGONAL_MANHATAN_3_DISTANCE 1


namespace solution::lpmvs
{

__device__ __forceinline__  float angle_difference(
    float angle1,
    float angle2
)
{
    return M_PI_2 - fabsf(fabsf(angle1 - angle2) - M_PI_2);
}

__device__ __forceinline__ void normalize(float3& v)
{
    const float norm_squared = pow2(v.x) + pow2(v.y) + pow2(v.z);
    const float inverse_sqrt = rsqrtf(norm_squared);
    v.x *= inverse_sqrt;
    v.y *= inverse_sqrt;
    v.z *= inverse_sqrt;
}


__device__ __forceinline__ float random_between(
    curandState* cs,
    const float min,
    const float max
)
{
    return (curand_uniform(cs) * (max - min) + min);
}

__device__ __forceinline__ void get_world_coord_from_pixel(
    const float& px,
    const float& py,
    const float& depth,
    const float* cam_m_inv,
    const float4 cam_P_col34,
    float3& pt_w)
{
    float4 pt;
    pt.x = depth * px - cam_P_col34.x;
    pt.y = depth * py - cam_P_col34.y;
    pt.z = depth - cam_P_col34.z;
    matvecmul4(cam_m_inv, pt, pt_w);
}

__device__ __forceinline__ void generate_ray_towards_pixel(
    const float px,
    const float py,
    const float camera_focal_length,
    const float* const camera_m_inv,
    const float4 camera_P_col34,
    const float4 camera_origin,
    float3& screen_sample_world_coord,
    float3& direction,
    float3& momentum
)
{
    get_world_coord_from_pixel(
        px,
        py,
        camera_focal_length,
        camera_m_inv,
        camera_P_col34,
        screen_sample_world_coord);

    plucker_line3(camera_origin, screen_sample_world_coord, direction, momentum);
}


__device__ __forceinline__ void get_pixel_coord_from_world_coord(
    const float* proj_mat,
    const float3& world_coord,
    float2* pixel
)
{
    float3 pos;
    matvecmul4_hom(proj_mat, world_coord, pos);
    pixel->x = pos.x / pos.z;
    pixel->y = pos.y / pos.z;
}

__device__ __forceinline__ void project_point_to_camera_space(
    const float* R,
    const float4 t4,
    const float3 world_coord,
    float3& point_local_coord
)
{
    matvecmul4(R, world_coord, point_local_coord);
    add4(point_local_coord, t4);
}


__device__ __forceinline__ void find_closest_point_on_l1_to_l2_dir2(
    const float3& direction_l1,
    const float3& momentum_l1,
    const float2& direction2_l2,
    const float4& momentum_depth_l2,
    float3& closest_point_world_coord
)
{
    float3 cross_d1d2;
    cross_product_dir2(direction_l1, direction2_l2, cross_d1d2);
    float3 aux1;
    float3 aux2;
    cross_product_dir2_1(direction2_l2, cross_d1d2, aux1);
    cross_product(-momentum_l1, aux1, aux2);
    aux1 = direction_l1;
    scale_vec4(aux1, dot_product(momentum_depth_l2, cross_d1d2));
    add4(aux2, aux1);
    vecdiv4(aux2, get_pow2_normf3(cross_d1d2));
    closest_point_world_coord.x = aux2.x;
    closest_point_world_coord.y = aux2.y;
    closest_point_world_coord.z = aux2.z;
}

__device__ __forceinline__ void find_closest_point_on_l1_to_l2(
    const float4& direction_l1,
    const float4& momentum_l1,
    const float4& direction_l2,
    const float4& momentum_l2,
    float4& closest_point_world_coord
)
{
    float4 cross_d1d2;
    cross_product(direction_l1, direction_l2, cross_d1d2);
    float4 aux1;
    float4 aux2;
    cross_product(direction_l2, cross_d1d2, aux1);
    cross_product(-momentum_l1, aux1, aux2);
    aux1 = direction_l1;
    scale_vec4(aux1, dot_product(momentum_l2, cross_d1d2));
    add4(aux2, aux1);
    vecdiv4(aux2, get_pow2_normf3(cross_d1d2));
    closest_point_world_coord.x = aux2.x;
    closest_point_world_coord.y = aux2.y;
    closest_point_world_coord.z = aux2.z;
}

__device__ __forceinline__ void calc_nvcc_element_float(
    float* __restrict__ elm1,
    float* __restrict__ elm2,
    float* __restrict__ dot_acc,
    float* __restrict__ norm1_acc,
    float* __restrict__ norm2_acc
)
{
    (*dot_acc) += (*elm1) * (*elm2);
    (*norm1_acc) += (*elm1) * (*elm1);
    (*norm2_acc) += (*elm2) * (*elm2);
}


__device__ __forceinline__ void calc_nvcc_element_float4(
    float4* __restrict__ elm1,
    float4* __restrict__ elm2,
    float4* __restrict__ dot_acc,
    float4* __restrict__ norm1_acc,
    float4* __restrict__ norm2_acc
)
{
    (*dot_acc).x += (*elm1).x * (*elm2).x;
    (*norm1_acc).x += (*elm1).x * (*elm1).x;
    (*norm2_acc).x += (*elm2).x * (*elm2).x;

    (*dot_acc).y += (*elm1).y * (*elm2).y;
    (*norm1_acc).y += (*elm1).y * (*elm1).y;
    (*norm2_acc).y += (*elm2).y * (*elm2).y;

    (*dot_acc).z += (*elm1).z * (*elm2).z;
    (*norm1_acc).z += (*elm1).z * (*elm1).z;
    (*norm2_acc).z += (*elm2).z * (*elm2).z;
}

__device__ __forceinline__ void calc_nvcc_acc_float(
    float* __restrict__ acc,
    float* __restrict__ dot_acc,
    float* __restrict__ norm1_acc,
    float* __restrict__ norm2_acc
)
{
    *acc = (*dot_acc) / (sqrtf((*norm1_acc) * (*norm2_acc)) + 1e-8);
}


__device__ __forceinline__ void calc_nvcc_acc_float4(
    float* __restrict__ acc,
    float4* __restrict__ dot_acc,
    float4* __restrict__ norm1_acc,
    float4* __restrict__ norm2_acc
)
{
    float4 aux;
    aux.x = (*dot_acc).x / (sqrtf((*norm1_acc).x * (*norm2_acc).x) + 1e-8);
    aux.y = (*dot_acc).y / (sqrtf((*norm1_acc).y * (*norm2_acc).y) + 1e-8);
    aux.z = (*dot_acc).z / (sqrtf((*norm1_acc).z * (*norm2_acc).z) + 1e-8);
    (*acc) = (aux.x + aux.y + aux.z) / 3.f;
}

__device__ __forceinline__ bool is_visible_in_neighboring_view(
    const float px,
    const float py,
    const float depth,
    const float* ref_cam_m_inv,
    const float4 ref_cam_P_col34,
    const float* neighbor_cam_R,
    const float4 neighbor_cam_t4,
    const float* neighbor_proj_mat,
    const int img_rows,
    const int img_cols,
    cudaTextureObject_t neighbor_initial_depth_map,
    const float occlusion_depth_thresh,
    // for debug
    const int idx_ref
)
{
    if (depth == 0 || depth != depth)
        return false;
    
    // get world coordinate of the pixel
    float3 world_coord;
    get_world_coord_from_pixel(px, py, depth, ref_cam_m_inv, ref_cam_P_col34, world_coord);
    
    // transform from the world coordinate to the neighboring camera coordinate
    float3 point_local_coord;
    project_point_to_camera_space(neighbor_cam_R, neighbor_cam_t4, world_coord, point_local_coord);

    // also get the pixel coordinates in the neighboring view
    float2 pixel_coord;
    get_pixel_coord_from_world_coord(neighbor_proj_mat, world_coord, &pixel_coord);

    if (pixel_coord.y < 0 || pixel_coord.y >= img_rows || pixel_coord.x < 0 || pixel_coord.x >= img_cols)
		return false;

    // check whether the projected depth and the input depth are close
    const float neighbor_depth = texat(neighbor_initial_depth_map, pixel_coord.x, pixel_coord.y);

    if (neighbor_depth == 0 || neighbor_depth != neighbor_depth)
        return false;
    
    if (fabsf(neighbor_depth - point_local_coord.z) < occlusion_depth_thresh)
        return true;

    return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
template< typename T >
__device__ __forceinline__ float multiview_cost(
    device_dataframe* ddf,
    const int2& p, // pixel location
    const float2& direction,
    const float4& momentum_depth
)
{
    
    int idx_ref = ddf->id_ref;

    // projecting 3d line into 2d image space of the reference camera
    float3 reverse_dir;
    float4 reverse_momentum_depth;
    plucker_line_reverse_dir2(direction,
        momentum_depth,
        reverse_dir,
        reverse_momentum_depth);

    // 2D line parameters a,b of ax + by + c = 0
    float2 ref_line_2d;
    plucker_line_project_to_2d(reverse_dir,
        reverse_momentum_depth,
        ddf->line_proj_mat[idx_ref],
        ref_line_2d);

    // Initialize cost values
    float cost_geometric = 0.0f;
    float sum_of_gamma = 1e-8f;
    float cost_image = 0.0f;
    float num_neighbor = ddf->view_selection_subset_number + 1e-8f;

    // results of atanf will be in radians, in the interval [-pi / 2, pi / 2]
    float angle = atanf(-ref_line_2d.x / ref_line_2d.y);
    float cost = 0.f;
    float confidence_sum = 0.f;

    // computing the cost for the reference view
// #pragma unroll // if you define the half_samples_num in the params namespace as a constexpr, you can unroll this loop, in this case compilation takes forever but provides descent speed up!
    for (int i{ -ddf->half_samples_num }; i <= ddf->half_samples_num; i++)
    {
        float x = i * cosf(angle) / ddf->half_samples_num * ddf->sampling_radius + p.x;
        float y = i * sinf(angle) / ddf->half_samples_num * ddf->sampling_radius + p.y;

        float ori = texat(ddf->imgs_ori[0], x, y);
        float conf = texat(ddf->imgs_conf[0], x, y);

        cost += angle_difference(ori, angle) * conf;
        confidence_sum += conf;
    }
    cost /= (confidence_sum + 1e-8);

    cost_geometric += (1 + num_neighbor) * cost;
    sum_of_gamma += (1 + num_neighbor);

    float2 neighbor_line_2d;
    float neighbor_angle;
    int idx_neighbor;
    float img_cost;

    // variables for computing NCC
    T dot_acc;
    T norm1_acc;
    T norm2_acc;

    // computing cost for the neighboring views
    for (int j = 0; j < ddf->view_selection_subset_number; j++)
    {
        idx_neighbor = ddf->view_selection_subset[j];

        // check if the pixel is visible in the neighboring view if initial depth values are provided
        if (ddf->cu_array_imgs_initial_depth[0])
        {
            if (ddf->cu_array_imgs_initial_depth[j+1])
            {
                float ref_init_depth = texat(ddf->imgs_initial_depth[0], p.x, p.y);
                if (!is_visible_in_neighboring_view((float)p.x, (float)p.y, 
                        ref_init_depth, 
                        ddf->M_inv[idx_ref], 
                        ddf->P_col34[idx_ref], 
                        ddf->R[idx_neighbor], 
                        ddf->t4[idx_neighbor], 
                        ddf->P[idx_neighbor],
                        ddf->rows,
                        ddf->cols,
                        ddf->imgs_initial_depth[j+1],
                        ddf->occlusion_depth_thresh,
                        idx_ref)
                    )
                    {
                        
                        continue;
                    }
            }
        }

        cost = 0.f;
        confidence_sum = 0.f;
        img_cost = 0.f;
        if constexpr (sizeof(T) == sizeof(float))
        {
            dot_acc = 0.f;
            norm1_acc = 0.f;
            norm2_acc = 0.f;
        }
        else if constexpr (sizeof(T) == sizeof(float4))
        {
            dot_acc.x = dot_acc.y = dot_acc.z = dot_acc.w = 0.f;
            norm1_acc.x = norm1_acc.y = norm1_acc.z = norm1_acc.w = 0.f;
            norm2_acc.x = norm2_acc.y = norm2_acc.z = norm2_acc.w = 0.f;
        }
        // else : error, unknown type T

        plucker_line_project_to_2d(reverse_dir,
            reverse_momentum_depth,
            ddf->line_proj_mat[idx_neighbor],
            neighbor_line_2d);

        neighbor_angle = atanf(-neighbor_line_2d.x / neighbor_line_2d.y);

// #pragma unroll // if you define the half_samples_num in the params namespace as a constexpr, you can unroll this loop, in this case compilation takes forever but provides descent speed up!
        for (int i{  -ddf->half_samples_num }; i <= ddf->half_samples_num; ++i)
        {
            // reference pixel coordinates
            float x = i * cosf(angle) / ddf->half_samples_num * ddf->sampling_radius + p.x;
            float y = i * sinf(angle) / ddf->half_samples_num * ddf->sampling_radius + p.y;

            // Once we obtain Sp,0, we shoot rays from the reference view’s origin towards each sample in Sp,0.
            float3 screen_sample_world_coord;
            float3 ray_direction;
            float3 ray_momentum;
            generate_ray_towards_pixel(
                x,
                y,
                ddf->fx[idx_ref],
                ddf->M_inv[idx_ref],
                ddf->P_col34[idx_ref],
                ddf->c4[idx_ref],
                screen_sample_world_coord,
                ray_direction,
                ray_momentum);

            // We find intersection points with the 3D line Lp
            float3 sample_world_coord;
            find_closest_point_on_l1_to_l2_dir2(ray_direction, ray_momentum, direction, momentum_depth, sample_world_coord);

            // and re-project the points into j-th view and get corresponding 2D samples Sp,i
            float2 pixel_coord;
            get_pixel_coord_from_world_coord(ddf->P[idx_neighbor], sample_world_coord, &pixel_coord);

            
            // check whether any mask is provided for the neighboring view, if so, 
            // check whether the projected pixel in the neighboring view ends up on the background, if so,
            // consider a large amount for the cost value
            float mask_val = -1.f;
            if (ddf->cu_array_imgs_mask[j+1])
            {
                mask_val = texat(ddf->imgs_mask[j + 1], pixel_coord.x, pixel_coord.y);
            }

            // no masks available or projected pixel is on the foreground
            if (mask_val < -0.001f || mask_val > 0.001f)
            {
                float ori = texat(ddf->imgs_ori[j + 1], pixel_coord.x, pixel_coord.y);
                float conf = texat(ddf->imgs_conf[j + 1], pixel_coord.x, pixel_coord.y);
                T img_val = tex2D<T>(ddf->imgs[j + 1], pixel_coord.x + 0.5f, pixel_coord.y + 0.5f);
                T ref_img_val = tex2D<T>(ddf->imgs[0], x + 0.5f, y + 0.5f);

                if constexpr (sizeof(T) == sizeof(float))
                calc_nvcc_element_float(
                    (float*)&ref_img_val,
                    (float*)&img_val,
                    (float*)&dot_acc,
                    (float*)&norm1_acc,
                    (float*)&norm2_acc);
                else if constexpr (sizeof(T) == sizeof(float4))
                    calc_nvcc_element_float4(
                        (float4*)&ref_img_val,
                        (float4*)&img_val,
                        (float4*)&dot_acc,
                        (float4*)&norm1_acc,
                        (float4*)&norm2_acc);
                // else : error, unknown type T

                cost += angle_difference(ori, neighbor_angle) * conf;
                confidence_sum += conf;

            }
            else // projected pixel is on the background region
            {
                cost += 10000.f;
                confidence_sum += 250.f;
            }
            
        }

        cost /= (confidence_sum + 1e-8);

        if constexpr (sizeof(T) == sizeof(float))
            calc_nvcc_acc_float(
                &img_cost,
                (float*)&dot_acc,
                (float*)&norm1_acc,
                (float*)&norm2_acc);
        else if constexpr (sizeof(T) == sizeof(float4))
            calc_nvcc_acc_float4(
                &img_cost,
                (float4*)&dot_acc,
                (float4*)&norm1_acc,
                (float4*)&norm2_acc);
        // else : error, unknown type T

        // Flip ncc value range [-1, 1] to [1, 0]
        img_cost *= -0.5f;
        img_cost += 0.5f;

        cost_geometric += cost;
        cost_image += img_cost;

        sum_of_gamma += 1.f;
        
    }

    cost_geometric /= sum_of_gamma;
    cost_image /= num_neighbor;

    float total_cost = ((1 - ddf->alpha) * cost_geometric + ddf->alpha * cost_image);

    return total_cost;

}


template< typename T >
__device__ __forceinline__ void spatial_propagation(
    device_dataframe* ddf,
    // reference pixel
    const int2 p,
    // reference view camera info
    const float ref_cam_focal_length,
    const float* ref_cam_m_inv,
    const float4 ref_cam_P_col34,
    const float4 ref_cam_origin,
    const float* ref_cam_rotation_mat,
    const float4 ref_cam_t4,
    // neighboring view u direction momentum and depth values
    const float2 neighbor_direction,
    const float4 neighbor_momentum_depth,
    // output
    float* cost_now,
    float2* direction_now,
    float4* momentum_depth_now
)
{
    // We first shoot a ray from the camera center of the reference view through the reference pixel.
    float3 screen_sample_world_coord;
    float3 direction;
    float3 momentum;
    generate_ray_towards_pixel(
        (float)p.x,
        (float)p.y,
        ref_cam_focal_length,
        ref_cam_m_inv,
        ref_cam_P_col34,
        ref_cam_origin,
        screen_sample_world_coord,
        direction,
        momentum);

    // We then find a 3D point on the ray that has the minimum distance to the 3D line of the neighboring pixel.
    float3 closest_point_world_coord;
    find_closest_point_on_l1_to_l2_dir2(
        direction,
        momentum,
        neighbor_direction,
        neighbor_momentum_depth,
        closest_point_world_coord);

    // check for depth validity 
    float3 closest_ref_cam_coord;
    project_point_to_camera_space(ref_cam_rotation_mat, ref_cam_t4, closest_point_world_coord, closest_ref_cam_coord);
    if (closest_ref_cam_coord.z < ddf->minimum_depth || closest_ref_cam_coord.z > ddf->maximum_depth)
        return;

    // store the line parameters going through the closest point on the 
    float2 direction2;
    float4 momentum_depth;
    // camera ray to the 3d line in the neighboring pixel
    plucker_line3_dir2(closest_point_world_coord, neighbor_direction, direction2, momentum_depth);

    // storing the depth value in the w component of momentum_depth
    momentum_depth.w = closest_ref_cam_coord.z;

    // calculate the new multi-view cost with the new line hypothesis
    float new_cost = multiview_cost<T>(ddf, p, direction2, momentum_depth);

    // accept the new line hypothesis if it enhances the cost 
    if (new_cost < *cost_now)
    {
        *direction_now = direction2;
        *momentum_depth_now = momentum_depth;
        *cost_now = new_cost;
    }
}

template< typename T >
__device__ __forceinline__ static void line_refinement(
    device_dataframe* ddf,
    const int2 p,
    const float delta_depth_amp,
    const float delta_direction_amp
)
{
    const int rows = ddf->rows;
    const int cols = ddf->cols;

    if (p.x >= cols)
        return;
    if (p.y >= rows)
        return;

    int idx_ref = ddf->id_ref;
    const int p_index = p.y * cols + p.x;

    // Read from global memory
    float cost_now = ddf->cost[p_index];
    float2 direction_now = ddf->direction[p_index];
    float4 momentum_depth_now = ddf->momentum_depth[p_index];
    
    curandState local_rand_state = ddf->cs[p_index];

    const float max_delta_depth = delta_depth_amp;
    float delta_direction = delta_direction_amp;

    // divide delta by 4 instead of 2 for less iterations
    for (float delta_depth = max_delta_depth; delta_depth >= 0.0001f; delta_depth *= 0.25f)
    {
        float depth_change = random_between(&local_rand_state, -delta_depth, delta_depth);

        // clamping the new random depth value between the specified limits
        float new_depth = fminf(fmaxf(momentum_depth_now.w + depth_change, ddf->minimum_depth), ddf->maximum_depth);

        float3 new_direction;
        new_direction.x = direction_now.x + random_between(&local_rand_state, -delta_direction, delta_direction);
        new_direction.y = direction_now.y + random_between(&local_rand_state, -delta_direction, delta_direction);
        // new_direction.z = sqrtf(1 - pow2(direction_now.x) - pow2(direction_now.y)) + random_between(&local_rand_state, -delta_direction, delta_direction);

        new_direction.z = sqrtf(1 - pow2(direction_now.x) - pow2(direction_now.y));
        if (new_direction.z != new_direction.z)
            new_direction.z = 0.f;
        new_direction.z += random_between(&local_rand_state, -delta_direction, delta_direction);

        normalize(new_direction);

        float3 world_coord_with_new_depth;
        get_world_coord_from_pixel(
            (float)p.x,
            (float)p.y,
            new_depth,
            ddf->M_inv[idx_ref],
            ddf->P_col34[idx_ref],
            world_coord_with_new_depth);

        float2 new_line_direction;
        float4 new_line_momentum_depth;
        plucker_line3_dir2(world_coord_with_new_depth, new_direction, new_line_direction, new_line_momentum_depth);
        // storing the depth value in the w component of new momentum_depth
        new_line_momentum_depth.w = new_depth;
        // calculate the new multi-view cost with the new line hypothesis
        float new_cost = multiview_cost<T>(ddf, p, new_line_direction, new_line_momentum_depth);

        // accept the new random line parameters if they reduce the multi-view cost 
        if (new_cost < cost_now)
        {
            direction_now = new_line_direction;
            momentum_depth_now = new_line_momentum_depth;
            cost_now = new_cost;
        }

        delta_direction *= 0.25f;
    }

    // Save to global memory
    ddf->cost[p_index] = cost_now;
    ddf->direction[p_index] = direction_now;
    ddf->momentum_depth[p_index] = momentum_depth_now;

    // copy rand state back to global memory
    ddf->cs[p_index] = local_rand_state;
}

template< typename T >
__device__ __forceinline__ void checkerboard_spatial_prop(
    device_dataframe* ddf,
    const int2 p
)
{
    const int rows = ddf->rows;
    const int cols = ddf->cols;

    if (p.x >= cols)
        return;
    if (p.y >= rows)
        return;

    int idx_ref = ddf->id_ref;
    const int p_index = p.y * cols + p.x;

    // Read from global memory
    float cost_now = ddf->cost[p_index];
    float2 direction_now = ddf->direction[p_index];
    float4 momentum_depth_now = ddf->momentum_depth[p_index];

    int2 np; // neighbor pixel
    int np_index;

#define SPATIAL_PROPAGATION(p_neighbor_index) \
    spatial_propagation<T>( \
        ddf,\
        p,\
        ddf->fx[idx_ref],\
        ddf->M_inv[idx_ref],\
        ddf->P_col34[idx_ref],\
        ddf->c4[idx_ref],\
        ddf->R[idx_ref],\
        ddf->t4[idx_ref],\
        ddf->direction[p_neighbor_index],\
        ddf->momentum_depth[p_neighbor_index],\
        &cost_now,\
        &direction_now,\
        &momentum_depth_now)   


    // axis-aligned neighbors
    // up
    if (p.y > 0) 
    {
        np.y = p.y - 1;
        np.x = p.x;
        np_index = p_index - cols;
        SPATIAL_PROPAGATION(np_index);
    }
    // down
    if (p.y < rows - 1) 
    {
        np.y = p.y + 1;
        np.x = p.x;
        np_index = p_index + cols;
        SPATIAL_PROPAGATION(np_index);
    }
    // left
    if (p.x > 0) 
    {
        np.y = p.y;
        np.x = p.x - 1;
        np_index = p_index - 1;
        SPATIAL_PROPAGATION(np_index);
    }
    // right
    if (p.x < cols - 1) 
    {
        np.y = p.y;
        np.x = p.x + 1;
        np_index = p_index + 1;
        SPATIAL_PROPAGATION(np_index);
    }

#ifdef EXTRA_POINTS_AXIS_ALIGNED_3_PIXEL_FAR
    // axis-aligned neighbors 3 pixels apart
    // up
    if (p.y > 2)
    {
        np.y = p.y - 3;
        np.x = p.x;
        np_index = p_index - cols * 3;
        SPATIAL_PROPAGATION(np_index);
    }
    // down
    if (p.y < rows - 3)
    {
        np.y = p.y + 3;
        np.x = p.x;
        np_index = p_index + cols * 3;
        SPATIAL_PROPAGATION(np_index);
    }
    // left
    if (p.x > 2)
    {
        np.y = p.y;
        np.x = p.x - 3;
        np_index = p_index - 3;
        SPATIAL_PROPAGATION(np_index);
    }
    // right
    if (p.x < cols - 3)
    {
        np.y = p.y;
        np.x = p.x + 3;
        np_index = p_index + 3;
        SPATIAL_PROPAGATION(np_index);
    }
#endif

#ifdef EXTRA_POINTS_AXIS_ALIGNED_5_PIXEL_FAR
    // axis-aligned neighbors 5 pixels apart
    // up
    if (p.y > 4)
    {
        np.y = p.y - 5;
        np.x = p.x;
        np_index = p_index - cols * 5;
        SPATIAL_PROPAGATION(np_index);
    }
    // down
    if (p.y < rows - 5)
    {
        np.y = p.y + 5;
        np.x = p.x;
        np_index = p_index + cols * 5;
        SPATIAL_PROPAGATION(np_index);
    }
    // left
    if (p.x > 4)
    {
        np.y = p.y;
        np.x = p.x - 5;
        np_index = p_index - 5;
        SPATIAL_PROPAGATION(np_index);
    }
    // right
    if (p.x < cols - 5)
    {
        np.y = p.y;
        np.x = p.x + 5;
        np_index = p_index + 5;
        SPATIAL_PROPAGATION(np_index);
    }
#endif

#ifdef EXTRA_POINTS_DIAGONAL_MANHATAN_3_DISTANCE
    // diagonal neighbors with manhatan distance 3
    // up_right_right
    if (p.y > 0 && p.x < cols - 2) 
    {
        np.y = p.y - 1;
        np.x = p.x + 2;
        np_index = np.y * cols + np.x;
        SPATIAL_PROPAGATION(np_index);
    }
    // down_right_right
    if (p.y < rows - 1 && p.x < cols - 2) 
    {
        np.y = p.y + 1;
        np.x = p.x + 2;
        np_index = np.y * cols + np.x;
        SPATIAL_PROPAGATION(np_index);
    }
    // up_left_left
    if (p.y > 0 && p.x > 1) 
    {
        np.y = p.y - 1;
        np.x = p.x - 2;
        np_index = np.y * cols + np.x;
        SPATIAL_PROPAGATION(np_index);
    }
    // down_left_left
    if (p.y < rows - 1 && p.x > 1) 
    {
        np.y = p.y + 1;
        np.x = p.x - 2;
        np_index = np.y * cols + np.x;
        SPATIAL_PROPAGATION(np_index);
    }
    // up_up_right
    if (p.x < cols - 1 && p.y>2) 
    {
        np.y = p.y - 2;
        np.x = p.x + 1;
        np_index = np.y * cols + np.x;
        SPATIAL_PROPAGATION(np_index);
    }
    // up_up_left
    if (p.x > 0 && p.y > 2) 
    {
        np.y = p.y - 2;
        np.x = p.x - 1;
        np_index = np.y * cols + np.x;
        SPATIAL_PROPAGATION(np_index);
    }
    // down_down_left
    if (p.x > 0 && p.y < rows - 2) 
    {
        np.y = p.y + 2;
        np.x = p.x - 1;
        np_index = np.y * cols + np.x;
        SPATIAL_PROPAGATION(np_index);
    }
    // down_down_right
    if (p.x < cols - 1 && p.y < rows - 2) 
    {
        np.y = p.y + 2;
        np.x = p.x + 1;
        np_index = np.y * cols + np.x;
        SPATIAL_PROPAGATION(np_index);
    }
#endif

#undef SPATIAL_PROPAGATION

    // Save to global memory
    ddf->cost[p_index] = cost_now;
    ddf->direction[p_index] = direction_now;
    ddf->momentum_depth[p_index] = momentum_depth_now;

}

template <typename T>
__global__ void black_spatial_propagation(
    device_dataframe* ddf
)
{
    int2 p;
    p.x = blockIdx.x * blockDim.x + threadIdx.x;
    p.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (threadIdx.x % 2 == 0)
        p.y = p.y * 2;
    else
        p.y = p.y * 2 + 1;

    // check whether any mask is provided, if so, only calculate for the masked pixels
    if (ddf->cu_array_imgs_mask[0])
    {
        float mask_val = tex2D<float>(ddf->imgs_mask[0], p.x + 0.5f, p.y + 0.5f);
        if (mask_val < 0.001f)
            return;
    }

    checkerboard_spatial_prop<T>(ddf, p);
}

template <typename T>
__global__ void black_line_refinement(
    device_dataframe* ddf,
    const float delta_depth_amp,
    const float delta_direction_amp
)
{
    int2 p;
    p.x = blockIdx.x * blockDim.x + threadIdx.x;
    p.y = blockIdx.y * blockDim.y + threadIdx.y;
    if (threadIdx.x % 2 == 0)
        p.y = p.y * 2;
    else
        p.y = p.y * 2 + 1;

    // check whether any mask is provided, if so, only calculate for the masked pixels
    if (ddf->cu_array_imgs_mask[0])
    {
        float mask_val = tex2D<float>(ddf->imgs_mask[0], p.x + 0.5f, p.y + 0.5f);
        if (mask_val < 0.001f)
            return;
    }

    line_refinement<T>(ddf, p, delta_depth_amp, delta_direction_amp);
}

template <typename T>
__global__ void red_spatial_propagation(
    device_dataframe* ddf
)
{
    int2 p;
    p.x = blockIdx.x * blockDim.x + threadIdx.x;
    p.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (threadIdx.x % 2 == 0)
        p.y = p.y * 2 + 1;
    else
        p.y = p.y * 2;

    if (ddf->cu_array_imgs_mask[0])
    {
        float mask_val = tex2D<float>(ddf->imgs_mask[0], p.x + 0.5f, p.y + 0.5f);
        if (mask_val < 0.001f)
            return;
    }

    checkerboard_spatial_prop<T>(ddf, p);
}

template <typename T>
__global__ void red_line_refinement(
    device_dataframe* ddf,
    const float delta_depth_amp,
    const float delta_direction_amp
)
{
    int2 p;
    p.x = blockIdx.x * blockDim.x + threadIdx.x;
    p.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (threadIdx.x % 2 == 0)
        p.y = p.y * 2 + 1;
    else
        p.y = p.y * 2;

    // check whether any mask is provided, if so, only calculate for the masked pixels
    if (ddf->cu_array_imgs_mask[0])
    {
        float mask_val = tex2D<float>(ddf->imgs_mask[0], p.x + 0.5f, p.y + 0.5f);
        if (mask_val < 0.001f)
            return;
    }

    line_refinement<T>(ddf, p, delta_depth_amp, delta_direction_amp);
}
    


__device__ __forceinline__ void random_sphere(
    curandState* cs,
    float2* v
)
{
    float theta = 2 * M_PI * random_between(cs, 0.f, 1.f);
    float phi = acosf(1 - 2 * random_between(cs, 0.f, 1.f));
    v->x = sinf(phi) * cosf(theta);
    v->y = sinf(phi) * sinf(theta);
}

__global__ void setup_curand_init_kernel(
    device_dataframe* ddf
)
{
    int2 p;
    p.x = blockIdx.x * blockDim.x + threadIdx.x;
    p.y = blockIdx.y * blockDim.y + threadIdx.y;

    const int index = p.y * ddf->cols + p.x;

    // each thread gets a different sequence number, no offset
    curand_init(1234, index, 0, &ddf->cs[index]);
    // curand_init(clock64(), index, 0, &ddf->cs[index]);
}

template <typename T>
__global__ void init_kernel(
    device_dataframe* ddf,
    const float* __restrict__ initial_depths,
    const float* __restrict__ initial_directions,
    const float delta_depth_amp,
    const float delta_dir_amp
)
{
    int2 p;
    p.x = blockIdx.x * blockDim.x + threadIdx.x;
    p.y = blockIdx.y * blockDim.y + threadIdx.y;

    const int rows = ddf->rows;
    const int cols = ddf->cols;

    if (p.x >= cols)
        return;
    if (p.y >= rows)
        return;

    // float mask_val = texat(ddf->imgs_mask[0], p.x, p.y);
    // if (mask_val < 0.01f)
    //     return;

    const int index = p.y * cols + p.x;
    curandState local_rand_state = ddf->cs[index];

    float2 direction;
    direction.x = direction.y = 0.f;
    float4 momentum_depth;
    momentum_depth.x = momentum_depth.y = momentum_depth.z = momentum_depth.w = 0.f;

    // if initial depths are provided, use them
    // otherwise initialize them randomly 
    if (initial_depths)
    {
        momentum_depth.w = initial_depths[index];
        if (delta_depth_amp > 0.f)
        {
            float depth_change = random_between(&local_rand_state, -delta_depth_amp, delta_depth_amp);
            momentum_depth.w = fminf(fmaxf(momentum_depth.w + depth_change, ddf->minimum_depth), ddf->maximum_depth);
        }
       
       
    }
    
    if (momentum_depth.w == 0 || momentum_depth.w != momentum_depth.w)
        momentum_depth.w = random_between(&local_rand_state,  ddf->minimum_depth, ddf->maximum_depth);
    
    
    float3 world_coord;
    get_world_coord_from_pixel((float)p.x, (float)p.y,
    momentum_depth.w, 
    ddf->M_inv[ddf->id_ref], 
    ddf->P_col34[ddf->id_ref], 
    world_coord);
    
    // if initial directions are provided, use them
    // otherwise initialize them randomly 
    float3 initial_dir;
    initial_dir.x = initial_dir.y = initial_dir.z = 0.f;

    if (initial_directions)
    {
        initial_dir.x = initial_directions[index + 0 * rows * cols];
        initial_dir.y = initial_directions[index + 1 * rows * cols];
        initial_dir.z = initial_directions[index + 2 * rows * cols];

        normalize(initial_dir);
        initial_dir.x *= (initial_dir.z >= 0 ? 1.f : -1.f);
        initial_dir.y *= (initial_dir.z >= 0 ? 1.f : -1.f);

        if (delta_dir_amp > 0.f)
        {
            initial_dir.x = initial_dir.x + random_between(&local_rand_state, -delta_dir_amp, delta_dir_amp);
            initial_dir.y = initial_dir.y + random_between(&local_rand_state, -delta_dir_amp, delta_dir_amp);
            initial_dir.z = initial_dir.z + random_between(&local_rand_state, -delta_dir_amp, delta_dir_amp);
            normalize(initial_dir);
            initial_dir.x *= (initial_dir.z >= 0 ? 1.f : -1.f);
            initial_dir.y *= (initial_dir.z >= 0 ? 1.f : -1.f);
        }
        
    }
    
    float initial_dir_norm = get_pow2_normf3(initial_dir);
    if (fabsf(initial_dir_norm - 1.f) > 0.0001f || initial_dir_norm != initial_dir_norm)
    {
        float2 random_vector;
        random_sphere(&local_rand_state, &random_vector);
        plucker_line3_dir2(world_coord, random_vector, direction, momentum_depth);
    }
    else
    {
        plucker_line3_dir2(world_coord, initial_dir, direction, momentum_depth);
        
    }

    ddf->direction[index] = direction;
    ddf->momentum_depth[index] = momentum_depth;

    // calculating the initial cost of the assigned random values
    ddf->cost[index] = multiview_cost<T>(ddf, p, direction, momentum_depth);

    // copy rand state back to global memory
    ddf->cs[index] = local_rand_state;
}


__global__ void get_pixel_locations(
    device_dataframe* ddf,
    float* __restrict__ final_positions,
    float* __restrict__ final_directions
)
{
    int2 p;
    p.x = blockIdx.x * blockDim.x + threadIdx.x;
    p.y = blockIdx.y * blockDim.y + threadIdx.y;

    const int rows = ddf->rows;
    const int cols = ddf->cols;

    if (p.x >= cols)
        return;
    if (p.y >= rows)
        return;

    // check whether any mask is provided, if so, only calculate for the masked pixels
    if (ddf->cu_array_imgs_mask[0])
    {
        float mask_val = tex2D<float>(ddf->imgs_mask[0], p.x + 0.5f, p.y + 0.5f);
        if (mask_val < 0.001f)
            return;
    }

    int idx_ref = ddf->id_ref;
    const int p_index = p.y * cols + p.x;

    float2 direction = ddf->direction[p_index];
    float pixel_depth = ddf->momentum_depth[p_index].w;
    float3 world_coord;
    get_world_coord_from_pixel(
        (float)p.x,
        (float)p.y,
        pixel_depth,
        ddf->M_inv[idx_ref],
        ddf->P_col34[idx_ref],
        world_coord);    

    // placing pixel depth in the fourth element
    final_positions[p_index + 0 * rows * cols] = world_coord.x;
    final_positions[p_index + 1 * rows * cols] = world_coord.y;
    final_positions[p_index + 2 * rows * cols] = world_coord.z;
    final_positions[p_index + 3 * rows * cols] = pixel_depth;

    final_directions[p_index + 0 * rows * cols] = direction.x;
    final_directions[p_index + 1 * rows * cols] = direction.y;
    float direction_z = sqrtf(1 - pow2(direction.x) - pow2(direction.y));
    if (direction_z != direction_z)
        direction_z = 0.f;
    final_directions[p_index + 2 * rows * cols] = direction_z;
        
}

        

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
    const float* initial_depths,
    const float* initial_directions,
    // if you want to print additional informations
    const bool verbose
)
{
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    device_dataframe* ddf = get_ddf_instance();

    // load the given lpmvs parameters
    ddf->alpha = alpha;
    ddf->iterations = iterations;
    ddf->minimum_depth = minimum_depth;
    ddf->maximum_depth = maximum_depth;
    ddf->sampling_radius = sampling_radius;
    ddf->half_samples_num = (samples_num / 2);
    ddf->occlusion_depth_thresh = occulusion_depth_thresh;
    
    ddf->id_ref = ref_view_id;
    ddf->view_selection_subset_number = view_selection_subset_number;
    CUDA_SAFE_CALL(cudaMemcpy(ddf->view_selection_subset, view_selection_subset_cpu, view_selection_subset_number * sizeof(int), cudaMemcpyHostToDevice));
    
    int rows = ddf->rows;
    int cols = ddf->cols;
    CUDA_SAFE_CALL(cudaMalloc(&ddf->cs, rows * cols * sizeof(curandState)));

    // now send the dataframe to gpu
    device_dataframe* gpu_frame{ nullptr };
    CUDA_SAFE_CALL(cudaMalloc(&gpu_frame, sizeof(device_dataframe)));
    CUDA_SAFE_CALL(cudaMemcpy(gpu_frame, ddf, sizeof(device_dataframe), cudaMemcpyHostToDevice));

    // some timing related variables
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    float milliseconds = 0.f;

    dim3 grid_size_initrand = dim3(div_round_up(cols, 16), div_round_up(rows, 16));
    dim3 block_size_initrand = dim3(16, 16);

    int block_w = 16;
    int block_h = block_w / 2;
    dim3 grid_size = dim3(div_round_up(cols, block_w), div_round_up(rows, block_h));
    dim3 block_size = dim3(block_w, block_h);

    CUDA_SAFE_CALL(cudaEventRecord(start));

    if (verbose) printf("Random initialization ... \n");

    setup_curand_init_kernel <<< grid_size_initrand, block_size_initrand >>> (gpu_frame);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERRORS("setup_curand_init_kernel kernel launch failure");


    init_kernel<float> <<< grid_size_initrand, block_size_initrand >>> (gpu_frame, initial_depths, initial_directions, delta_depth_amp, deleta_dir_amp);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERRORS("init kernel launch failure");

    for (int it{ 0 }; it < ddf->iterations; ++it)
    {
        float depth_perturbation_amplitude = 0.f;
        if (delta_depth_amp > 0)
            depth_perturbation_amplitude = 0.5f * delta_depth_amp / ((float)(it+1));
        else
            depth_perturbation_amplitude = 0.5f * (maximum_depth - minimum_depth) / ((float)(it+1));

        float direction_perturbation_amplitude = 0.f;
        if (deleta_dir_amp > 0)
            direction_perturbation_amplitude = 1.f;
        else
            direction_perturbation_amplitude = deleta_dir_amp;

        if (verbose) printf("Iteration %d \n", it + 1);
        if (verbose) printf("Black cells spatial propation ... \n");
        black_spatial_propagation<float> <<< grid_size, block_size >>> (gpu_frame);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERRORS("black_spatial_propagation kernel launch failure");

        if (verbose) printf("Black cells line refinement ... \n");
        black_line_refinement<float> <<< grid_size, block_size >>> (gpu_frame, depth_perturbation_amplitude, direction_perturbation_amplitude);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERRORS("black_line_refinement kernel launch failure");

        if (verbose) printf("Red cells spatial propation ... \n");
        red_spatial_propagation<float> <<< grid_size, block_size >>> (gpu_frame);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERRORS("red_spatial_propagation kernel launch failure");

        if (verbose) printf("Red cells line refinement ... \n");
        red_line_refinement<float> <<< grid_size, block_size >>> (gpu_frame, depth_perturbation_amplitude, direction_perturbation_amplitude);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERRORS("red_line_refinement kernel launch failure");

    }

    CUDA_SAFE_CALL(cudaEventRecord(stop));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, start, stop));

    if (verbose) printf("\n");
    if (verbose) printf("Kernel operation completed !!! \n");
    if (verbose) printf("Ellapsed time (GPU): %16.10f ms\n", milliseconds);
    if (verbose) printf("\n");

    get_pixel_locations <<< grid_size, block_size >>> (gpu_frame, final_positions, final_directions);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERRORS("get_pixel_locations kernel launch failure");

    CUDA_SAFE_CALL(cudaFree(ddf->cs));
    CUDA_SAFE_CALL(cudaFree(gpu_frame));
    gpu_frame = nullptr;

    ddf->reset_containers();

    return 0;

}

}
