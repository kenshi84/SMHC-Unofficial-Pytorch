#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <assert.h>

#include "line_filtering.h"
#include "device_dataframe.h"

namespace solution::line_filtering
{

__device__ __forceinline__ void get_pixel_coord_from_world_coord(
const float* proj_mat,
const float4& world_coord,
float2* pixel
)
{
	float4 pos;
	matvecmul4_hom(proj_mat, world_coord, pos);
	pixel->x = pos.x / pos.z;
	pixel->y = pos.y / pos.z;
}

__global__ void line_filtering_kernel(
	device_dataframe* ddf,
    const float position_consistency_threshold,
    const float direction_diff_threshold_rad,
    const int consistent_neighboring_views_threshold
)
{
	int2 p;
	//each block is assigned to a row of an image, iy index of y value                  
	p.y = blockIdx.y;
	//each thread is assigned to a pixel of a row, ix index of x value
	p.x = threadIdx.x + blockIdx.x * blockDim.x;

	int img_cols = ddf->cols;
	int img_rows = ddf->rows;

	// Ignore launch padding before accessing any pixel buffers.
	if (p.x >= img_cols || p.y >= img_rows)
		return;

	int ref_pixel_idx = p.y * img_cols + p.x;
	float4 ref_pos = ddf->positions[0][ref_pixel_idx];
	float4 ref_dir = ddf->directions[0][ref_pixel_idx];

    if (ddf->cu_array_imgs_mask[0])
    {
        float mask_val = tex2D<float>(ddf->imgs_mask[0], p.x + 0.5f, p.y + 0.5f);
        if (mask_val < 0.001f)
            return;
    }

	float2 neighbor_pixel_coord;
	int consistent_neighbors = 0;
	
	// also accumulate the color value for consistent neighbors, this is going to be later used in GS optimization
	// float color_accum = tex2D<float>(ddf->imgs[0], p.x + 0.5f, p.y + 0.5f);

	for (int i = 0; i < ddf->view_selection_subset_number; ++i)
	{
		int idx_neighbor = ddf->view_selection_subset[i];

		get_pixel_coord_from_world_coord(ddf->P[idx_neighbor], ref_pos, &neighbor_pixel_coord);
		int2 p_neighbor;
		p_neighbor.y = __float2int_rz(neighbor_pixel_coord.y);
		p_neighbor.x = __float2int_rz(neighbor_pixel_coord.x);

		if (p_neighbor.y < 0 || p_neighbor.y >= img_rows || p_neighbor.x < 0 || p_neighbor.x >= img_cols)
			continue;

		int neighbor_pixel_idx = p_neighbor.y * img_cols + p_neighbor.x;

		float4 neighbor_pos = ddf->positions[i + 1][neighbor_pixel_idx];
		float4 neighbor_dir = ddf->directions[i + 1][neighbor_pixel_idx];

		sub4(neighbor_pos, ref_pos);
		float pos_diff = get_normf3(neighbor_pos);
		float dir_diff = acosf(dot_product(neighbor_dir, ref_dir) / (get_normf3(neighbor_dir) * get_normf3(ref_dir)));

		// in case neighbor_dir and ref_dir are completely aligned acosf() can generate nan
		if (dir_diff != dir_diff)
			dir_diff = 0.f;
		dir_diff = M_PI_2 - fabsf(dir_diff - M_PI_2);

		if (pos_diff < position_consistency_threshold && dir_diff < direction_diff_threshold_rad)
		{
			consistent_neighbors += 1;
			// color_accum += tex2D<float>(ddf->imgs[i + 1], p_neighbor.x + 0.5f, p_neighbor.y + 0.5f);
		}
	}

	if (consistent_neighbors >= consistent_neighboring_views_threshold)
	{
		ddf->consistency_flag[ref_pixel_idx] = 1;
		// storing the accumulated average color in the fourth element of the direction data
		// color_accum /= (consistent_neighbors + 1);
		// ddf->directions[0][ref_pixel_idx].w = color_accum;
	}

}

// constructing the intermediate maps
__global__ void get_intermediate_dir_depth_map(
    device_dataframe* __restrict__ ddf,
    float* __restrict__ int_map
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

    const int p_index = p.y * cols + p.x;
    uint8_t consistent_estimation = ddf->consistency_flag[p_index];

    // the first channel in the int_map contains the estimated depth values
    // if the value is negative it is an indication of no reliable consistent
    // depth estimation from the previous round of lpmvs.
    const float int_map_depth = int_map[p_index + 0 * rows * cols];

    if (consistent_estimation == 1 && int_map_depth < 1.e-6f)
    {
        float4 direction_estimate = ddf->directions[0][p_index];
    	float4 position_estimate = ddf->positions[0][p_index];

        int_map[p_index + 0 * rows * cols] = position_estimate.w;
    	int_map[p_index + 1 * rows * cols] = direction_estimate.x;
    	int_map[p_index + 2 * rows * cols] = direction_estimate.y;
    }
    else if (consistent_estimation == 0 && int_map_depth < 1.e-6f)
    {
        int_map[p_index + 0 * rows * cols] = -1.f;
        int_map[p_index + 1 * rows * cols] = tex2D<float>(ddf->imgs_ori[0], p.x + 0.5f, p.y + 0.5f);
        int_map[p_index + 2 * rows * cols] = tex2D<float>(ddf->imgs_conf[0], p.x + 0.5f, p.y + 0.5f);

    }
    else if (consistent_estimation == 1 && int_map_depth > 1.e-6f)
    {
        // turning off (!) the consistency flag here to prevent reaccumulation of already
        // collected points in the filtered point cloud
        ddf->consistency_flag[p_index] = 0;
    }
    
}


int run(
    const int ref_view_id,
    const int view_selection_subset_number,
    const int* view_selection_subset_cpu,
    float* int_map,
    //
    const float position_consistency_threshold,
    const float direction_diff_threshold_rad,
    const int consistent_neighboring_views_threshold,
    //
    const bool verbose
)
{
    device_dataframe* ddf = get_ddf_instance();
    
    ddf->id_ref = ref_view_id;
    ddf->view_selection_subset_number = view_selection_subset_number;
    CUDA_SAFE_CALL(cudaMemcpy(ddf->view_selection_subset, view_selection_subset_cpu, view_selection_subset_number * sizeof(int), cudaMemcpyHostToDevice));

    int rows = ddf->rows;
    int cols = ddf->cols;

    device_dataframe* gpu_frame{ nullptr };
    CUDA_SAFE_CALL(cudaMalloc(&gpu_frame, sizeof(device_dataframe)));
    CUDA_SAFE_CALL(cudaMemcpy(gpu_frame, ddf, sizeof(device_dataframe), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    float milliseconds = 0.f;

    dim3 block_size = dim3(1024);
    dim3 grid_size = dim3(div_round_up(cols, 1024), rows);

    if (verbose) printf("Line filtering ... \n");
    CUDA_SAFE_CALL(cudaEventRecord(start));

    line_filtering_kernel <<< grid_size, block_size >>> (
        gpu_frame,
        position_consistency_threshold,
        direction_diff_threshold_rad,
        consistent_neighboring_views_threshold
    );

    cudaDeviceSynchronize();
    CUDA_CHECK_ERRORS("line filtering kernel launch failure");

    CUDA_SAFE_CALL(cudaEventRecord(stop));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, start, stop));

    if (verbose) printf("\n");
    if (verbose) printf("Kernel operation completed !!! \n");
    if (verbose) printf("Ellapsed time (GPU): %16.10f ms\n", milliseconds);
    if (verbose) printf("\n");


    int block_w = 32;
    int block_h = block_w;
    grid_size = dim3(div_round_up(cols, block_w), div_round_up(rows, block_h));
    block_size = dim3(block_w, block_h);

    get_intermediate_dir_depth_map <<< grid_size, block_size >>> (gpu_frame, int_map);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERRORS("get_intermediate_dir_depth_map kernel launch failure");

    CUDA_SAFE_CALL(cudaFree(gpu_frame));
    gpu_frame = nullptr;
    ddf->reset_containers();

    return 0;
   
}

}