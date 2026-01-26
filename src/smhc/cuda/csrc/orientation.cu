#include <cstdint>
#include "auxiliary.h"
#include "orientation.h"


namespace solution::gabor
{

    __device__ __forceinline__ float 
    get_pixel_value(
        const float* img,
        const int img_rows,
        const int img_cols,
        const int x,
        const int y
    )
    {
        if (x >= img_cols || y >= img_rows || x < 0 || y < 0) 
        {
           return 0.0f;
        } 
        else 
        {
           return img[y * img_cols + x];
        }
    }

    __global__ void 
    gabor_filter_tiled_kernel(
        const float* __restrict__ img,
        const int img_rows,
        const int img_cols, 
        const float* __restrict__ all_kernels,
        const int kernel_size, 
        const int num_kernels,
        float* __restrict__ result,
        int* __restrict__ max_resp_indices
    )
    {
        int2 p;
        p.x = blockIdx.x * blockDim.x + threadIdx.x;
        p.y = blockIdx.y * blockDim.y + threadIdx.y;

        const int start_x = blockIdx.x * blockDim.x;
        const int start_y = blockIdx.y * blockDim.y;

        int n = kernel_size / 2;
        int K2 = kernel_size * kernel_size;
        // the first part of the shared memory will be storing the image tile pixels
        // the rest of the K2 part will store the kernel paramters
        extern __shared__ __align__(sizeof(float)) float s_data[]; // [height][width]

#define CALC_FLATTENED_INDEX(x, y, col) (x) + (y) * (col)
        
        int thread_id = CALC_FLATTENED_INDEX(threadIdx.x, threadIdx.y, blockDim.x);
        int shared_array_width = blockDim.x + kernel_size - 1;
        int shared_array_width2 = shared_array_width * shared_array_width;
        const int num_blocks = (shared_array_width2 + blockDim.x * blockDim.y - 1) / (blockDim.x * blockDim.y);

        for (int b = 0; b < num_blocks; ++b)
        {
            int tid = b * (blockDim.x * blockDim.y) + thread_id;
            if (tid < shared_array_width2)
            {
                int local_x = tid % shared_array_width;
                int local_y = tid / shared_array_width;

                int x = start_x + local_x;
                int y = start_y + local_y;

                s_data[CALC_FLATTENED_INDEX(local_x, local_y, shared_array_width)] = get_pixel_value(img, img_rows, img_cols, x-n, y-n);
            }
        }

        float sum = 0.0f;
        float max_resp = 0.f;
        int max_resp_index = 0;
        for (int k = 0; k < num_kernels; ++k)
        {
        
            if (thread_id < K2)
                s_data[shared_array_width2 + thread_id] = all_kernels[k * K2 + thread_id];
        
            __syncthreads();

            sum = 0.0f;
            if (p.y < img_rows && p.x < img_cols)
            {
                for (int i = 0; i < kernel_size; ++i) // kernel row
                {
                    for (int j = 0; j < kernel_size; ++j) // kernel column
                    {
                        // s_img[threadIdx.y + i][threadIdx.x + j] * s_kernel[i][j]
                        sum += s_data[CALC_FLATTENED_INDEX(threadIdx.x + j, threadIdx.y + i, shared_array_width)] \
                            * s_data[shared_array_width2 + CALC_FLATTENED_INDEX(j, i, kernel_size)];
                    }
                }
                
                result[((uint64_t)k) * img_cols * img_rows + CALC_FLATTENED_INDEX(p.x, p.y, img_cols)] = sum;
                sum = fabsf(sum);
                if (sum > max_resp)
                {
                    max_resp = sum;
                    max_resp_index = k;
                }
            }

            __syncthreads();

        }

        if (p.y < img_rows && p.x < img_cols)
        {
            max_resp_indices[CALC_FLATTENED_INDEX(p.x, p.y, img_cols)] = max_resp_index;
        }
        

#undef CALC_FLATTENED_INDEX

    }

    __global__ void ori_img_conf_kernel(
        const float* __restrict__ ori_responses,
        const int* __restrict__ max_resp_indices,
        const int img_rows, 
        const int img_cols, 
        const int kernel_size, 
        const int num_kernels,
        float* __restrict__ orientation_img,
        float* __restrict__ confindence_img     
    )
    {

        //each block is assigned to a row of an image, iy index of y value                  
        int iy = blockIdx.y;

        //each thread is assigned to a pixel of a row, ix index of x value
        int ix = threadIdx.x + blockIdx.x * blockDim.x;

        //idx global index (all blocks) of the image pixel 
        int idx = iy * img_cols + ix;

        if (iy < img_rows && ix < img_cols)
        {
            float responses_sum = 0.f;
            float variance = 0.f;
            float angle_diff = 0.f;
            float current_resp = 0.f;
            float best_angle = max_resp_indices[idx] * (M_PI / (float)num_kernels);

            for (int k = 0; k < num_kernels; k++)
            {
                current_resp = fabsf(ori_responses[((uint64_t)k) * img_cols * img_rows + idx]);
                responses_sum += current_resp;
                angle_diff = M_PI_2 - fabsf(fabsf(best_angle - (k * (M_PI / (float)num_kernels))) - M_PI_2);
                variance += angle_diff * angle_diff * current_resp;
            }

            variance /= (responses_sum + 1.e-6f);
            confindence_img[idx] = (variance > 1.e-3f) ? 1 / (variance * variance) : 1.f;
            orientation_img[idx] = (max_resp_indices[idx] * M_PI / (float)num_kernels - M_PI_2);
        }

    }


    void calculate_orientation_maps(
        const float* img_data,
        const int img_rows,
        const int img_cols,
        const float* gb_kernels,
        const int kernels_num,
        const int kernels_size,
        float* ori2d,
        float* conf
    )
    {
        const int k2 = kernels_size * kernels_size;

        float* d_all_results;
        CUDA_SAFE_CALL(cudaMalloc(&d_all_results, ((uint64_t)kernels_num) * img_rows * img_cols * sizeof(float)));

        int* d_max_resp_index;
        CUDA_SAFE_CALL(cudaMalloc(&d_max_resp_index, img_rows * img_cols * sizeof(int)));

        int block_width = 32;
        int block_height = block_width;

        dim3 grid_size = dim3(div_round_up(img_cols, block_width), div_round_up(img_rows, block_height));
        dim3 block_size = dim3(block_width, block_height);

        int shared_load_w = block_width + kernels_size - 1;
        int shared_load_h = block_height + kernels_size - 1;

        int needed_shared_memory_size = (k2 + shared_load_w * shared_load_h) * sizeof(float);

        gabor_filter_tiled_kernel <<< grid_size, block_size, needed_shared_memory_size >>> (
            img_data,
            img_rows,
            img_cols,
            gb_kernels,
            kernels_size,
            kernels_num,
            d_all_results,
            d_max_resp_index
        );

        cudaDeviceSynchronize();
        CUDA_CHECK_ERRORS("gabor filter tiled kernel launch failure!");

        block_size = dim3(1024);
        grid_size = dim3(div_round_up(img_cols, 1024), img_rows);

        ori_img_conf_kernel <<< grid_size, block_size >>> (
            d_all_results,
            d_max_resp_index,
            img_rows,
            img_cols,
            kernels_size,
            kernels_num,
            ori2d,
            conf
        );

        cudaDeviceSynchronize();
        CUDA_CHECK_ERRORS("orientation and cofidence kernel launch failure!");

        // clean up
        CUDA_SAFE_CALL(cudaFree(d_all_results));
        CUDA_SAFE_CALL(cudaFree(d_max_resp_index));


    }
}


// about the extracted orientations
// angle zero is corresponding to this gabor filter configuration
//      --------------------------> x
//      |           +
//      |         - + - 
//      |       + - + - + 
//      |     - + - + - + -
//      |       + - + - + 
//      |         - + -   
//      |           +
//      V
//      y
// 
// therefore, it is detecting lines with vertical directions, i.e.
//      --------------------------> x
//      |           |
//      |           |  
//      |           |   
//      |           | 
//      |           |  
//      |           |    
//      |           |
//      V
//      y
//
// at angle pi / 4 the gabor filter configuration and the detected line direction would be as,
//      --------------------------> x       --------------------------> x
//      |                   +               |                    /
//      |               - + -               |                  /  
//      |           + - + - +               |                /   
//      |        - + - + - + -              |              / 
//      |        + - + - +                  |            /  
//      |        - + -                      |          /    
//      |        +                          |        /
//      V                                   V
//      y                                   y
//
// considering the positive x and y direction the line angle detected would be -pi / 4
// at angle 3pi / 4 the gabor filter configuration and the detected line direction would be as,
//      --------------------------> x       --------------------------> x
//      |    +                              |    \
//      |    - + -                          |      \  
//      |     + - + - +                     |        \
//      |    - + - + - + -                  |          \ 
//      |        + - + - +                  |            \  
//      |           - + -                   |              \    
//      |               +                   |                \
//      V                                   V
//      y                                   y
//
// considering the positive x and y direction the line angle detected would be pi / 4
// 
// the conversion formula between the gabor kernel angle and detected line angle would be as,
//      line_angle = gabor_anlge - pi / 2
//