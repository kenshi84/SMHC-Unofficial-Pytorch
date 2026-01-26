#include <algorithm>
#include <stdexcept>

#include "auxiliary.h"
#include "device_dataframe.h"

namespace solution
{
    namespace
    {
        void add_image_to_cuda_texture(
            const int img_idx,
            const int img_rows,
            const int img_cols,
            const float* img_data,
            cudaTextureObject_t texs[],
            cudaArray_t cu_arrays[],
            cudaTextureFilterMode filter_mode=cudaFilterModeLinear
        )
        {
            // create channel with floating point type
            cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();

            // allocate array with correct size and number of channels
            CUDA_SAFE_CALL(cudaMallocArray(&cu_arrays[img_idx], &channel_desc, img_cols, img_rows));
        
            CUDA_SAFE_CALL(
                cudaMemcpy2DToArray(cu_arrays[img_idx], 0, 0, img_data, img_cols * sizeof(float), img_cols * sizeof(float), img_rows, cudaMemcpyHostToDevice)
            );
        
            // specify texture
            struct cudaResourceDesc res_desc;
            memset(&res_desc, 0, sizeof(res_desc));
            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = cu_arrays[img_idx];
        
            // specify texture object parameters
            struct cudaTextureDesc tex_desc;
            memset(&tex_desc, 0, sizeof(tex_desc));
            tex_desc.addressMode[0] = cudaAddressModeClamp; // cudaAddressModeWrap
            tex_desc.addressMode[1] = cudaAddressModeClamp; // cudaAddressModeWrap
            tex_desc.filterMode = filter_mode;
            tex_desc.readMode = cudaReadModeElementType;
            tex_desc.normalizedCoords = 0;
        
            // create texture object
            CUDA_SAFE_CALL(cudaCreateTextureObject(&(texs[img_idx]), &res_desc, &tex_desc, nullptr));

            return;
        }
    
        // a single instance to be hold during the entire algorithm run
        // I think this would do for the time being and above all it prevents a long list of inputs to each kernel call later
        device_dataframe* ddf{ nullptr };
        host_dataframe* hdf{ nullptr };
        
    } // anonymous namespace

    host_dataframe::host_dataframe(const int img_rows, const int img_cols)
    {
        rows = img_rows;
        cols = img_cols;
        
        for (int i{ 0 }; i < max_cameras_num; ++i)
        {
            P_cpu[i] = (float*)malloc(sizeof(float) * 3 * 4);
            M_inv_cpu[i] = (float*)malloc(sizeof(float) * 3 * 3);
            K_cpu[i] = (float*)malloc(sizeof(float) * 3 * 3);
            K_inv_cpu[i] = (float*)malloc(sizeof(float) * 3 * 3);
            R_cpu[i] = (float*)malloc(sizeof(float) * 3 * 3);
            R_inv_cpu[i] = (float*)malloc(sizeof(float) * 3 * 3);
            line_proj_mat_cpu[i] = (float*)malloc(sizeof(float) * 3 * 6);
        }

        std::fill(imgs_gray_cpu, imgs_gray_cpu + max_cameras_num, nullptr);
        std::fill(imgs_ori_cpu, imgs_ori_cpu + max_cameras_num, nullptr);
        std::fill(imgs_conf_cpu, imgs_conf_cpu + max_cameras_num, nullptr);
        std::fill(imgs_mask_cpu, imgs_mask_cpu + max_cameras_num, nullptr);
    }

    host_dataframe::~host_dataframe()
    {

    }

    void host_dataframe::destroy()
    {
        for (int i{ 0 }; i < max_cameras_num; ++i)
        {
            free(P_cpu[i]            );
            free(M_inv_cpu[i]        );
            free(K_cpu[i]            );
            free(K_inv_cpu[i]        );
            free(R_cpu[i]            );
            free(R_inv_cpu[i]        );
            free(line_proj_mat_cpu[i]);
            
            // dont free these, cuase their memory management is with torch
            // we just were rude enough (!) to steal some pointers to their data
            imgs_gray_cpu[i] = nullptr;
            imgs_ori_cpu[i] = nullptr;
            imgs_conf_cpu[i] = nullptr;
            imgs_mask_cpu[i] = nullptr;
        }
    }


    device_dataframe::device_dataframe(
        const int img_rows, 
        const int img_cols
    )
    {
        // total number of pixels
        const int n = img_rows * img_cols;
        rows = img_rows;
        cols = img_cols;

        std::fill(imgs, imgs + max_neighboring_views, 0);
        std::fill(cu_array_imgs, cu_array_imgs + max_neighboring_views, nullptr);

        std::fill(imgs_mask, imgs_mask + max_neighboring_views, 0);
        std::fill(cu_array_imgs_mask, cu_array_imgs_mask + max_neighboring_views, nullptr);

        std::fill(imgs_ori, imgs_ori + max_neighboring_views, 0);
        std::fill(cu_array_imgs_ori, cu_array_imgs_ori + max_neighboring_views, nullptr);

        std::fill(imgs_conf, imgs_conf + max_neighboring_views, 0);
        std::fill(cu_array_imgs_conf, cu_array_imgs_conf + max_neighboring_views, nullptr);

        std::fill(imgs_initial_depth, imgs_initial_depth + max_neighboring_views, 0);
        std::fill(cu_array_imgs_initial_depth, cu_array_imgs_initial_depth + max_neighboring_views, nullptr);

        for (int i{ 0 }; i < max_cameras_num; ++i)
        {
            CUDA_SAFE_CALL(cudaMalloc(&P[i], sizeof(float) * 3 * 4));
            CUDA_SAFE_CALL(cudaMalloc(&M_inv[i], sizeof(float) * 3 * 3));
            CUDA_SAFE_CALL(cudaMalloc(&K[i], sizeof(float) * 3 * 3));
            CUDA_SAFE_CALL(cudaMalloc(&K_inv[i], sizeof(float) * 3 * 3));
            CUDA_SAFE_CALL(cudaMalloc(&R[i], sizeof(float) * 3 * 3));
            CUDA_SAFE_CALL(cudaMalloc(&R_inv[i], sizeof(float) * 3 * 3));
            CUDA_SAFE_CALL(cudaMalloc(&line_proj_mat[i], sizeof(float) * 3 * 6));
        }

        CUDA_SAFE_CALL(cudaMalloc(&view_selection_subset, sizeof(int) * max_neighboring_views));
        CUDA_SAFE_CALL(cudaMalloc(&direction, sizeof(float2) * n));
        CUDA_SAFE_CALL(cudaMalloc(&momentum_depth, sizeof(float4) * n));
        CUDA_SAFE_CALL(cudaMalloc(&cost, sizeof(float) * n));
        // CUDA_SAFE_CALL(cudaMalloc(&debug_data, sizeof(float4) * n));

        CUDA_SAFE_CALL(cudaMemset(direction, 0, sizeof(float2) * n));
        CUDA_SAFE_CALL(cudaMemset(momentum_depth, 0, sizeof(float4) * n));
        CUDA_SAFE_CALL(cudaMemset(cost, 0, sizeof(float) * n));
        // CUDA_SAFE_CALL(cudaMemset(debug_data, 0, sizeof(float4) * n));

        // printf("max neighboring views in the device dataframe struct is: %d\n", max_neighboring_views);

        for (int i{ 0 }; i < 1 + max_neighboring_views; ++i)
        {
            CUDA_SAFE_CALL(cudaMalloc(&directions[i], sizeof(float4) * n));
            CUDA_SAFE_CALL(cudaMalloc(&positions[i], sizeof(float4) * n));
        
            CUDA_SAFE_CALL(cudaMemset(directions[i], 0, sizeof(float4) * n));
            CUDA_SAFE_CALL(cudaMemset(positions[i], 0, sizeof(float4) * n));
        }

        CUDA_SAFE_CALL(cudaMalloc(&consistency_flag, sizeof(uint8_t) * n));
        CUDA_SAFE_CALL(cudaMemset(consistency_flag, 0, sizeof(uint8_t) * n));

    }

    device_dataframe::~device_dataframe() {}

    void device_dataframe::destroy_image_data()
    {
        for (int i = 0; i < max_neighboring_views; i++)
        {
            if (cu_array_imgs[i])
            {
                CUDA_SAFE_CALL(cudaFreeArray(cu_array_imgs[i]));
                CUDA_SAFE_CALL(cudaDestroyTextureObject(imgs[i]));
                cu_array_imgs[i] = nullptr;
                imgs[i] = 0;
            }

            if (cu_array_imgs_mask[i])
            {
                CUDA_SAFE_CALL(cudaFreeArray(cu_array_imgs_mask[i]));
                CUDA_SAFE_CALL(cudaDestroyTextureObject(imgs_mask[i]));
                cu_array_imgs_mask[i] = nullptr;
                imgs_mask[i] = 0;
            }

            if (cu_array_imgs_ori[i])
            {
                CUDA_SAFE_CALL(cudaFreeArray(cu_array_imgs_ori[i]));
                CUDA_SAFE_CALL(cudaDestroyTextureObject(imgs_ori[i]));
                cu_array_imgs_ori[i] = nullptr;
                imgs_ori[i] = 0;
            }

            if (cu_array_imgs_conf[i])
            {
                CUDA_SAFE_CALL(cudaFreeArray(cu_array_imgs_conf[i]));
                CUDA_SAFE_CALL(cudaDestroyTextureObject(imgs_conf[i]));
                cu_array_imgs_conf[i] = nullptr;
                imgs_conf[i] = 0;
            }

            if (cu_array_imgs_initial_depth[i])
            {
                CUDA_SAFE_CALL(cudaFreeArray(cu_array_imgs_initial_depth[i]));
                CUDA_SAFE_CALL(cudaDestroyTextureObject(imgs_initial_depth[i]));
                cu_array_imgs_initial_depth[i] = nullptr;
                imgs_initial_depth[i] = 0;
            }
        }
    }

    void device_dataframe::reset_containers()
    {
        destroy_image_data();

        CUDA_SAFE_CALL(cudaMemset(view_selection_subset, 0, view_selection_subset_number * sizeof(int)));

        CUDA_SAFE_CALL(cudaMemset(direction, 0, rows * cols * sizeof(float2)));
        CUDA_SAFE_CALL(cudaMemset(momentum_depth, 0, rows * cols * sizeof(float4)));

        CUDA_SAFE_CALL(cudaMemset(cost, 0, rows * cols * sizeof(float)));
        // CUDA_SAFE_CALL(cudaMemset(debug_data, 0, rows * cols * sizeof(float4)));

        for (int i{ 0 }; i < 1 + max_neighboring_views; ++i)
        {
            CUDA_SAFE_CALL(cudaMemset(directions[i], 0, sizeof(float4) * rows * cols));
            CUDA_SAFE_CALL(cudaMemset(positions[i], 0, sizeof(float4) * rows * cols));
        }
        CUDA_SAFE_CALL(cudaMemset(consistency_flag, 0, sizeof(uint8_t) * rows * cols));

    }

    void device_dataframe::destroy()
    {
        destroy_image_data();
    
        for (int i{ 0 }; i < max_cameras_num; ++i)
        {
            CUDA_SAFE_CALL(cudaFree(P[i]));
            CUDA_SAFE_CALL(cudaFree(M_inv[i]));
            CUDA_SAFE_CALL(cudaFree(K[i]));
            CUDA_SAFE_CALL(cudaFree(K_inv[i]));
            CUDA_SAFE_CALL(cudaFree(R[i]));
            CUDA_SAFE_CALL(cudaFree(R_inv[i]));
            CUDA_SAFE_CALL(cudaFree(line_proj_mat[i]));
        }
    
        CUDA_SAFE_CALL(cudaFree(view_selection_subset));
        CUDA_SAFE_CALL(cudaFree(direction));
        CUDA_SAFE_CALL(cudaFree(momentum_depth));
    
        CUDA_SAFE_CALL(cudaFree(cost));
        // CUDA_SAFE_CALL(cudaFree(debug_data));

        for (int i{ 0 }; i < 1 + max_neighboring_views; ++i)
        {
            CUDA_SAFE_CALL(cudaFree(directions[i]));
            CUDA_SAFE_CALL(cudaFree(positions[i]));
        }

        CUDA_SAFE_CALL(cudaFree(consistency_flag));
    }

    void send_camera_parameters_to_dataframes(
        device_dataframe*   ddf,
        host_dataframe*     hdf,
        const int           cam_idx,
        const float*        cam_P_cpu,
        const float4*       cam_P_col34_cpu,
        const float*        cam_M_inv_cpu,
        const float*        cam_K_cpu,
        const float*        cam_K_inv_cpu,
        const float*        cam_R_cpu,
        const float4*       cam_t4_cpu,
        const float*        cam_R_inv_cpu,
        const float4*       cam_c4_cpu,
        const float*        cam_line_proj_mat_cpu,
        const float         cam_fx_cpu,
        const float         cam_fy_cpu
    )
    {
        const int i = cam_idx;

        CUDA_SAFE_CALL(cudaMemcpy(ddf->P[i],               cam_P_cpu,              12 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ddf->M_inv[i],           cam_M_inv_cpu,          9 * sizeof(float),  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ddf->K[i],               cam_K_cpu,              9 * sizeof(float),  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ddf->K_inv[i],           cam_K_inv_cpu,          9 * sizeof(float),  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ddf->R[i],               cam_R_cpu,              9 * sizeof(float),  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ddf->R_inv[i],           cam_R_inv_cpu,          9 * sizeof(float),  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ddf->line_proj_mat[i],   cam_line_proj_mat_cpu,  18 * sizeof(float), cudaMemcpyHostToDevice));

        memcpy(hdf->P_cpu[i], cam_P_cpu, 12 * sizeof(float));
        memcpy(hdf->M_inv_cpu[i], cam_M_inv_cpu, 9 * sizeof(float));
        memcpy(hdf->K_cpu[i], cam_K_cpu, 9 * sizeof(float));
        memcpy(hdf->K_inv_cpu[i], cam_K_inv_cpu, 9 * sizeof(float));
        memcpy(hdf->R_cpu[i], cam_R_cpu, 9 * sizeof(float));
        memcpy(hdf->R_inv_cpu[i], cam_R_inv_cpu, 9 * sizeof(float));
        memcpy(hdf->line_proj_mat_cpu[i], cam_line_proj_mat_cpu, 18 * sizeof(float));

        ddf->P_col34[i].x = (*cam_P_col34_cpu).x;
        ddf->P_col34[i].y = (*cam_P_col34_cpu).y;
        ddf->P_col34[i].z = (*cam_P_col34_cpu).z;

        ddf->c4[i].x = (*cam_c4_cpu).x;
        ddf->c4[i].y = (*cam_c4_cpu).y;
        ddf->c4[i].z = (*cam_c4_cpu).z;

        ddf->t4[i].x = (*cam_t4_cpu).x;
        ddf->t4[i].y = (*cam_t4_cpu).y;
        ddf->t4[i].z = (*cam_t4_cpu).z;

        ddf->fx[i] = cam_fx_cpu;
        ddf->fy[i] = cam_fy_cpu;

        ////////////////////////////////////////////////////////////
        hdf->P_col34[i].x = (*cam_P_col34_cpu).x;
        hdf->P_col34[i].y = (*cam_P_col34_cpu).y;
        hdf->P_col34[i].z = (*cam_P_col34_cpu).z;
        hdf->c4[i].x = (*cam_c4_cpu).x;
        hdf->c4[i].y = (*cam_c4_cpu).y;
        hdf->c4[i].z = (*cam_c4_cpu).z;
        hdf->t4[i].x = (*cam_t4_cpu).x;
        hdf->t4[i].y = (*cam_t4_cpu).y;
        hdf->t4[i].z = (*cam_t4_cpu).z;
        hdf->fx[i] = cam_fx_cpu;
        hdf->fy[i] = cam_fy_cpu;
        
    }

    void load_camera_parameters(
        const int cam_idx,
        const int img_height,
        const int img_width,
        const float* cams_P_cpu,
        const float4* cams_P_col34_cpu,
        const float* cams_M_inv_cpu,
        const float* cams_K_cpu,
        const float* cams_K_inv_cpu,
        const float* cams_R_cpu,
        const float4* cams_t4_cpu,
        const float* cams_R_inv_cpu,
        const float4* cams_c4_cpu,
        const float* cams_line_proj_mat_cpu,
        const float  cams_fx_cpu,
        const float  cams_fy_cpu
    )
    {
        if (!ddf)
            ddf = new device_dataframe(img_height, img_width);

        if (!hdf)
            hdf = new host_dataframe(img_height, img_width);

        send_camera_parameters_to_dataframes(
            ddf,
            hdf,
            cam_idx,
            cams_P_cpu,
            cams_P_col34_cpu,
            cams_M_inv_cpu,
            cams_K_cpu,
            cams_K_inv_cpu,
            cams_R_cpu,
            cams_t4_cpu,
            cams_R_inv_cpu,
            cams_c4_cpu,
            cams_line_proj_mat_cpu,
            cams_fx_cpu,
            cams_fy_cpu
        );


    }

    void load_img_data(
        const int img_idx,
        const int img_height,
        const int img_width,
        const float* img_gray,
        const float* img_ori,
        const float* img_conf,
        const float* img_mask,
        const float* initial_depth_map,
        const bool load_to_cpu_containers
    )
    {
        if (!ddf)
            throw std::runtime_error("Device dataframe is not yet initialized. Make sure to first upload the camera data and then load all image data!");

        if (!load_to_cpu_containers)
        {
            add_image_to_cuda_texture(img_idx, img_height, img_width, img_gray, ddf->imgs, ddf->cu_array_imgs);
            add_image_to_cuda_texture(img_idx, img_height, img_width, img_ori, ddf->imgs_ori, ddf->cu_array_imgs_ori, cudaFilterModePoint);
            add_image_to_cuda_texture(img_idx, img_height, img_width, img_conf, ddf->imgs_conf, ddf->cu_array_imgs_conf, cudaFilterModePoint);

            if (img_mask)
                add_image_to_cuda_texture(img_idx, img_height, img_width, img_mask, ddf->imgs_mask, ddf->cu_array_imgs_mask);

            if (initial_depth_map)
                add_image_to_cuda_texture(img_idx, img_height, img_width, initial_depth_map, ddf->imgs_initial_depth, ddf->cu_array_imgs_initial_depth, cudaFilterModePoint);
        }
        else
        {
            if (!hdf)
                throw std::runtime_error("Dataframe is not yet initialized. Make sure to first upload the camera data and then load all image data!");

            hdf->imgs_gray_cpu[img_idx] = (float*)img_gray;
            hdf->imgs_ori_cpu[img_idx] = (float*)img_ori;
            hdf->imgs_conf_cpu[img_idx] = (float*)img_conf;
            hdf->imgs_mask_cpu[img_idx] = (float*)img_mask;
        }

    }

    void load_lpmvs_estimations(
        const int img_idx,
        const float* estimated_positions,
        const float* estimated_direcitons
    )
    {
        if (!ddf)
            throw std::runtime_error("Device dataframe is not yet initialized. Make sure to first upload the camera data and then load all image data!!");
        
        uint32_t load_elements_byte = ddf->rows * ddf->cols * sizeof(float4);
        CUDA_SAFE_CALL(cudaMemcpy(ddf->directions[img_idx], estimated_direcitons, load_elements_byte, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ddf->positions[img_idx], estimated_positions, load_elements_byte, cudaMemcpyHostToDevice));
    }

    device_dataframe* get_ddf_instance()
    {
        if (!ddf)
            throw std::runtime_error("Device dataframe is not yet initialized. Make sure to first upload the camera data and then load all image data!!");
        return ddf;
    }

    host_dataframe* get_hdf_instance()
    {
        if (!hdf)
            throw std::runtime_error("Dataframe is not yet initialized. Make sure to first upload the camera data and then load all image data!!");
        return hdf;
    }

    void destroy_ddf_instance()
    {
        if (ddf)
        {
            ddf->destroy();
            delete ddf;
            ddf = nullptr;

            hdf->destroy();
            delete hdf;
            hdf = nullptr;
        }
    }


}