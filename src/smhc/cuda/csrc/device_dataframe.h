#ifndef DEVICE_DATAFRAME_H_INCLUDED
#define DEVICE_DATAFRAME_H_INCLUDED

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>

namespace solution
{
    constexpr int max_cameras_num{ 256 };
    constexpr int max_neighboring_views{ 10 };

    struct __align__(128) device_dataframe
    {

        device_dataframe(const int img_rows, const int img_cols);
        ~device_dataframe();
        // copy constructor
        explicit device_dataframe(const device_dataframe&) = delete;
        // copy assignment
        device_dataframe& operator=(const device_dataframe&) = delete;
        // move constructor
        explicit device_dataframe(device_dataframe&&) = delete;
        // move assignment
        device_dataframe& operator=(device_dataframe&&) = delete;

        void destroy_image_data();
        void destroy();
        void reset_containers();

        curandState* cs{ nullptr };

        cudaTextureObject_t imgs[1+max_neighboring_views];
        cudaArray* cu_array_imgs[1+max_neighboring_views];

        cudaTextureObject_t imgs_mask[1+max_neighboring_views];
        cudaArray* cu_array_imgs_mask[1+max_neighboring_views];

        cudaTextureObject_t imgs_ori[1+max_neighboring_views];
        cudaArray* cu_array_imgs_ori[1+max_neighboring_views];

        cudaTextureObject_t imgs_conf[1+max_neighboring_views];
        cudaArray* cu_array_imgs_conf[1+max_neighboring_views];

        cudaTextureObject_t imgs_initial_depth[1+max_neighboring_views];
        cudaArray* cu_array_imgs_initial_depth[1+max_neighboring_views];

        // cameras parameters
        // projection formula x_31 = P_34 X_41 = K33 [ R_33 | t_31 ] X_41 = [ M_33 | P_col34_31] X_41
        // extrinsic matrix : [ R | t4]
        // extrinsic matrix inverse : [ R_inv | c4]
        float*  P[max_cameras_num];
        float4  P_col34[max_cameras_num];
        float*  M_inv[max_cameras_num];
        float*  K[max_cameras_num];
        float*  K_inv[max_cameras_num];
        float*  R[max_cameras_num];
        float4  t4[max_cameras_num];
        float*  R_inv[max_cameras_num];
        float4  c4[max_cameras_num];
        float*  line_proj_mat[max_cameras_num];
        float   fx[max_cameras_num];
        float   fy[max_cameras_num];
        
        int     id_ref{ 0 };
        int     cols{ 0 };
        int     rows{ 0 };

        int*    view_selection_subset{ nullptr };
        int     view_selection_subset_number{ 0 };

        // lpmvs algorithm parameters
        float   alpha{ 0.f };
        int     iterations{ 0 };
        float   minimum_depth{ 0.f };
        float   maximum_depth{ 0.f };
        float   sampling_radius{ 0.f };
        int     half_samples_num{ 0 };
        float   occlusion_depth_thresh{ 0.f };
        bool    use_zncc{ true };
        float   zncc_min_variance{ 1e-6f };

        // line parameters
        float2* direction{ nullptr };
        float4* momentum_depth{ nullptr }; // 3 values for momentum and last value represent depth
        float*  cost{ nullptr };

        // for debug purposes
        // float4* debug_data{ nullptr };

        // these are needed for line filtering step
        float4* directions[1 + max_neighboring_views]; // the first element would be reference image data
        float4* positions[1 + max_neighboring_views]; // the first element would be reference image data
        uint8_t* consistency_flag;

    };

    struct host_dataframe
    {
        host_dataframe(const int img_rows, const int img_cols);
        ~host_dataframe();
        // copy constructor
        explicit host_dataframe(const host_dataframe&) = delete;
        // copy assignment
        host_dataframe& operator=(const host_dataframe&) = delete;
        // move constructor
        explicit host_dataframe(host_dataframe&&) = delete;
        // move assignment
        host_dataframe& operator=(host_dataframe&&) = delete;

        void destroy();

        // these are needed for multi-view hair growing
        float*  P_cpu[max_cameras_num];
        float*  M_inv_cpu[max_cameras_num];
        float*  K_cpu[max_cameras_num];
        float*  K_inv_cpu[max_cameras_num];
        float*  R_cpu[max_cameras_num];
        float*  R_inv_cpu[max_cameras_num];
        float*  line_proj_mat_cpu[max_cameras_num];

        float4  P_col34[max_cameras_num];
        float4  t4[max_cameras_num];
        float4  c4[max_cameras_num];
        float   fx[max_cameras_num];
        float   fy[max_cameras_num];
        
        int     cols{ 0 };
        int     rows{ 0 };
        
        // these will only contain pointers to the corespoding contiguous torch tensors.
        // make sure to not delete the tensors in the python side or any where else till
        // this step of the algorithm is finished completely.
        // despite this being a bit risky, it saves you from additional memory requirements.
        float* imgs_gray_cpu[max_cameras_num];
        float* imgs_conf_cpu[max_cameras_num];
        float* imgs_ori_cpu[max_cameras_num];
        float* imgs_mask_cpu[max_cameras_num];
        // didn't find any use case for depth priors in the multi-view hair growing step
    };


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
    );

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
    );

    void load_img_data(
        const int img_idx,
        const int img_height,
        const int img_width,
        const float* img_gray,
        const float* img_ori=nullptr,
        const float* img_conf=nullptr,
        const float* img_mask=nullptr,
        const float* initial_depth_map=nullptr,
        const bool load_to_cpu_containers=false
    );

    void load_lpmvs_estimations(
        const int img_idx,
        const float* estimated_positions,
        const float* estimated_direcitons
    );

    device_dataframe* get_ddf_instance();
    host_dataframe* get_hdf_instance();

    void destroy_ddf_instance();
}

#endif