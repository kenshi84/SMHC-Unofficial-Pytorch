// Exercise the production texture sampling and multi-view cost on a known line.
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "lpmvs.cu"

__global__ void evaluate_cost(solution::device_dataframe* frame, float* output)
{
    const int2 pixel = {8, 8};
    const float2 direction = {1.f, 0.f};
    // The line passes through (8, 8, 1) and runs parallel to the X axis.
    const float4 momentum_depth = {0.f, 1.f, -8.f, 1.f};
    *output = solution::lpmvs::multiview_cost<float>(frame, pixel, direction, momentum_depth);
}

int main(int argc, char** argv)
{
    constexpr int size = 16;
    const float identity[] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
    const float projection[] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f};
    const float line_projection[] = {
        0.f, 0.f, 1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f, 0.f, 0.f,
        1.f, 0.f, 0.f, 0.f, 0.f, 0.f,
    };
    const float4 zero = {0.f, 0.f, 0.f, 0.f};
    int neighbors;
    std::cin >> neighbors;
    std::vector<int> indices(neighbors);
    std::vector<float> orientation(size * size, 0.f);
    std::vector<float> confidence(size * size, 10.f);
    for (int camera = 0; camera <= neighbors; ++camera)
    {
        solution::load_camera_parameters(
            camera, size, size, projection, &zero, identity, identity, identity,
            identity, &zero, identity, &zero, line_projection, 1.f, 1.f);
        std::vector<float> image(size * size, 0.f);
        std::vector<float> mask(size * size, 0.f);
        float values[5], mask_values[5];
        for (float& value : values)
            std::cin >> value;
        for (float& value : mask_values)
            std::cin >> value;
        for (int y = 0; y < size; ++y)
            for (int x = 6; x <= 10; ++x)
            {
                image[y * size + x] = values[x - 6];
                mask[y * size + x] = mask_values[x - 6];
            }
        float depth;
        std::cin >> depth;
        std::vector<float> depths(size * size, depth);
        solution::load_img_data(
            camera, size, size, image.data(), orientation.data(), confidence.data(),
            mask.data(), depth > 0.f ? depths.data() : nullptr);
        if (camera > 0)
            indices[camera - 1] = camera;
    }

    auto* frame = solution::get_ddf_instance();
    frame->alpha = 1.f; // Isolate the image cost from the orientation cost.
    frame->sampling_radius = 2.f;
    frame->half_samples_num = 2;
    frame->occlusion_depth_thresh = 0.001f;
    frame->view_selection_subset_number = neighbors;
    frame->use_zncc = std::atoi(argv[1]) != 0;
    frame->zncc_min_variance = std::strtof(argv[2], nullptr);
    CUDA_SAFE_CALL(cudaMemcpy(frame->view_selection_subset, indices.data(),
        indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    solution::device_dataframe* device_frame;
    float* output;
    CUDA_SAFE_CALL(cudaMalloc(&device_frame, sizeof(*frame)));
    CUDA_SAFE_CALL(cudaMemcpy(device_frame, frame, sizeof(*frame), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc(&output, sizeof(float)));
    evaluate_cost<<<1, 1>>>(device_frame, output);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    float cost;
    CUDA_SAFE_CALL(cudaMemcpy(&cost, output, sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << std::setprecision(9) << cost << '\n';
    CUDA_SAFE_CALL(cudaFree(output));
    CUDA_SAFE_CALL(cudaFree(device_frame));
    solution::destroy_ddf_instance();
}
