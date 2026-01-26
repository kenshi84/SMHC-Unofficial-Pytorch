#ifndef HAIR_MODELS_H_INCLUDED
#define HAIR_MODELS_H_INCLUDED

#include <string>
#include <functional>

namespace solution
{
    namespace hair_models
    {
        int load_dot_hair_model_file(
            const std::string& model_file,
            const float positions_scale,
            std::function<float* (size_t)> positions_buffer,
            std::function<float* (size_t)> directions_buffer
        );

        int load_dot_bin_model_file(
            const std::string& model_file,
            const float positions_scale,
            std::function<float* (size_t)> positions_buffer,
            std::function<float* (size_t)> points_per_strand_buffer
        );
    }
}

#endif