#ifndef ORIENTATION_H_INCLUDED
#define ORIENTATION_H_INCLUDED

namespace solution
{
    namespace gabor
    {
        void calculate_orientation_maps(
            const float* img_data,
            const int img_rows,
            const int img_cols,
            const float* gb_kernels,
            const int kernels_num,
            const int kernels_size,
            float* ori2d,
            float* conf
        );
    }
}


#endif