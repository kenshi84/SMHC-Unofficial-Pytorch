#include <torch/extension.h>

#include "solution.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{

    m.def("calculate_orientation_maps", &solution::calculate_orientation_maps);
    
    m.def("load_camera_params", &solution::load_camera_parameters_to_dataframe);
    m.def("load_image_data", &solution::load_img_data_to_dataframe);
    m.def("load_patchmatch_estimations", &solution::load_lpmvs_estimations_to_dataframe);
    
    m.def("run_line_based_patchmatch_mvs", &solution::run_lpmvs);

    m.def("filter_pc", &solution::run_line_filtering);
    
    m.def("fuse_pc", &solution::run_line_fusion);

    m.def("generate_strand_segments", &solution::run_strand_segment_generation);

    m.def("run_multiview_hair_growing", &solution::run_multiview_hair_growing);

    m.def("calculate_hair_metrics", &solution::calculate_hair_metrics);

    m.def("destroy_dataframe", &solution::destroy_dataframe);

    m.def("load_dot_hair_model", &solution::load_dot_hair_model);
    m.def("load_dot_bin_model", &solution::load_dot_bin_model);

}