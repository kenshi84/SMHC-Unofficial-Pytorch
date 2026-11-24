from typing import Callable


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        from ._backend import _C
        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


calculate_orientation_maps = _make_lazy_cuda_func("calculate_orientation_maps")

load_camera_params = _make_lazy_cuda_func("load_camera_params")
load_image_data = _make_lazy_cuda_func("load_image_data")
load_patchmatch_estimations = _make_lazy_cuda_func("load_patchmatch_estimations")

run_line_based_patchmatch_mvs = _make_lazy_cuda_func("run_line_based_patchmatch_mvs")

filter_pc = _make_lazy_cuda_func("filter_pc")

fuse_pc = _make_lazy_cuda_func("fuse_pc")

generate_strand_segments = _make_lazy_cuda_func("generate_strand_segments")

run_multiview_hair_growing = _make_lazy_cuda_func("run_multiview_hair_growing")

calculate_hair_metrics = _make_lazy_cuda_func("calculate_hair_metrics")

destroy_dataframe = _make_lazy_cuda_func("destroy_dataframe")

load_dot_hair_model = _make_lazy_cuda_func("load_dot_hair_model")
load_dot_bin_model = _make_lazy_cuda_func("load_dot_bin_model")
load_dot_data_model = _make_lazy_cuda_func("load_dot_data_model")
