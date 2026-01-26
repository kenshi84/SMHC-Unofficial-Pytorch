#ifndef POINT_CLOUD_STRUCTS_H_INCLUDED
#define POINT_CLOUD_STRUCTS_H_INCLUDED

#include <cuda_runtime.h>
#include "cukd/builder.h"


namespace solution::pc_structs
{

struct detailed
{
    float3 direction;
    // 1 byte for split dimenstion used to save the widest 
    // dimension chosen in each node of the tree
    uint8_t split_dim;
    uint8_t mark_to_delete; // used in strand generation step
    uint8_t dummy2;
    uint8_t dummy3;

    float3 position;
    float dummy4;

    __host__ __device__ 
    detailed() {}

    __host__ __device__ 
    detailed(float3 d, float3 p)
        : direction{d}, position{p}, split_dim{0}, mark_to_delete{0}, dummy2{0}, dummy3{0}, dummy4{0.f} {}

    __host__ __device__ 
    detailed(float default_d, float default_p)
        : split_dim{0}, mark_to_delete{0}, dummy2{0}, dummy3{0}, dummy4{0.f}
    {
        direction.x = direction.y = direction.z = default_d;
        position.x = position.y = position.z = default_p;
    }
};

// this struct is needed by the cukd builders
// have a look at this example https://github.com/ingowald/cudaKDTree?tab=readme-ov-file#example-3-trees-with-support-arbitrary-dimension-splits
struct detailed_traits
    : public cukd::default_data_traits<float3>
{
    using point_t = float3;

    static inline __device__ __host__
        float3 get_point(const detailed& data)
    {
        return data.position;
    }

    static inline __device__ __host__
        float  get_coord(const detailed& data, int dim)
    {
        return cukd::get_coord(get_point(data), dim);
    }

    enum { has_explicit_dim = true };
    static inline __device__ __host__
        int  get_dim(const detailed& p)
    {
        return p.split_dim;
    }
    static inline __device__ __host__
        void set_dim(detailed& p, int dim)
    {
        p.split_dim = dim;
    }
};


struct plain
{
    float4 direction;
    float4 position;

    plain() {}

    plain(const float4& dir, const float4& pos) 
        : direction{ dir }, position{pos} {}

    plain(const plain& o)
        : direction{ o.direction }, position{ o.position } {}

    plain(const plain* o)
        : direction{ o->direction }, position{ o->position } {}

    plain(float default_d, float default_p)
    {
        direction.x = direction.y = direction.z = default_d;
        position.x = position.y = position.z = default_p;
    }

};

}

#endif
