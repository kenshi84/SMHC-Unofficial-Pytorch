#ifndef AUXILIARY_H_INCLUDED
#define AUXILIARY_H_INCLUDED

// for stderr
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923f
#endif

#ifndef OMP_NUM_THREADS
#define OMP_NUM_THREADS 32
#endif


#define pow2(x) ((x)*(x))
#define pow3(x) ((x)*(x)*(x))
#define pow4(x) ((x)*(x)*(x)*(x))
#define get_pow2_norm(x,y) (pow2(x)+pow2(y))
#define get_normf(x,y) (sqrtf((pow2(x) + pow2(y))))
#define get_normf3(v) (sqrtf((pow2(v.x) + pow2(v.y) + pow2(v.z))))
#define get_pow2_normf3(v) ((pow2(v.x) + pow2(v.y) + pow2(v.z)))


#define cross_product(in1, in2, out) \
out.x = in1.y*in2.z - in1.z*in2.y; \
out.y = in1.z*in2.x - in1.x*in2.z; \
out.z = in1.x*in2.y - in1.y*in2.x; 


#define dot_product(in1, in2) \
( in1.x*in2.x + in1.y*in2.y + in1.z*in2.z )


#define norm2(in) \
( in.x*in.x + in.y*in.y + in.z*in.z )

#define scale_vec4(vec, s) \
vec.x = vec.x * s; vec.y = vec.y * s; vec.z = vec.z * s;

#define set_equal_vec3(v0, v1) \
v0.x = v1.x; v0.y = v1.y; v0.z = v1.z;

#define zero_init_vec3(v) \
v.x = 0.f; v.y = 0.f; v.z = 0.f;

#define vecdiv4(v,k) \
v.x = v.x / k; \
v.y = v.y / k; \
v.z = v.z / k;

#define sub4(v0,v1) \
v0.x = v0.x - v1.x; \
v0.y = v0.y - v1.y; \
v0.z = v0.z - v1.z;

#define add4(v0,v1) \
v0.x = v0.x + v1.x; \
v0.y = v0.y + v1.y; \
v0.z = v0.z + v1.z;

#define subout(v0,v1,vout) \
vout.x = v0.x - v1.x; \
vout.y = v0.y - v1.y; \
vout.z = v0.z - v1.z;

#define addscaleout(v0, s, v1, out) \
out.x = v0.x + s * v1.x; \
out.y = v0.y + s * v1.y; \
out.z = v0.z + s * v1.z;

#define matvecmul4(m, v, out) \
out.x = \
m[0] * v.x + \
m[1] * v.y + \
m[2] * v.z; \
out.y = \
m[3] * v.x + \
m[4] * v.y + \
m[5] * v.z; \
out.z = \
m[6] * v.x + \
m[7] * v.y + \
m[8] * v.z;


#define matvecmul4_hom(m, v, out) \
out.x = \
m[0] * v.x + \
m[1] * v.y + \
m[2] * v.z + m[3]; \
out.y = \
m[4] * v.x + \
m[5] * v.y + \
m[6] * v.z + m[7]; \
out.z = \
m[8] * v.x + \
m[9] * v.y + \
m[10] * v.z + m[11];


// from https://stackoverflow.com/questions/35922268/create-an-identifier-at-compile-time-that-has-the-line-number-as-part-of-it
#define CAT_(a, b) a ## b
#define CAT(a, b) CAT_(a, b)
#define VARNAME(Var) CAT(Var, __LINE__)

// in2 is normalized and has only x and y coordinates
#define cross_product_dir2(in1, in2, out) \
float VARNAME(s) = sqrtf(1 - pow2(in2.x) - pow2(in2.y)); \
VARNAME(s) = (VARNAME(s) != VARNAME(s)) ? 0.f : VARNAME(s); \
out.x = in1.y*VARNAME(s) - in1.z*in2.y; \
out.y = in1.z*in2.x - in1.x*VARNAME(s); \
out.z = in1.x*in2.y - in1.y*in2.x;

// in1 is normalized and has only x and y coordinates
#define cross_product_dir2_1(in1, in2, out) \
float VARNAME(s) = sqrtf(1 - pow2(in1.x) - pow2(in1.y)); \
VARNAME(s) = (VARNAME(s) != VARNAME(s)) ? 0.f : VARNAME(s); \
out.x = in1.y*in2.z - VARNAME(s)*in2.y; \
out.y = VARNAME(s)*in2.x - in1.x*in2.z; \
out.z = in1.x*in2.y - in1.y*in2.x;

// values stored in out1 represent the direction 
// first 3 values stored in out2 represent the momentum
// vec1 is position of a point on the line and vec2 is the (probably float2) direction of the line
// s variable is to check for nan values of sqrtf(1 - pow2(vec2.x) - pow2(vec2.y))
#define plucker_line3_dir2(vec1, vec2, out1, out2) \
float VARNAME(s) = sqrtf(1 - pow2(vec2.x) - pow2(vec2.y)); \
VARNAME(s) = (VARNAME(s) != VARNAME(s)) ? 0.f : VARNAME(s); \
out1.x = vec2.x; \
out1.y = vec2.y; \
out2.x = vec1.y * VARNAME(s) - vec2.y * vec1.z; \
out2.y = vec1.z * vec2.x - VARNAME(s) * vec1.x; \
out2.z = vec1.x * vec2.y - vec2.x * vec1.y;

// 3 first values stored in out1 represent the direction 
// other 3 values stored in out2 represent the momentum
// vec1 and vec2 are two points along the line 
#define plucker_line3(vec1, vec2, out1, out2) \
out1.x = vec2.x - vec1.x; \
out1.y = vec2.y - vec1.y; \
out1.z = vec2.z - vec1.z; \
out2.x = vec1.y * vec2.z - vec2.y * vec1.z; \
out2.y = vec1.z * vec2.x - vec2.z * vec1.x; \
out2.z = vec1.x * vec2.y - vec2.x * vec1.y;

// See also [Hartley and Zisserman, Eq.(8.4)]
// Reversing order and dot product represents the product (L|Lˆ) which defined in [Hartley and Zisserman, Eq.(3.13)]
#define plucker_line_reverse(in1, in2, out1, out2) \
out1.x = in2.z; \
out1.y = in2.y; \
out1.z = in2.x; \
out2.x = in1.z; \
out2.y = in1.y; \
out2.z = in1.x;

#define plucker_line_reverse_dir2(in1, in2, out1, out2) \
float VARNAME(s) = sqrtf(1 - pow2(in1.x) - pow2(in1.y)); \
VARNAME(s) = (VARNAME(s) != VARNAME(s)) ? 0.f : VARNAME(s); \
out1.x = in2.z; \
out1.y = in2.y; \
out1.z = in2.x; \
out2.x = VARNAME(s); \
out2.y = in1.y; \
out2.z = in1.x;

// 2D line parameters of ax + by + c = 0
// a = out[0], b = out[1], c = out[2]
#define plucker_line_project_to_2d(in1, in2, proj, out) \
out.x = proj[0]*in1.x + proj[1]*in1.y + proj[2]*in1.z + proj[3]*in2.x + proj[4]*in2.y + proj[5]* in2.z; \
out.y = proj[6]*in1.x + proj[7]*in1.y + proj[8]*in1.z + proj[9]*in2.x + proj[10]*in2.y + proj[11]* in2.z;
// out.z = proj[12]*in1.x + proj[13]*in1.y + proj[14]*in1.z + proj[15]*in2.x + proj[16]*in2.y + proj[17]* in2.z;

// texture utils
#define texat(tex,x, y) \
tex2D<float>(tex, x+0.5f, y+0.5f)
#define texatf4(tex,x, y) \
tex2D<float4>(tex, x+0.5f, y+0.5f)

// error checking macro
#define CUDA_CHECK_ERRORS(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, cudaGetErrorString(__err), __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call) \
    do { \
        cudaError_t err = call; \
        if (cudaSuccess != err) { \
            fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err) ); \
            exit(1); \
        } \
    } while (0)


inline int div_round_up(int a, int b)
{
	return (a + b - 1) / b;
}

#endif