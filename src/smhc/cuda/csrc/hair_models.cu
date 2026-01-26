// some parts of this file is derived from [cyCodeBase] 
// (https://www.cemyuksel.com/cyCodeBase/index.html)
// Copyright (c) 2016, Cem Yuksel <cem@cemyuksel.com>
// Licensed under the MIT License.

// some parts of this file is derived from [HairStrandsRendering] 
// (https://github.com/Jhonve/HairStrandsRendering)
// Licensed under the MIT License.

#include <sstream>
#include <vector>
#include <fstream>
#include <filesystem>


#include "hair_models.h"
#include "cyHairFile.h"
#include "auxiliary.h"

namespace solution::hair_models
{

int load_dot_hair_model_file(
    const std::string& model_file,
    const float positions_scale,
    std::function<float* (size_t)> positions_buffer,
    std::function<float* (size_t)> points_per_strand_buffer
    // const std::string& output_obj_file
)
{
    cyHairFile hairfile;

    // Load the hair model
    // this part is taken from https://www.cemyuksel.com/cyCodeBase/soln/using_hair_files.html
    int result = hairfile.LoadFromFile(model_file.c_str());
    // Check for errors
    switch (result) {
    case CY_HAIR_FILE_ERROR_CANT_OPEN_FILE:
        printf("Error: Cannot open hair file!\n");
        return -1;
    case CY_HAIR_FILE_ERROR_CANT_READ_HEADER:
        printf("Error: Cannot read hair file header!\n");
        return -1;
    case CY_HAIR_FILE_ERROR_WRONG_SIGNATURE:
        printf("Error: File has wrong signature!\n");
        return -1;
    case CY_HAIR_FILE_ERROR_READING_SEGMENTS:
        printf("Error: Cannot read hair segments!\n");
        return -1;
    case CY_HAIR_FILE_ERROR_READING_POINTS:
        printf("Error: Cannot read hair points!\n");
        return -1;
    case CY_HAIR_FILE_ERROR_READING_COLORS:
        printf("Error: Cannot read hair colors!\n");
        return -1;
    case CY_HAIR_FILE_ERROR_READING_THICKNESS:
        printf("Error: Cannot read hair thickness!\n");
        return -1;
    case CY_HAIR_FILE_ERROR_READING_TRANSPARENCY:
        printf("Error: Cannot read hair transparency!\n");
        return -1;
    default:
        printf("Hair file \"%s\" loaded.\n", model_file.c_str());
    }

    int hairCount = hairfile.GetHeader().hair_count;
    int pointCount = hairfile.GetHeader().point_count;
    unsigned short* hair_segments = hairfile.GetSegmentsArray();
    float* points = hairfile.GetPointsArray();
    unsigned int d_segments = hairfile.GetHeader().d_segments;

    printf("Number of hair strands = %d\n", hairCount);
    printf("Number of hair points = %d\n", pointCount);

    // scaling the positions
    for (unsigned int i{0}; i < pointCount; ++i)
    {
        points[i*3]*=positions_scale;
        points[i*3+1]*=positions_scale;
        points[i*3+2]*=positions_scale;
    }

    // sending the data to the specified output tensors
    float* output_positions = positions_buffer(3 * pointCount);
    float* output_points_per_strand = points_per_strand_buffer(hairCount);

    memcpy(output_positions, points, 3 * sizeof(float)  * pointCount);
    // memcpy(output_directions, dirs, 3 * sizeof(float) * pointCount);

    for (unsigned int i = 0; i < hairCount; ++i) 
    {
        // hair segments number is actually one less than the number of points per strand
	    output_points_per_strand[i] = (hair_segments) ? (hair_segments[i]+1) : (d_segments+1);
       
    }

    return 0;

}

// this function is a slightly different version of the load_bin function in the HairStrandRendering repo
// https://github.com/Jhonve/HairStrandsRendering/blob/main/HairStrandsRendering/elems/strands.cpp#L260
int load_dot_bin_model_file(
    const std::string& model_file,
    const float positions_scale,
    std::function<float* (size_t)> positions_buffer,
    std::function<float* (size_t)> points_per_strand_buffer
)
{
    FILE* f = fopen(model_file.c_str(), "rb");
    if (!f)
    {
        fprintf(stderr, "Couldn't open %s\n", model_file.c_str());
        return -1;
    }

    int num_strands = 0;    
    fread(&num_strands, 4, 1, f);

    int valid_strands_num = 0;
    int total_points_num = 0;

    std::vector<std::vector<float3>> strands_points;
    std::vector<int> points_num_per_strand;

    for (int i_strand = 0; i_strand < num_strands; ++i_strand)
    {
        int num_points = 0;
        fread(&num_points, 4, 1, f);
        
        std::vector<float3> strand_points(num_points, {0.f, 0.f, 0.f});

        float dummy = 0.0f;
        for (int j_point = 0; j_point < num_points; j_point++)
        {
            fread(&strand_points[j_point].x, 4, 1, f);
            fread(&strand_points[j_point].y, 4, 1, f);
            fread(&strand_points[j_point].z, 4, 1, f);
            fread(&dummy, 4, 1, f); // nx unused
            fread(&dummy, 4, 1, f); // ny unused
            fread(&dummy, 4, 1, f); // nz unused
            fread(&dummy, 4, 1, f); // label unused
        }

        bool valid_strand = true;
        for (int j_point = 0; j_point < num_points - 1; j_point++)
        {
            float3 aux;
            subout(strand_points[j_point], strand_points[j_point + (size_t)1], aux)
            if (get_normf3(aux) < 1.e-4f)
            {
                valid_strand = false;
                break;
            }
        }
        if (valid_strand)
        {
            strands_points.push_back(strand_points);
            points_num_per_strand.push_back(num_points);
            valid_strands_num++;
            total_points_num += num_points;
        }
    }

    fclose(f);

    // sending the data to the specified output tensors
    float* output_positions = positions_buffer(3 * total_points_num);
    float* output_points_per_strand = points_per_strand_buffer(valid_strands_num);

    int points_index_acc = 0;
    for (int i = 0; i < valid_strands_num; ++i)
    {
        int points_num_in_strand = points_num_per_strand[i];
        output_points_per_strand[i] = points_num_in_strand;

        std::vector<float3>& strand = strands_points[i];

        for (int j = 0; j < points_num_in_strand; ++j)
        {
            output_positions[points_index_acc*3] = strand[j].x * positions_scale;
            output_positions[points_index_acc*3+1] = strand[j].y * positions_scale;
            output_positions[points_index_acc*3+2] = strand[j].z * positions_scale;
            points_index_acc++;
        }
    }

    return 0;
}

}