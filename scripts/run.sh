#!/bin/bash

USAGE="Usage: $0 <path_to_input_dataset> <path_to_output_folder>"
export GPU="0"
export PROJECT_DIR="$(dirname "$PWD")"
# export DISABLE_TYPES=1

# define colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW="\033[1;33m"
NC='\033[0m'

PATH_TO_INPUT_DATASET=$1
if [ -z "$PATH_TO_INPUT_DATASET" ]; then
    echo -e "${RED}error:${NC} missing required argument, please provide the path to the input dataset."
    echo "$USAGE"
    exit 1
fi
if [ ! -d "$PATH_TO_INPUT_DATASET" ]; then
    echo -e "${RED}error:${NC} input dataset at '${PATH_TO_INPUT_DATASET}' not found."
    exit 1
fi
echo -e "${GREEN}validated:${NC} found the input dataset at '${PATH_TO_INPUT_DATASET}' ..."

PATH_TO_OUPUT_FOLDER=$2
if [ -z "$PATH_TO_OUPUT_FOLDER" ]; then
    echo -e "${RED}error:${NC} missing required argument, please provide the desired output folder."
    echo "$USAGE"
    exit 1
fi
if [ ! -d "$PATH_TO_OUPUT_FOLDER" ]; then
    echo -e "${YELLOW}warning:${NC} output directory '${PATH_TO_OUPUT_FOLDER}' not found, attempting to create ..."
    
    if mkdir "$PATH_TO_OUPUT_FOLDER"; then
        echo -e "${YELLOW}info:${NC} ouput directory created at '${PATH_TO_OUPUT_FOLDER}'."
    else
        echo -e "${RED}error:${NC} failed to create the ouput directory at '${PATH_TO_OUPUT_FOLDER}', check permissions."
        exit 1
    fi
fi
echo -e "${GREEN}validated:${NC} ouputs are going to be saved at '${PATH_TO_OUPUT_FOLDER}' ..."


eval "$(conda shell.bash hook)"

conda deactivate && conda activate smhc_torch

# running for synthetic models
cd $PROJECT_DIR
CUDA_VISIBLE_DEVICES="$GPU" python main.py \
    --dataset_path $PATH_TO_INPUT_DATASET \
    --output_path $PATH_TO_OUPUT_FOLDER \
    --use_hair_masks --calc_quantitative_measures


# # running for real-world captures of "AVA" dataset
# cd $PROJECT_DIR
# CUDA_VISIBLE_DEVICES="$GPU" python main.py \
#     --dataset_path $PATH_TO_INPUT_DATASET \
#     --output_path $PATH_TO_OUPUT_FOLDER \
#     --neighboring_views_maximum_angle_diff 35 \
#     --lpmvs_maximum_depth 1.9 \
#     --lpmvs_minimum_depth 0.1 \
#     --mvhg_background_color_threshold 0.04 \
#     --mvhg_minimum_views_with_valid_2d_direction 4 \
#     --mvhg_valid_pixels_number_threshold 5 \
#     --mvhg_minimum_views_with_valid_2d_direction 4

    

