#!/bin/bash
USAGE="Usage: $0 <path_to_blender_executable> <path_to_input_hair_model> <path_to_input_head_model> <path_to_output_folder>"
export GPU="0"
export PROJECT_DIR="$(dirname "$PWD")"
# export DISABLE_TYPES=1

# define colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW="\033[1;33m"
NC='\033[0m'

PATH_TO_BLENDER=$1
if [ -z "$PATH_TO_BLENDER" ]; then
    echo -e "${RED}error:${NC} missing required argument, please provide the path to blender executable."
    echo "$USAGE"
    exit 1
fi
if [ ! -f "$PATH_TO_BLENDER" ]; then
    echo -e "${RED}error:${NC} file '${PATH_TO_BLENDER}' not found."
    exit 1
fi
echo -e "${GREEN}validated:${NC} found blender executable at '${PATH_TO_BLENDER}' ..."


PATH_TO_INPUT_HAIR_MODEL=$2
if [ -z "$PATH_TO_INPUT_HAIR_MODEL" ]; then
    echo -e "${RED}error:${NC} missing required argument, please provide the path to the input hair model."
    echo "$USAGE"
    exit 1
fi
if [ ! -f "$PATH_TO_INPUT_HAIR_MODEL" ]; then
    echo -e "${RED}error:${NC} file '${PATH_TO_INPUT_HAIR_MODEL}' not found."
    exit 1
fi
echo -e "${GREEN}validated:${NC} found the input hair model at '${PATH_TO_INPUT_HAIR_MODEL}' ..."


PATH_TO_INPUT_HEAD_MODEL=$3
if [ -z "$PATH_TO_INPUT_HEAD_MODEL" ]; then
    echo -e "${RED}error:${NC} missing required argument, please provide the path to the input head model."
    echo "$USAGE"
    exit 1
fi
if [ ! -f "$PATH_TO_INPUT_HEAD_MODEL" ]; then
    echo -e "${RED}error:${NC} file '${PATH_TO_INPUT_HEAD_MODEL}' not found."
    exit 1
fi
echo -e "${GREEN}validated:${NC} found the input head model at '${PATH_TO_INPUT_HEAD_MODEL}' ..."


PATH_TO_OUPUT_FOLDER=$4
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


# use this to activate conda environments
eval "$(conda shell.bash hook)"

conda deactivate && conda activate smhc_torch


# extracting and interpolating hair models to prepare them for rendering
cd $PROJECT_DIR/datasets/synthetic_data
CUDA_VISIBLE_DEVICES="$GPU" python load_hair_models.py \
    --input_path $PATH_TO_INPUT_HAIR_MODEL \
    --finest_distance_measure 0.00199 \
    --output_model_folder $PATH_TO_OUPUT_FOLDER \
    --save_hair_model


# render hair model
cd $PROJECT_DIR/datasets/synthetic_data
$PATH_TO_BLENDER -b ./synthetic_scene.blend -P ./render_dataset.py -- --cycles-device OPTIX +CPU -- --args \
    $PATH_TO_INPUT_HEAD_MODEL \
    $PATH_TO_OUPUT_FOLDER \
    2048 2048 0.0002 1024 -1 \
    $CONDA_PREFIX/lib/python3.11/site-packages/


# extracting hair surface
cd $PROJECT_DIR/datasets/synthetic_data
CUDA_VISIBLE_DEVICES="$GPU" python load_hair_models.py \
    --input_path $PATH_TO_OUPUT_FOLDER/strands_interpolated.ply \
    --cameras_calibration_path $PATH_TO_OUPUT_FOLDER/calibration.json \
    --finest_distance_measure 0.00099 \
    --hair_surface_depth 0.0004 \
    --output_model_folder $PATH_TO_OUPUT_FOLDER \
    --extract_hair_surface