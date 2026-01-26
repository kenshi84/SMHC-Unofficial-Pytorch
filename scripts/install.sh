#!/bin/bash

# use this to activate conda environments
eval "$(conda shell.bash hook)"
conda deactivate
conda create -n smhc_torch python=3.11
conda activate smhc_torch
conda install cuda -c nvidia/label/cuda-12.4.0

cd ..
conda env update --file environment.yml
pip install --no-build-isolation -e ".[dev]"
