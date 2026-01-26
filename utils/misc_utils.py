# Part of this file is derived from [Image-GS] 
# (https://github.com/NYU-ICL/image-gs)
# Copyright (c) 2025 NYU Immersive Computing Lab
# Licensed under the MIT License.

from argparse import ArgumentParser

import yaml


# code taken from https://github.com/NYU-ICL/image-gs/blob/main/utils/misc_utils.py
def load_cfg(cfg_path: str, parser: ArgumentParser) -> ArgumentParser:
    with open(cfg_path, "r", encoding="utf-8") as file:
        cfg: dict = yaml.safe_load(file)
    for key, value in cfg.items():
        if value is None:
            raise ValueError("'None' is not a supported value in the config file")
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true", default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    return parser

# code taken from https://github.com/NYU-ICL/image-gs/blob/main/utils/misc_utils.py
def save_cfg(path: str, args, mode="w"):
    with open(path, mode=mode, encoding="utf-8") as file:
        print("#################### Config Parameters ####################", file=file)
        yaml.dump(vars(args), file, default_flow_style=False)
