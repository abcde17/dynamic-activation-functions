"""
Example: python script_create_config_cntd.py 31D 
                [--init 1 1]
                [--seed 1]

Explanation for 31D:
    3 => model_size = 760M
    1 => dataset_size = 1B
    D => variant = bootstrap
"""

import argparse
import os
from pathlib import Path
from os.path import join, abspath, dirname, isfile
from typing import Dict, List, Union

ROOT_DIR = abspath(dirname(dirname(__file__)))

from helper_functions import get_model_size, get_dataset_size, get_variant, get_global_batch_size


# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------
def parse_input(args: argparse.Namespace) -> Dict[str, Union[float, str]]:
    exp = args.exp
    assert len(exp) == 3, f'ERROR! exp = {exp} needs to have length 3, e.g. 11A'
    params = {}
    params['seed'] = args.seed

    assert len(args.init) == 2, f'ERROR! need to specify 2 initial values, e.g. --init 1 2'
    params['init'] = args.init[0]
    params['init_attn'] = args.init[1]
    params['init_str'] = str(args.init[0]).replace('.', 'p') + 'x' + str(args.init[1]).replace('.', 'p')

    params['model_size_float'], params['model_size_str'] = get_model_size(exp[0])
    params['dataset_size_float'], params['dataset_size_str'], params['dataset_size_iter'] = get_dataset_size(exp[1])
    params['variant_str'] = get_variant(exp[2])
    params['exp_name'] = f'exp{exp}-{params["model_size_str"]}-{params["dataset_size_str"]}-{params["variant_str"]}-i{params["init_str"]}-s{params["seed"]}'
    params['config_file_path'] = join(ROOT_DIR, 'config', f'{params["exp_name"]}.py')
    params['config_file_path_cntd2'] = params['config_file_path'].replace('.py', '_cntd2.py')
    return params


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main(_args):
    print("--- parse input ---")
    params = parse_input(_args)
    for k, v in params.items():
        print(f'{k}: {v}')
    print()

    print("--- check existence of previous config file ---")
    if isfile(params['config_file_path']):
        print(f"> file {params['config_file_path']} exists.")
    else:
        raise ValueError(f"ERROR! file {params['config_file_path']} does not exist.")
    print()

    print("--- read previous config file ---")
    with open(params['config_file_path'], 'r') as f:
        lines = f.readlines()

    print("--- modify previous config file -> cntd config file ---")
    print("\n> BEFORE")
    print(lines)

    assert lines[3] == '\n', f'ERROR! 4th row of file is {lines[3]}, expected it to be empty'
    lines[3] = "init_from = 'resume'\n"
    for l, line in enumerate(lines):
        if line.startswith('wandb_run_name ='):
            line_number_wandb_run_name = l
            break
    lines[line_number_wandb_run_name] = lines[line_number_wandb_run_name].replace("'\n", "_cntd2'\n")

    print("\n> AFTER")
    print(lines)


    print("--- write cntd config file ---")
    with open(params['config_file_path_cntd2'], 'w') as f:
        for line in lines:
            f.write(line)
    print(f"> wrote config file {params['config_file_path_cntd2']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('--init', nargs='+', default=[1.])
    parser.add_argument('--seed', default=1, type=int)
    args = parser.parse_args()
    args.init = [float(elem) for elem in args.init]
    main(args)
