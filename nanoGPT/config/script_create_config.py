"""
Example: python script_create_config.py 31D 
                [--init 1 1]
                [--seed 1]
                [--wandb dyaf]
                [--tokenizer gpt2]
                [--dataset openwebtext]
                [--out_dir_main_path /path/to/store/checkpoints]

Explanation for 31D:
    3 => model_size = 760M
    1 => dataset_size = 1B
    D => variant = bootstrap
"""

import argparse
import os
from pathlib import Path
from os.path import join, abspath, dirname
from typing import Dict, List, Union

ROOT_DIR = abspath(dirname(dirname(__file__)))

from helper_functions import get_model_size, get_dataset_size, get_variant, get_global_batch_size
from settings import BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, BLOCK_SIZE, GPUS, LOG_INTERVAL


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

    params['tokenizer'] = args.tokenizer
    params['dataset'] = args.dataset
    params['wandb'] = args.wandb
    params['model_size_float'], params['model_size_str'] = get_model_size(exp[0])
    params['dataset_size_float'], params['dataset_size_str'], params['dataset_size_iter'] = get_dataset_size(exp[1])
    params['variant_str'] = get_variant(exp[2])
    params['exp_name'] = f'exp{exp}-{params["model_size_str"]}-{params["dataset_size_str"]}-{params["variant_str"]}-i{params["init_str"]}-s{params["seed"]}'
    params['config_file_path'] = join(ROOT_DIR, 'config', f'{params["exp_name"]}.py')
    if args.out_dir_main_path is None:
        cwd = Path(os.getcwd())
        params['out_dir_main_path'] = cwd.parent
    else:
        params['out_dir_main_path'] = args.out_dir_main_path
    return params

# -----------------------------------------------------------
# SECTION FUNCTIONS
# -----------------------------------------------------------
def get_section_types(params: Dict[str, Union[float, str]]) -> List[str]:
    variant_str = params['variant_str']
    variant_str_pure = variant_str.split('residual')[0].split('projection')[0].split('adapter')[0]
    _lines = [
        "# --- types ---",
        f"normalization = '{variant_str_pure}'",
        ""
    ]
    if "dyt" in variant_str:
        _lines[-1] = f"init_alpha = {params['init']}"
        _lines += [""]
        if 'init_attn' in params:
            _lines[-1] = f"init_alpha_attn = {params['init_attn']}"
            _lines += [""]
    elif "dyisru" in variant_str:
        _lines[-1] = f"init_beta = {params['init']}"
        _lines += [""]
        if 'init_attn' in params:
            _lines[-1] = f"init_beta_attn = {params['init_attn']}"
            _lines += [""]
    _lines[-1] = f"seed = {params['seed']}"
    _lines += [""]
    
    return _lines

def get_section_experiment(params: Dict[str, Union[float, str]]) -> List[str]:
    compile = True
    wandb = params['wandb']
    exp_name = params['exp_name']
    out_dir_main_path = params['out_dir_main_path']

    _lines = [
        "# --- experiment ---",
        "wandb_log = True",
        f"wandb_project = '{wandb}'",
        f"wandb_run_name = '{exp_name}'",
        f"out_dir = '{out_dir_main_path}/output/{exp_name}'",
        f"compile = {compile}",
        "",
    ]
    return _lines

def get_section_batch_size(params: Dict[str, Union[float, str]]) -> List[str]:
    model_size_str = params["model_size_str"]

    batch_size = BATCH_SIZE[model_size_str]
    gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS[model_size_str]
    block_size = BLOCK_SIZE[model_size_str]
    gpus = GPUS['node']
    global_batch_size = get_global_batch_size(model_size_str)

    _lines = [
        "# --- batch size ---",
        f"# {batch_size} batch size * {block_size} block size * {gradient_accumulation_steps} gradaccum * {gpus} GPUs ~ {global_batch_size}",
        f"gradient_accumulation_steps = {gradient_accumulation_steps}*{gpus}",
        f"batch_size = {batch_size}",
        f"block_size = {block_size}",
        "",
    ]
    return _lines

def get_section_dataset_size(params: Dict[str, Union[float, str]]) -> List[str]:
    dataset_size_iter = params['dataset_size_iter']
    model_size_str = params['model_size_str']
    iterations = params['dataset_size_iter']
    global_batch_size = get_global_batch_size(model_size_str)
    global_tokens = global_batch_size * int(dataset_size_iter)
    _lines = [
        "# --- dataset size ---",
        f"# tokens ~ {global_batch_size} * {dataset_size_iter} ~ {global_tokens/10**9:.1f}B",
        f"max_iters = {iterations}",
        f"lr_decay_iters = {iterations}",
        "",
    ]
    return _lines

def get_section_checkpointing(params: Dict[str, Union[float, str]]) -> List[str]:
    iterations = params['dataset_size_iter']
    log_interval = LOG_INTERVAL[iterations]
    _lines = [
        "# --- checkpointing ---",
        f"eval_interval = {int(int(iterations)/10)}",
        "eval_iters = 100",
        f"log_interval = {log_interval}",
        "",
    ]
    return _lines

def get_section_optimizer(params: Dict[str, Union[float, str]]) -> List[str]:
    _lines = [
        "# --- optimizer ---",
        "optimizer_core = 'adamw'",
        "optimizer_embedding = 'adamw'",
        "weight_decay = 1e-1  # general",
        "grad_clip = 1.0  # general; clip gradients at this value, or disable if == 0.0",
        "beta1 = 0.9  # adamw",
        "beta2 = 0.95  # adamw",
        "# momentum = 0  # sgd",
        "",
    ]
    return _lines

def get_section_model(params: Dict[str, Union[float, str]]) -> List[str]:
    model_size_str = params['model_size_str']
    if model_size_str == "125M":
        _lines = [
            "# --- model ---",
            "# 125M",
            "n_layer = 12",
            "n_head = 12",
            "n_embd = 768",
            "learning_rate = 3e-4",
            "min_lr = 3e-5",
            "",
        ]
    elif model_size_str == "355M":
        _lines = [
            "# --- model ---",
            "# 355M",
            "n_layer = 24",
            "n_head = 16",
            "n_embd = 1024",
            "learning_rate = 3e-4",
            "min_lr = 3e-5",
            "",
        ]
    elif model_size_str == "760M":
        _lines = [
            "# --- model ---",
            "# 760M",
            "n_layer = 24",
            "n_head = 16",
            "n_embd = 1536",
            "learning_rate = 2.5e-4",
            "min_lr = 2.5e-5",
            "",
        ]
    elif model_size_str == "1300M":
        _lines = [
            "# --- model ---",
            "# 1300M",
            "n_layer = 24",
            "n_head = 16",
            "n_embd = 2048",
            "learning_rate = 2.0e-4",
            "min_lr = 2.0e-5",
            "",
        ]
    else:
        raise Exception(f"config section 'model' not defined for model_size_str = {model_size_str}")
    return _lines

def get_section_hyperparameters(params: Dict[str, Union[float, str]]) -> List[str]:
    _lines = [
        "# --- hyperparameters ---",
        "warmup_iters = 100  # not super necessary potentially"
        "",
    ]
    return _lines

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main(_args):
    print("--- parse input ---")
    params = parse_input(_args)
    for k, v in params.items():
        print(f'{k}: {v}')
    print()

    print("--- create lines ---")
    lines = ["#", "#", "#", ""]
    lines += get_section_types(params)
    lines += get_section_experiment(params)
    lines += get_section_batch_size(params)
    lines += get_section_dataset_size(params)
    lines += get_section_checkpointing(params)
    lines += get_section_optimizer(params)
    lines += get_section_model(params)
    lines += get_section_hyperparameters(params)
    print(lines)
    print()

    print("--- write config file ---")
    with open(params['config_file_path'], 'w') as f:
        for line in lines:
            f.write(line + '\n')
    print(f"> wrote config file {params['config_file_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('--init', nargs='+', default=[1.])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--wandb', default='dyaf_prod', type=str)
    parser.add_argument('--tokenizer', default='gpt2', type=str)
    parser.add_argument('--dataset', default='openwebtext', type=str)
    parser.add_argument('--out_dir_main_path', default=None, type=str)
    args = parser.parse_args()
    args.init = [float(elem) for elem in args.init]
    main(args)
