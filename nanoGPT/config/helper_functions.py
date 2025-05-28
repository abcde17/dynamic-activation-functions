
from typing import Tuple
from settings import BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, BLOCK_SIZE, GPUS


def get_global_batch_size(model_size_str: str) -> int:
    batch_size = BATCH_SIZE[model_size_str]
    gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS[model_size_str]
    block_size = BLOCK_SIZE[model_size_str]
    gpus = GPUS['node']
    return batch_size * block_size * gradient_accumulation_steps * gpus


def get_model_size(_exp: str) -> Tuple[float, str]:
    _exp = int(_exp)
    if _exp == 1:
        return 0.125, '125M'
    elif _exp == 2:
        return 0.355, '355M'
    elif _exp == 3:
        return 0.760, '760M'
    elif _exp == 4:
        return 1.300, '1300M'
    else:
        raise Exception(f'ERROR! model size not defined for input {_exp}')

def get_dataset_size(_exp: str) -> Tuple[float, str]:
    _exp = int(_exp)
    if _exp == 0:
        return 1., '1k', '1000'
    elif _exp == 1:
        return 10., '10k', '10000'
    elif _exp == 2:
        return 50., '50k', '50000'
    elif _exp == 3:
        return 100., '100k', '100000'
    elif _exp == 4:
        return 150., '150k', '150000'
    elif _exp == 5:
        return 200., '200k', '200000'
    elif _exp == 6:
        return 250., '250k', '250000'
    else:
        raise Exception(f'ERROR! dataset size not defined for input {_exp}')

def get_variant(_exp: str) -> str:
    if _exp == 'A':
        return 'ln'
    elif _exp == 'B':
        return 'dyt'
    elif _exp == 'C':
        return 'dyisrusp'
    elif _exp == 'D':
        return 'dytsp'
    elif _exp == 'E':
        return 'dyisru'
    else:
        raise Exception(f'ERROR! variant not defined for input {_exp}')
