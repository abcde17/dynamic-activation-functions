
import json
from os.path import isfile
import numpy as np
import pandas as pd
import wandb

from helpers import Helpers
from settings import ENTITY, PROJECT


class AlphaBeta:

    def __init__(self, 
                 entity: str, 
                 project: str, 
                 runs_1d: dict[str, str], 
                 runs_2d: dict[str, str], 
                 debug: bool = False, 
                 verbose: bool = False, 
                 overwrite: bool = False):
        self.api = wandb.Api()
        self.entity = entity
        self.project = project
        self.runs_1d = runs_1d
        self.runs_2d = runs_2d
        self.debug = debug
        self.verbose = verbose
        self.overwrite = overwrite
        
    def get_alpha_beta(self, run_name: str, diagonal: bool = True) -> dict[str, float]:
        """
        args:
            run_name: e.g. 'exp12B-125M-50k-dyt-i0p5-s1'
    
        return:
            alpha_beta: e.g. {'mean': 3.3424, 'std': 3.3424, 'median': 3.3424, 'min': 3.3424, 'max': 3.3424}
        """
        run_id = self.runs_2d[run_name] if diagonal is False else self.runs_1d[run_name]
        path_alpha_beta = f'./wandb/{run_name}.json'
        if isfile(path_alpha_beta) and self.overwrite is False:
            if self.verbose:
                print(f'> read {path_alpha_beta}')
            with open(path_alpha_beta, 'r') as f:
                alpha_beta = json.load(f)
        else:
            steps = Helpers.steps_from_run_name(run_name)
            run = self.api.run(f"{self.entity}/{self.project}/{run_id}")
            df = pd.DataFrame(run.history(pandas=False))
            alpha_beta_cols = [col for col in df.columns if col.startswith('alpha') or col.startswith('beta')]
            alpha_beta_dict = {col: None for col in alpha_beta_cols}
            for alpha_beta_col in alpha_beta_cols:
                _alpha_beta = df[df['iter'] == steps][alpha_beta_col].values
                assert len(_alpha_beta) == 1, f'ERROR! alpha_beta = {_alpha_beta} should be single float number'
                alpha_beta_dict[alpha_beta_col] = float(_alpha_beta[0])
            alpha_beta_dict = {int(k.lstrip('alpha/alpha_').lstrip('beta/beta_')): v for k, v in alpha_beta_dict.items()}
            alpha_beta_dict = {k: v for k, v in sorted(alpha_beta_dict.items(), key=lambda item: item[0])}
            alpha_beta_values = [v for v in alpha_beta_dict.values()]

            alpha_beta = {
                'mean': float(np.mean(alpha_beta_values)),
                'std': float(np.std(alpha_beta_values)),
                'median': float(np.median(alpha_beta_values)),
                'max': float(np.max(alpha_beta_values)),
                'min': float(np.min(alpha_beta_values)),
                'layers': alpha_beta_dict,
            }
    
            if self.debug:
                print(f'{run_name}: {alpha_beta}')
    
            if self.verbose:
                print(f'> write {path_alpha_beta}')

            with open(path_alpha_beta, 'w') as f:
                json.dump(alpha_beta, f)
        return alpha_beta


if __name__ == '__main__':
    RUNS_1D = {
        'exp12B-125M-50k-dyt-i0p25-s1': 'djm847bw',
        'exp12C-125M-50k-dyisrusp-i-5p0-s1': 'w03cv0cj',
    }
    RUNS_2D = {
        'exp12B-125M-50k-dytx-i0p25x0p25-s1': '896lvjjv',
    }

    DEBUG = False
    VERBOSE = True
    OVERWRITE = False

    alpha_beta_instance = AlphaBeta(ENTITY, PROJECT, RUNS_1D, RUNS_2D, debug=DEBUG, verbose=VERBOSE, overwrite=OVERWRITE)
    for run_name in RUNS_1D:
        if not 'ln' in run_name:
            alpha_beta = alpha_beta_instance.get_alpha_beta(run_name)
            print(f'alpha_beta for {run_name}:')
            print(alpha_beta)