
from os.path import isfile
import numpy as np
import pandas as pd
import wandb

from settings import ENTITY, PROJECT


class TimePerStep:

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
        
    def get_time_per_step(self, run_name: str, diagonal: bool = True) -> float:
        """
        args:
            run_name: e.g. 'exp12B-125M-50k-dyt-i0p5-s1'
    
        return:
            time_per_step: e.g. 1000.3
        """
        run_id = self.runs_2d[run_name] if diagonal is False else self.runs_1d[run_name]
        path_time_per_step = f'./wandb/{run_name}-TIME_PER_STEP.npy'
        if isfile(path_time_per_step) and self.overwrite is False:
            if self.verbose:
                print(f'> read {path_time_per_step}')
            with open(path_time_per_step, 'rb') as f:
                time_per_step = float(np.load(f))
        else:
            run = self.api.run(f"{self.entity}/{self.project}/{run_id}")
            df = pd.DataFrame(run.history(pandas=False))
            time_per_step = df['time/per_step'].values[-1]
    
            if self.debug:
                print(f'{run_name}: {time_per_step:.3f}')
    
            if self.verbose:
                print(f'> write {path_time_per_step}')
            with open(path_time_per_step, 'wb') as f:
                np.save(f, time_per_step)
        return time_per_step

if __name__ == '__main__':
    RUNS_1D = {
        'exp12A-125M-50k-ln-i1p0-s1': 'ecvdsgbj',
    }
    RUNS_2D = {
        'exp12B-125M-50k-dytx-i0p25x0p25-s1': '896lvjjv',
    }
    time_per_step = TimePerStep(ENTITY, PROJECT, RUNS_1D, RUNS_2D, debug=False, verbose=False, overwrite=True)
    for run_name in RUNS_1D:
        time_per_step = time_per_step.get_time_per_step(run_name)
        print(f'time/step for {run_name}: {time_per_step:.3f}')