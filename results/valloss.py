
from os.path import isfile
import numpy as np
import pandas as pd
import wandb

from helpers import Helpers

# SETTINGS
import sys
from os.path import abspath
source_path = abspath('..')
if not source_path in sys.path:
    sys.path.append(source_path)
    from settings import ENTITY, PROJECT


class ValLoss:

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
        
    def get_val_loss(self, run_name: str, diagonal: bool = True) -> float:
        """
        args:
            run_name: e.g. 'exp12B-125M-50k-dyt-i0p5-s1'
    
        return:
            steps: e.g. 3.342463493347168
        """
        run_id = self.runs_2d[run_name] if diagonal is False else self.runs_1d[run_name]
        path_val_loss = f'./wandb/{run_name}.npy'
        if isfile(path_val_loss) and self.overwrite is False:
            if self.verbose:
                print(f'> read {path_val_loss}')
            with open(path_val_loss, 'rb') as f:
                val_loss = float(np.load(f))
        else:
            steps = Helpers.steps_from_run_name(run_name)
            run = self.api.run(f"{self.entity}/{self.project}/{run_id}")
            df = pd.DataFrame(run.history(pandas=False))
            val_loss = df[df['iter'] == steps]['val/loss'].values
            assert len(val_loss) == 1, f'ERROR! val_loss = {val_loss} should be single float number'
            val_loss = float(val_loss[0])
    
            if self.debug:
                print(f'{run_name}: {val_loss:.3f}')
    
            if self.verbose:
                print(f'> write {path_val_loss}')
            with open(path_val_loss, 'wb') as f:
                np.save(f, val_loss)
        return val_loss

if __name__ == '__main__':
    RUNS_1D = {
        'exp12A-125M-50k-ln-i1p0-s1': 'ecvdsgbj',
    }
    RUNS_2D = {
        'exp12B-125M-50k-dytx-i0p25x0p25-s1': '896lvjjv',
    }
    val_loss = ValLoss(ENTITY, PROJECT, RUNS_1D, RUNS_2D, debug=False, verbose=False, overwrite=False)
    for run_name in RUNS_1D:
        loss = val_loss.get_val_loss(run_name)
        print(f'val_loss for {run_name}: {loss:.3f}')