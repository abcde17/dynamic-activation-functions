# Results

## 1. Preparation

### Create and Activate a Virtual Environment

```
# e.g. using conda
conda create -n dyaf python=3.11
conda activate dyaf
```

### Install Dependencies

```
pip install numpy scipy matplotlib scienceplots jupyter
```

## 2. Analysis of Main Experiments (Section 5)

```
jupyter notebook
```

### A. Get the run names & ids from Weights & Biases

Specify your W&B entity and project in `settings.py`. Then run

```
python get_wandb_run_ids.py FILTER
```

This script, run with e.g. `FILTER = exp12`, will print the combinations `run_name: run_id` where `run_name` contains the specified filter.

### B. Register the run names & ids

The combinations `run_name: run_id` need to be copied and pasted to 
- the `RUNS_1D` dictionary in `data/runs_1d.py`
- the `RUNS_2D` dictionary in `data/runs_2d.py`

1D refers to the same initial value for all dynamic activation functions, 2D to different ones for those related to attention blocks. Runs present in `RUNS_1D` may also be present in `RUNS_2D`, but not the other way around.

### C. Analyze runs

The notebook `Results.ipynb` may then be used to analyze the results of the main experiments (Section 5).


## 3. Initial Values (Appendix C)

The notebook `Theory_Initial_Values.ipynb` can be used to reproduce the results and figures from Appendix C.