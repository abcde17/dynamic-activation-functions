
# nanogpt-dyaf

This repository is based on the [nanoGPT](https://github.com/karpathy/nanoGPT) commit [93a43d9](https://github.com/karpathy/nanogpt/commit/93a43d9a5c22450bbf06e78da2cb6eeef084b717).

Our changes include the implementation and use of dynamic activation functions (DyT, DyISRU) in addition to layer normalization (LN). 
They have been added in a single commit [TODO](https://github.com/abcde17/dynamic-activation-functions/commit/TODO) in order to facilitate comparison.

## 1. Preparation

### Create and Activate a Virtual Environment

```
# e.g. using conda
conda create -n dyaf python=3.11
conda activate dyaf
```

### Install Dependencies

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

### Data Preparation (OpenWebText)

```
cd data/openwebtext
python prepare.py
```

## 2. Training

### Training Config Creation

```
cd config
python script_create_config.py 12B --init 1 1 --seed 1
```

The main argument (`12B` above) contains two numbers and a letter. 

- The first number represents the model size N: 
    - 1: N = 125M
    - 3: N = 760M

- The second number represents the dataset size D:
    - 2: D =  5B
    - 5: D = 20B

- The letter represents the dynamic activation function being used:
    - A: LN
    - B: DyT
    - C: DyISRU (Softplus parametrization)
    - D: DyT (Softplus parametrization)
    - E: DyISRU

The argument `--init` takes two numbers (`1 1` above), the initial value for dynamic activation functions after non-attention and attention blocks, respectively.

The argument `--seed` specifies the random seed for reproducibility.

See the documentation at the top of `config/script_create_config.py` as well as `config/helper_functions.py` for more optional arguments that can be specified.

The above command will create a config file at `config/exp12B-125M-50k-dyt-i1p0x1p0-s1.py`. It can be used to train a N = 125M model for 10k steps (D = 5B), using DyT with initial values 1 for all alphas. The employed seed is 1.


### Execution

Specify your W&B entity and project in `settings.py` and make sure to be logged in to Weights & Biases: 

```
wandb login
```

To train on 4 GPUs:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py config/exp12B-125M-50k-dyt-i1p0x1p0-s1.py
```