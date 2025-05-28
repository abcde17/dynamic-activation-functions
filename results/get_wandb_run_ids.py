"""
Script to get run_name / run_id combinations from Weights & Biases where run_name contains specified filter.

Usage:
python get_wandb_run_ids.py FILTER

Example:
python get_wandb_run_ids.py exp12

Example Output (print):
    'exp12B-125M-50k-dyt-i0p25-s1': 'djm847bw',
    'exp12B-125M-50k-dyt-i0p5-s1': 'plmc10vw',
    [..]
"""

import argparse
import wandb

from settings import ENTITY, PROJECT

def main(args):
    print(f'filter: {args.filter}\n')
    api = wandb.Api()
    runs = api.runs(path=f"{ENTITY}/{PROJECT}")
    runs_filtered = [(elem.name, elem.id) for elem in runs if args.filter in elem.name]
    runs_filtered = {tup[0]: tup[1] for tup in runs_filtered}
    runs_filtered = {k: v for k, v in sorted(runs_filtered.items(), key=lambda item: item[0])}
    print("run_name: run_id")
    for k, v in runs_filtered.items():
        print(f"    '{k}': '{v}',")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filter', type=str)
    args = parser.parse_args()
    main(args)    


