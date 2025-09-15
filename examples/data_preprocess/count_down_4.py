"""
Preprocess dataset for countdown-4 dataset to parquet format
"""

import re
import os
from typing import List, Tuple
from random import randint, seed, choice
import argparse

import datasets
from tqdm import tqdm

from verl.utils.hdfs_io import copy, makedirs

def make_prefix(dp):
    target = dp['target']
    numbers = dp['nums']
    # NOTE: also need to change reward_score/countdown.py
    prefix = f"""Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Let's think step by step and return the final answer in <answer> </answer> tags. For example <answer> (1 + 2) / 3 </answer>."""
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/countdown-4')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=8192)
    parser.add_argument('--dev_size', type=int, default=512)
    parser.add_argument('--test_size', type=int, default=1024)

    args = parser.parse_args()

    data_source = 'Jiayi-Pan/Countdown-Tasks-4'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    # NOTE: there is only train split, so we use train split as both train/dev/test
    dataset = datasets.load_dataset(data_source, split="train", trust_remote_code=True)
    train_range = range(args.train_size)
    dev_range = range(args.train_size, args.train_size + args.dev_size)
    test_range = range(args.train_size + args.dev_size, args.train_size + args.dev_size + args.test_size)
    train_dataset = dataset.select(train_range)
    dev_dataset = dataset.select(dev_range)
    test_dataset = dataset.select(test_range)

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example)
            solution = {
                "target": example['target'],
                "numbers": example['nums']
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    dev_dataset = dev_dataset.map(function=make_map_fn('dev'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    print(len(train_dataset), len(dev_dataset), len(test_dataset))
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    dev_dataset.to_parquet(os.path.join(local_dir, 'dev.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 