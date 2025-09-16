import os
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns


EVAL_FOLDER = './eval'
TASKS = ["gsm8k"]
SEEDS = [41, 42, 43]
ALGORITHMS = ["grpo", "unlikely", "elliptical"]
CHECKPOINT_TYPE = "best_pass@1"

def process_data(data):
    pass_at_k = defaultdict(list)
    for d in data:
        for key, v in d.items():
            for k in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                if key.endswith(f"reward/pass@{k}/mean"):
                    pass_at_k[k].append(v)

    for k in pass_at_k.keys():
        assert len(pass_at_k[k]) == len(SEEDS)

    pass_at_k_sem = {k: stats.sem(v) for k, v in pass_at_k.items()}
    pass_at_k = {k: np.mean(v) for k, v in pass_at_k.items()}
    
    return pass_at_k, pass_at_k_sem

def main():
    eval_folders = os.listdir(EVAL_FOLDER)
    
    sns.set_style("whitegrid")
    for task in TASKS:
        for algorithm in ALGORITHMS:
            folders = [f for f in eval_folders if task in f and algorithm in f]
            data = []
            for folder in folders:
                with open(os.path.join(EVAL_FOLDER, folder, CHECKPOINT_TYPE, "eval.json"), "r") as f:
                    data.append(json.load(f))

            pass_at_k, pass_at_k_sem = process_data(data)

            # plot the data
            xs = list(pass_at_k.keys())
            ys = np.array([pass_at_k[k] for k in xs])
            sems = np.array([pass_at_k_sem[k] for k in xs])
            plt.plot(xs, ys, label=algorithm)
            plt.fill_between(xs, ys - sems, ys + sems, alpha=0.2)

    plt.ylim(top=1.0)
    plt.xlim(left=1, right=256)
    plt.legend()
    plt.xscale("log", base=2)
    x_ticks = [2**i for i in range(int(np.log2(max(xs))) + 1)]
    x_tick_labels = [f"$2^{{{i}}}$" for i in range(int(np.log2(max(xs))) + 1)]
    plt.xticks(x_ticks, x_tick_labels)
    plt.xlabel("k")
    plt.ylabel("Pass@k")
    plt.title(f"{task} {CHECKPOINT_TYPE}")
    plt.savefig(f"{task}_{CHECKPOINT_TYPE}.pdf")
    plt.close()


if __name__ == "__main__":
    main()