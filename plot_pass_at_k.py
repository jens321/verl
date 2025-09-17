import os
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns


EVAL_FOLDER = './eval'
TASKS = ["math", "gsm8k", "gsm8k"]
SEEDS = [41, 42, 43]
ALGORITHMS = ["grpo", "unlikely", "elliptical"]
CHECKPOINT_TYPE = "best_pass@1"

def process_data(data, algorithm):
    pass_at_k = defaultdict(list)
    for d in data:
        for key, v in d.items():
            for k in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                if key.endswith(f"reward/pass@{k}/mean"):
                    pass_at_k[k].append(v)

    if algorithm != "untrained":
        for k in pass_at_k.keys():
            assert len(pass_at_k[k]) == len(SEEDS)

    pass_at_k_sem = {k: stats.sem(v) for k, v in pass_at_k.items()}
    pass_at_k = {k: np.mean(v) for k, v in pass_at_k.items()}
    
    return pass_at_k, pass_at_k_sem

def main():
    eval_folders = os.listdir(EVAL_FOLDER)
    
    sns.set_style("whitegrid")
    # make figure with 3 subplots in a row
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for i, (ax, task) in enumerate(zip(axs, TASKS)):
        for algorithm in ALGORITHMS:
            folders = [f for f in eval_folders if task in f and algorithm in f]
            if len(folders) == 0:
                continue

            data = []
            for folder in folders:
                if algorithm == "untrained":
                    with open(os.path.join(EVAL_FOLDER, folder, "eval.json"), "r") as f:
                        data.append(json.load(f))
                else:
                    with open(os.path.join(EVAL_FOLDER, folder, CHECKPOINT_TYPE, "eval.json"), "r") as f:
                        data.append(json.load(f))

            pass_at_k, pass_at_k_sem = process_data(data, algorithm)

            # plot the data
            xs = list(pass_at_k.keys())
            ys = np.array([pass_at_k[k] for k in xs])
            ax.plot(xs, ys, label=algorithm)

            if algorithm != "untrained":
                sems = np.array([pass_at_k_sem[k] for k in xs])
                ax.fill_between(xs, ys - sems, ys + sems, alpha=0.2)

            if task == 'gsm8k' and algorithm == 'untrained':
                ax.set_ylim(top=1.0, bottom=0.7)
            else:
                ax.set_ylim(top=1.0)
            ax.set_xlim(left=1, right=256)
            ax.legend()
            ax.set_xscale("log", base=2)
            x_ticks = [2**i for i in range(int(np.log2(max(xs))) + 1)]
            x_tick_labels = [f"$2^{{{i}}}$" for i in range(int(np.log2(max(xs))) + 1)]
            ax.set_xticks(x_ticks, x_tick_labels)
            ax.set_xlabel("k")
            if i == 0:
                ax.set_ylabel("Pass@k")
            ax.set_title(f"{task} {CHECKPOINT_TYPE}")

    plt.tight_layout()
    plt.savefig(os.path.join("figures", f"rl_pass_at_k_{TASKS}_{CHECKPOINT_TYPE}.pdf"))
    plt.close()


if __name__ == "__main__":
    main()