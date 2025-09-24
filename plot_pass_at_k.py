import os
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns

EVAL_FOLDER = './eval'
TASKS = ["math", "gsm8k", "dapo-with-aime2425"]
SEEDS = [41, 42, 43]
ALGORITHMS = ["grpo", "elliptical", "unlikely", "untrained"]
CHECKPOINT_TYPE = "best_pass@1"

FACE_COLOR = "#F7F7FF"
MARKER = "o"
LINEWIDTH = 1.275
MARKERSIZE = 6
MARKEREDGEWIDTH = 0.9

LABEL_FONT_SIZE = 11
TITLE_FONT_SIZE = 13
TICK_LABEL_FONT_SIZE = 11
LEGEND_FONT_SIZE = 10

TASK_TO_NICE_NAME = {
    "math": "MATH",
    "gsm8k": "GSM8K",
    "dapo-with-aime2425": "AIME 2024",
    "countdown-4": "Countdown",
}

ALGO_TO_COLOR = {
    "grpo": sns.color_palette("deep")[-1],
    "untrained": sns.color_palette("deep")[7],
    "elliptical": sns.color_palette("colorblind")[2],
    "unlikely": sns.color_palette("deep")[1],
}

ALGO_TO_NICE_NAME = {
    "grpo": "GRPO",
    "untrained": "Base Model",
    "elliptical": r"RepExp (ours)",
    "unlikely": "Unlikeliness",
}

TASK_ALGO_TO_MARKER_DELTA = {
    "math": {
        "grpo": -0.05,
        "unlikely": -0.1,
    },
    "dapo-with-aime2425": {
        "grpo": -0.17,
        "unlikely": -0.15,
    },
    "countdown-4": {
        "grpo": -0.05,
        "unlikely": -0.05,
    },
    "gsm8k": {
        "grpo": -0.03,
        "unlikely": -0.02,
    }
}

TASK_TO_MOVE_UP = {
    "math": 0.004,
    "gsm8k": 0.002,
    "dapo-with-aime2425": 0.008,
    "countdown-4": 0.008,
}

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

def plot_arrow(ax, method_1_xs, method_1_ys, method_2_ys, color, y_min, marker_delta: float = 0.0, task: str = "math", algo: str = "grpo"):
    y_min_overlap = max(method_1_ys.min(), method_2_ys.min())
    y_max_overlap = min(method_1_ys.max(), method_2_ys.max())

    y_grid = np.linspace(y_min_overlap, y_max_overlap, 400)
    x_method_1_at_y = np.interp(y_grid, method_1_ys, method_1_xs)
    x_method_2_at_y = np.interp(y_grid, method_2_ys, method_1_xs)
    # Work in log2(k) space to match the axis scaling
    dx_log = np.abs(np.log2(x_method_2_at_y) - np.log2(x_method_1_at_y))
    # max_idx = int(np.argmax(dx_log))
    max_idx = -1
    y_star = y_grid[max_idx]
    x_random_star = x_method_1_at_y[max_idx]
    x_elliptical_star = x_method_2_at_y[max_idx]

    # Draw double-headed arrow between the two x positions at y_star
    ax.annotate(
        '',
        xy=(x_random_star, y_star + marker_delta),
        xytext=(x_elliptical_star, y_star + marker_delta),
        arrowprops=dict(arrowstyle='<->', color=color, lw=1.2)
    )

    # Place a small label at the geometric mean x to look centered on a log-x axis
    mid_x = np.sqrt(x_random_star * x_elliptical_star)
    ax.text(
        mid_x,
        y_star + marker_delta + TASK_TO_MOVE_UP[task],  # move up slightly
        r"$\mathbf{\times" + f"{(x_random_star / x_elliptical_star):.1f}" + "}$",
        color=color,
        ha='center',
        va='bottom',
        fontsize=8
    )

    if task == 'dapo-with-aime2425' and algo == 'grpo':
        print(x_random_star, x_elliptical_star)

    # vertical line at x_random_star & x_elliptical_star, up to y_star
    if marker_delta != 0:
        ax.vlines(x_random_star + (3 if algo == 'unlikely' else -3), ymin=y_min, ymax=y_star, color=color, linestyle='dashed', linewidth=1.0)
        ax.vlines(x_elliptical_star, ymin=y_min, ymax=y_star, color=color, linestyle='dashed', linewidth=1.0)

def main():
    eval_folders = os.listdir(EVAL_FOLDER)

    algo_to_xs = {}
    algo_to_ys = {}
    
    sns.set_style("whitegrid")
    # make figure with 3 subplots in a row
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
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

            algo_to_xs[algorithm] = np.array(list(pass_at_k.keys()))
            algo_to_ys[algorithm] = np.array([pass_at_k[k] for k in algo_to_xs[algorithm]])

            # plot the data
            xs = list(pass_at_k.keys())
            ys = np.array([pass_at_k[k] for k in xs])
            ax.plot(
                xs, 
                ys, 
                color=ALGO_TO_COLOR[algorithm], 
                label=algorithm, 
                markeredgecolor=FACE_COLOR, 
                marker=MARKER, 
                linewidth=LINEWIDTH, 
                markersize=MARKERSIZE, 
                markeredgewidth=MARKEREDGEWIDTH,
                alpha=1.0 if algorithm != "untrained" else 0.8
            )

            if algorithm != "untrained":
                sems = np.array([pass_at_k_sem[k] for k in xs])
                ax.fill_between(xs, ys - sems, ys + sems, alpha=0.2, color=ALGO_TO_COLOR[algorithm])

            
            if task == 'math':
                y_min = 0.7
                ax.set_ylim(top=0.95, bottom=y_min)
            elif task == 'gsm8k':
                y_min = 0.925
                ax.set_ylim(top=0.995, bottom=y_min)
            elif task == 'dapo-with-aime2425':
                y_min = 0.1
                ax.set_ylim(bottom=y_min, top=0.63)
            ax.set_xlim(left=2**(-0.2), right=2**(8.2))
            # ax.legend()
            ax.set_xscale("log", base=2)
            x_ticks = [2**i for i in range(int(np.log2(max(xs))) + 1)]
            x_tick_labels = [f"$2^{{{i}}}$" for i in range(int(np.log2(max(xs))) + 1)]
            ax.set_xticks(x_ticks, x_tick_labels)
            ax.set_xlabel("k", fontsize=LABEL_FONT_SIZE)
            if i == 0:
                ax.set_ylabel("Pass@k", fontsize=LABEL_FONT_SIZE)
            ax.set_title(f"{TASK_TO_NICE_NAME[task]}", fontsize=TITLE_FONT_SIZE)

        for _label in ax.get_xticklabels():
            _label.set_fontsize(TICK_LABEL_FONT_SIZE)
        for _label in ax.get_yticklabels():
            _label.set_fontsize(TICK_LABEL_FONT_SIZE)

        # repexp vs. grpo
        plot_arrow(ax, algo_to_xs["grpo"], algo_to_ys["grpo"], algo_to_ys["elliptical"], ALGO_TO_COLOR["grpo"], y_min, marker_delta=TASK_ALGO_TO_MARKER_DELTA[task]["grpo"], task=task, algo='grpo')
        plot_arrow(ax, algo_to_xs["unlikely"], algo_to_ys["unlikely"], algo_to_ys["elliptical"], ALGO_TO_COLOR["unlikely"], y_min, marker_delta=TASK_ALGO_TO_MARKER_DELTA[task]["unlikely"], task=task, algo='unlikely')

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D(
            [0], [0],
            color=ALGO_TO_COLOR[algo],
            marker=MARKER,
            linestyle='-',
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            markeredgewidth=MARKEREDGEWIDTH,
            markeredgecolor=FACE_COLOR,
            label=ALGO_TO_NICE_NAME[algo]
        )
        for algo in ALGORITHMS
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(ALGORITHMS),
        bbox_to_anchor=(0.5, -0.07),
        fontsize=LEGEND_FONT_SIZE
    )

    plt.tight_layout()
    plt.savefig(os.path.join("figures", f"rl_pass_at_k_{TASKS}_{CHECKPOINT_TYPE}.pdf"), bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()