import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

import matplotlib
import wandb

sns.set(font="serif", font_scale=1.4)
sns.set_style(
    "white",
    {
        "font.family": "serif",
        "font.weight": "normal",
        "font.serif": ["Times", "Palatino", "serif"],
        "axes.facecolor": "white",
        "lines.markeredgewidth": 1,
    },
)


def setup_plot():
    fig = plt.figure(dpi=100, figsize=(5.0, 3.0))
    ax = plt.subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.tick_params(axis="both", which="minor", labelsize=15)
    ax.tick_params(direction="in")


def plot(env, irl_metrics, last_return, steps, sz, filename):
    setup_plot()
    plot_wall_clock_time = (
        False  # Set to True to plot wall clock time instead of timesteps
    )
    if plot_wall_clock_time:
        start_time = wandb.run.start_time
        current_time = time.time()
        run_duration = current_time - start_time
        sz = run_duration / steps
        x_axis_label = "Wall clock time (s)"
    else:
        x_axis_label = "Timesteps"
    irl_mean = irl_metrics.mean(axis=-1)
    irl_std_err = np.std(irl_metrics, axis=-1) / np.sqrt(irl_metrics.shape[-1])
    plt.plot(np.arange(steps) * sz, irl_mean, label="IRL", color="#F79646")
    plt.fill_between(
        np.arange(steps) * sz,
        irl_mean - irl_std_err,
        irl_mean + irl_std_err,
        color="#F79646",
        alpha=0.1,
    )
    plt.axhline(last_return, label="Expert", color="#008000", linestyle="--")
    plt.ylabel("Reward")
    plt.xlabel(x_axis_label)
    plt.title(f"{env}")
    plt.legend(ncol=1, fontsize=12, loc="lower right")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
