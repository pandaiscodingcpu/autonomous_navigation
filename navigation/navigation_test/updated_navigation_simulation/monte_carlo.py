"""Monte Carlo: random start heading, ground wind speed/direction and sensor noise.

Compares landing accuracy with and without wind compensation (same random conditions).

    python monte_carlo.py [n_runs]
"""
import sys
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from simulation import run_simulation, summarize


def random_config(base: Config, rng):
    speed = rng.uniform(1.0, 4.0)                       # ground wind [m/s] (up to ~6.4 aloft)
    direction = rng.uniform(0, 2 * np.pi)
    return replace(base,
                   start_heading_deg=rng.uniform(0, 360),
                   wind_ground=(speed * np.cos(direction), speed * np.sin(direction)))


def run_batch(n_runs=20, master_seed=7):
    base = Config()
    rng = np.random.default_rng(master_seed)
    rows = {True: [], False: []}
    for i in range(n_runs):
        cfg = random_config(base, rng)
        seed = int(rng.integers(1_000_000))
        for comp in (True, False):
            c = replace(cfg, use_wind_comp=comp)
            s = summarize(run_simulation(c, seed), c)
            rows[comp].append(s)
        print(f"run {i + 1:2d}/{n_runs}: miss with comp = {rows[True][-1]['miss_distance_m']:5.1f} m,"
              f"  without = {rows[False][-1]['miss_distance_m']:5.1f} m", flush=True)
    return base, rows


def report(base, rows, path="figures/fig8_monte_carlo.png"):
    tN, tE = base.target
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    colors = {True: "tab:blue", False: "tab:gray"}
    names = {True: "with wind compensation", False: "without wind compensation"}
    print()
    for comp in (True, False):
        miss = np.array([r["miss_distance_m"] for r in rows[comp]])
        wind = np.array([r["wind_rmse_mps"] for r in rows[comp]])
        print(f"{names[comp]:28s} miss: mean {miss.mean():5.1f} m, median {np.median(miss):5.1f} m, "
              f"90th pct {np.percentile(miss, 90):5.1f} m, max {miss.max():5.1f} m   "
              f"| wind RMSE {wind.mean():.2f} m/s")
        dN = np.array([r["landing_N"] for r in rows[comp]]) - tN
        dE = np.array([r["landing_E"] for r in rows[comp]]) - tE
        ax[0].plot(dE, dN, "o", color=colors[comp], alpha=0.8, label=names[comp])
        ax[1].plot(np.sort(miss), np.linspace(0, 1, len(miss)), drawstyle="steps-post",
                   color=colors[comp], label=names[comp])
    for r in (10, 25, 50):
        ax[0].add_patch(plt.Circle((0, 0), r, fill=False, ls=":", color="r", alpha=0.5))
    ax[0].plot(0, 0, "r*", ms=16)
    ax[0].set_aspect("equal"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[0].set_xlabel("East of target [m]"); ax[0].set_ylabel("North of target [m]")
    ax[0].set_title("Touchdown points relative to the target")
    ax[1].set_xlabel("miss distance [m]"); ax[1].set_ylabel("cumulative fraction")
    ax[1].set_title("Miss-distance CDF"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    base, rows = run_batch(n)
    report(base, rows)
