'''
The code is AI generated
'''



"""Run the full closed-loop simulation and generate all figures.

    python run_simulation.py
"""
from dataclasses import replace

import numpy as np

from config import Config
from plotting import make_all
from simulation import run_simulation, summarize

SEED = 1


def print_summary(title, s):
    print(f"\n--- {title} ---")
    print(f"  flight time          : {s['flight_time_s']:.1f} s")
    print(f"  touchdown (N, E)     : ({s['landing_N']:.1f}, {s['landing_E']:.1f}) m")
    print(f"  MISS DISTANCE        : {s['miss_distance_m']:.1f} m")
    print(f"  wind RMSE            : {s['wind_rmse_mps']:.2f} m/s")
    print(f"  airspeed RMSE        : {s['airspeed_rmse_mps']:.2f} m/s")
    print(f"  heading RMSE         : {s['heading_rmse_deg']:.2f} deg")
    print(f"  descent-rate RMSE    : {s['descent_rmse_mps']:.3f} m/s")
    print(f"  altitude RMSE        : {s['altitude_rmse_m']:.2f} m")


if __name__ == "__main__":
    cfg = Config()                                   # wind compensation ON
    cfg_off = replace(cfg, use_wind_comp=False)      # same seed, compensation OFF

    log = run_simulation(cfg, SEED)
    log_off = run_simulation(cfg_off, SEED)

    print(f"Start: alt {cfg.start_alt:.0f} m at {cfg.start_pos}, target (N, E) = {cfg.target}")
    print_summary("with wind compensation (EKF wind estimate in the heading command)", summarize(log, cfg))
    print_summary("without wind compensation (aim straight at the target)", summarize(log_off, cfg_off))

    files = make_all(log, cfg, "figures", log_nocomp=log_off)
    np.savez_compressed("figures/sim_log.npz", **log)
    print("\nFigures:", *files, sep="\n  ")