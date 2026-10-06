"""Check the PID + plant in isolation: heading step response (true heading, no wind, no noise)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from parafoil import Parafoil
from pid import PID
from utils import wrap_pi


def step_response(cfg: Config, step_deg=90.0, t_end=30.0):
    dt = cfg.dt
    plant = Parafoil(cfg)
    plant.psi = 0.0
    pid = PID(cfg.kp, cfg.ki, cfg.kd, dt * cfg.ctrl_div, 1.0, cfg.i_limit, cfg.d_tau)
    target = np.deg2rad(step_deg)
    t, psi, u = [], [], []
    u_cmd = 0.0
    for k in range(int(t_end / dt)):
        if k % cfg.ctrl_div == 0:
            u_cmd = pid.update(wrap_pi(target - plant.psi))
        plant.step(u_cmd, (0.0, 0.0), dt)
        t.append((k + 1) * dt)
        psi.append(plant.psi)
        u.append(u_cmd)
    return np.array(t), np.rad2deg(np.unwrap(psi)), np.array(u)


def metrics(t, psi_deg, step_deg):
    final = psi_deg[-1]
    overshoot = max(0.0, (psi_deg.max() - step_deg) / step_deg * 100)
    t10 = t[np.argmax(psi_deg >= 0.1 * step_deg)]
    t90 = t[np.argmax(psi_deg >= 0.9 * step_deg)]
    outside = np.where(np.abs(psi_deg - step_deg) > 0.02 * step_deg)[0]
    settle = t[outside[-1]] if len(outside) else 0.0
    return dict(final_deg=final, overshoot_pct=overshoot, rise_time_s=t90 - t10,
                settling_2pct_s=settle, steady_state_err_deg=step_deg - final)


if __name__ == "__main__":
    cfg = Config()
    step = 90.0
    t, psi, u = step_response(cfg, step)
    m = metrics(t, psi, step)
    print(f"PID step test ({step:.0f} deg): " + ", ".join(f"{k}={v:.2f}" for k, v in m.items()))

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(t, psi, label="heading")
    ax[0].axhline(step, color="r", ls="--", label="target heading")
    ax[0].set_ylabel("heading [deg]")
    ax[0].set_title(f"PID heading step response  (Kp={cfg.kp}, Ki={cfg.ki}, Kd={cfg.kd})")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(t, u)
    ax[1].set_ylabel("servo command"); ax[1].set_xlabel("time [s]"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/fig0_pid_step_response.png", dpi=110)
    print("saved figures/fig0_pid_step_response.png")
