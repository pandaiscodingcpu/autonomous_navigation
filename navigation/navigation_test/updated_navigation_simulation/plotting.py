"""All figures for the simulation (matplotlib, non-interactive backend)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F401  (registers 3D projection)

from simulation import landing_point
from utils import wrap_pi

DPI = 110
C_TRUE, C_EST, C_CMD, C_RAW = "tab:blue", "tab:orange", "tab:green", "0.7"


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _deg_aligned(ref, x):
    """Unwrap `ref` (rad) and express `x` on the same branch; return both in degrees."""
    r = np.unwrap(ref)
    return np.rad2deg(r), np.rad2deg(r + wrap_pi(x - r))


def _samples(t, x):
    m = np.isfinite(x)
    return t[m], x[m]


def _grid(ax, title, ylabel, xlabel=None):
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(alpha=0.3)


# ---------------------------------------------------------------- 3D trajectory
def plot_trajectory_3d(log, cfg, path, log_nocomp=None):
    s = slice(None, None, 5)
    E, N, h, t = log["true_pE"][s], log["true_pN"][s], log["true_h"][s], log["t"][s]
    tN, tE = cfg.target
    lN, lE = landing_point(log)

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection="3d")

    pts = np.column_stack([E, N, h]).reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(segs, cmap="viridis", norm=Normalize(0, t[-1]), linewidth=2)
    lc.set_array(t[:-1])
    ax.add_collection3d(lc)
    fig.colorbar(lc, ax=ax, shrink=0.55, pad=0.08, label="time [s]")

    ax.plot(log["est_pE"][s], log["est_pN"][s], log["true_h"][s], "--", color=C_EST, lw=0.8,
            label="EKF estimate")
    ax.plot(E, N, np.zeros_like(h), color="0.55", lw=1, label="ground shadow")
    if log_nocomp is not None:
        ax.plot(log_nocomp["true_pE"][s], log_nocomp["true_pN"][s], log_nocomp["true_h"][s],
                color="0.6", lw=0.8, alpha=0.8, label="no wind compensation")

    ax.plot([cfg.start_pos[1]], [cfg.start_pos[0]], [cfg.start_alt], "o", color="k", ms=8,
            label="release")
    ax.plot([tE], [tN], [0], "*", color="r", ms=16, label="target")
    ax.plot([lE], [lN], [0], "X", color="m", ms=11, label="touchdown")
    ax.plot([tE, tE], [tN, tN], [0, cfg.start_alt], ":", color="r", lw=0.8)

    allE = np.concatenate([E, [tE, lE]])
    allN = np.concatenate([N, [tN, lN]])
    cE, cN = (allE.max() + allE.min()) / 2, (allN.max() + allN.min()) / 2
    half = max(allE.max() - allE.min(), allN.max() - allN.min()) / 2 * 1.1 + 5
    ax.set_xlim(cE - half, cE + half)
    ax.set_ylim(cN - half, cN + half)
    ax.set_zlim(0, cfg.start_alt * 1.02)
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]"); ax.set_zlabel("Altitude [m]")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-58)
    ax.set_title(f"3D flight path - miss distance {np.hypot(lN - tN, lE - tE):.1f} m", fontsize=12)
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, path)


# ---------------------------------------------------------------- ground track
def plot_ground_track(log, cfg, path, log_nocomp=None):
    tN, tE = cfg.target
    lN, lE = landing_point(log)
    fig, ax = plt.subplots(figsize=(8.5, 8))
    if log_nocomp is not None:
        nN, nE = landing_point(log_nocomp)
        ax.plot(log_nocomp["true_pE"], log_nocomp["true_pN"], color="0.65", lw=1,
                label=f"no wind comp. (miss {np.hypot(nN - tN, nE - tE):.1f} m)")
        ax.plot([nE], [nN], "x", color="0.4", ms=9)
    ax.plot(log["true_pE"], log["true_pN"], color=C_TRUE, lw=1.3,
            label=f"truth, wind comp. (miss {np.hypot(lN - tN, lE - tE):.1f} m)")
    ax.plot(log["est_pE"], log["est_pN"], "--", color=C_EST, lw=0.8, label="EKF estimate")
    ax.plot(*cfg.start_pos[::-1], "ko", ms=8, label="release")
    ax.plot([tE], [tN], "r*", ms=18, label="target")
    ax.plot([lE], [lN], "mX", ms=11, label="touchdown")
    for r in (10, 25):
        ax.add_patch(plt.Circle((tE, tN), r, fill=False, ls=":", color="r", alpha=0.5))
    ax.set_aspect("equal")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.set_title("Ground track (top view)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    return _save(fig, path)


# ---------------------------------------------------------------- navigation / control
def plot_navigation_control(log, cfg, path):
    t = log["t"]
    fig, ax = plt.subplots(3, 2, figsize=(13, 10), sharex=True)

    tr, es = _deg_aligned(log["true_psi"], log["est_psi"])
    _, cm = _deg_aligned(log["true_psi"], log["psi_cmd"])
    ax[0, 0].plot(t, tr, color=C_TRUE, label="true heading")
    ax[0, 0].plot(t, es, "--", color=C_EST, label="EKF heading")
    ax[0, 0].plot(t, cm, color=C_CMD, lw=0.8, alpha=0.8, label="commanded heading")
    _grid(ax[0, 0], "Heading (unwrapped)", "deg"); ax[0, 0].legend(fontsize=8)

    ax[0, 1].plot(t, np.rad2deg(log["err"]), color=C_RAW, lw=0.8, label="signed error")
    ax[0, 1].plot(t, np.rad2deg(log["abs_err"]), color="tab:red", lw=1, label="|error angle|")
    _grid(ax[0, 1], "Error angle = |commanded heading - heading|  (PID input)", "deg")
    ax[0, 1].legend(fontsize=8)

    ax[1, 0].plot(t, log["u_cmd"], color=C_CMD, label="PID command")
    ax[1, 0].plot(t, log["true_u"], "--", color=C_TRUE, label="actual servo (lagged)")
    _grid(ax[1, 0], "Servo command", "normalised  (+ = right)"); ax[1, 0].legend(fontsize=8)

    ax[1, 1].plot(t, log["pid_p"], label="P")
    ax[1, 1].plot(t, log["pid_i"], label="I")
    ax[1, 1].plot(t, log["pid_d"], label="D")
    _grid(ax[1, 1], "PID terms", "servo units"); ax[1, 1].legend(fontsize=8)

    ax[2, 0].plot(t, log["dist_true"], color=C_TRUE, label="true")
    ax[2, 0].plot(t, log["dist_est"], "--", color=C_EST, lw=0.8, label="from estimate")
    _grid(ax[2, 0], "Distance to target", "m", "time [s]"); ax[2, 0].legend(fontsize=8)

    ax[2, 1].plot(t, log["gyro"], color=C_RAW, lw=0.5, label="raw gyro")
    ax[2, 1].plot(t, log["est_rate"], color=C_EST, lw=0.8, label="gyro - estimated bias")
    ax[2, 1].plot(t, log["true_r"], color=C_TRUE, lw=1, label="true yaw rate")
    _grid(ax[2, 1], "Yaw rate", "rad/s", "time [s]"); ax[2, 1].legend(fontsize=8)
    return _save(fig, path)


# ---------------------------------------------------------------- wind / EKF
def plot_wind_estimation(log, cfg, path):
    t = log["t"]
    fig, ax = plt.subplots(4, 2, figsize=(13, 13), sharex=True)

    for col, (name, lab) in enumerate((("wN", "North"), ("wE", "East"))):
        a = ax[0, col]
        a.plot(t, log["true_" + name], color=C_TRUE, label="true (incl. gusts)")
        a.plot(t, log["est_" + name], "--", color=C_EST, label="EKF estimate")
        a.fill_between(t, log["est_" + name] - 3 * log["std_" + name],
                       log["est_" + name] + 3 * log["std_" + name], color=C_EST, alpha=0.2,
                       label="±3σ")
        _grid(a, f"Wind {lab} component", "m/s"); a.legend(fontsize=8)

    ts = np.hypot(log["true_wN"], log["true_wE"])
    es = np.hypot(log["est_wN"], log["est_wE"])
    ax[1, 0].plot(t, ts, color=C_TRUE, label="true"); ax[1, 0].plot(t, es, "--", color=C_EST, label="EKF")
    _grid(ax[1, 0], "Wind speed", "m/s"); ax[1, 0].legend(fontsize=8)

    td = np.rad2deg(np.arctan2(log["true_wE"], log["true_wN"])) % 360
    ed = np.rad2deg(np.arctan2(log["est_wE"], log["est_wN"])) % 360
    ax[1, 1].plot(t, td, color=C_TRUE, label="true"); ax[1, 1].plot(t, ed, "--", color=C_EST, label="EKF")
    _grid(ax[1, 1], "Wind direction (blowing towards, clockwise from North)", "deg")
    ax[1, 1].legend(fontsize=8)

    ax[2, 0].plot(t, log["true_Va"], color=C_TRUE, label="true"); ax[2, 0].plot(t, log["est_Va"], "--", color=C_EST, label="EKF")
    _grid(ax[2, 0], "Horizontal airspeed", "m/s"); ax[2, 0].legend(fontsize=8)

    ax[2, 1].plot(t, log["true_bias"], color=C_TRUE, label="true"); ax[2, 1].plot(t, log["est_bias"], "--", color=C_EST, label="EKF")
    _grid(ax[2, 1], "Gyro bias", "rad/s"); ax[2, 1].legend(fontsize=8)

    he = np.rad2deg(wrap_pi(log["est_psi"] - log["true_psi"]))
    ax[3, 0].plot(t, he, color=C_EST, lw=0.8)
    ax[3, 0].fill_between(t, -3 * np.rad2deg(log["std_psi"]), 3 * np.rad2deg(log["std_psi"]),
                          color=C_EST, alpha=0.2, label="±3σ")
    _grid(ax[3, 0], "Heading error (EKF - true)", "deg", "time [s]"); ax[3, 0].legend(fontsize=8)

    ax[3, 1].plot(t, log["est_pN"] - log["true_pN"], lw=0.8, label="North")
    ax[3, 1].plot(t, log["est_pE"] - log["true_pE"], lw=0.8, label="East")
    _grid(ax[3, 1], "Position error (EKF - true)", "m", "time [s]"); ax[3, 1].legend(fontsize=8)
    return _save(fig, path)


# ---------------------------------------------------------------- descent KF
def plot_descent_kf(log, cfg, path):
    t = log["t"]
    tb, hb = _samples(t, log["baro_h"])
    tb_true = np.interp(tb, t, log["true_h"])
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

    ax[0, 0].plot(t, log["true_h"], color=C_TRUE, label="true")
    ax[0, 0].plot(t, log["est_h"], "--", color=C_EST, label="1D KF")
    _grid(ax[0, 0], "Altitude", "m"); ax[0, 0].legend(fontsize=8)

    ax[0, 1].plot(tb, hb - tb_true, ".", color=C_RAW, ms=2, label=f"raw baro (σ={cfg.baro_sigma} m)")
    ax[0, 1].plot(t, log["est_h"] - log["true_h"], color=C_EST, lw=0.9, label="1D KF")
    _grid(ax[0, 1], "Altitude error", "m"); ax[0, 1].legend(fontsize=8)

    ax[1, 0].plot(t, log["true_vd"], color=C_TRUE, label="true")
    ax[1, 0].plot(t, log["est_vd"], "--", color=C_EST, label="1D KF (no dh/dt)")
    _grid(ax[1, 0], "Descent velocity (positive = down)", "m/s", "time [s]"); ax[1, 0].legend(fontsize=8)

    ax[1, 1].plot(t, log["est_vd"] - log["true_vd"], color=C_EST, lw=0.9)
    _grid(ax[1, 1], "Descent velocity error (KF - true)", "m/s", "time [s]")
    return _save(fig, path)


# ---------------------------------------------------------------- raw sensor errors
def plot_sensors(log, cfg, path):
    t = log["t"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    ax[0, 0].plot(t, log["gyro"] - log["true_r"], color=C_RAW, lw=0.5)
    ax[0, 0].plot(t, log["true_bias"], color=C_TRUE, lw=1, label="true bias")
    _grid(ax[0, 0], "Gyro error (raw - true rate)", "rad/s"); ax[0, 0].legend(fontsize=8)

    tm, m = _samples(t, log["mag_psi"])
    ax[0, 1].plot(tm, np.rad2deg(wrap_pi(m - np.interp(tm, t, np.unwrap(log["true_psi"])))), ".", ms=2)
    _grid(ax[0, 1], "Magnetometer heading error", "deg")

    tb, b = _samples(t, log["baro_h"])
    ax[0, 2].plot(tb, b - np.interp(tb, t, log["true_h"]), ".", ms=2)
    _grid(ax[0, 2], "Barometric altitude error", "m")

    tg, gN = _samples(t, log["gps_pN"]); _, gE = _samples(t, log["gps_pE"])
    ax[1, 0].plot(tg, gN - np.interp(tg, t, log["true_pN"]), ".", ms=3, label="North")
    ax[1, 0].plot(tg, gE - np.interp(tg, t, log["true_pE"]), ".", ms=3, label="East")
    _grid(ax[1, 0], "GPS position error", "m", "time [s]"); ax[1, 0].legend(fontsize=8)

    _, vN = _samples(t, log["gps_vN"]); _, vE = _samples(t, log["gps_vE"])
    ax[1, 1].plot(tg, vN - np.interp(tg, t, log["true_vN"]), ".", ms=3, label="North")
    ax[1, 1].plot(tg, vE - np.interp(tg, t, log["true_vE"]), ".", ms=3, label="East")
    _grid(ax[1, 1], "GPS velocity error", "m/s", "time [s]"); ax[1, 1].legend(fontsize=8)

    ax[1, 2].plot(tg, np.hypot(vN, vE), ".", ms=3, color=C_RAW, label="GPS ground speed")
    ax[1, 2].plot(t, np.hypot(log["true_vN"], log["true_vE"]), color=C_TRUE, lw=0.8, label="true ground speed")
    _grid(ax[1, 2], "Ground speed", "m/s", "time [s]"); ax[1, 2].legend(fontsize=8)
    return _save(fig, path)


# ---------------------------------------------------------------- wind profile
def plot_wind_profile(log, cfg, path):
    h = log["true_h"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    ts = np.hypot(log["true_wN"], log["true_wE"]); es = np.hypot(log["est_wN"], log["est_wE"])
    ax[0].plot(ts, h, color=C_TRUE, label="true"); ax[0].plot(es, h, "--", color=C_EST, label="EKF")
    _grid(ax[0], "Wind speed vs altitude", "altitude [m]", "m/s"); ax[0].legend(fontsize=8)
    td = np.rad2deg(np.arctan2(log["true_wE"], log["true_wN"])) % 360
    ed = np.rad2deg(np.arctan2(log["est_wE"], log["est_wN"])) % 360
    ax[1].plot(td, h, color=C_TRUE, label="true"); ax[1].plot(ed, h, "--", color=C_EST, label="EKF")
    _grid(ax[1], "Wind direction vs altitude", "", "deg (towards)"); ax[1].legend(fontsize=8)
    return _save(fig, path)


def make_all(log, cfg, outdir, log_nocomp=None):
    p = lambda n: f"{outdir}/{n}"
    return [
        plot_trajectory_3d(log, cfg, p("fig1_trajectory_3d.png"), log_nocomp),
        plot_ground_track(log, cfg, p("fig2_ground_track.png"), log_nocomp),
        plot_navigation_control(log, cfg, p("fig3_navigation_control.png")),
        plot_wind_estimation(log, cfg, p("fig4_wind_estimation.png")),
        plot_descent_kf(log, cfg, p("fig5_descent_kf.png")),
        plot_sensors(log, cfg, p("fig6_sensors.png")),
        plot_wind_profile(log, cfg, p("fig7_wind_profile.png")),
    ]
