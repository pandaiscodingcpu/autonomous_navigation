"""Closed-loop simulation of the CanSat para-glider.

  truth plant -> sensors -> estimator (EKF + descent KF) -> navigation -> PID -> servo -> plant
"""
import numpy as np

from config import Config
from estimator import StateEstimator
from navigation import Navigator
from parafoil import Parafoil
from pid import PID
from sensors import Sensors
from utils import wrap_pi
from wind import WindField

KEYS = [
    "t",
    # truth
    "true_pN", "true_pE", "true_h", "true_psi", "true_Va", "true_vd", "true_vN", "true_vE",
    "true_wN", "true_wE", "true_r", "true_u", "true_bias",
    # raw sensors (NaN when no new sample)
    "gyro", "gps_pN", "gps_pE", "gps_vN", "gps_vE", "mag_psi", "baro_h",
    # estimates
    "est_pN", "est_pE", "est_h", "est_vd", "est_psi", "est_Va", "est_wN", "est_wE",
    "est_bias", "std_wN", "std_wE", "std_psi", "est_rate",
    # navigation + control
    "bearing", "psi_cmd", "err", "abs_err", "dist_est", "dist_true",
    "u_cmd", "pid_p", "pid_i", "pid_d",
]


def run_simulation(cfg: Config = None, seed: int = 0):
    """Run one descent. Returns a dict of numpy arrays (one entry per KEYS)."""
    cfg = cfg or Config()
    rng = np.random.default_rng(seed)
    dt = cfg.dt

    wind = WindField(cfg, rng)
    plant = Parafoil(cfg)
    sensors = Sensors(cfg, rng)
    est = StateEstimator(cfg)
    nav = Navigator(cfg.target, cfg.use_wind_comp, cfg.hold_radius)
    pid = PID(cfg.kp, cfg.ki, cfg.kd, dt * cfg.ctrl_div, 1.0, cfg.i_limit, cfg.d_tau)
    target = np.asarray(cfg.target, float)

    # ---- initial sensor snapshot -> initialise the estimator ----
    w0 = wind.at(plant.h, 0.0)
    v0 = (plant.cfg.va0 * np.cos(plant.psi) + w0[0], plant.cfg.va0 * np.sin(plant.psi) + w0[1])
    est.initialize(sensors.gps(plant.pN, plant.pE, *v0), sensors.mag(plant.psi),
                   sensors.baro(plant.h))

    log = {k: [] for k in KEYS}
    nav_out = None
    u_cmd = 0.0
    n_max = int(cfg.t_max / dt)
    nan = np.nan

    for k in range(n_max):
        # ---- controller (50 Hz) uses ESTIMATED quantities only ----
        if k % cfg.ctrl_div == 0:
            nav_out = nav.update(est.position, est.heading, est.wind, est.airspeed)
            u_cmd = pid.update(nav_out.error)

        # ---- truth plant ----
        w = wind.at(plant.h, dt)
        plant.step(u_cmd, w, dt)

        # ---- sensors + estimator ----
        gyro = sensors.gyro(plant.r, dt)
        est.predict(gyro)

        gps = (nan,) * 4
        mag = baro = nan
        if k % cfg.gps_div == 0:
            gps = sensors.gps(plant.pN, plant.pE, plant.vN, plant.vE)
            est.update_gps(*gps)
        if k % cfg.mag_div == 0:
            mag = sensors.mag(plant.psi)
            est.update_mag(mag)
        if k % cfg.baro_div == 0:
            baro = sensors.baro(plant.h)
            est.update_baro(baro)

        # ---- logging ----
        sg = est.sigmas
        row = dict(
            t=(k + 1) * dt,
            true_pN=plant.pN, true_pE=plant.pE, true_h=plant.h, true_psi=plant.psi,
            true_Va=plant.Va, true_vd=plant.sink, true_vN=plant.vN, true_vE=plant.vE,
            true_wN=w[0], true_wE=w[1], true_r=plant.r, true_u=plant.u, true_bias=sensors.bias,
            gyro=gyro, gps_pN=gps[0], gps_pE=gps[1], gps_vN=gps[2], gps_vE=gps[3],
            mag_psi=mag, baro_h=baro,
            est_pN=est.position[0], est_pE=est.position[1], est_h=est.altitude,
            est_vd=est.descent_velocity, est_psi=est.heading, est_Va=est.airspeed,
            est_wN=est.wind[0], est_wE=est.wind[1], est_bias=est.gyro_bias,
            std_wN=sg["wN"], std_wE=sg["wE"], std_psi=sg["psi"],
            est_rate=gyro - est.gyro_bias,
            bearing=nav_out.bearing, psi_cmd=nav_out.psi_cmd, err=nav_out.error,
            abs_err=nav_out.abs_error, dist_est=nav_out.dist,
            dist_true=float(np.hypot(*(target - (plant.pN, plant.pE)))),
            u_cmd=u_cmd, pid_p=pid.p_term, pid_i=pid.i_term, pid_d=pid.d_term,
        )
        for key in KEYS:
            log[key].append(row[key])

        if plant.h <= 0.0:                      # touched down
            break

    return {k: np.asarray(v, float) for k, v in log.items()}


def landing_point(log):
    """Interpolate the touchdown position (altitude = 0) from the last two samples."""
    h = log["true_h"]
    f = h[-2] / (h[-2] - h[-1]) if len(h) > 1 and h[-2] != h[-1] else 1.0
    N = log["true_pN"][-2] + f * (log["true_pN"][-1] - log["true_pN"][-2])
    E = log["true_pE"][-2] + f * (log["true_pE"][-1] - log["true_pE"][-2])
    return float(N), float(E)


def summarize(log, cfg: Config):
    """Key performance numbers for one run."""
    N, E = landing_point(log)
    tN, tE = cfg.target
    t = log["t"]
    m = t > 30.0                                   # skip the convergence transient
    d = lambda a, b: log[a][m] - log[b][m]
    wind_err = np.hypot(d("est_wN", "true_wN"), d("est_wE", "true_wE"))
    return dict(
        flight_time_s=float(t[-1]),
        landing_N=N, landing_E=E,
        miss_distance_m=float(np.hypot(N - tN, E - tE)),
        wind_rmse_mps=float(np.sqrt(np.mean(wind_err ** 2))),
        airspeed_rmse_mps=float(np.sqrt(np.mean(d("est_Va", "true_Va") ** 2))),
        heading_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(wrap_pi(d("est_psi", "true_psi")) ** 2)))),
        descent_rmse_mps=float(np.sqrt(np.mean(d("est_vd", "true_vd") ** 2))),
        altitude_rmse_m=float(np.sqrt(np.mean(d("est_h", "true_h") ** 2))),
        mean_abs_error_deg=float(np.rad2deg(np.mean(log["abs_err"]))),
    )
