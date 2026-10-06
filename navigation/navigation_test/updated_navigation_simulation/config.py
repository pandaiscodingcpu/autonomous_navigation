"""All tunable parameters of the CanSat para-glider simulation in one place.

Frame: local North/East/Up, origin at the release point.
Heading psi is measured clockwise from North [rad].
Servo command u in [-1, 1]: +1 = full right turn (psi increases).
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ---------------- timing ----------------
    dt: float = 0.01          # base loop (gyro + EKF predict) = 100 Hz
    ctrl_div: int = 2         # controller every 2 steps  -> 50 Hz
    gps_div: int = 10         # GPS every 10 steps        -> 10 Hz
    mag_div: int = 5          # magnetometer              -> 20 Hz
    baro_div: int = 5         # barometer / altitude      -> 20 Hz
    t_max: float = 900.0      # safety stop [s]

    # ---------------- mission ----------------
    start_alt: float = 1000.0             # [m]
    start_pos: tuple = (0.0, 0.0)         # (North, East) [m]
    start_heading_deg: float = 30.0       # initial heading [deg]
    target: tuple = (-45.0, 50.0)         # landing target (North, East) [m]

    # ---------------- para-glider (truth model) ----------------
    va0: float = 8.0               # nominal horizontal airspeed [m/s]
    glide_ratio: float = 2.0       # G/D: horizontal speed / sink speed
    r_max: float = 0.5             # yaw rate at full servo deflection [rad/s]
    tau_servo: float = 0.25        # servo response time constant [s]
    tau_canopy: float = 0.8        # canopy yaw-rate response time constant [s]
    va_turn_loss: float = 0.10     # airspeed lost at full deflection (fraction)
    sink_turn_gain: float = 0.10   # extra sink at full deflection (fraction)

    # ---------------- wind (truth model) ----------------
    wind_ground: tuple = (2.5, -1.5)   # mean wind at ground (N, E) [m/s]
    wind_shear_per_km: float = 0.6     # speed increase per km of altitude (fraction)
    wind_rot_deg_per_km: float = 25.0  # direction veer per km of altitude [deg]
    gust_sigma: float = 0.4            # gust std per axis [m/s]
    gust_tau: float = 10.0             # gust correlation time [s]

    # ---------------- sensors (truth noise) ----------------
    gyro_sigma: float = 0.01           # [rad/s] per sample
    gyro_bias: float = 0.02            # initial gyro bias [rad/s]
    gyro_bias_walk: float = 1e-4       # [rad/s^2/sqrt(Hz)]
    gps_pos_sigma: float = 2.0         # [m]
    gps_vel_sigma: float = 0.1         # [m/s]
    mag_sigma_deg: float = 3.0         # [deg]
    baro_sigma: float = 1.0            # [m]

    # ---------------- estimator tuning ----------------
    ekf_gyro_noise: float = 0.005
    ekf_gyro_bias_walk: float = 1e-4
    ekf_va_walk: float = 0.2
    ekf_wind_walk: float = 0.05
    ekf_pos_noise: float = 0.05
    ekf_alt_noise: float = 0.1
    glide_ratio_est: float = 2.0       # G/D used for the airspeed pseudo-measurement
    airspeed_sigma: float = 1.0        # trust in that pseudo-measurement [m/s]
    dkf_accel_sigma: float = 0.3       # 1D descent KF: sink-rate "acceleration" noise [m/s^2]

    # ---------------- navigation + PID ----------------
    use_wind_comp: bool = True         # crab-angle correction from the wind estimate
    hold_radius: float = 3.0           # inside this radius keep the last command [m]
    kp: float = 1.5
    ki: float = 0.02
    kd: float = 0.6
    d_tau: float = 0.15                # derivative low-pass time constant [s]
    i_limit: float = 0.3               # integrator clamp (in servo units)
