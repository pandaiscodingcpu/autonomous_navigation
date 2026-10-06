"""Truth plant: para-glider with servo lag, canopy yaw-rate lag and wind drift."""
import numpy as np
from utils import wrap_pi


class Parafoil:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pN, self.pE = cfg.start_pos
        self.h = cfg.start_alt
        self.psi = float(wrap_pi(np.deg2rad(cfg.start_heading_deg)))
        self.u = 0.0                                 # actual servo position
        self.r = 0.0                                 # yaw rate [rad/s]
        self.Va = cfg.va0
        self.sink = cfg.va0 / cfg.glide_ratio
        self.vN = self.vE = 0.0

    def step(self, u_cmd, wind, dt):
        c = self.cfg
        u_cmd = float(np.clip(u_cmd, -1.0, 1.0))
        self.u += (u_cmd - self.u) * dt / c.tau_servo                # servo lag
        self.r += (c.r_max * self.u - self.r) * dt / c.tau_canopy    # canopy response
        self.psi = float(wrap_pi(self.psi + self.r * dt))

        a = abs(self.u)
        self.Va = c.va0 * (1.0 - c.va_turn_loss * a)
        self.sink = self.Va / c.glide_ratio * (1.0 + c.sink_turn_gain * a)

        self.vN = self.Va * np.cos(self.psi) + wind[0]
        self.vE = self.Va * np.sin(self.psi) + wind[1]
        self.pN += self.vN * dt
        self.pE += self.vE * dt
        self.h -= self.sink * dt
