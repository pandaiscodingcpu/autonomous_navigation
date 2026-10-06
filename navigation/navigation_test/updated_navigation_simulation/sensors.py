"""Noisy sensor models (gyro, GPS, magnetometer, barometer)."""
import numpy as np
from utils import wrap_pi


class Sensors:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.bias = cfg.gyro_bias            # true gyro bias (random walk)

    def gyro(self, r_true, dt):
        c = self.cfg
        self.bias += self.rng.normal(0.0, c.gyro_bias_walk * np.sqrt(dt))
        return r_true + self.bias + self.rng.normal(0.0, c.gyro_sigma)

    def gps(self, pN, pE, vN, vE):
        c = self.cfg
        n = self.rng.normal
        return (pN + n(0, c.gps_pos_sigma), pE + n(0, c.gps_pos_sigma),
                vN + n(0, c.gps_vel_sigma), vE + n(0, c.gps_vel_sigma))

    def mag(self, psi):
        return float(wrap_pi(psi + self.rng.normal(0.0, np.deg2rad(self.cfg.mag_sigma_deg))))

    def baro(self, h):
        return h + self.rng.normal(0.0, self.cfg.baro_sigma)
