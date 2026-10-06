"""Truth wind model: mean wind with altitude shear and veer, plus correlated gusts."""
import numpy as np


class WindField:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.gust = np.zeros(2)

    def mean(self, h):
        c = self.cfg
        km = h / 1000.0
        w0 = np.asarray(c.wind_ground, float)
        speed = np.linalg.norm(w0) * (1.0 + c.wind_shear_per_km * km)
        direction = np.arctan2(w0[1], w0[0]) + np.deg2rad(c.wind_rot_deg_per_km) * km
        return speed * np.array([np.cos(direction), np.sin(direction)])

    def at(self, h, dt):
        """Wind velocity (N, E) [m/s] at altitude h; advances the gust process by dt."""
        c = self.cfg
        a = np.exp(-dt / c.gust_tau)
        self.gust = a * self.gust + c.gust_sigma * np.sqrt(1 - a * a) * self.rng.normal(size=2)
        return self.mean(h) + self.gust
