"""Navigation: bearing/distance to target and wind-compensated heading command."""
from dataclasses import dataclass
import numpy as np
from utils import wrap_pi


@dataclass
class NavOutput:
    dist: float       # distance to target [m]
    bearing: float    # line-of-sight bearing to target [rad]
    psi_cmd: float    # commanded heading (bearing, crab-corrected if enabled) [rad]
    error: float      # signed error angle  psi_cmd - psi  in [-pi, pi]
    abs_error: float  # |error|


class Navigator:
    def __init__(self, target, use_wind_comp=True, hold_radius=3.0):
        self.target = np.asarray(target, float)
        self.use_wind_comp = use_wind_comp
        self.hold_radius = hold_radius
        self._last_cmd = None

    def update(self, pos, psi, wind, va):
        dN, dE = self.target - np.asarray(pos)
        dist = float(np.hypot(dN, dE))
        bearing = float(np.arctan2(dE, dN))

        if dist < self.hold_radius and self._last_cmd is not None:
            psi_cmd = self._last_cmd                 # too close: bearing is just noise
        else:
            psi_cmd = bearing
            if self.use_wind_comp:
                # wind component perpendicular to the line of sight (positive = towards the right)
                w_perp = -wind[0] * np.sin(bearing) + wind[1] * np.cos(bearing)
                ratio = np.clip(w_perp / max(va, 1.0), -0.95, 0.95)
                # point upwind by the crab angle so the ground track runs at the target:
                #   Va*sin(psi - bearing) + w_perp = 0
                psi_cmd = bearing - np.arcsin(ratio)
            psi_cmd = float(wrap_pi(psi_cmd))
        self._last_cmd = psi_cmd

        err = float(wrap_pi(psi_cmd - psi))
        return NavOutput(dist, bearing, psi_cmd, err, abs(err))
