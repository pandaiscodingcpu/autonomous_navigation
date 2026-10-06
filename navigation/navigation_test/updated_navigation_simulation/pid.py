"""PID controller for the heading loop (angle-aware, with anti-windup)."""
import numpy as np
from utils import wrap_pi


class PID:
    def __init__(self, kp, ki, kd, dt, out_limit=1.0, i_limit=0.3, d_tau=0.15):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.out_limit = out_limit
        self.i_limit = i_limit
        self.alpha = dt / (d_tau + dt)        # derivative low-pass coefficient
        self.reset()

    def reset(self):
        self.i = 0.0
        self.d_filt = 0.0
        self.prev_err = None
        self.p_term = self.i_term = self.d_term = 0.0

    def update(self, error):
        """error: signed angle error [rad]. Returns servo command in [-out_limit, out_limit]."""
        # derivative of the (wrapped) error, low-pass filtered
        if self.prev_err is None:
            de = 0.0
        else:
            de = wrap_pi(error - self.prev_err) / self.dt
        self.prev_err = error
        self.d_filt += self.alpha * (de - self.d_filt)

        p = self.kp * error
        d = self.kd * self.d_filt
        i_new = self.i + self.ki * error * self.dt

        u_unsat = p + i_new + d
        u = float(np.clip(u_unsat, -self.out_limit, self.out_limit))

        # anti-windup: only keep integrating when not saturated,
        # or when the error is pulling the output back out of saturation
        if u == u_unsat or np.sign(error) != np.sign(u_unsat):
            self.i = float(np.clip(i_new, -self.i_limit, self.i_limit))

        self.p_term, self.i_term, self.d_term = p, self.i, d
        return u
