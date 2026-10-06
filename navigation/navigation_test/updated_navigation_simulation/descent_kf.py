"""1D Kalman filter for altitude and descent velocity (no dh/dt differencing).

State  x = [h, vd]
  h  : altitude [m]
  vd : descent speed, positive downward [m/s]   (dh/dt = -vd)

Model: constant descent speed driven by white-noise acceleration, i.e.
  h[k+1]  = h[k] - vd[k] dt - 0.5 a dt^2
  vd[k+1] = vd[k] + a dt
Measurement: barometric altitude z = h + noise.
"""
import numpy as np


class DescentKF:
    def __init__(self, dt, accel_sigma=0.3, baro_sigma=1.0):
        self.dt = dt
        self.F = np.array([[1.0, -dt],
                           [0.0, 1.0]])
        g = np.array([[-0.5 * dt * dt], [dt]])
        self.Q = accel_sigma ** 2 * (g @ g.T)
        self.H = np.array([[1.0, 0.0]])
        self.R = baro_sigma ** 2
        self.x = np.zeros(2)
        self.P = np.diag([baro_sigma ** 2, 3.0 ** 2])

    def reset(self, h0, vd0):
        self.x = np.array([h0, vd0], float)
        self.P = np.diag([self.R, 3.0 ** 2])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, h_meas):
        y = h_meas - self.x[0]                       # innovation
        S = self.P[0, 0] + self.R
        K = self.P[:, 0] / S
        self.x = self.x + K * y
        I_KH = np.eye(2) - np.outer(K, self.H[0])    # Joseph form
        self.P = I_KH @ self.P @ I_KH.T + self.R * np.outer(K, K)
        return y, S

    @property
    def h(self):
        return self.x[0]

    @property
    def vd(self):
        return self.x[1]
